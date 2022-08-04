from pathy import dataclass
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Optional, List

from einops.layers.torch import Rearrange
from einops import rearrange


from .recurrent_decoder import *
from .VM.fast_guided_filter import FastGuidedFilterRefiner
from .VM.deep_guided_filter import DeepGuidedFilterRefiner

from .backbone import *
from .basic_block import *
from .bottleneck_fusion import *
from .trimap_fusion import *
from .decoder import *
from .util import collaborate_fuse
from .module import *

class STCNEncoder(nn.Module):
    def __init__(
        self, 
        backbone_arch='mobilenetv3_large_100', 
        backbone_pretrained=True, 
        out_indices=(0, 1, 2, 3),
        ch_key=32
        ):
        super().__init__()

        self.key_encoder  = Backbone(backbone_arch, backbone_pretrained, out_indices, in_chans=3)
        self.value_encoder  = Backbone(backbone_arch, backbone_pretrained, [3], in_chans=4)
        self.channels = self.key_encoder.channels
        self.key_proj = Projection(self.channels[-1], ch_key)
        self.feat_fuse = FeatureFusion3(self.channels[-1]*2, self.channels[-1])
        self.memory = MemoryReader()
        # self.ch_value = ch_value

    def forward(self, qimgs, mimgs, masks):
        q = qimgs.size(1)
        m = mimgs.size(1)
        feats, keys = self.encode_key(torch.cat([qimgs, mimgs], dim=1))
        qk, mk = keys.split((q, m), dim=2) # b, ch_key, t, h, w

        mv = self.encode_value(mimgs, masks, feats[-1][:, q:]) # b, c, t, h, w
        mv = self.query_value(qk, mk, mv)

        feats_q = [f[:, :q] for f in feats] # b, t_q, ch_feat_i, h, w
        return feats_q, mv

    def encode_key(self, imgs):
        feats = self.key_encoder(imgs)
        key = self.key_proj(feats[3]).transpose(1, 2)
        return feats, key

    def encode_value(self, imgs, masks, f16):
        val = self.value_encoder(torch.cat([imgs, masks], dim=2))[0]
        val = self.feat_fuse(torch.cat([val, f16], dim=2)).transpose(1, 2) # b, c, t, h, w
        return val
        
    def query_value(self, qk, mk, mv): 
        A = self.memory.get_affinity(mk, qk)
        mv = self.memory.readout(A, mv)
        return mv

class STCNFuseMatting(nn.Module):
    def __init__(
        self, 
        backbone_arch='mobilenetv3_large_100', 
        backbone_pretrained=True, 
        ch_bottleneck=128,
        ch_key=32,
        ch_seg=[96, 48, 16],
        ch_mat=32,
        seg_gru=ConvGRU,
        refiner='deep_guided_filter',
        trimap_fusion='default',
        bottleneck_fusion='default',
    ):
        super().__init__()
        assert refiner in ['fast_guided_filter', 'deep_guided_filter']

        # Encoder
        self.backbone = Backbone(backbone_arch, backbone_pretrained, (0, 1, 2, 3), in_chans=3)
        self.trimap_fuse = {
            'default': TrimapGatedFusion,
            'naive': TrimapNaiveFusion,
            'bn': TrimapGatedFusionBN,
            'fullres': TrimapGatedFusionFullRes,
            'intrimap_only': TrimapGatedFusionInTrimapOnly,
            'intrimap_only_fullres': TrimapGatedFusionInTrimapOnlyFullres,
            'small': TrimapGatedFusionSmall,
            'fullgate': TrimapGatedFusionFullGate,
        }[trimap_fusion](self.backbone.channels)
        
        self.bottleneck_fuse = BottleneckFusion(self.backbone.channels[-1], ch_key, self.backbone.channels[-1], ch_bottleneck)

        self.feat_channels = list(self.backbone.channels)
        self.feat_channels[-1] = ch_bottleneck

        self.avgpool = AvgPool(3)
        
        # Decoder
        self.ch_seg = ch_seg # 8, 4, out
        self.seg_decoder = SegmentationDecoderTo4x_bottleneck_gru(
            self.feat_channels, self.ch_seg, gru=seg_gru)
        self.seg_project = Projection(self.ch_seg[-1], 3)

        self.ch_mat = ch_mat
        self.mat_decoder = MattingDecoderFrom4x(
            self.feat_channels, 
            [ch_bottleneck] + self.ch_seg, 
            ch_feat=self.ch_mat, ch_out=self.ch_mat, gru=ConvGRU)
        self.mat_project = Projection(self.ch_mat, 1)
        self.default_rec = [self.seg_decoder.default_rec, self.mat_decoder.default_rec, None]
        # print(self.default_rec)
        if refiner == 'deep_guided_filter':
            self.refiner = DeepGuidedFilterRefiner(self.ch_mat)
        else:
            self.refiner = FastGuidedFilterRefiner()

    def forward(self, 
        qimgs: Tensor, mimgs: Tensor, masks: Tensor,
        rec_seg = None,
        rec_mat = None,
        rec_bottleneck = None,
        downsample_ratio: float = 1,
        segmentation_pass: bool = False
    ):
        if rec_mat is None:
            rec_seg, rec_mat, rec_bottleneck = self.default_rec
        is_refine, qimg_sm, mimg_sm, mask_sm = self._smaller_input(qimgs, mimgs, masks, downsample_ratio)

        # Encode
        q = qimg_sm.size(1)
        feats = self.backbone(torch.cat([qimg_sm, mimg_sm], dim=1))
        # zip func is unavailable in torch.jit
        feats_q = [f[:, :q] for f in feats] # b, t_q, ch_feat_i, h, w
        feats_m = [f[:, q:] for f in feats] # b, t_q, ch_feat_i, h, w

        value_m = self.trimap_fuse(mimg_sm, mask_sm, feats_m) # b, c, t, h, w
        feats_q[-1], rec_bottleneck = self.bottleneck_fuse(feats_q[-1], feats_m[-1], value_m, rec_bottleneck)
    
        return self.decode(qimgs, qimg_sm, feats_q, segmentation_pass, is_refine, rec_seg, rec_mat, rec_bottleneck)

    def forward_with_memory(self, 
        qimgs: Tensor, m_feat16: Tensor, m_value: Tensor,
        rec_seg = None,
        rec_mat = None,
        rec_bottleneck = None,
        downsample_ratio: float = 1,
        segmentation_pass: bool = False
    ):
        if rec_mat is None:
            rec_bottleneck, rec_seg, rec_mat = self.default_rec
        
        if is_refine := (downsample_ratio != 1):
            qimg_sm = self._interpolate(qimgs, scale_factor=downsample_ratio)
        else:
            qimg_sm = qimgs

        # Encode
        feats_q = self.backbone(qimg_sm)
        feats_q[-1], rec_bottleneck = self.bottleneck_fuse(feats_q[-1], m_feat16, m_value, rec_bottleneck)
    
        return self.decode(qimgs, qimg_sm, feats_q, segmentation_pass, is_refine, rec_seg, rec_mat, rec_bottleneck)

    def decode(self, 
        qimgs, qimg_sm, feats_q, 
        segmentation_pass,
        is_refine,
        rec_seg = None,
        rec_mat = None,
        rec_bottleneck = None,
    ):
        # Decode
        qimg_sm_avg = self.avgpool(qimg_sm) # 2, 4, 8
        # Segmentation
        hid, *remain = self.seg_decoder(
            qimg_sm, qimg_sm_avg[1], qimg_sm_avg[2],
            *feats_q[2:], 
            *rec_seg)
        # hid, rec1, rec2 ..., inter-feats
        rec_seg, feat_seg = remain[:-1], remain[-1]
        out_seg = self.seg_project(hid)
        out_seg = F.interpolate(out_seg.flatten(0, 1), scale_factor=(4, 4), mode='bilinear').unflatten(0, out_seg.shape[:2])
        if segmentation_pass:
            return [out_seg, rec_seg]

        # Matting
        hid, *remain = self.mat_decoder(
            qimg_sm, qimg_sm_avg[0], 
            *feats_q[:2],
            feat_seg[0], feat_seg[2],
            *rec_mat
        )
        rec_mat, feats = remain[:-1], remain[-1]
        out_mat = torch.sigmoid(self.mat_project(hid))
        
        if is_refine:
            # fgr_residual, pha = self.refiner(src, src_sm, fgr_residual, pha, hid)
            pass

        return [out_seg, out_mat, collaborate_fuse(out_seg, out_mat), [rec_seg, rec_mat, rec_bottleneck], feats]

    def _smaller_input(self,
        query_img: Tensor,
        memory_img: Tensor,
        memory_mask: Tensor,
        downsample_ratio = 1
    ):
        if is_refine := (downsample_ratio != 1):
            qimg_sm = self._interpolate(query_img, scale_factor=downsample_ratio)
            mimg_sm = self._interpolate(memory_img, scale_factor=downsample_ratio)
            mask_sm = self._interpolate(memory_mask, scale_factor=downsample_ratio)
        else:
            qimg_sm = query_img
            mimg_sm = memory_img
            mask_sm = memory_mask
        return is_refine, qimg_sm, mimg_sm, mask_sm

    def encode_key(self, imgs):
        feats = self.backbone(imgs)
        key = self.trimap_fuse(feats[3]).transpose(1, 2)
        return feats, key

    def encode_imgs_to_value(self, imgs, masks):
        feat = self.backbone(imgs)
        values = self.trimap_fuse(imgs, masks, feat) # b, c, t, h, w
        return feat[-1], values

    def _interpolate(self, x: Tensor, scale_factor: float):
        if x.ndim == 5:
            B, T = x.shape[:2]
            x = F.interpolate(x.flatten(0, 1), scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
            x = x.unflatten(0, (B, T))
        else:
            x = F.interpolate(x, scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
        return x

class STCNFuseMatting_gru_before_fuse(STCNFuseMatting):
    def __init__(self, backbone_arch='mobilenetv3_large_100', backbone_pretrained=True, ch_bottleneck=128, ch_key=32, seg_gru=ConvGRU, refiner='deep_guided_filter'):
        super().__init__(backbone_arch, backbone_pretrained, ch_bottleneck, ch_key, seg_gru, refiner)
        del self.bottleneck_fuse
        self.bottleneck_fuse = BottleneckFusionGRU(
            self.backbone.channels[-1], ch_key, self.backbone.channels[-1], ch_bottleneck)

        del self.seg_decoder
        self.seg_decoder = SegmentationDecoderTo4x(
            self.feat_channels, self.ch_seg, gru=seg_gru)
        

class STCNFuseMatting_big(STCNFuseMatting):
    def __init__(self):
        super().__init__(ch_seg=[96, 64, 32], ch_mat=32)
    
class STCNFuseMatting_fullres_mat(STCNFuseMatting):
    def __init__(self, backbone_arch='mobilenetv3_large_100', backbone_pretrained=True, ch_bottleneck=128, ch_key=32, ch_seg=[96, 48, 16], ch_mat=32, seg_gru=ConvGRU, refiner='deep_guided_filter', trimap_fusion='default', bottleneck_fusion='default'):
        super().__init__(backbone_arch, backbone_pretrained, ch_bottleneck, ch_key, ch_seg, ch_mat, seg_gru, refiner, trimap_fusion, bottleneck_fusion)
    
        del self.mat_decoder
        self.mat_decoder = MattingDecoderFrom4x_fullres_chattn(
            self.feat_channels, 
            [ch_bottleneck] + self.ch_seg, 
            ch_feat=self.ch_mat, ch_out=self.ch_mat, gru=ConvGRU
        )

class STCNFuseMatting_SameDec(STCNFuseMatting):
    def __init__(self, backbone_arch='mobilenetv3_large_100', backbone_pretrained=True, ch_bottleneck=128, ch_key=32, ch_seg=[96, 48, 16], ch_mat=32, seg_gru=ConvGRU, refiner='deep_guided_filter', trimap_fusion='default', bottleneck_fusion='default'):
        super().__init__(backbone_arch, backbone_pretrained, ch_bottleneck, ch_key, ch_seg, ch_mat, seg_gru, refiner, trimap_fusion, bottleneck_fusion)
    
        del self.mat_decoder
        del self.seg_decoder
        del self.mat_project
        del self.seg_project

        ch_decode = [96, 48, 32, 16]
        self.mat_decoder = RecurrentDecoder(self.feat_channels, ch_decode)
        self.seg_decoder = RecurrentDecoder(self.feat_channels, ch_decode)

        self.mat_project = Projection(ch_decode[-1], 1)
        self.seg_project = Projection(ch_decode[-1], 3)
        self.default_rec = [[None]*4]*2 + [None]

    def decode(self, 
        qimgs, qimg_sm, feats_q, 
        segmentation_pass,
        is_refine,
        rec_seg = None,
        rec_mat = None,
        rec_bottleneck = None,
    ):
        # Decode
        qimg_sm_avg = self.avgpool(qimg_sm) # 2, 4, 8
        # Segmentation
        hid, *remain = self.seg_decoder(qimg_sm, *feats_q, *rec_seg)
        # hid, rec1, rec2 ..., inter-feats
        rec_seg, feat_seg = remain[:-1], remain[-1]
        out_seg = self.seg_project(hid)
        if segmentation_pass:
            return [out_seg, rec_seg]

        # Matting
        hid, *remain = self.seg_decoder(qimg_sm, *feats_q, *rec_mat)
        rec_mat, feats = remain[:-1], remain[-1]
        out_mat = torch.sigmoid(self.mat_project(hid))
        
        if is_refine:
            # fgr_residual, pha = self.refiner(src, src_sm, fgr_residual, pha, hid)
            pass

        return [out_seg, out_mat, collaborate_fuse(out_seg, out_mat), [rec_seg, rec_mat, rec_bottleneck], feats]

class STCNEncoder(nn.Module):
    def __init__(
        self, 
        backbone_arch='mobilenetv3_large_100', 
        backbone_pretrained=True, 
        out_indices=(0, 1, 2, 3),
        ch_key=32
        ):
        super().__init__()

        self.key_encoder  = Backbone(backbone_arch, backbone_pretrained, out_indices, in_chans=3)
        self.value_encoder  = Backbone(backbone_arch, backbone_pretrained, [3], in_chans=4)
        self.channels = self.key_encoder.channels
        self.key_proj = Projection(self.channels[-1], ch_key)
        self.feat_fuse = FeatureFusion3(self.channels[-1]*2, self.channels[-1])
        self.memory = MemoryReader()
        # self.ch_value = ch_value

    def forward(self, qimgs, mimgs, masks):
        q = qimgs.size(1)
        m = mimgs.size(1)
        feats, keys = self.encode_key(torch.cat([qimgs, mimgs], dim=1))
        qk, mk = keys.split((q, m), dim=2) # b, ch_key, t, h, w

        mv = self.encode_value(mimgs, masks, feats[-1][:, q:]) # b, c, t, h, w
        mv = self.query_value(qk, mk, mv)

        feats_q = [f[:, :q] for f in feats] # b, t_q, ch_feat_i, h, w
        return feats_q, mv

    def encode_key(self, imgs):
        feats = self.key_encoder(imgs)
        key = self.key_proj(feats[3]).transpose(1, 2)
        return feats, key

    def encode_value(self, imgs, masks, f16):
        val = self.value_encoder(torch.cat([imgs, masks], dim=2))[0]
        val = self.feat_fuse(torch.cat([val, f16], dim=2)).transpose(1, 2) # b, c, t, h, w
        return val
        
    def query_value(self, qk, mk, mv): 
        A = self.memory.get_affinity(mk, qk)
        mv = self.memory.readout(A, mv)
        return mv

class STCNGatedEncoder(STCNEncoder):
    def __init__(self, backbone_arch='mobilenetv3_large_100', backbone_pretrained=True, out_indices=(0, 1, 2, 3), ch_key=32):
        super().__init__(backbone_arch, backbone_pretrained, out_indices, ch_key)
    
        del self.value_encoder
        del self.feat_fuse
        self.avg2d = AvgPool(num=3)
        self.gated_convs = nn.ModuleList([
            nn.Sequential(
                GatedConv2d(self.channels[i]+1+(self.channels[i-1] if i > 0 else 0), self.channels[i], 3, 1, 1),
                nn.Conv2d(self.channels[i], self.channels[i], 3, 1, 1),
                nn.LeakyReLU(0.2),
                nn.AvgPool2d((2, 2)),
            )
            for i in range(3)
        ])
        self.feat_fuse = FeatureFusion3(self.channels[-1]+self.channels[-2], self.channels[-1])

    def forward(self, qimgs, mimgs, masks):
        q = qimgs.size(1)
        m = mimgs.size(1)
        feats, keys = self.encode_key(torch.cat([qimgs, mimgs], dim=1))
        qk, mk = keys.split((q, m), dim=2) # b, ch_key, t, h, w
        feats_q, feats_m = zip(*[f.split([q, m], dim=1) for f in feats])

        mv = self.encode_value(masks, feats_m) # b, c, t, h, w
        mv = self.query_value(qk, mk, mv)

        return feats_q, mv

    def encode_value(self, masks, feats):
        b, t = feats[0].shape[:2]
        masks_sm = self.avg2d(masks)
        mf = [torch.cat([feats[i], masks_sm[i]], dim=2).flatten(0, 1) for i in range(3)]
        for i in range(3):
            val = mf[0] if i == 0 else torch.cat([val, mf[i]], dim=1) 
            val = self.gated_convs[i](val)
        val = val.unflatten(0, (b, t))
        val = self.feat_fuse(torch.cat([val, feats[-1]], dim=2)).transpose(1, 2) # b, c, t, h, w
        return val

class STCNGatedEncoder2(STCNEncoder):
    def __init__(self, backbone_arch='mobilenetv3_large_100', backbone_pretrained=True, out_indices=(0, 1, 2, 3), ch_key=32):
        super().__init__(backbone_arch, backbone_pretrained, out_indices, ch_key)
        del self.feat_fuse
        del self.value_encoder
        self.avg2d = AvgPool(num=4)
        self.gated_convs = nn.ModuleList([
            nn.Sequential(
                GatedConv2d(self.channels[i]+1+(self.channels[i] if i > 0 else 0), self.channels[min(3, i+1)], 3, 1, 1),
                nn.Conv2d(self.channels[min(3, i+1)], self.channels[min(3, i+1)], 3, 1, 1),
                nn.LeakyReLU(0.2),
                nn.AvgPool2d((2, 2)) if i < 3 else nn.Identity(),
            )
            for i in range(4)
        ])

    def forward(self, qimgs, mimgs, masks):
        q = qimgs.size(1)
        m = mimgs.size(1)
        feats, keys = self.encode_key(torch.cat([qimgs, mimgs], dim=1))
        qk, mk = keys.split((q, m), dim=2) # b, ch_key, t, h, w
        feats_q, feats_m = zip(*[f.split([q, m], dim=1) for f in feats])

        mv = self.encode_value(masks, feats_m) # b, c, t, h, w
        mv = self.query_value(qk, mk, mv)

        return feats_q, mv

    def encode_value(self, masks, feats):
        b, t = feats[0].shape[:2]
        masks_sm = self.avg2d(masks)
        mf = [torch.cat([feats[i], masks_sm[i]], dim=2).flatten(0, 1) for i in range(4)]
        for i in range(4):
            val = mf[0] if i == 0 else torch.cat([val, mf[i]], dim=1) 
            val = self.gated_convs[i](val)
        val = val.unflatten(0, (b, t))
        return val.transpose(1, 2)

class MattingDecoder(nn.Module):
    def __init__(self,
                feat_channels=[],
                refiner: str = 'deep_guided_filter'
        ):
        super().__init__()
        assert refiner in ['fast_guided_filter', 'deep_guided_filter']
        
        ch_bottleneck = 128
        # print(feat_channels)
        self.aspp = LRASPP(feat_channels[0], ch_bottleneck)
        decoder_ch = list(feat_channels)
        decoder_ch[0] = ch_bottleneck
        decoder_ch.reverse()
        self.decoder = RecurrentDecoder_(decoder_ch, [80, 40, 32, 16])
            
        self.project_mat = Projection(16, 1)
        # self.project_mat = Projection(16, 4)
        self.project_seg = Projection(16, 1)

        if refiner == 'deep_guided_filter':
            self.refiner = DeepGuidedFilterRefiner()
        else:
            self.refiner = FastGuidedFilterRefiner()
        
    def forward(self,
                src: Tensor,
                f4: Tensor,
                f3: Tensor,
                f2: Tensor,
                r1: Optional[Tensor] = None,
                r2: Optional[Tensor] = None,
                r3: Optional[Tensor] = None,
                r4: Optional[Tensor] = None,
                downsample_ratio: float = 1,
                segmentation_pass: bool = False):
        
        if downsample_ratio != 1:
            src_sm = self._interpolate(src, scale_factor=downsample_ratio)
        else:
            src_sm = src
        
        # print(f4.shape, fuse.shape)
        f4 = self.aspp(f4)
        hid, *rec = self.decoder(src_sm, f2, f3, f4, r1, r2, r3, r4)
        
        if not segmentation_pass:
            return [torch.sigmoid(self.project_mat(hid)), rec] # memory of gru
            fgr_residual, pha = self.project_mat(hid).split([3, 1], dim=-3)
            if downsample_ratio != 1:
                fgr_residual, pha = self.refiner(src, src_sm, fgr_residual, pha, hid)
            fgr = fgr_residual + src
            fgr = fgr.clamp(0., 1.)
            pha = pha.clamp(0., 1.)
            return [fgr, pha, *rec]
        else:
            seg = self.project_seg(hid)
            return [seg, rec] # memory of gru

class DualMattingNetwork(nn.Module):
    def __init__(self,
                backbone_arch: str = 'mobilenetv3_large_100',
                backbone_pretrained = True,
                refiner: str = 'deep_guided_filter',
                is_output_fg = False,
                gru=ConvGRU
        ):
        super().__init__()
        assert refiner in ['fast_guided_filter', 'deep_guided_filter']
        
        self.backbone = Backbone(
            backbone_arch=backbone_arch, 
            backbone_pretrained=backbone_pretrained, in_chans=4,
            out_indices=list(range(4))) # only for 4 stages
        
        ch_bottleneck = 128
        self.aspp = LRASPP(self.backbone.channels[-1]*2, ch_bottleneck)
        
        decoder_ch = list(self.backbone.channels)
        decoder_ch[-1] = ch_bottleneck
        self.out_ch = [80, 40, 32, 16]
        self.decoder = RecurrentDecoder(decoder_ch, self.out_ch, gru=gru)
            
        # self.project_mat = Projection(16, 1)
        self.is_output_fg = is_output_fg
        self.project_mat = Projection(self.out_ch[-1], 4 if is_output_fg else 1)
        self.project_seg = Projection(self.out_ch[-1], 1)

        if refiner == 'deep_guided_filter':
            self.refiner = DeepGuidedFilterRefiner()
        else:
            self.refiner = FastGuidedFilterRefiner()
        self.default_rec = [None]*4
    def forward(self,
                query_img: Tensor,
                memory_img: Tensor,
                memory_mask: Tensor,
                r1: Optional[Tensor] = None,
                r2: Optional[Tensor] = None,
                r3: Optional[Tensor] = None,
                r4: Optional[Tensor] = None,
                downsample_ratio: float = 1,
                segmentation_pass: bool = False):
        
        # B, T, C, H, W
        num_query = query_img.size(1)
        imgs = torch.cat([query_img, memory_img], 1) # T
        shape = list(query_img.shape)
        shape[2] = 1
        masks = torch.cat([torch.zeros(shape, device=query_img.device), memory_mask], 1) # T
        src = torch.cat([imgs, masks], 2) # C

        if downsample_ratio != 1:
            src_sm = self._interpolate(src, scale_factor=downsample_ratio)
        else:
            src_sm = src
        
        f1, f2, f3, f4 = self.backbone(src_sm)
        # print(f4.shape, fuse.shape)
        f4_q, f4_m = f4.split([num_query, 1], dim=1)
        f4_m = torch.repeat_interleave(f4_m, num_query, dim=1)
        # f4_m = torch.zeros_like(f4_q) # TODO
        f4 = self.aspp(torch.cat([f4_q, f4_m], 2))
        # print(f4.shape, f3[:, :num_query].shape)
        src_sm = src_sm[:, :num_query, :3] # get the query rgb back
        f1 = f1[:, :num_query]
        f2 = f2[:, :num_query]
        f3 = f3[:, :num_query]
        hid, *rec = self.decoder(src_sm, f1, f2, f3, f4, r1, r2, r3, r4)
        rec, feats = rec[:-1], rec[-1]
        if not segmentation_pass:
            if self.is_output_fg:
                fgr_residual, pha = self.project_mat(hid).split([3, 1], dim=-3)
                src = src[:, :num_query, :3] # get the query rgb back
                if downsample_ratio != 1:
                    fgr_residual, pha = self.refiner(src, src_sm, fgr_residual, pha, hid)
                # fgr = fgr_residual + torch.sigmoid(src)
                # pha = torch.sigmoid(pha)

                fgr = fgr_residual + src
                fgr = fgr.clamp(0., 1.)
                pha = pha.clamp(0., 1.)
                return [pha, fgr, rec, feats]

            return [torch.sigmoid(self.project_mat(hid)), rec, feats] # memory of gru
        else:
            seg = self.project_seg(hid)
            return [seg, rec] # memory of gru

    def _interpolate(self, x: Tensor, scale_factor: float):
        if x.ndim == 5:
            B, T = x.shape[:2]
            x = F.interpolate(x.flatten(0, 1), scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
            x = x.unflatten(0, (B, T))
        else:
            x = F.interpolate(x, scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
        return x


class MattingOutput(nn.Module):
    def __init__(self, in_ch=16, is_output_fg=True, refiner='deep_guided_filter'):
        super().__init__()
        self.is_output_fg = is_output_fg
        self.project_mat = Projection(in_ch, 4 if is_output_fg else 1)
        self.project_seg = Projection(in_ch, 1)

        if refiner == 'deep_guided_filter':
            self.refiner = DeepGuidedFilterRefiner()
        else:
            self.refiner = FastGuidedFilterRefiner()

    def forward(self, img, img_sm, hid, refine=False, segmentation_pass=False):
        if segmentation_pass:
            return [self.project_seg(hid)]
        
        if self.is_output_fg:
            fgr_residual, pha = self.project_mat(hid).split([3, 1], dim=-3)
            if refine:
                fgr_residual, pha = self.refiner(img, img_sm, fgr_residual, pha, hid)

            # fgr = fgr_residual + torch.sigmoid(src)
            # pha = torch.sigmoid(pha)

            fgr = fgr_residual + img
            fgr = fgr.clamp(0., 1.)
            pha = pha.clamp(0., 1.)
            return pha, fgr

        return [torch.sigmoid(self.project_mat(hid))]
        # return [self.project_mat(hid).clamp(0., 1.)]


class FuseTemporal(nn.Module):
    def __init__(self, ch_feats, ch_out):
        super().__init__()
        self.ch_feats_out = list(ch_feats)
        self.ch_feats_out[-1] = ch_out

        # self.attn = nn.ModuleList([
        #     ChannelAttention(ch_feats[i]*2, self.ch_feats_out[i])
        #     for i in range(4)
        # ])
        self.avgpool = AvgPool(num=4)
        
        k = [3, 3, 3]
        self.align_conv = nn.ModuleList([
            AlignDeformConv2d(ch_feats[i], ch_feats[i], ch_feats[i], 
                kernel_size=k[i], stride=1, padding=k[i]//2)
            for i in range(3)
        ])

        self.fuse = nn.ModuleList([
            FeatureFusion3(ch_feats[i]*2, ch_feats[i])
            for i in range(3)
        ])
        # self.fuse.append(LRASPP(ch_feats[-1]*2, ch_out))
        self.aspp = LRASPP(ch_feats[-1]*2, ch_out)

        
        
    def forward(self, feats, feats_prev, masks_prev):
        # num_q = feats_q[0].size(1)
        # B, num_q = feats_q[0].shape[:2]
        # fg_masks = [(m>1e-4).float() for m in self.avgpool(masks_m)]
        # bg_masks = [(m<(1-1e-4)).float() for m in ]
        masks = self.avgpool(masks_prev)
        feats_ret = []
        extra_out = []
        for i in range(3):
            mask = masks[i]
            fc = feats[i]
            fp = feats_prev[i]
            fp = self.align_conv[i](fp, fc)
            f = self.fuse[i](torch.cat([fc, fp], dim=2))
            feats_ret.append(f)
            
        
        feats_ret.append(self.aspp(torch.cat([feats[3], feats_prev[3]], dim=2)))
        return feats_ret, extra_out
    
    @staticmethod
    def weighted_mean(x, w):
        return (x*w).sum(dim=(-2, -1)) / (w.sum(dim=(-2, -1)) + 1e-5)

class MattingPropFramework(nn.Module):
    def __init__(
        self,
        encoder: Backbone,
        middle: FuseTemporal,
        decoder: RecurrentDecoder,
        out: MattingOutput,
    ):
        ''' 
        encoder(qimg, mimg, mmask) -> qfeats, mfeats
        middle(qfeats, mfeats, mmask) -> feats
        decoder(qimg, qimfeats, memory_feat) -> out_feats, memory_feat
        out(qimg, qimg_sm, out_feats, is_refine, seg_pass) -> segmentation or pha, (fgr)
        '''
        super().__init__()
        self.encoder = encoder
        self.middle = middle
        self.decoder = decoder
        self.out = out
        self.is_output_fg = self.out.is_output_fg

    def forward(self,
            imgs: Tensor,
            imgs_prev: Tensor,
            masks_prev: Tensor,
            # memory_feat: list = [None, None, None, None],
            r1=None, r2=None, r3=None, r4=None,
            downsample_ratio: float = 1,
            segmentation_pass: bool = False
        ):
        
        # B, T, C, H, W
        img_sm, mask_sm, prev_sm, feats, feats_prev, is_refine = \
            self.encode_inference(imgs, imgs_prev, masks_prev, downsample_ratio)
        
        feats, middle_out = self.middle(feats, feats_prev, mask_sm)

        # out_feats, *_ = self.decoder(img_sm, *feats, *memory_feat)
        out_feats, *_ = self.decoder(img_sm, *feats, r1, r2, r3, r4)
        memory_feat, inner_feats = _[:-1], _[-1]

        ret = self.out(imgs, img_sm, out_feats, is_refine, segmentation_pass)
        return *ret, memory_feat, middle_out
    
    @staticmethod
    def cur_to_prev(cur):
        return cur[:, [0] + list(range(cur.size(1)-1))]

    def encode_training(self, imgs, masks, downsample_ratio):
        masks = self.cur_to_prev(masks)

        if is_refine := (downsample_ratio != 1):
            img_sm = self._interpolate(imgs, scale_factor=downsample_ratio)
            mask_sm = self._interpolate(masks, scale_factor=downsample_ratio)
        else:
            img_sm = imgs
            mask_sm = masks
        
        prev_sm = self.cur_to_prev(img_sm)
        feats, feats_prev = self.encoder(img_sm, prev_sm, mask_sm)
        return img_sm, mask_sm, prev_sm, feats, feats_prev, is_refine

    def encode_inference(self, imgs, imgs_prev, masks_prev, downsample_ratio):
        if is_refine := (downsample_ratio != 1):
            img_sm = self._interpolate(imgs, scale_factor=downsample_ratio)
            mask_sm = self._interpolate(masks_prev, scale_factor=downsample_ratio)
            prev_sm = self._interpolate(imgs_prev, scale_factor=downsample_ratio)
        else:
            img_sm = imgs
            prev_sm = imgs_prev
            mask_sm = masks_prev
        
        feats, feats_prev = self.encoder(img_sm, prev_sm, mask_sm)
        return img_sm, mask_sm, prev_sm, feats, feats_prev, is_refine

    def _interpolate(self, x: Tensor, scale_factor: float):
        if x.ndim == 5:
            B, T = x.shape[:2]
            x = F.interpolate(x.flatten(0, 1), scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
            x = x.unflatten(0, (B, T))
        else:
            x = F.interpolate(x, scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
        return x

class GFM_FuseVM(nn.Module):
    def __init__(self,
                backbone_arch: str = 'mobilenetv3_large_100',
                backbone_pretrained = True,
                refiner: str = 'deep_guided_filter',
                is_output_fg = False,
                gru=ConvGRU,
                fuse='GFMFuse',
                decoder_wo_img=False,
                out_ch=[80, 40, 32, 16]
        ):
        super().__init__()
        assert refiner in ['fast_guided_filter', 'deep_guided_filter']
        
        self.backbone = RGBMaskEncoder(
            backbone_arch=backbone_arch, 
            backbone_pretrained=backbone_pretrained,
            out_indices=list(range(4))) # only for 4 stages
        

        ch_bottleneck = 128
        self.fuse = {
            'GFMFuse': GFMFuse,
            'GFMFuse2': GFMFuse2,
            'GFMFuse3': GFMFuse3,
        }[fuse](self.backbone.channels, ch_bottleneck)
        self.decoder_ch = decoder_ch = list(self.fuse.ch_feats_out)
        
        self.out_ch = out_ch
        self.decoder_focus = RecurrentDecoder(decoder_ch, self.out_ch, gru=gru, no_img_skip=decoder_wo_img)
        self.is_output_fg = is_output_fg
        self.project_mat = Projection(self.out_ch[-1], 4 if is_output_fg else 1)

        self.decoder_glance = RecurrentDecoder(decoder_ch, self.out_ch, gru=gru, no_img_skip=decoder_wo_img)
        self.project_seg = Projection(self.out_ch[-1], 3)
        self.default_rec = [[None]*4]*2

        if refiner == 'deep_guided_filter':
            self.refiner = DeepGuidedFilterRefiner()
        else:
            self.refiner = FastGuidedFilterRefiner()
        
    def forward(self,
                query_img: Tensor,
                memory_img: Tensor,
                memory_mask: Tensor,
                rec_glance = [None, None, None, None],
                rec_focus = [None, None, None, None],
                downsample_ratio: float = 1,
                segmentation_pass: bool = False):
        
        # B, T, C, H, W
        num_query = query_img.size(1)
        is_refine, qimg_sm, mimg_sm, mask_sm = self._smaller_input(query_img, memory_img, memory_mask)
        
        feats_q, feats_m = self.backbone(qimg_sm, mimg_sm, mask_sm)
        feats_gl, feats_fc = self.fuse(feats_q, feats_m, mask_sm)
        
        hid, *rec = self.decoder_glance(qimg_sm, *feats_gl, *rec_glance)
        rec_glance, feats = rec[:-1], rec[-1]
        out_glance = self.project_seg(hid)
        if segmentation_pass:
            return [out_glance, rec_glance]

        hid, *rec = self.decoder_focus(qimg_sm, *feats_fc, *rec_focus)
        rec_focus, feats = rec[:-1], rec[-1]
        out_focus = torch.sigmoid(self.project_mat(hid))
        return [out_glance, out_focus, self._collaborate(out_glance, out_focus), [rec_glance, rec_focus], feats]
    
    def _smaller_input(self,
        query_img: Tensor,
        memory_img: Tensor,
        memory_mask: Tensor,
        downsample_ratio = 1
    ):
        if is_refine := (downsample_ratio != 1):
            qimg_sm = self._interpolate(query_img, scale_factor=downsample_ratio)
            mimg_sm = self._interpolate(memory_img, scale_factor=downsample_ratio)
            mask_sm = self._interpolate(memory_mask, scale_factor=downsample_ratio)
        else:
            qimg_sm = query_img
            mimg_sm = memory_img
            mask_sm = memory_mask
        return is_refine, qimg_sm, mimg_sm, mask_sm

    def _interpolate(self, x: Tensor, scale_factor: float):
        if x.ndim == 5:
            B, T = x.shape[:2]
            x = F.interpolate(x.flatten(0, 1), scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
            x = x.unflatten(0, (B, T))
        else:
            x = F.interpolate(x, scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
        return x

    @staticmethod
    def _collaborate(out_glance, out_focus):
        val, idx = torch.sigmoid(out_glance).max(dim=2, keepdim=True) # ch
        # (bg, t, fg)
        tran_mask = idx.clone() == 1
        fg_mask = idx.clone() == 2
        return out_focus*tran_mask + fg_mask

class GFMFuse(nn.Module):
    def __init__(self, ch_feats, ch_out):
        super().__init__()
        self.bottleneck = PSP(ch_feats[-1]*2, ch_feats[-1]*2//4, ch_out)
        self.attn = nn.ModuleList([
            ChannelAttention(ch_feats[i]*2, ch_out if i == 3 else ch_feats[i])
            for i in range(4)
        ])

        self.sa = ConvSelfAttention(ch_feats[-1], head=1, patch_size=1, drop_p=0)
        self.avgpool = AvgPool(num=4)
        self.ch_feats_out = list(ch_feats)
        self.ch_feats_out[-1] = ch_out
        
    def forward(self, feats_q, feats_m, masks_m):
        # num_q = feats_q[0].size(1)
        # B, num_q = feats_q[0].shape[:2]
        # fg_masks = [(m>1e-4).float() for m in self.avgpool(masks_m)]
        # bg_masks = [(m<(1-1e-4)).float() for m in ]
        masks = self.avgpool(masks_m)
        feats_glance = []
        feats_focus = []
        for i in range(4):
            fq = feats_q[i]
            fm = feats_m[i]
            mask = masks[i]
            # B, T, 1, H, W
            ch_attn = self.attn[i](torch.cat([
                self.weighted_mean(fm, mask),
                self.weighted_mean(fm, 1-mask)
                ], dim=2))
            ch_attn = ch_attn.view(*ch_attn.shape, 1, 1)
            if i == 3:
                sa, A = self.sa(fq, fm)
                f = self.bottleneck(torch.cat([fq, sa], dim=2)) * ch_attn
                feats_glance.append(f)
                feats_focus.append(f)
            else:
                feats_glance.append(fq*ch_attn)
                feats_focus.append(fq)

        
        return feats_glance, feats_focus
    
    @staticmethod
    def weighted_mean(x, w, keepdim=False):
        return (x*w).sum(dim=(-2, -1), keepdim=keepdim) / (w.sum(dim=(-2, -1), keepdim=keepdim) + 1e-5)

class GFMFuse2(nn.Module):
    def __init__(self, ch_feats, ch_out):
        super().__init__()
        self.fuse = FeatureFusion3(ch_feats[-1]*2, ch_out)
        self.bottleneck = PSP(ch_out, ch_out//4, ch_out)
        self.attn = nn.ModuleList([
            ChannelAttention(ch_feats[i]*2, ch_feats[i])
            for i in range(3)
        ])

        self.sa = ConvSelfAttention(ch_feats[-1], head=2, patch_size=1, drop_p=0)
        self.avgpool = AvgPool(num=4)
        self.ch_feats_out = list(ch_feats)
        self.ch_feats_out[-1] = ch_out
        
    def forward(self, feats_q, feats_m, masks_m):
        # num_q = feats_q[0].size(1)
        # B, num_q = feats_q[0].shape[:2]
        # fg_masks = [(m>1e-4).float() for m in self.avgpool(masks_m)]
        # bg_masks = [(m<(1-1e-4)).float() for m in ]
        masks = self.avgpool(masks_m)
        feats_glance = []
        feats_focus = []
        for i in range(4):
            fq = feats_q[i]
            fm = feats_m[i]
            # B, T, 1, H, W
            if i == 3:
                sa, A = self.sa(fq, fm)
                f = self.fuse(torch.cat([fq, sa], dim=2))
                f = self.bottleneck(f)
                feats_glance.append(f)
                feats_focus.append(f)
            else:
                mask = masks[i]
                ch_attn = self.attn[i](torch.cat([
                    self.weighted_mean(fm, mask),
                    self.weighted_mean(fm, 1-mask)
                    ], dim=2))
                ch_attn = ch_attn.view(*ch_attn.shape, 1, 1)
                feats_glance.append(fq*ch_attn)
                feats_focus.append(fq)

        
        return feats_glance, feats_focus
    
    @staticmethod
    def weighted_mean(x, w, keepdim=False):
        return (x*w).sum(dim=(-2, -1), keepdim=keepdim) / (w.sum(dim=(-2, -1), keepdim=keepdim) + 1e-5)

class GFMFuse3(GFMGatedFuse):
    def __init__(self, ch_feats, ch_out):
        super().__init__(ch_feats, ch_out)
        del self.gated_convs
        
    def forward(self, feats_q, feats_m, masks_m):
        # masks = self.avgpool(masks_m)

        f4_m = feats_m[3]
        f4_q = feats_q[3]
        f4_m, A = self.sa(f4_q, f4_m)

        f4 = self.fuse(torch.cat([f4_q, f4_m], dim=2))
        f4 = self.bottleneck(f4)

        feats = list(feats_q)
        feats[3] = f4
        return feats, feats


class STCN_GFM_VM(nn.Module):
    def __init__(self,
                backbone_arch: str = 'mobilenetv3_large_100',
                backbone_pretrained = True,
                refiner: str = 'deep_guided_filter',
                is_output_fg = False,
                gru=ConvGRU,
                fuse='STCNFuse'
        ):
        super().__init__()
        assert refiner in ['fast_guided_filter', 'deep_guided_filter']
        
        self.backbone = STCNEncoder(
            backbone_arch=backbone_arch, 
            backbone_pretrained=backbone_pretrained,
            out_indices=list(range(4))) # only for 4 stages
        
        ch_bottleneck = 128
        self.fuse = {
            'STCNFuse': STCNFuse,
        }[fuse](self.backbone.channels, ch_bottleneck)
        decoder_ch = list(self.fuse.ch_feats_out)
        
        self.out_ch = [80, 40, 32, 16]
        self.decoder_focus = RecurrentDecoder(decoder_ch, self.out_ch, gru=gru)
        self.is_output_fg = is_output_fg
        self.project_mat = Projection(self.out_ch[-1], 4 if is_output_fg else 1)

        self.decoder_glance = RecurrentDecoder(decoder_ch, self.out_ch, gru=gru)
        self.project_seg = Projection(self.out_ch[-1], 3)


        if refiner == 'deep_guided_filter':
            self.refiner = DeepGuidedFilterRefiner()
        else:
            self.refiner = FastGuidedFilterRefiner()
        
    def forward(self,
                query_img: Tensor,
                memory_img: Tensor,
                memory_mask: Tensor,
                rec_glance = [None, None, None, None],
                rec_focus = [None, None, None, None],
                downsample_ratio: float = 1,
                segmentation_pass: bool = False):
        
        # B, T, C, H, W
        num_query = query_img.size(1)
        if is_refine := (downsample_ratio != 1):
            qimg_sm = self._interpolate(query_img, scale_factor=downsample_ratio)
            mimg_sm = self._interpolate(memory_img, scale_factor=downsample_ratio)
            mask_sm = self._interpolate(memory_mask, scale_factor=downsample_ratio)
        else:
            qimg_sm = query_img
            mimg_sm = memory_img
            mask_sm = memory_mask
        
        
        feats_q, feat_m = self.backbone(qimg_sm, mimg_sm, mask_sm)
        feats_gl, feats_fc = self.fuse(feats_q, feat_m)
        
        hid, *rec = self.decoder_glance(qimg_sm, *feats_gl, *rec_glance)
        rec_glance, feats = rec[:-1], rec[-1]
        out_glance = self.project_seg(hid)
        if segmentation_pass:
            return [out_glance, rec_glance]

        hid, *rec = self.decoder_focus(qimg_sm, *feats_fc, *rec_focus)
        rec_focus, feats = rec[:-1], rec[-1]
        out_focus = torch.sigmoid(self.project_mat(hid))
        return [out_glance, out_focus, self._collaborate(out_glance, out_focus), [rec_glance, rec_focus], feats]

    def _interpolate(self, x: Tensor, scale_factor: float):
        if x.ndim == 5:
            B, T = x.shape[:2]
            x = F.interpolate(x.flatten(0, 1), scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
            x = x.unflatten(0, (B, T))
        else:
            x = F.interpolate(x, scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
        return x

    @staticmethod
    def _collaborate(out_glance, out_focus):
        val, idx = torch.sigmoid(out_glance).max(dim=2, keepdim=True) # ch
        # (bg, t, fg)
        tran_mask = idx.clone() == 1
        fg_mask = idx.clone() == 2
        return out_focus*tran_mask + fg_mask

class GatedSTCN_GFM_VM(STCN_GFM_VM):
    def __init__(self, 
    backbone_arch: str = 'mobilenetv3_large_100', 
    backbone_pretrained=True, 
    refiner: str = 'deep_guided_filter', 
    is_output_fg=False, 
    gru=ConvGRU, 
    fuse='STCNFuse',
    encoder=1
    ):
        super().__init__(backbone_arch, backbone_pretrained, refiner, is_output_fg, gru, fuse)
        assert refiner in ['fast_guided_filter', 'deep_guided_filter']
        
        self.backbone = {
            1: STCNGatedEncoder,
            2: STCNGatedEncoder2
        }[encoder](
            backbone_arch=backbone_arch, 
            backbone_pretrained=backbone_pretrained,
            out_indices=list(range(4))) # only for 4 stages

class STCNFuse(nn.Module):
    def __init__(self, ch_feats, ch_out):
        super().__init__()
        self.feat_fuse = FeatureFusion(ch_feats[-1]*2, ch_feats[-1])
        self.bottleneck = PSP(ch_feats[-1], ch_feats[-1]//4, ch_out)
        
        self.sa = ConvSelfAttention(ch_feats[-1], head=1, patch_size=1, drop_p=0)
        self.avgpool = AvgPool(num=4)
        self.ch_feats_out = list(ch_feats)
        self.ch_feats_out[-1] = ch_out
        
    def forward(self, feats_q, feat_m):
        # feats_glance = []
        # feats_focus = []
        f4 = self.feat_fuse(torch.cat([feats_q[3], feat_m], dim=2))
        f4 = self.bottleneck(f4)
        feats = list(feats_q)
        feats[3] = f4
        return feats, feats
        # return feats_glance, feats_focus

class GFM_GatedFuseVM(GFM_FuseVM):
    def __init__(self, 
        backbone_arch: str = 'mobilenetv3_large_100', 
        backbone_pretrained=True, 
        refiner: str = 'deep_guided_filter', 
        is_output_fg=False, 
        gru=AttnGRU, 
        fuse='GFMGatedFuse',
        decoder_wo_img=False,
        out_ch=[]
        ):
        if len(out_ch) > 0:
            super().__init__(
                backbone_arch, 
                backbone_pretrained, 
                refiner, 
                is_output_fg, 
                gru, 
                decoder_wo_img=decoder_wo_img,
                out_ch=out_ch)
        else:
            super().__init__(
                backbone_arch, 
                backbone_pretrained, 
                refiner, 
                is_output_fg, 
                gru, 
                decoder_wo_img=decoder_wo_img)
        
        self.backbone = RGBEncoder(
            backbone_arch=backbone_arch, 
            backbone_pretrained=backbone_pretrained,
            out_indices=list(range(4))) # only for 4 stages

        ch_bottleneck = self.ch_bottleneck = 128
        self.fuse = {
            'GFMGatedFuse': GFMGatedFuse,
            'sameqk': GFMGatedFuse_sameqk,
            'samekv': GFMGatedFuse_samekv,
            'splitstage': GFMGatedFuse_splitstage,
            'splitfuse': GFMGatedFuse_splitfeatfuse,
            'sameqk_head1': GFMGatedFuse_sameqk_head1,
            'head1': GFMGatedFuse_head1,
            'dropout': lambda *a: GFMGatedFuse(*a, dropout=0.1),
            '3chmask': lambda *a: GFMGatedFuse(*a, ch_mask=3),
            'naivefuse': GFMNaiveFuse,
            'inputmaskonly': GFMGatedFuse_inputmaskonly,
            'bn': GFMGatedFuse_bn,
            'naive_h1sqk': GFMNaiveFuse_h1sqk,
            'grufuse': GFMGatedFuse_grufuse,
            'ff2': GFMGatedFuse_h1sqk_ff2,
        }[fuse](self.backbone.channels, ch_bottleneck)

        self.default_rec = [[None]*4]*2

class GFM_GatedFuseVM_big(GFM_GatedFuseVM):
    def __init__(self):
        out_ch = [128, 64, 64, 32]
        super().__init__(gru=ConvGRU, out_ch=out_ch)
    

class GFM_GatedFuseVM_4xfoucs(GFM_GatedFuseVM):
    def __init__(self, 
        backbone_arch: str = 'mobilenetv3_large_100', 
        backbone_pretrained=True, 
        refiner: str = 'deep_guided_filter', 
        is_output_fg=False, 
        gru=AttnGRU2,
        fuse='GFMGatedFuse'
    ):
        super().__init__(backbone_arch, backbone_pretrained, refiner, is_output_fg, gru, fuse)
        self.decoder_focus = RecurrentDecoder4x(
            self.decoder_ch, 
            self.out_ch[::-1]+[self.decoder_ch[-1]], 
            self.out_ch[-1], ch_feat=32, gru=ConvGRU)
        self.avgpool = AvgPool(1)
        self.default_rec = [[None]*4, [None]]

    def forward(self,
                query_img: Tensor,
                memory_img: Tensor,
                memory_mask: Tensor,
                rec_glance = None,
                rec_focus = None,
                downsample_ratio: float = 1,
                segmentation_pass: bool = False):
        if rec_glance is None:
            rec_glance, rec_focus = self.default_rec

        # B, T, C, H, W
        num_query = query_img.size(1)
        is_refine, qimg_sm, mimg_sm, mask_sm = self._smaller_input(query_img, memory_img, memory_mask, downsample_ratio)
        imgs_sm = self.avgpool(qimg_sm)
        
        feats_q, feats_m = self.backbone(qimg_sm, mimg_sm, mask_sm)
        feats_gl, feats_fc = self.fuse(feats_q, feats_m, mask_sm)
        
        hid, *rec = self.decoder_glance(qimg_sm, *feats_gl, *rec_glance)
        rec_glance, feats = rec[:-1], rec[-1]
        # print([f.shape for f in feats], self.out_ch)
        out_glance = self.project_seg(hid)
        if segmentation_pass:
            return [out_glance, rec_glance]

        hid, *rec = self.decoder_focus(
            qimg_sm, imgs_sm[0], 
            *feats_fc[:3],
            hid, *feats, *rec_focus)
        rec_focus, feats = rec[:-1], rec[-1]
        out_focus = torch.sigmoid(self.project_mat(hid))
        return [out_glance, out_focus, self._collaborate(out_glance, out_focus), [rec_glance, rec_focus], feats]

class GFM_GatedFuseVM_4xfoucs_1(GFM_GatedFuseVM_4xfoucs):
    def __init__(self, 
        backbone_arch: str = 'mobilenetv3_large_100', 
        backbone_pretrained=True, 
        refiner: str = 'deep_guided_filter', 
        is_output_fg=False, 
        gru=AttnGRU2,
        fuse='GFMGatedFuse'
    ):
        super().__init__(backbone_arch, backbone_pretrained, refiner, is_output_fg, gru, fuse)
        del self.decoder_focus
        self.decoder_focus = RecurrentDecoder4x_1(
            self.decoder_ch, 
            self.out_ch[::-1]+[self.decoder_ch[-1]], 
            self.out_ch[-1], ch_feat=32, gru=ConvGRU)
        self.default_rec = [[None]*4, [None]*2]


class GFM_GatedFuseVM_4xfoucs_2(GFM_GatedFuseVM_4xfoucs):
    def __init__(self, 
        backbone_arch: str = 'mobilenetv3_large_100', 
        backbone_pretrained=True, 
        refiner: str = 'deep_guided_filter', 
        is_output_fg=False, 
        gru=AttnGRU2,
        fuse='GFMGatedFuse'
    ):
        super().__init__(backbone_arch, backbone_pretrained, refiner, is_output_fg, gru, fuse)
        del self.decoder_focus
        self.decoder_focus = RecurrentDecoder4x_2(
            self.decoder_ch, 
            self.out_ch[::-1]+[self.decoder_ch[-1]], 
            self.out_ch[-1], ch_feat=32, gru=ConvGRU)
        self.default_rec = [[None]*4, [None]*2]

class GatedVM(nn.Module):
    def __init__(self,
                backbone_arch: str = 'mobilenetv3_large_100',
                backbone_pretrained = True,
                refiner: str = 'deep_guided_filter',
                is_output_fg = False,
                gru=ConvGRU
        ):
        super().__init__()
        assert refiner in ['fast_guided_filter', 'deep_guided_filter']
        
        self.backbone = Backbone(
            backbone_arch=backbone_arch, 
            backbone_pretrained=backbone_pretrained, in_chans=3,
            out_indices=list(range(4))) # only for 4 stages
        
        ch_bottleneck = 128
        self.channels = self.backbone.channels
        self.gated_convs = nn.ModuleList([
            nn.Sequential(
                GatedConv2d(self.channels[i]+1+(self.channels[i-1] if i > 0 else 0), self.channels[i], 3, 1, 1),
                nn.Conv2d(self.channels[i], self.channels[i], 3, 1, 1),
                nn.LeakyReLU(0.2),
            )
            for i in range(4)
        ])

        self.aspp = LRASPP(self.backbone.channels[-1]*2, ch_bottleneck)
        
        decoder_ch = list(self.backbone.channels)
        decoder_ch[-1] = ch_bottleneck
        self.out_ch = [80, 40, 32, 16]
        self.decoder = RecurrentDecoder(decoder_ch, self.out_ch, gru=gru)
        self.avgpool = AvgPool(num=4)
        self.avg = nn.AvgPool2d((2, 2))
        # self.project_mat = Projection(16, 1)
        self.is_output_fg = is_output_fg
        self.project_mat = Projection(self.out_ch[-1], 4 if is_output_fg else 1)
        self.project_seg = Projection(self.out_ch[-1], 1)

        if refiner == 'deep_guided_filter':
            self.refiner = DeepGuidedFilterRefiner()
        else:
            self.refiner = FastGuidedFilterRefiner()
        
    def forward(self,
                query_img: Tensor,
                memory_img: Tensor,
                memory_mask: Tensor,
                r1: Optional[Tensor] = None,
                r2: Optional[Tensor] = None,
                r3: Optional[Tensor] = None,
                r4: Optional[Tensor] = None,
                downsample_ratio: float = 1,
                segmentation_pass: bool = False):
        
        # B, T, C, H, W
        num_query = query_img.size(1)
        num_memory = memory_img.size(1)
        src = torch.cat([query_img, memory_img], 1) # T
        # shape = list(query_img.shape)
        # shape[2] = 1
        # masks = torch.cat([torch.zeros(shape, device=query_img.device), memory_mask], 1) # T
        # src = torch.cat([imgs, masks], 2) # C

        if downsample_ratio != 1:
            src_sm = self._interpolate(src, scale_factor=downsample_ratio)
            mask_sm = self._interpolate(memory_mask, scale_factor=downsample_ratio)
        else:
            src_sm = src
            mask_sm = memory_mask
        
        feats = self.backbone(src_sm)
        feats_q, feats_m = zip(*[f.split([num_query, num_memory], dim=1) for f in feats])

        masks = self.avgpool(mask_sm)
        f4_m = feats_m[0]
        b, t = f4_m.shape[:2]
        f4_m = self.gated_convs[0](torch.cat([f4_m, masks[0]], dim=2).flatten(0, 1))
        for i in range(1, 4):
            f4_m = torch.cat([self.avg(f4_m), feats_m[i].flatten(0, 1), masks[i].flatten(0, 1)],  dim=1)
            f4_m = self.gated_convs[i](f4_m)
        f4_m = f4_m.unflatten(0, (b, t))

        # print(f4.shape, fuse.shape)
        # f4_q, f4_m = f4.split([num_query, 1], dim=1)
        f4_m = torch.repeat_interleave(f4_m, num_query, dim=1)
        # f4_m = torch.zeros_like(f4_q) # TODO
        f4 = self.aspp(torch.cat([feats_q[3], f4_m], 2))
        f1, f2, f3 = feats_q[:3]
        # print(f4.shape, f3[:, :num_query].shape)
        src_sm = src_sm[:, :num_query, :3] # get the query rgb back
        f1 = f1[:, :num_query]
        f2 = f2[:, :num_query]
        f3 = f3[:, :num_query]
        hid, *rec = self.decoder(src_sm, f1, f2, f3, f4, r1, r2, r3, r4)
        rec, feats = rec[:-1], rec[-1]
        if not segmentation_pass:
            if self.is_output_fg:
                fgr_residual, pha = self.project_mat(hid).split([3, 1], dim=-3)
                src = src[:, :num_query, :3] # get the query rgb back
                if downsample_ratio != 1:
                    fgr_residual, pha = self.refiner(src, src_sm, fgr_residual, pha, hid)
                # fgr = fgr_residual + torch.sigmoid(src)
                # pha = torch.sigmoid(pha)

                fgr = fgr_residual + src
                fgr = fgr.clamp(0., 1.)
                pha = pha.clamp(0., 1.)
                return [pha, fgr, rec, feats]

            return [torch.sigmoid(self.project_mat(hid)), rec, feats] # memory of gru
        else:
            seg = self.project_seg(hid)
            return [seg, rec] # memory of gru

    def _interpolate(self, x: Tensor, scale_factor: float):
        if x.ndim == 5:
            B, T = x.shape[:2]
            x = F.interpolate(x.flatten(0, 1), scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
            x = x.unflatten(0, (B, T))
        else:
            x = F.interpolate(x, scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
        return x

class GFM_GatedFuseVM_to4xglance_4xfocus(GFM_GatedFuseVM):
    def __init__(self, 
        backbone_arch: str = 'mobilenetv3_large_100', 
        backbone_pretrained=True, 
        refiner: str = 'deep_guided_filter', 
        is_output_fg=False, 
        gru=AttnGRU2,
        fuse='GFMGatedFuse',
        ch_focus_decoder=32,
    ):
        super().__init__(backbone_arch, backbone_pretrained, refiner, is_output_fg, gru, fuse)
        del self.decoder_glance
        del self.decoder_focus
        del self.project_seg
        del self.project_mat

        self.out_ch = [96, 64, 48, 32]
        # self.out_ch = [80, 40, 32, 16]
        self.decoder_glance = RecurrentDecoderTo4x(self.decoder_ch, self.out_ch[:3], gru=gru)
        self.is_output_fg = is_output_fg
        self.project_seg = Projection(self.out_ch[2], 3)

        self.decoder_focus = RecurrentDecoderFrom4x(
            self.decoder_ch, 
            self.out_ch[::-1]+[self.decoder_ch[-1]], 
            self.out_ch[-1], ch_feat=ch_focus_decoder, gru=ConvGRU)
        self.project_mat = Projection(self.out_ch[-1], 4 if is_output_fg else 1)
        self.avgpool = AvgPool(3)
        self.default_rec = [[None]*3, [None]]

    def forward(self,
                query_img: Tensor,
                memory_img: Tensor,
                memory_mask: Tensor,
                rec_glance = None,
                rec_focus = None,
                downsample_ratio: float = 1,
                segmentation_pass: bool = False):
        if rec_glance is None:
            rec_glance, rec_focus = self.default_rec

        # B, T, C, H, W
        num_query = query_img.size(1)
        is_refine, qimg_sm, mimg_sm, mask_sm = self._smaller_input(query_img, memory_img, memory_mask, downsample_ratio)
        imgs_sm = self.avgpool(qimg_sm)
        
        feats_q, feats_m = self.backbone(qimg_sm, mimg_sm, mask_sm)
        feats_gl, feats_fc = self.fuse(feats_q, feats_m, mask_sm)
        hid, *rec = self.decoder_glance(qimg_sm, imgs_sm[1], imgs_sm[2], feats_gl[2], feats_gl[3], *rec_glance)
        rec_glance, feats = rec[:-1], rec[-1]
        # print([f.shape for f in feats], self.out_ch)
        out_glance = self.project_seg(hid)
        if segmentation_pass:
            return [out_glance, rec_glance]

        hid, *rec = self.decoder_focus(
            qimg_sm, imgs_sm[0], 
            *feats_fc[:2],
            feats[0], feats[2], *rec_focus)
        rec_focus, feats = rec[:-1], rec[-1]
        out_focus = torch.sigmoid(self.project_mat(hid))
        return [out_glance, out_focus, self._collaborate(out_glance, out_focus), [rec_glance, rec_focus], feats]

class GFM_GatedFuseVM_to4xglance_4xfocus_2(GFM_GatedFuseVM_to4xglance_4xfocus):
    def __init__(self, backbone_arch: str = 'mobilenetv3_large_100', backbone_pretrained=True, refiner: str = 'deep_guided_filter', is_output_fg=False, gru=ConvGRU, fuse='GFMGatedFuse'):
        super().__init__(backbone_arch, backbone_pretrained, refiner, is_output_fg, gru, fuse)
        del self.decoder_glance
        del self.decoder_focus
        del self.project_seg
        del self.project_mat

        self.out_ch = [96, 64, 48, 32]
        # self.out_ch = [80, 40, 32, 16]
        self.decoder_glance = RecurrentDecoderTo4x_2(self.decoder_ch, self.out_ch[:3], gru=gru)
        self.is_output_fg = is_output_fg
        self.project_seg = Projection(self.out_ch[2], 3)

        self.decoder_focus = RecurrentDecoderFrom4x_2(
            self.decoder_ch, 
            self.out_ch[::-1]+[self.decoder_ch[-1]], 
            self.out_ch[-1], ch_feat=32, gru=ConvGRU)
        self.project_mat = Projection(self.out_ch[-1], 4 if is_output_fg else 1)
        self.avgpool = AvgPool(3)
        self.default_rec = [[None]*2, [None]*2]

class GFM_GatedFuseVM_to4xglance_4xfocus_4(GFM_GatedFuseVM_to4xglance_4xfocus):
    def __init__(self, backbone_arch: str = 'mobilenetv3_large_100', backbone_pretrained=True, refiner: str = 'deep_guided_filter', is_output_fg=False, gru=AttnGRU2, fuse='GFMGatedFuse'):
        super().__init__(backbone_arch, backbone_pretrained, refiner, is_output_fg, gru, fuse)
        del self.decoder_focus
        del self.project_mat

        # self.out_ch = [96, 64, 48, 32]
        # # self.out_ch = [80, 40, 32, 16]
        # self.decoder_glance = RecurrentDecoderTo4x_2(self.decoder_ch, self.out_ch[:3], gru=gru)
        # self.is_output_fg = is_output_fg
        # self.project_seg = Projection(self.out_ch[2], 3)

        self.decoder_focus = RecurrentDecoderFrom4x_4(
            self.decoder_ch, 
            self.out_ch[::-1]+[self.decoder_ch[-1]], 
            self.out_ch[-1], ch_feat=32, gru=ConvGRU)
        self.project_mat = Projection(self.out_ch[-1], 4 if is_output_fg else 1)
        # self.avgpool = AvgPool(3)
        # self.default_rec = [[None]*2, [None]*2]


class GFM_GatedFuseVM_to4xglance_4xfocus_3(GFM_GatedFuseVM_to4xglance_4xfocus_2):
    def __init__(self, backbone_arch: str = 'mobilenetv3_large_100', backbone_pretrained=True, refiner: str = 'deep_guided_filter', is_output_fg=False, gru=AttnGRU2, fuse='GFMGatedFuse'):
        super().__init__(backbone_arch, backbone_pretrained, refiner, is_output_fg, gru, fuse)
        del self.decoder_glance

        self.decoder_glance = RecurrentDecoderTo4x_3(self.decoder_ch, self.out_ch[:3], gru=gru)
    
    def forward(self,
                query_img: Tensor,
                memory_img: Tensor,
                memory_mask: Tensor,
                rec_glance = None,
                rec_focus = None,
                downsample_ratio: float = 1,
                segmentation_pass: bool = False):
        if rec_glance is None:
            rec_glance, rec_focus = self.default_rec

        # B, T, C, H, W
        num_query = query_img.size(1)
        is_refine, qimg_sm, mimg_sm, mask_sm = self._smaller_input(query_img, memory_img, memory_mask, downsample_ratio)
        imgs_sm = self.avgpool(qimg_sm)
        
        feats_q, feats_m = self.backbone(qimg_sm, mimg_sm, mask_sm)
        feats_gl, feats_fc = self.fuse(feats_q, feats_m, mask_sm)
        hid, *rec = self.decoder_glance(qimg_sm, imgs_sm[1], imgs_sm[2], feats_gl[2], feats_gl[3], *rec_glance)
        rec_glance, feats = rec[:-1], rec[-1]
        # print([f.shape for f in feats], self.out_ch)
        out_glance = self.project_seg(hid)
        out_glance = F.interpolate(out_glance.flatten(0, 1), size=qimg_sm.shape[-2:], mode='bilinear').unflatten(0, qimg_sm.shape[:2])
        if segmentation_pass:
            return [out_glance, rec_glance]

        hid, *rec = self.decoder_focus(
            qimg_sm, imgs_sm[0], 
            *feats_fc[:2],
            feats[0], feats[2], *rec_focus)
        rec_focus, feats = rec[:-1], rec[-1]
        out_focus = torch.sigmoid(self.project_mat(hid))
        return [out_glance, out_focus, self._collaborate(out_glance, out_focus), [rec_glance, rec_focus], feats]

class GFM_GatedFuseVM_to4xglance_4xfocus_5(GFM_GatedFuseVM_to4xglance_4xfocus_3):
    def __init__(self, backbone_arch: str = 'mobilenetv3_large_100', backbone_pretrained=True, refiner: str = 'deep_guided_filter', is_output_fg=False, gru=AttnGRU2, fuse='GFMGatedFuse'):
        super().__init__(backbone_arch, backbone_pretrained, refiner, is_output_fg, gru, fuse)
        del self.decoder_focus

        self.decoder_focus = RecurrentDecoderFrom4x_5(
            self.decoder_ch, 
            self.out_ch[::-1]+[self.decoder_ch[-1]], 
            self.out_ch[-1], ch_feat=32, gru=ConvGRU)
        self.project_mat = Projection(self.out_ch[-1], 1)
        self.default_rec = [[None]*2, [None]]

class GFM_GatedFuseVM_to4xglance_4xfocus_6(GFM_GatedFuseVM_to4xglance_4xfocus_3):
    def __init__(self, backbone_arch: str = 'mobilenetv3_large_100', backbone_pretrained=True, refiner: str = 'deep_guided_filter', is_output_fg=False, gru=AttnGRU2, fuse='GFMGatedFuse'):
        super().__init__(backbone_arch, backbone_pretrained, refiner, is_output_fg, gru, fuse)
        del self.decoder_focus

        self.decoder_focus = RecurrentDecoderFrom4x_6(
            self.decoder_ch, 
            self.out_ch[::-1]+[self.decoder_ch[-1]], 
            self.out_ch[-1], ch_feat=32, gru=ConvGRU)
        self.project_mat = Projection(self.out_ch[-1], 1)
        self.default_rec = [[None]*2, [None]]

class GFM_GatedFuseVM_to4xglance_4xfocus_7(GFM_GatedFuseVM_to4xglance_4xfocus_3):
    def __init__(self, backbone_arch: str = 'mobilenetv3_large_100', backbone_pretrained=True, refiner: str = 'deep_guided_filter', is_output_fg=False, gru=AttnGRU2, fuse='GFMGatedFuse'):
        super().__init__(backbone_arch, backbone_pretrained, refiner, is_output_fg, gru, fuse)
        del self.decoder_focus

        self.decoder_focus = RecurrentDecoderFrom4x_7(
            self.decoder_ch, 
            self.out_ch[::-1]+[self.decoder_ch[-1]], 
            self.out_ch[-1], ch_feat=32, gru=ConvGRU)
        self.project_mat = Projection(self.out_ch[-1], 1)
        self.default_rec = [[None]*2, [None]]


class GFM_GatedFuseVM_to4xglance_4xfocus_3_fusefeature(GFM_GatedFuseVM_to4xglance_4xfocus_2):
    def __init__(self, backbone_arch: str = 'mobilenetv3_large_100', backbone_pretrained=True, refiner: str = 'deep_guided_filter', is_output_fg=False, gru=AttnGRU2, fuse='GFMGatedFuse'):
        super().__init__(backbone_arch, backbone_pretrained, refiner, is_output_fg, gru, fuse)
        del self.decoder_glance

        self.decoder_glance = RecurrentDecoderTo4x_3(self.decoder_ch, self.out_ch[:3], gru=gru)
        self.project_fuse = nn.Sequential(
            nn.Conv2d(self.out_ch[2]+self.out_ch[-1], 32, 3, 1, 1),
            nn.ReLU(True),
            cbam.CBAM(32, reduction_ratio=4),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(16, 1, 1, bias=False),
        )
    
    def forward(self,
                query_img: Tensor,
                memory_img: Tensor,
                memory_mask: Tensor,
                rec_glance = None,
                rec_focus = None,
                downsample_ratio: float = 1,
                segmentation_pass: bool = False):
        if rec_glance is None:
            rec_glance, rec_focus = self.default_rec

        # B, T, C, H, W
        num_query = query_img.size(1)
        is_refine, qimg_sm, mimg_sm, mask_sm = self._smaller_input(query_img, memory_img, memory_mask, downsample_ratio)
        imgs_sm = self.avgpool(qimg_sm)
        
        feats_q, feats_m = self.backbone(qimg_sm, mimg_sm, mask_sm)
        feats_gl, feats_fc = self.fuse(feats_q, feats_m, mask_sm)
        hid_gl, *rec = self.decoder_glance(qimg_sm, imgs_sm[1], imgs_sm[2], feats_gl[2], feats_gl[3], *rec_glance)
        rec_glance, feats = rec[:-1], rec[-1]
        # print([f.shape for f in feats], self.out_ch)
        out_glance = self.project_seg(hid_gl)
        out_glance = F.interpolate(out_glance.flatten(0, 1), size=qimg_sm.shape[-2:], mode='bilinear').unflatten(0, qimg_sm.shape[:2])
        if segmentation_pass:
            return [out_glance, rec_glance]

        hid_fc, *rec = self.decoder_focus(
            qimg_sm, imgs_sm[0], 
            *feats_fc[:2],
            feats[0], feats[2], *rec_focus)
        rec_focus, feats = rec[:-1], rec[-1]
        out_focus = torch.sigmoid(self.project_mat(hid_fc))
        hid_gl = F.interpolate(hid_gl.flatten(0, 1), size=qimg_sm.shape[-2:], mode='bilinear')
        out_final = torch.sigmoid(self.project_fuse(torch.cat([hid_gl, hid_fc.flatten(0, 1)], dim=1))).unflatten(0, qimg_sm.shape[:2])
        return [out_glance, out_focus, out_final, [rec_glance, rec_focus], feats]



class GFM_GatedFuseVM_to4xglance_4xfocus_3_fullresgate(GFM_GatedFuseVM_to4xglance_4xfocus_3):
    def __init__(self, backbone_arch: str = 'mobilenetv3_large_100', backbone_pretrained=True, refiner: str = 'deep_guided_filter', is_output_fg=False, gru=AttnGRU2, fuse='GFMGatedFuse'):
        super().__init__(backbone_arch, backbone_pretrained, refiner, is_output_fg, gru, fuse)
        del self.fuse
        self.fuse = GFMGatedFuse_fullres(self.backbone.channels, self.ch_bottleneck)
    
    def forward(self,
                query_img: Tensor,
                memory_img: Tensor,
                memory_mask: Tensor,
                rec_glance = None,
                rec_focus = None,
                downsample_ratio: float = 1,
                segmentation_pass: bool = False):
        if rec_glance is None:
            rec_glance, rec_focus = self.default_rec

        # B, T, C, H, W
        num_query = query_img.size(1)
        is_refine, qimg_sm, mimg_sm, mask_sm = self._smaller_input(query_img, memory_img, memory_mask, downsample_ratio)
        imgs_sm = self.avgpool(qimg_sm)
        
        feats_q, feats_m = self.backbone(qimg_sm, mimg_sm, mask_sm)
        feats_gl, feats_fc = self.fuse(feats_q, feats_m, mimg_sm, mask_sm)
        hid, *rec = self.decoder_glance(qimg_sm, imgs_sm[1], imgs_sm[2], feats_gl[2], feats_gl[3], *rec_glance)
        rec_glance, feats = rec[:-1], rec[-1]
        # print([f.shape for f in feats], self.out_ch)
        out_glance = self.project_seg(hid)
        out_glance = F.interpolate(out_glance.flatten(0, 1), size=qimg_sm.shape[-2:], mode='bilinear').unflatten(0, qimg_sm.shape[:2])
        if segmentation_pass:
            return [out_glance, rec_glance]

        hid, *rec = self.decoder_focus(
            qimg_sm, imgs_sm[0], 
            *feats_fc[:2],
            feats[0], feats[2], *rec_focus)
        rec_focus, feats = rec[:-1], rec[-1]
        out_focus = torch.sigmoid(self.project_mat(hid))
        return [out_glance, out_focus, self._collaborate(out_glance, out_focus), [rec_glance, rec_focus], feats]