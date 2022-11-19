from pathy import dataclass
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Optional, List

from einops.layers.torch import Rearrange
from einops import rearrange


from .recurrent_decoder import *
from .fast_guided_filter import FastGuidedFilterRefiner
from .VM.deep_guided_filter import DeepGuidedFilterRefiner

from .backbone import *
from .basic_block import *
from .bottleneck_fusion import *
from .trimap_fusion import *
from .decoder import *
from .util import *
from .module import *
from util.tensor_util import pad_divide_by, unpad

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
        trimap_fusion='default',
        seg_decoder='4x',
        mat_decoder='4x',
        ch_mask=1,
        bottleneck_fusion='default',
    ):
        super().__init__()

        # Encoder
        self.backbone = Backbone(backbone_arch, backbone_pretrained, (0, 1, 2, 3), in_chans=3)
        self.trimap_fuse = {
            'default': TrimapGatedFusion,
            'naive': TrimapNaiveFusion,
            'bn': TrimapGatedFusionBN,
            'bn2': TrimapGatedFusionBN2,
            'in': TrimapGatedFusionIN,
            'gn': TrimapGatedFusionGN,
            'fullres': TrimapGatedFusionFullRes,
            'intrimap_only': TrimapGatedFusionInTrimapOnly,
            'intrimap_only_fullres': TrimapGatedFusionInTrimapOnlyFullres,
            'small': TrimapGatedFusionSmall,
            'fullgate': TrimapGatedFusionFullGate,
            'backbone': TrimapFusionBackbone,
        }[trimap_fusion](self.backbone.channels, ch_mask=ch_mask)
        
        self.bottleneck_fuse = {
            'default': BottleneckFusion,
            'l2': lambda *args: BottleneckFusion(*args, affinity='l2'),
            'f16': BottleneckFusion_f16,
            'gate': lambda *args: BottleneckFusion_gate(*args, affinity='l2'),
            '1236': BottleneckFusion_PPM1236,
            'woppm': BottleneckFusion_woPPM,
            'wocbam': BottleneckFusion_woCBAM,
            'wocbamppm': BottleneckFusion_woCBAMPPM,
        }[bottleneck_fusion](self.backbone.channels[-1], ch_key, self.backbone.channels[-1], ch_bottleneck)

        self.feat_channels = list(self.backbone.channels)
        self.feat_channels[-1] = ch_bottleneck

        self.avgpool = AvgPool(3)
        
        # Decoder
        self.ch_seg = ch_seg # 8, 4, out
        self.seg_decoder = {
            '4x': SegmentationDecoderTo4x,
            '4x_2': SegmentationDecoderTo4x_2,
            '4x_3': SegmentationDecoderTo4x_3,
            '4x-gru': SegmentationDecoderTo4x_woGRU,
        }[seg_decoder](self.feat_channels, self.ch_seg, gru=seg_gru)

        self.seg_project = Projection(self.ch_seg[-1], 3)
        self.seg_upsample = self.seg_decoder.output_stride > 1
        self.seg_stride = [self.seg_decoder.output_stride]*2

        self.ch_mat = ch_mat
        self.mat_decoder = {
            '4x': MattingDecoderFrom4x,
            '4x_2': MattingDecoderFrom4x_2,
        }[mat_decoder](self.feat_channels, 
            [ch_bottleneck] + self.ch_seg, 
            ch_feat=self.ch_mat, ch_out=self.ch_mat, gru=ConvGRU
        )
        self.mat_project = Projection(self.ch_mat, 1)
        self.default_rec = [self.seg_decoder.default_rec, self.mat_decoder.default_rec, None]
        # print(self.default_rec)
        self.refiner = FastGuidedFilterRefiner(self.ch_mat)
        

    def forward(self, 
        qimgs: Tensor, mimgs: Tensor, masks: Tensor,
        rec_seg = None,
        rec_mat = None,
        rec_bottleneck = None,
        downsample_ratio: float = 1,
        segmentation_pass: bool = False,
        replace_given_seg: bool = False,
    ):
        if rec_seg is None:
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
        replace_seg = mask_sm if replace_given_seg else None
        return self.decode(qimgs, qimg_sm, feats_q, segmentation_pass, is_refine, rec_seg, rec_mat, rec_bottleneck, replace_seg=replace_seg)

    def forward_with_memory(self, 
        qimgs: Tensor, m_feat16: Tensor, m_value: Tensor,
        rec_seg = None,
        rec_mat = None,
        rec_bottleneck = None,
        downsample_ratio: float = 1,
        segmentation_pass: bool = False,
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
        replace_seg = None,
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
        if self.seg_upsample:
            out_seg = F.interpolate(out_seg.flatten(0, 1), scale_factor=self.seg_stride, mode='bilinear').unflatten(0, out_seg.shape[:2])
        if segmentation_pass:
            return [out_seg, rec_seg]

        # Matting
        hid, *remain = self.mat_decoder(
            qimg_sm, *qimg_sm_avg[:2], 
            *feats_q[:2],
            feat_seg[0], feat_seg[2],
            *rec_mat
        )
        rec_mat, feats = remain[:-1], remain[-1]
        out_mat = torch.sigmoid(self.mat_project(hid))
        if replace_seg is None:
            out_collab = collaborate_fuse(out_seg, out_mat)
        else:
            t = replace_seg.size(1)
            out_collab = torch.zeros_like(out_mat)
            out_collab[:, :t] = collaborate_fuse_trimap(replace_seg, out_mat[:, :t])
            if out_mat.size(1) > t:
                out_collab[:, t:] = collaborate_fuse(out_seg[:, t:], out_mat[:, t:])


        if is_refine:
            # self.tmp_out_collab = F.interpolate(out_collab.detach().flatten(0, 1), qimgs.shape[-2:], mode='bilinear', align_corners=False).unflatten(0, qimgs.shape[:2])
            # out_collab = self.tmp_out_collab
            out_collab = self.refiner(qimgs, qimg_sm, out_collab)

        return [out_seg, out_mat, out_collab, [rec_seg, rec_mat, rec_bottleneck], feats]

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

    def encode_imgs_to_value(self, imgs, masks, downsample_ratio = 1):
        if downsample_ratio != 1:
            imgs = self._interpolate(imgs, scale_factor=downsample_ratio)
            masks = self._interpolate(masks, scale_factor=downsample_ratio)
        feat = self.backbone(imgs)
        values = self.trimap_fuse(imgs, masks, feat) # b, c, t, h, w
        return feat[-1], values

    def _interpolate(self, x: Tensor, scale_factor: float):
        if x.ndim == 5:
            B, T = x.shape[:2]
            x = F.interpolate(x.flatten(0, 1), scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
            x, _ = pad_divide_by(x, 16)
            x = x.unflatten(0, (B, T))
        else:
            x = F.interpolate(x, scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
            x, _ = pad_divide_by(x, 16)
        return x

class STCNFuseMatting_big(STCNFuseMatting):
    def __init__(self):
        super().__init__(ch_seg=[96, 64, 32], ch_mat=32)
    

class STCNFuseMatting_fullres_mat(STCNFuseMatting):
    def __init__(
        self, 
        backbone_arch='mobilenetv3_large_100', 
        backbone_pretrained=True, 
        ch_bottleneck=128,
        ch_key=32,
        ch_seg=[96, 48, 16],
        ch_mat=[64, 32, 16, 8],
        seg_gru=ConvGRU,
        trimap_fusion='default',
        bottleneck_fusion='default',
        seg_decoder='4x',
        mat_decoder='4x',
        ch_mask=1,
    ):
        super().__init__(
        backbone_arch=backbone_arch, 
        backbone_pretrained=backbone_pretrained, 
        ch_bottleneck=ch_bottleneck,
        ch_key=ch_key,
        ch_seg=ch_seg,
        ch_mat=32,
        seg_gru=seg_gru,
        trimap_fusion=trimap_fusion,
        bottleneck_fusion=bottleneck_fusion,
        seg_decoder=seg_decoder,
        ch_mask=ch_mask,
    )
        
        self.ch_mat = ch_mat
        self.mat_decoder = {
            '4x': MattingDecoderFrom4x_feats,
            '4x_2': MattingDecoderFrom4x_feats_2,
            '4x_3': MattingDecoderFrom4x_feats_3,
            'naive': MattingDecoderFrom4x_feats_naive,
            'naive-gru': MattingDecoderFrom4x_feats_naive_woGRU,
            'naive2': MattingDecoderFrom4x_feats_naive2,
            'naive3': MattingDecoderFrom4x_feats_naive3,
            'naive4': MattingDecoderFrom4x_feats_naive4,
        }[mat_decoder](self.feat_channels, 
            [ch_bottleneck] + self.ch_seg, 
            self.ch_mat, gru=ConvGRU
        )
        self.mat_project = Projection(self.ch_mat[-1], 1)
        self.default_rec = [self.seg_decoder.default_rec, self.mat_decoder.default_rec, None]

class STCNFuseMatting_fullres_mat_big(STCNFuseMatting_fullres_mat):
    def __init__(self,
        trimap_fusion='default',
        bottleneck_fusion='default',
        seg_decoder='4x',
        mat_decoder='4x',
        ch_mask=1,
        ):
        super().__init__(
            backbone_arch='mixnet_xl', 
            backbone_pretrained=False,
            ch_bottleneck=128, ch_key=32,
            # ch_seg=[96, 48, 16],
            # ch_mat=[64, 32, 16, 8],
            ch_seg=[128, 96, 32],
            ch_mat=[128, 96, 64, 32],
            trimap_fusion=trimap_fusion, 
            bottleneck_fusion=bottleneck_fusion, 
            seg_decoder=seg_decoder, mat_decoder=mat_decoder, ch_mask=ch_mask)

class STCNFuseMatting_SameDec(STCNFuseMatting):
    def __init__(self, backbone_arch='mobilenetv3_large_100', backbone_pretrained=True, ch_bottleneck=128, ch_key=32, ch_seg=[96, 48, 16], ch_mat=32, seg_gru=ConvGRU, trimap_fusion='default', seg_decoder='4x', mat_decoder='4x'):
        super().__init__(backbone_arch, backbone_pretrained, ch_bottleneck, ch_key, ch_seg, ch_mat, seg_gru, trimap_fusion, seg_decoder, mat_decoder)
    
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



class STCNFuseMatting_SingleDec(STCNFuseMatting):
    def __init__(
        self, backbone_arch='mobilenetv3_large_100', backbone_pretrained=True, ch_bottleneck=128, ch_key=32, ch_seg=[96, 48, 16], ch_mat=32, seg_gru=ConvGRU, trimap_fusion='default', seg_decoder='4x', mat_decoder='4x',
        ch_decode = [96, 48, 32, 16]
    ):
        super().__init__(backbone_arch, backbone_pretrained, ch_bottleneck, ch_key, ch_seg, ch_mat, seg_gru, trimap_fusion, seg_decoder, mat_decoder)
        
    
        del self.mat_decoder
        del self.seg_decoder
        del self.mat_project
        del self.seg_project

        # ch_decode = [128, 96, 64, 32]
        self.decoder = RecurrentDecoder(self.feat_channels, ch_decode)

        self.mat_project = Projection(ch_decode[-1], 1)
        self.seg_project = Projection(ch_decode[-1], 1)
        self.default_rec = [[None]*4, None] + [None]
        # only use one rec

    def decode(self, 
        qimgs, qimg_sm, feats_q, 
        segmentation_pass,
        is_refine,
        rec_seg = None,
        rec_mat = None,
        rec_bottleneck = None,
    ):
        # Decode
        # qimg_sm_avg = self.avgpool(qimg_sm) # 2, 4, 8
        # Segmentation
        hid, *remain = self.decoder(qimg_sm, *feats_q, *rec_seg)
        # hid, rec1, rec2 ..., inter-feats
        rec_seg, feat_seg = remain[:-1], remain[-1]
        out_seg = self.seg_project(hid)
        if segmentation_pass:
            return [out_seg, rec_seg]

        # Matting
        out_mat = torch.sigmoid(self.mat_project(hid))
        
        if is_refine:
            # fgr_residual, pha = self.refiner(src, src_sm, fgr_residual, pha, hid)
            pass

        return [out_mat, [rec_seg, rec_mat, rec_bottleneck], feat_seg]

class SingleTrimapPropagation(nn.Module):
    def __init__(
        self, 
        backbone_arch='mobilenetv3_large_100', 
        backbone_pretrained=True, 
        ch_bottleneck=128,
        ch_key=32,
        ch_decode = [96, 48, 32, 16],
        ch_mask=1,
    ):
        super().__init__()

        # Encoder
        self.backbone = Backbone(backbone_arch, backbone_pretrained, (0, 1, 2, 3), in_chans=3)
        self.trimap_fuse = TrimapGatedFusionGN(self.backbone.channels, ch_mask=ch_mask)
        self.bottleneck_fuse = BottleneckFusion(self.backbone.channels[-1], ch_key, self.backbone.channels[-1], ch_bottleneck)

        self.feat_channels = list(self.backbone.channels)
        self.feat_channels[-1] = ch_bottleneck

        self.avgpool = AvgPool(3)
        
        # Decoder
        self.ch_seg = ch_decode # 8, 4, out
        self.decoder = RecurrentDecoder(self.feat_channels, ch_decode)
        self.project = Projection(self.ch_seg[-1], 3)

        self.default_rec = self.decoder.default_rec
        

    def forward(self, qimgs: Tensor, mimgs: Tensor, masks: Tensor, rec_seg = None):
        if rec_seg is None:
            rec_seg = self.default_rec

        # Encode
        q = qimgs.size(1)
        feats = self.backbone(torch.cat([qimgs, mimgs], dim=1))
        # zip func is unavailable in torch.jit
        feats_q = [f[:, :q] for f in feats] # b, t_q, ch_feat_i, h, w
        feats_m = [f[:, q:] for f in feats] # b, t_q, ch_feat_i, h, w

        value_m = self.trimap_fuse(mimgs, masks, feats_m) # b, c, t, h, w
        feats_q[-1], _ = self.bottleneck_fuse(feats_q[-1], feats_m[-1], value_m, None)
        hid, *remain = self.decoder(qimgs, *feats_q, *rec_seg)
        # hid, rec1, rec2 ..., inter-feats
        rec_seg, feat_seg = remain[:-1], remain[-1]
        out_seg = self.project(hid)
        if out_seg.size(-1) < qimgs.size(-1):
            out_seg = F.interpolate(out_seg.flatten(0, 1), size=qimgs.shape[-2:], mode='bilinear').unflatten(0, out_seg.shape[:2])
        return out_seg, rec_seg

    def forward_with_memory(self, qimgs: Tensor, m_feat16: Tensor, m_value: Tensor, rec_seg = None):
        if rec_seg is None:
            rec_seg = self.default_rec
        
        # Encode
        feats_q = self.backbone(qimgs)
        feats_q[-1], _ = self.bottleneck_fuse(feats_q[-1], m_feat16, m_value, None)
        hid, *remain = self.decoder(qimgs, *feats_q, *rec_seg)
        # hid, rec1, rec2 ..., inter-feats
        rec_seg, feat_seg = remain[:-1], remain[-1]
        out_seg = self.project(hid)
        if out_seg.size(-1) < qimgs.size(-1):
            out_seg = F.interpolate(out_seg.flatten(0, 1), size=qimgs.shape[-2:], mode='bilinear').unflatten(0, out_seg.shape[:2])
        return out_seg, rec_seg

    def encode_imgs_to_value(self, imgs, masks):
        feat = self.backbone(imgs)
        values = self.trimap_fuse(imgs, masks, feat) # b, c, t, h, w
        return feat[-1], values

class SingleTrimapPropagation4x(SingleTrimapPropagation):
    def __init__(self, backbone_arch='mobilenetv3_large_100', backbone_pretrained=True, ch_bottleneck=128, ch_key=32, ch_decode=[96, 48, 32, 16], ch_mask=1):
        super().__init__(backbone_arch, backbone_pretrained, ch_bottleneck, ch_key, ch_decode, ch_mask)
        del self.decoder
        # Decoder
        self.ch_seg = ch_decode # 8, 4, out
        self._decoder = SegmentationDecoderTo4x(self.feat_channels, ch_decode)
        self.project = Projection(self.ch_seg[2], 3)

        self.default_rec = self._decoder.default_rec
        self.decoder = lambda qimgs, f2, f4, f8, f16, *rec_seg: \
            self._decoder(qimgs, *self.avgpool(qimgs)[1:3], f8, f16, *rec_seg)    

class SingleMatting(nn.Module):
    def __init__(
        self, 
        backbone_arch='mobilenetv3_large_100', 
        backbone_pretrained=True, 
        ch_bottleneck = 128,
        ch_decode = [96, 48, 32, 16],
        ch_mask=1,
    ):
        super().__init__()

        # Encoder
        self.backbone = Backbone(backbone_arch, backbone_pretrained, (0, 1, 2, 3), in_chans=3+ch_mask)
        
        self.feat_channels = list(self.backbone.channels)
        ch16 = self.feat_channels[-1]
        self.feat_channels[-1] = ch_bottleneck

        self.ppm = PSP(ch16, ch16//4, ch_bottleneck, (1, 2, 3, 6))

        self.avgpool = AvgPool(3)
        
        # Decoder
        self.ch_seg = ch_decode # 8, 4, out
        self.decoder = RecurrentDecoder(self.feat_channels, ch_decode)
        self.project = nn.Sequential(
            Projection(self.ch_seg[-1], 1),
            nn.Sigmoid()
        )

        self.default_rec = self.decoder.default_rec
        

    def forward(self, imgs: Tensor, masks: Tensor, rec = None):
        if rec is None:
            rec = self.default_rec

        # Encode
        feats = self.backbone(torch.cat([imgs, masks], dim=2))
        feats[-1] = self.ppm(feats[-1])
        hid, *remain = self.decoder(imgs, *feats, *rec)
        # hid, rec1, rec2 ..., inter-feats
        rec, feat = remain[:-1], remain[-1]
        out = self.project(hid)
        return out, rec


class SeperateNetwork(nn.Module):
    def __init__(self, seg='default'):
        super().__init__()
        self.seg = {
            'default': SingleTrimapPropagation,
            '4x': SingleTrimapPropagation4x,
        }[seg]()

        self.mat = SingleMatting()
        self.refiner = FastGuidedFilterRefiner()
        self.default_rec = [self.seg.default_rec, self.mat.default_rec]
        

    def forward(self, 
        qimgs: Tensor, mimgs: Tensor, masks: Tensor,
        rec_seg = None,
        rec_mat = None,
        downsample_ratio: float = 1,
        segmentation_pass: bool = False
    ):
        is_refine, qimg_sm, mimg_sm, mask_sm = self._smaller_input(qimgs, mimgs, masks, downsample_ratio)
        out_seg, rec_seg = self.seg(qimg_sm, mimg_sm, mask_sm[:, [0]], rec_seg)
        if segmentation_pass:
            return out_seg, rec_seg, None
        out_mat, rec_mat = self.mat(qimg_sm, mask_sm, rec_mat)
        out_collab = collaborate_fuse(out_seg, out_mat)

        if is_refine:
            self.tmp_out_collab = F.interpolate(out_collab.detach().flatten(0, 1), qimgs.shape[-2:], mode='bilinear', align_corners=False).unflatten(0, qimgs.shape[:2])
            out_collab = self.refiner(qimgs, qimg_sm, out_collab)

        return out_seg, out_mat, out_collab, [rec_seg, rec_mat], None

    def forward_with_memory(self, 
        qimgs: Tensor, m_feat16: Tensor, m_value: Tensor,
        rec_seg = None,
        rec_mat = None,
        downsample_ratio: float = 1,
        segmentation_pass: bool = False
    ):
        if rec_seg is None:
            rec_seg, rec_mat = self.default_rec
        
        if is_refine := (downsample_ratio != 1):
            qimg_sm = self._interpolate(qimgs, scale_factor=downsample_ratio)
        else:
            qimg_sm = qimgs

        # Encode
        out_seg, rec_seg = self.seg.forward_with_memory(qimgs, m_feat16, m_value, rec_seg)

        if segmentation_pass:
            return out_seg, rec_seg, None
        out_mat, rec_mat = self.mat(qimg_sm, self._seg_to_trimap(out_seg), rec_mat)
        out_collab = collaborate_fuse(out_seg, out_mat)

        if is_refine:
            self.tmp_out_collab = F.interpolate(out_collab.detach().flatten(0, 1), qimgs.shape[-2:], mode='bilinear', align_corners=False).unflatten(0, qimgs.shape[:2])
            out_collab = self.refiner(qimgs, qimg_sm, out_collab)

        return out_seg, out_mat, out_collab, [rec_seg, rec_mat], None

    def encode_imgs_to_value(self, imgs, masks, downsample_ratio = 1):
        if downsample_ratio != 1:
            imgs = self._interpolate(imgs, scale_factor=downsample_ratio)
            masks = self._interpolate(masks, scale_factor=downsample_ratio)
        return self.seg.encode_imgs_to_value(imgs, masks)

    @staticmethod
    def _seg_to_trimap(logit):
        val, idx = torch.sigmoid(logit).max(dim=2, keepdim=True) # ch
        # (bg, t, fg)
        tran_mask = idx == 1
        fg_mask = idx == 2
        return tran_mask*0.5 + fg_mask

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