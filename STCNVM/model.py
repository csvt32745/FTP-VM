from pathy import dataclass
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Optional, List

import timm
from einops.layers.torch import Rearrange
from einops import rearrange

from STCNVM.module import GlobalMatch

from .recurrent_decoder import RecurrentDecoder, Projection, RecurrentDecoder4x, ConvGRU, RecurrentDecoder_, RecurrentDecoderTo8x, RecurrentDecoder8x
from .VM.fast_guided_filter import FastGuidedFilterRefiner
from .VM.deep_guided_filter import DeepGuidedFilterRefiner

from .STCN.network import *
from .STCN.modules import *
from .module import *
from .decoder import FramewiseDecoder, FramewiseDecoder4x, FramewiseDecoder8x
from .memory_bank import MemoryBank

from .backbone import *
from .basic_block import *
    
class MattingNetwork(nn.Module):
    def __init__(self,
                fuse_channels: int = 32,
                backbone_arch: str = 'mobilenetv3_large_100',
                backbone_pretrained = True,
                refiner: str = 'deep_guided_filter'
        ):
        super().__init__()
        assert refiner in ['fast_guided_filter', 'deep_guided_filter']
        
        self.backbone = Backbone(
            backbone_arch=backbone_arch, 
            backbone_pretrained=backbone_pretrained,
            out_indices=list(range(4))) # only for 4 stages
        
        ch_bottleneck = 128
        self.aspp = LRASPP(self.backbone.channels[-1]+fuse_channels, ch_bottleneck)

        decoder_ch = list(self.backbone.channels)
        decoder_ch[-1] = ch_bottleneck
        self.decoder = RecurrentDecoder(decoder_ch, [80, 40, 32, 16])
            
        self.project_mat = Projection(16, 1)
        # self.project_mat = Projection(16, 4)
        self.project_seg = Projection(16, 1)

        if refiner == 'deep_guided_filter':
            self.refiner = DeepGuidedFilterRefiner()
        else:
            self.refiner = FastGuidedFilterRefiner()
        
    def forward(self,
                src: Tensor,
                fuse: Tensor,
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
        
        f1, f2, f3, f4 = self.backbone(src_sm)
        # print(f4.shape, fuse.shape)
        f4 = self.aspp(torch.cat([f4, fuse], 2))
        hid, *rec = self.decoder(src_sm, f1, f2, f3, f4, r1, r2, r3, r4)
        
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

class STCN_Encoder(nn.Module):
    def __init__(self, ch_value=128, ch_key=64):
        super().__init__()
        self.key_encoder = KeyEncoder(backbone_arch='mobilenetv3_large_100')
        self.value_encoder = ValueEncoder(
            ch_fuse=self.key_encoder.channels[-1], ch_out=ch_value,
            backbone_arch='mobilenetv3_large_100') 

        # Projection from f16 feature space to key space
        self.key_proj = KeyProjection(self.key_encoder.channels[-1], keydim=ch_key)

        # Compress f16 a bit to use in decoding later on
        self.key_comp = nn.Conv2d(self.key_encoder.channels[-1], ch_value, kernel_size=3, padding=1)

        self.memory = MemoryReader()
        # self.decoder = Decoder()
        self.out_channel = ch_value*2

        self.to4d = Rearrange('b t c h w -> (b t) c h w')

    def encode_key(self, frame): 
        # input: b*t*c*h*w
        b, t = frame.shape[:2]

        f4, f8, f16 = self.key_encoder(frame.flatten(start_dim=0, end_dim=1))
        k16 = self.key_proj(f16)
        f16_thin = self.key_comp(f16)

        # B*C*T*H*W
        # print(k16.shape)
        # k16 = k16.view(b, t, *k16.shape[-3:]).transpose(1, 2).contiguous()
        k16 = rearrange(k16, '(b t) c h w -> b c t h w', b=b, t=t).contiguous()

        # B*T*C*H*W
        f16_thin = f16_thin.view(b, t, *f16_thin.shape[-3:])
        f16 = f16.view(b, t, *f16.shape[-3:])
        f8 = f8.view(b, t, *f8.shape[-3:])
        f4 = f4.view(b, t, *f4.shape[-3:])

        return k16, f16_thin, f16, f8, f4

    def encode_value(self, frame, kf16, mask):
        # Extract memory key/value for a frame
        # print(frame.shape, kf16.shape, mask.shape)
        if frame.ndim == 5:
            b, t = frame.shape[:2]
            f16 = self.value_encoder(self.to4d(frame), self.to4d(kf16), self.to4d(mask))
            f16 = rearrange(f16, '(b t) c h w -> b c t h w', b=b, t=t)
        else:
            f16 = self.value_encoder(frame, kf16, mask).unsqueeze(2)
        return f16 # B*CH_V*T*H*W

    def query_value(self, qk16, qv16, mk16, mv16): 
        # q - query, m - memory
        # qv16 is qf16_thin
        affinity = self.memory.get_affinity(mk16, qk16)
        v16 = self.memory.readout(affinity, mv16, qv16)
        return v16

    # def segment(self, qk16, qv16, qf8, qf4, mk16, mv16): 
    #     # q - query, m - memory
    #     # qv16 is f16_thin above
    #     affinity = self.memory.get_affinity(mk16, qk16)
    #     logits = self.decoder(self.memory.readout(affinity, mv16, qv16), qf8, qf4)
    #     prob = torch.sigmoid(logits)
    #     return logits, prob

    def forward(self, mode, *args, **kwargs):
        if mode == 'encode_key':
            return self.encode_key(*args, **kwargs)
        elif mode == 'encode_value':
            return self.encode_value(*args, **kwargs)
        # elif mode == 'segment':
        #     return self.segment(*args, **kwargs)
        else:
            raise NotImplementedError

class STCN_BASE(nn.Module):
    def __init__(self, stcn: STCN_Encoder):
        super().__init__()
        self.stcn = stcn

    def encode_key(self, *args, **kwargs):
        return self.stcn.encode_key(*args, **kwargs)
    
    def encode_value(self, *args, **kwargs):
        return self.stcn.encode_value(*args, **kwargs)

    def query_value(self, *args, **kwargs):
        return self.stcn.query_value(*args, **kwargs)

class STCN_VM(STCN_BASE):
    def __init__(self, ch_value=128, ch_key=64):
        super().__init__(STCN_Encoder(ch_value=ch_value, ch_key=ch_key))
        self.matting = MattingNetwork(fuse_channels=self.stcn.out_channel)
        
    def forward(self, 
        query_imgs, memory_imgs, memory_masks,
        gru_mems = [None, None, None, None], segmentation_pass=False
        ):
        # B, T, C, H, W
        num_q = query_imgs.size(1)
        # num_m = memory_imgs.size(1)
        k16, kf16_thin, kf16, kf8, kf4 = self.stcn.encode_key(torch.cat([query_imgs, memory_imgs], dim=1))
        mk = k16[:, :, num_q:]
        mv = self.stcn.encode_value(memory_imgs, kf16[:, num_q:], memory_masks)
        feat = self.stcn.query_value(k16[:, :, :num_q], kf16_thin[:, :num_q], mk, mv)
        return *self.matting(query_imgs, feat, *gru_mems, segmentation_pass=segmentation_pass), mk, mv
    
    def forward_with_mem_kv(self, 
        query_imgs, memory_k, memory_v,
        gru_mems = [None, None, None, None], segmentation_pass=False
        ):
        # B, T, C, H, W
        # num_q = query_imgs.size(1)
        # num_m = memory_imgs.size(1)
        k16, kf16_thin, kf16, kf8, kf4 = self.stcn.encode_key(query_imgs)
        feat = self.stcn.query_value(k16, kf16_thin, memory_k, memory_v)
        return *self.matting(query_imgs, feat, *gru_mems, segmentation_pass=segmentation_pass), memory_k, memory_v
    
    def forward_without_memory(self, 
        query_imgs, gru_mems = [None, None, None, None], segmentation_pass=False
        ):
        # B, T, C, H, W
        # num_q = query_imgs.size(1)
        # num_m = memory_imgs.size(1)
        k16, kf16_thin, kf16, kf8, kf4 = self.stcn.encode_key(query_imgs)
        # feat = self.stcn.query_value(k16, kf16_thin, memory_k, memory_v)
        feat = torch.cat([kf16_thin, torch.zeros_like(kf16_thin)], dim=2)
        return self.matting(query_imgs, feat, *gru_mems, segmentation_pass=segmentation_pass)

    def decode_with_query(self, mem_bank:MemoryBank, gru_mems, qimgs, qf8, qf4, qk16, qv16, segmentation_pass=False): 
        # k = mem_bank.num_objects

        # B T C H W
        readout_mem = mem_bank.match_memory(qk16)
        # qv16 = qv16.expand(k, -1, -1, -1)
        qv16 = torch.cat([readout_mem, qv16], 2) 
        return self.matting(qimgs, qv16, *gru_mems, segmentation_pass=segmentation_pass)

class STCN_Full(STCN_BASE):
    def __init__(self):
        # self.stcn = STCN_Encoder()
        super().__init__(STCN_Encoder())
        ch_dec = list(self.stcn.key_encoder.channels)
        ch_dec.reverse()
        ch_dec[0] = self.stcn.out_channel
        self.decoder = Decoder(ch_dec)

    def forward(self, 
        query_imgs, memory_imgs, memory_masks, segmentation_pass=False
        ):
        # B, T, C, H, W
        num_q = query_imgs.size(1)
        # num_m = memory_imgs.size(1)
        k16, kf16_thin, kf16, kf8, kf4 = self.encode_key(torch.cat([query_imgs, memory_imgs], dim=1))
        mk = k16[:, :, num_q:]
        mv = self.encode_value(memory_imgs, kf16[:, num_q:], memory_masks)
        feat = self.query_value(k16[:, :, :num_q], kf16_thin[:, :num_q], mk, mv)
        return self.decoder(query_imgs, feat, kf8[:, :num_q], kf4[:, :num_q], segmentation_pass), mk, mv

    def forward_with_mem_kv(self,
        query_imgs, mk, mv, segmentation_pass=False
        ):
        k16, kf16_thin, kf16, kf8, kf4 = self.encode_key(query_imgs)
        feat = self.query_value(k16, kf16_thin, mk, mv)
        return self.decoder(query_imgs, feat, kf8, kf4, segmentation_pass)

    def forward_add_mem_kv(self, 
        query_imgs, memory_imgs, memory_masks,
        mk, mv, segmentation_pass=False
        ):
        # B, T, C, H, W
        num_q = query_imgs.size(1)
        # num_m = memory_imgs.size(1)
        k16, kf16_thin, kf16, kf8, kf4 = self.encode_key(torch.cat([query_imgs, memory_imgs], dim=1))
        mk = torch.cat([mk, k16[:, :, num_q:]], dim=2).contiguous() # B, C, T, H, W
        # B, T, C, H, W
        mv = torch.cat([mv, self.encode_value(memory_imgs, kf16[:, num_q:], memory_masks)], dim=1).contiguous()
        feat = self.query_value(k16[:, :, :num_q], kf16_thin[:, :num_q], mk, mv)
        return self.decoder(query_imgs, feat, kf8[:, :num_q], kf4[:, :num_q], segmentation_pass), mk, mv
    
    def decode_with_query(self, mem_bank:MemoryBank, qimgs, qf8, qf4, qk16, qv16, segmentation_pass=False): 
        # k = mem_bank.num_objects

        # B T C H W
        readout_mem = mem_bank.match_memory(qk16)
        # qv16 = qv16.expand(k, -1, -1, -1)
        qv16 = torch.cat([readout_mem, qv16], 2) 

        return self.decoder(qimgs, qv16, qf8, qf4, segmentation_pass)

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

class STCN_RecDecoder(STCN_BASE):
    def __init__(self):
        super().__init__(STCN_Encoder())
        
        ch_dec = list(self.stcn.key_encoder.channels)
        ch_dec.reverse()
        ch_dec[0] = self.stcn.out_channel
        self.decoder = MattingDecoder(ch_dec)
    
    def forward(self, 
        query_imgs, memory_imgs, memory_masks,
        gru_mems = [None, None, None, None], segmentation_pass=False
        ):
        # B, T, C, H, W
        num_q = query_imgs.size(1)
        # num_m = memory_imgs.size(1)
        k16, kf16_thin, kf16, kf8, kf4 = self.stcn.encode_key(torch.cat([query_imgs, memory_imgs], dim=1))
        mk = k16[:, :, num_q:]
        mv = self.stcn.encode_value(memory_imgs, kf16[:, num_q:], memory_masks)
        feat = self.stcn.query_value(k16[:, :, :num_q], kf16_thin[:, :num_q], mk, mv)
        return *self.decoder(query_imgs, feat, kf8[:, :num_q], kf4[:, :num_q], *gru_mems, segmentation_pass=segmentation_pass), mk, mv

    def forward_with_mem_kv(self,
        query_imgs, mk, mv,
        gru_mems = [None, None, None, None], segmentation_pass=False
        ):
        k16, kf16_thin, kf16, kf8, kf4 = self.stcn.encode_key(query_imgs)
        feat = self.stcn.query_value(k16, kf16_thin, mk, mv)
        return *self.decoder(query_imgs, feat, kf8, kf4, *gru_mems, segmentation_pass=segmentation_pass), mk, mv

    def forward_add_mem_kv(self, 
        query_imgs, memory_imgs, memory_masks,
        mk, mv, gru_mems = [None, None, None, None], segmentation_pass=False
        ):
        # B, T, C, H, W
        num_q = query_imgs.size(1)
        # num_m = memory_imgs.size(1)
        k16, kf16_thin, kf16, kf8, kf4 = self.stcn.encode_key(torch.cat([query_imgs, memory_imgs], dim=1))
        mk = torch.cat([mk, k16[:, :, num_q:]], dim=2).contiguous() # B, C, T, H, W
        # B, T, C, H, W
        mv = torch.cat([mv, self.stcn.encode_value(memory_imgs, kf16[:, num_q:], memory_masks)], dim=1).contiguous()
        feat = self.stcn.query_value(k16[:, :, :num_q], kf16_thin[:, :num_q], mk, mv)
        return *self.decoder(query_imgs, feat, kf8[:, :num_q], kf4[:, :num_q], *gru_mems, segmentation_pass=segmentation_pass), mk, mv
    
    def decode_with_query(self, mem_bank:MemoryBank, gru_mems, qimgs, qf8, qf4, qk16, qv16, segmentation_pass=False): 
        # k = mem_bank.num_objects

        # B T C H W
        readout_mem = mem_bank.match_memory(qk16)
        # qv16 = qv16.expand(k, -1, -1, -1)
        qv16 = torch.cat([readout_mem, qv16], 2) 
        return self.decoder(qimgs, qv16, qf8, qf4, *gru_mems, segmentation_pass=segmentation_pass)

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
        # self.aspp = FeatureFusion(self.backbone.channels[-1]*2, ch_bottleneck)
        
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


class FuseMattingNetwork(nn.Module):
    def __init__(self,
                backbone_arch: str = 'mobilenetv3_large_100',
                backbone_pretrained = True,
                refiner: str = 'deep_guided_filter',
                is_output_fg = True,
                fuse_func = 'bottleneck_aspp'
        ):
        super().__init__()
        assert refiner in ['fast_guided_filter', 'deep_guided_filter']
        
        self.backbone_q = Backbone(
            backbone_arch=backbone_arch, 
            backbone_pretrained=backbone_pretrained, in_chans=3,
            out_indices=list(range(4))) # only for 4 stages
        
        self.backbone_m = Backbone(
            backbone_arch=backbone_arch, 
            backbone_pretrained=backbone_pretrained, in_chans=4,
            out_indices=list(range(4))) # only for 4 stages
        
        ch_bottleneck = 128
        self.fuse = {
            'bottleneck_aspp': FuseBottleneckASPP,
            'align_deformconv': FuseDfConvAlign,
            'cross_attention': FuseCrossAttention,
        }[fuse_func](self.backbone_q.channels, ch_bottleneck)
        
        decoder_ch = list(self.backbone_q.channels)
        decoder_ch[-1] = ch_bottleneck
        self.decoder = RecurrentDecoder(decoder_ch, [80, 40, 32, 16])
            
        # self.project_mat = Projection(16, 1)
        self.is_output_fg = is_output_fg
        self.project_mat = Projection(16, 4 if is_output_fg else 1)
        self.project_seg = Projection(16, 1)

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
        # num_query = query_img.size(1)
        mimgs = torch.cat([memory_img, memory_mask], 2) # C

        if downsample_ratio != 1:
            src_qimg = self._interpolate(query_img, scale_factor=downsample_ratio)
            src_mimg = self._interpolate(mimgs, scale_factor=downsample_ratio)
        else:
            src_qimg = query_img
            src_mimg = mimgs
        
        feats_q = self.backbone_q(src_qimg)
        # print(f4.shape, fuse.shape)
        feats_m = self.backbone_m(src_mimg)
        
        f1, f2, f3, f4 = self.fuse(feats_q, feats_m)
        # print(f4.shape, f3[:, :num_query].shape)
        hid, *rec = self.decoder(src_qimg, f1, f2, f3, f4, r1, r2, r3, r4)
        rec, feats = rec[:-1], rec[-1]
        if not segmentation_pass:
            if self.is_output_fg:
                fgr_residual, pha = self.project_mat(hid).split([3, 1], dim=-3)
                src = query_img # get the query rgb back
                if downsample_ratio != 1:
                    fgr_residual, pha = self.refiner(src, src_qimg, fgr_residual, pha, hid)
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

class FuseBottleneckASPP(nn.Module):
    def __init__(self, ch_feats, ch_out):
        super().__init__()
         # self.aspp = LRASPP(ch_feats[-1]*2, ch_out)
        self.aspp = FeatureFusion(ch_feats[-1]*2, ch_out)
    
    def forward(self, feats_q, feats_m):
        feats_ret = feats_q
        f4_m = torch.repeat_interleave(feats_m[3], feats_q[0].size(1), dim=1)
        feats_ret[3] = self.aspp(torch.cat([feats_q[3], f4_m], 2))
        return feats_ret
    
class FuseDfConvAlign(nn.Module):
    def __init__(self, ch_feats, ch_out):
        super().__init__()
        self.aspp = LRASPP(ch_feats[-1]*2, ch_out)
        k = 3
        kk = k//2
        self.align_dfconvs = nn.ModuleList([
            AlignDeformConv2d(ch_feats[i], ch_feats[i], ch_feats[i], 
                kernel_size=k, padding=kk, offset_group=4) #TODO
            for i in range(3)
        ])
        self.fusions = nn.ModuleList([
            FeatureFusion(ch_feats[i]*2, ch_feats[i])
            for i in range(3)
        ])
    
    def forward(self, feats_q, feats_m):
        num_q = feats_q[0].size(1)
        feats_ret = []
        for i in range(3):
            fq = feats_q[i]
            fm = torch.repeat_interleave(feats_m[i], num_q, dim=1)
            f = self.align_dfconvs[i](fm, fq)
            feats_ret.append(self.fusions[i](torch.cat([f, fq], dim=2)))

        fm = torch.repeat_interleave(feats_m[3], num_q, dim=1)
        feats_ret.append(self.aspp(torch.cat([fm, feats_q[3]], dim=2)))

        return feats_ret

class FuseCrossAttention(nn.Module):
    def __init__(self, ch_feats, ch_out):
        super().__init__()
        self.aspp = LRASPP(ch_feats[-1], ch_out)
        k = 3
        kk = k//2
        patch_size = [8, 4, 2, 1]
        self.conv_attn = nn.ModuleList([
            ConvSelfAttention(dim=ch_feats[i], attn_dim=32, head=1, patch_size=patch_size[i])
            for i in range(4)
        ])
        self.fusions = nn.ModuleList([
            FeatureFusion(ch_feats[i]*2, ch_feats[i])
            for i in range(4)
        ])
    
    def forward(self, feats_q, feats_m):
        # num_q = feats_q[0].size(1)
        B, num_q = feats_q[0].shape[:2]
        feats_ret = []
        for i in range(4):
            fq = feats_q[i]
            fm = feats_m[i]
            f = self.conv_attn[i](fq, fm)[0]
            feats_ret.append(self.fusions[i](torch.cat([f, fq], dim=2)))

        feats_ret[3] = self.aspp(feats_ret[3])
        return feats_ret

class BGAwareMatting(nn.Module):
    def __init__(self,
                backbone_arch: str = 'mobilenetv3_large_100',
                backbone_pretrained = True,
                refiner: str = 'deep_guided_filter',
                is_output_fg = True,
                fuse_func='bottleneck_aspp'
        ):
        super().__init__()
        assert refiner in ['fast_guided_filter', 'deep_guided_filter']
        
        self.backbone = Backbone(
            backbone_arch=backbone_arch, 
            backbone_pretrained=backbone_pretrained, in_chans=3,
            out_indices=list(range(4))) # only for 4 stages
        
        ch_bottleneck = 128
        # self.aspp = LRASPP(self.backbone.channels[-1]*2, ch_bottleneck)
        # # self.aspp = FeatureFusion(self.backbone.channels[-1]*2, ch_bottleneck)
        # self.fuse = {
        #     'bottleneck_aspp': FuseBottleneckASPP,
        #     'align_deformconv': FuseDfConvAlign,
        #     'cross_attention': FuseCrossAttention,
        # }[fuse_func](self.backbone.channels, ch_bottleneck)
        self.fuse = BGAwareFuse(self.backbone.channels, ch_bottleneck)
        decoder_ch = list(self.fuse.ch_feats_out)
        decoder_ch[-1] = ch_bottleneck
        self.decoder = RecurrentDecoder(decoder_ch, [80, 40, 32, 16])
            
        # self.project_mat = Projection(16, 1)
        self.is_output_fg = is_output_fg
        self.project_mat = Projection(16, 4 if is_output_fg else 1)
        self.project_seg = Projection(16, 1)

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
        src = torch.cat([query_img, memory_img], 1) # T
        if downsample_ratio != 1:
            src_sm = self._interpolate(src, scale_factor=downsample_ratio)
            mask_sm = self._interpolate(memory_mask, scale_factor=downsample_ratio)
        else:
            src_sm = src
            mask_sm = memory_mask
        
        feats = self.backbone(src_sm)
        feats_q, feats_m = zip(*[feat.split([num_query, 1], dim=1) for feat in feats])
        feats = self.fuse(feats_q, feats_m, mask_sm)

        # print(f4.shape, f3[:, :num_query].shape)
        src_sm = src_sm[:, :num_query, :3] # get the query rgb back
        
        hid, *rec = self.decoder(src_sm, *feats, r1, r2, r3, r4)
        rec, out_feats = rec[:-1], rec[-1]
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

class BGAwareFuse(nn.Module):
    def __init__(self, ch_feats, ch_out):
        super().__init__()
        self.aspp = LRASPP(ch_feats[-1]+2, ch_out)
        self.attn = nn.ModuleList([
            ChannelAttention(ch_feats[i]*2, ch_feats[i])
            for i in range(4)
        ])
        self.avgpool = AvgPool(num=4)
        patch_size = [16, 8, 8, 4]
        self.matches = nn.ModuleList([
            GlobalMatch(patch_size[i])
            for i in range(4)
        ])
        self.ch_feats_out = [f+2 for f in ch_feats]
        self.ch_feats_out[-1] = ch_out
        
    def forward(self, feats_q, feats_m, masks_m):
        # num_q = feats_q[0].size(1)
        # B, num_q = feats_q[0].shape[:2]
        # fg_masks = [(m>1e-4).float() for m in self.avgpool(masks_m)]
        # bg_masks = [(m<(1-1e-4)).float() for m in ]
        masks = self.avgpool(masks_m)
        feats_ret = []
        for i in range(4):
            fq = feats_q[i]
            mask = masks[i]
            fg_feat = feats_m[i]
            bg_feat = feats_m[i]
            # print(fg_mask.isnan().any(), bg_mask.isnan().any())
            # print(feats_m[i].isnan().any(), fg_feat.isnan().any(), bg_feat.isnan().any())
            # B, T, 1, H, W
            attn = self.attn[i](
                    torch.cat([
                    self.weighted_mean(fg_feat, mask),
                    self.weighted_mean(bg_feat, 1-mask)
                ], dim=2)
            )
            attn = attn.view(*attn.shape, 1, 1)

            f = torch.cat([
                attn*fq,
                self.matches[i](fq, fg_feat*(mask>1e-4).float()), 
                self.matches[i](fq, bg_feat*(mask<(1-1e-4)).float())], dim=2)
            feats_ret.append(f)

        feats_ret[3] = self.aspp(feats_ret[3])
        return feats_ret
    
    @staticmethod
    def weighted_mean(x, w):
        return (x*w).sum(dim=(-2, -1)) / (w.sum(dim=(-2, -1)) + 1e-5)


class BGAwareFuse2(nn.Module):
    def __init__(self, ch_feats, ch_out):
        super().__init__()
        self.ch_feats_out = list(ch_feats)
        self.ch_feats_out[-1] = ch_out

        self.attn = nn.ModuleList([
            ChannelAttention(ch_feats[i]*2, self.ch_feats_out[i])
            for i in range(4)
        ])
        self.avgpool = AvgPool(num=4)
        
        patch_size = [8, 4, 2, 1]
        self.conv_attn = nn.ModuleList([
            ConvSelfAttention(dim=ch_feats[i], attn_dim=32, head=1, patch_size=patch_size[i])
            for i in range(4)
        ])
        
        k = [3, 3, 3]
        self.align_conv = nn.ModuleList([
            AlignDeformConv2d(ch_feats[i]+1, ch_feats[i], ch_feats[i], 
                kernel_size=k[i], stride=1, padding=k[i]//2)
            for i in range(3)
        ])

        self.fuse = nn.ModuleList([
            FeatureFusion3(ch_feats[i]*2, ch_feats[i])
            for i in range(3)
        ])
        self.fuse.append(LRASPP(ch_feats[-1]*2, ch_out))
        # self.aspp = LRASPP(ch_feats[-1], ch_out)

        
        
    def forward(self, feats_q, feats_m, masks_m):
        # num_q = feats_q[0].size(1)
        # B, num_q = feats_q[0].shape[:2]
        # fg_masks = [(m>1e-4).float() for m in self.avgpool(masks_m)]
        # bg_masks = [(m<(1-1e-4)).float() for m in ]
        masks = self.avgpool(masks_m)
        feats_ret = []
        extra_out = []
        for i in range(4):
            mask = masks[i]
            fq = feats_q[i]
            fm = feats_m[i]

            ch_attn = self.attn[i](
                    torch.cat([
                    self.weighted_mean(fm, mask),
                    self.weighted_mean(fm, 1-mask)
                ], dim=2)
            )
            ch_attn = ch_attn.view(*ch_attn.shape, 1, 1)

            if i < 3:
                attn_val, A = self.conv_attn[i](fq, fm, mask)
                extra_out.append(attn_val[:, :, [-1]]) # B, T, 1, H, W
                attn_val = self.align_conv[i](attn_val, fq)
            else:
                attn_val, A = self.conv_attn[i](fq, fm)

            f = self.fuse[i](torch.cat([
                    fq,
                    attn_val
                ], dim=2))
            # f = ch_attn + attn_val
            feats_ret.append(f*ch_attn)
            # extra_out.append(A)
        
        # feats_ret[-1] = self.aspp(feats_ret[-1])
        return feats_ret, extra_out
    
    @staticmethod
    def weighted_mean(x, w):
        return (x*w).sum(dim=(-2, -1)) / (w.sum(dim=(-2, -1)) + 1e-5)

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

class MattingFramework(nn.Module):
    def __init__(
        self,
        encoder: Backbone,
        middle: BGAwareFuse,
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
            query_img: Tensor,
            memory_img: Tensor,
            memory_mask: Tensor,
            # memory_feat: list = [None, None, None, None],
            r1=None, r2=None, r3=None, r4=None,
            downsample_ratio: float = 1,
            segmentation_pass: bool = False
        ):
        
        # B, T, C, H, W
        # num_query = query_img.size(1)
        if is_refine := (downsample_ratio != 1):
            qimg_sm = self._interpolate(query_img, scale_factor=downsample_ratio)
            mimg_sm = self._interpolate(memory_img, scale_factor=downsample_ratio)
            mask_sm = self._interpolate(memory_mask, scale_factor=downsample_ratio)
        else:
            qimg_sm = query_img
            mimg_sm = memory_img
            mask_sm = memory_mask
        
        feats_q, feats_m = self.encoder(qimg_sm, mimg_sm, mask_sm)
        
        feats, middle_out = self.middle(feats_q, feats_m, mask_sm)

        # out_feats, *_ = self.decoder(qimg_sm, *feats, *memory_feat)
        out_feats, *_ = self.decoder(qimg_sm, *feats, r1, r2, r3, r4)
        memory_feat, inner_feats = _[:-1], _[-1]

        ret = self.out(query_img, qimg_sm, out_feats, is_refine, segmentation_pass)
        return *ret, memory_feat, middle_out

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


class GFM_VM(nn.Module):
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
        # self.aspp = FeatureFusion(self.backbone.channels[-1]*2, ch_bottleneck)
        
        decoder_ch = list(self.backbone.channels)
        decoder_ch[-1] = ch_bottleneck
        
        self.out_ch = [80, 40, 32, 16]
        self.decoder_focus = RecurrentDecoder(decoder_ch, self.out_ch, gru=gru)
        self.is_output_fg = is_output_fg
        self.project_mat = Projection(self.out_ch[-1], 4 if is_output_fg else 1)

        self.decoder_glance = FramewiseDecoder(decoder_ch, out_channel=3)

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
        f4_q, f4_m = f4.split([num_query, 1], dim=1)

        f4_m = torch.repeat_interleave(f4_m, num_query, dim=1)
        f4 = self.aspp(torch.cat([f4_q, f4_m], 2))
        
        src_sm = src_sm[:, :num_query, :3] # get the query rgb back
        f1 = f1[:, :num_query]
        f2 = f2[:, :num_query]
        f3 = f3[:, :num_query]

        out_glance = self.decoder_glance(src_sm, f1, f2, f3, f4)
        if segmentation_pass:
            return [out_glance]

        hid, *rec = self.decoder_focus(src_sm, f1, f2, f3, f4, r1, r2, r3, r4)
        rec, feats = rec[:-1], rec[-1]

        out_focus = torch.sigmoid(self.project_mat(hid))
        return [out_glance, out_focus, self._collaborate(out_glance, out_focus), rec, feats]

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


class GFM_VM_Rec(nn.Module):
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
        # self.aspp = FeatureFusion(self.backbone.channels[-1]*2, ch_bottleneck)
        
        decoder_ch = list(self.backbone.channels)
        decoder_ch[-1] = ch_bottleneck
        
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
        f4_q, f4_m = f4.split([num_query, 1], dim=1)

        f4_m = torch.repeat_interleave(f4_m, num_query, dim=1)
        f4 = self.aspp(torch.cat([f4_q, f4_m], 2))
        
        src_sm = src_sm[:, :num_query, :3] # get the query rgb back
        f1 = f1[:, :num_query]
        f2 = f2[:, :num_query]
        f3 = f3[:, :num_query]

        hid, *rec = self.decoder_glance(src_sm, f1, f2, f3, f4, *rec_glance)
        rec_glance, feats = rec[:-1], rec[-1]
        out_glance = self.project_seg(hid)
        if segmentation_pass:
            return [out_glance, rec_glance]

        hid, *rec = self.decoder_focus(src_sm, f1, f2, f3, f4, *rec_focus)
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

class GFM_FuseVM(nn.Module):
    def __init__(self,
                backbone_arch: str = 'mobilenetv3_large_100',
                backbone_pretrained = True,
                refiner: str = 'deep_guided_filter',
                is_output_fg = False,
                gru=ConvGRU,
                fuse='GFMFuse'
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

class PredictBG(nn.Module):
    def __init__(self, ch_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(ch_in, ch_in//2, 3, 1, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(ch_in//2, 3, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        if x.ndim == 5:
            B, T = x.shape[:2]
            return self.net(x.flatten(0, 1)).unflatten(0, (B, T))
        return self.net(x)

class BGVM(DualMattingNetwork):
    def __init__(self, backbone_arch: str = 'mobilenetv3_large_100', backbone_pretrained=True, refiner: str = 'deep_guided_filter', is_output_fg=False, gru=ConvGRU):
        super().__init__(backbone_arch, backbone_pretrained, refiner, is_output_fg, gru)

        self.ch_bg = self.out_ch[1]//2
        self.project_bg = PredictBG(self.ch_bg)
        self.bg_slice = (self.ch_bg//2, -self.ch_bg//2)

     
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
        rec, feats = rec[:-1], rec[-1] # 2 4 8 16
        out_bg = self.get_bg(feats)
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
                return [pha, fgr, rec, out_bg]

            return [torch.sigmoid(self.project_mat(hid)), rec, out_bg] # memory of gru
        else:
            seg = self.project_seg(hid)
            return [seg, rec, out_bg] # memory of gru
    
    def get_bg(self, feats):
        return self.project_bg(feats[1][:, :, self.bg_slice[0]:self.bg_slice[1]])

class BGVM_reconly(BGVM):
    def __init__(self, backbone_arch: str = 'mobilenetv3_large_100', backbone_pretrained=True, refiner: str = 'deep_guided_filter', is_output_fg=False, gru=ConvGRU):
        super().__init__(backbone_arch, backbone_pretrained, refiner, is_output_fg, gru)
        self.ch_bg = self.out_ch[1]//4
        self.project_bg = PredictBG(self.ch_bg)
        self.bg_slice = -self.ch_bg

    def get_bg(self, feats):
        return self.project_bg(feats[1][:, :, self.bg_slice:])

class GFM_Fuse8xVM(nn.Module):
    def __init__(self,
                backbone_arch: str = 'mobilenetv3_large_100',
                backbone_pretrained = True,
                refiner: str = 'deep_guided_filter',
                is_output_fg = False,
                gru=ConvGRU,
                fuse='GFMFuse8x'
        ):
        super().__init__()
        assert refiner in ['fast_guided_filter', 'deep_guided_filter']
        
        self.backbone = RGBMaskEncoder(
            backbone_arch=backbone_arch, 
            backbone_pretrained=backbone_pretrained,
            out_indices=list(range(4))) # only for 4 stages
        

        ch_bottleneck = 128
        self.fuse = {
            'GFMFuse8x': GFMFuse8x,
        }[fuse](self.backbone.channels, ch_bottleneck, gru)
        decoder_ch = list(self.fuse.ch_feats_out)
        # self.out_ch = [80, 40, 32, 16]
        self.out_ch = 16
        self.decoder_focus = RecurrentDecoder8x(decoder_ch, self.out_ch, gru=gru)
        self.is_output_fg = is_output_fg
        self.project_mat = Projection(self.out_ch, 4 if is_output_fg else 1)
        self.decoder_glance = FramewiseDecoder8x(decoder_ch, self.out_ch)
        
        self.project_seg = Projection(self.out_ch, 3)

        self.avgpool = AvgPool(3)
        if refiner == 'deep_guided_filter':
            self.refiner = DeepGuidedFilterRefiner()
        else:
            self.refiner = FastGuidedFilterRefiner()
        
    def forward(self,
                query_img: Tensor,
                memory_img: Tensor,
                memory_mask: Tensor,
                rec = [None, None, None, None],
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
        
        
        imgs_sm = self.avgpool(qimg_sm)
        feats_q, feats_m = self.backbone(qimg_sm, mimg_sm, mask_sm)
        feats_gl, feats_fc, r3r4 = self.fuse(imgs_sm, feats_q, feats_m, mask_sm, rec[2], rec[3])

        hid = self.decoder_glance(qimg_sm, *imgs_sm[:2], *feats_gl)
        out_glance = self.project_seg(hid)

        if segmentation_pass:
            return [out_glance, [None, None] + r3r4]

        hid, *rec = self.decoder_focus(qimg_sm, *imgs_sm[:2], *feats_fc, *rec[:2])
        r1r2, feats = rec[:-1], rec[-1]
        out_focus = torch.sigmoid(self.project_mat(hid))
        return [out_glance, out_focus, self._collaborate(out_glance, out_focus), r1r2+r3r4, feats]

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
    
class GFMFuse8x(nn.Module):
    # fuse and upsample to 8x
    def __init__(self, ch_feats, ch_bottleneck, gru=ConvGRU):
        super().__init__()
        self.ch_feats_out = list(ch_feats)
        self.ch_feats_out[-1] = ch_bottleneck

        self.sa = ConvSelfAttention(ch_feats[-1], head=1, patch_size=1, drop_p=0)
        self.bottleneck = PSP(ch_feats[-1]*2, ch_feats[-1]*2//4, ch_bottleneck)
        # self.attn = nn.ModuleList([
        #     ChannelAttention(ch_feats[i]*2, ch_bottleneck if i == 3 else ch_feats[i])
        #     for i in range(4)
        # ])

        self.decoder_8x = RecurrentDecoderTo8x(self.ch_feats_out, gru=gru)
        self.avgpool = AvgPool(num=4)
        
    def forward(self, imgs, feats_q, feats_m, masks_m, r3, r4):
        feats_q = list(feats_q)
        f4, A = self.sa(feats_q[3], feats_m[3])
        f4 = self.bottleneck(torch.cat([feats_q[3], f4], dim=2))
        f3, x4, r3, r4 = self.decoder_8x(imgs[2], feats_q[2], f4, r3, r4)
        feats_glance = feats_q[:2] + [f3]
        feats_focus = feats_q[:2] + [f3]

        return feats_glance, feats_focus, [r3, r4]
    
    @staticmethod
    def weighted_mean(x, w, keepdim=False):
        return (x*w).sum(dim=(-2, -1), keepdim=keepdim) / (w.sum(dim=(-2, -1), keepdim=keepdim) + 1e-5)

class GFMFuse8xBG(nn.Module):
    # fuse and upsample to 8x
    def __init__(self, ch_feats, ch_bottleneck, gru=ConvGRU):
        super().__init__()
        self.ch_feats_out = list(ch_feats)
        self.ch_feats_out[1] *= 2 # enc_feat + bg_feat
        self.ch_feats_out[3] = ch_bottleneck

        self.sa = ConvSelfAttention(ch_feats[-1], head=1, patch_size=1, drop_p=0)
        self.bottleneck = PSP(ch_feats[-1]*2, ch_feats[-1]*2//4, ch_bottleneck)
        # self.attn = nn.ModuleList([
        #     ChannelAttention(ch_feats[i]*2, ch_bottleneck if i == 3 else ch_feats[i])
        #     for i in range(4)
        # ])

        self.decoder_8x = RecurrentDecoderTo8x(self.ch_feats_out, gru=gru)
        self.avgpool = AvgPool(num=4)

        self.bg_up = UpsampleBlock(ch_feats[1]+3, ch_feats[2], ch_feats[1])
        self.bg_inpaint = BGInpaintingGRU(ch_feats[1])

        
        
    def forward(self, imgs, feats_q, feats_m, masks_m, r3, r4, rbg):
        f4, A = self.sa(feats_q[3], feats_m[3])
        f4 = self.bottleneck(torch.cat([feats_q[3], f4], dim=2))
        f3, x4, r3, r4 = self.decoder_8x(imgs[2], feats_q[2], f4, r3, r4)
        
        b, t = f3.shape[:2]
        bg_f2 = self.bg_up(torch.cat([feats_q[1], imgs[1]], dim=2).flatten(0, 1), f3.flatten(0, 1)).unflatten(0, (b, t))
        bg_f2, rbg, out_bg_mask = self.bg_inpaint(bg_f2, rbg)

        out_feats = [feats_q[0], torch.cat([feats_q[1], bg_f2], dim=-3), f3]
        feats_glance = out_feats
        feats_focus = out_feats

        return feats_glance, feats_focus, [r3, r4, rbg], out_bg_mask
    
    @staticmethod
    def weighted_mean(x, w, keepdim=False):
        return (x*w).sum(dim=(-2, -1), keepdim=keepdim) / (w.sum(dim=(-2, -1), keepdim=keepdim) + 1e-5)

class BGInpaintingGRU(nn.Module):
    # TODO: Additional BG gru inpainting block 
    # 1. coarse mask -> get BG feature
    # 2. soft attn (last feat & cur feat)
    # 3. spatial scaling (due to soft composition)
    # 4. decode to BG
    def __init__(self, ch_in: int, gru=AttnGRU):
        super().__init__()
        self.gru = gru(ch_in)
        self.pred_mask = nn.Sequential(
            nn.Conv2d(ch_in, 8, 3, 1, 1),
            nn.ReLU(True),
            # nn.LeakyReLU(0.2, True),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )
        self.pred_bg = PredictBG(ch_in)
        
    def forward(self, x, h=None):
        if x.ndim == 5:
            b, t = x.shape[:2]
            coarse_mask = self.pred_mask(x.flatten(0, 1)).unflatten(0, (b, t))
        else:
            coarse_mask = self.pred_mask(x)
        
        x, h = self.gru(x*(1-coarse_mask), h)
        bg = self.pred_bg(x)
        return x, h, torch.cat([bg, coarse_mask], dim=-3)

class BGInpaintingAttn(nn.Module):
    # TODO: Additional BG gru inpainting block 
    # 1. coarse mask -> get BG feature
    # 2. soft attn (last feat & cur feat)
    # 3. spatial scaling (due to soft composition)
    # 4. decode to BG
    def __init__(self, ch_in: int):
        super().__init__()
        self.attn = SoftCrossAttention(ch_in, 16, head=1)
        self.pred_feat_mask = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch_in, ch_in+1, 1)
        )
        self.pred_bg = PredictBG(ch_in)
        self.ch_in = ch_in

    def forward(self, x, h=None):
        if x.ndim == 5:
            b, t = x.shape[:2]
            f, mask = self.pred_feat_mask(x.flatten(0, 1)).split([self.ch_in, 1], dim=1)
            f = ((1-torch.sigmoid(mask))*f).unflatten(0, (b, t))
            mask = mask.unflatten(0, (b, t))
        else:
            f, mask = self.pred_feat_mask(x).split([self.ch_in, 1], dim=1)
            f = (1-torch.sigmoid(mask))*f
        


        bg = self.pred_bg(x)
        return x, h, torch.cat([bg, mask], dim=-3)

class GFM_BGFuse8xVM(nn.Module):
    def __init__(self,
                backbone_arch: str = 'mobilenetv3_large_100',
                backbone_pretrained = True,
                refiner: str = 'deep_guided_filter',
                is_output_fg = False,
                gru=ConvGRU,
                fuse='BG8x'
        ):
        super().__init__()
        assert refiner in ['fast_guided_filter', 'deep_guided_filter']
        
        self.backbone = RGBMaskEncoder(
            backbone_arch=backbone_arch, 
            backbone_pretrained=backbone_pretrained,
            out_indices=list(range(4))) # only for 4 stages
        

        ch_bottleneck = 128
        self.fuse = {
            'BG8x': GFMFuse8xBG,
        }[fuse](self.backbone.channels, ch_bottleneck, gru)
        decoder_ch = list(self.fuse.ch_feats_out)
        # print(decoder_ch)
        # self.out_ch = [80, 40, 32, 16]
        self.out_ch = 16
        self.decoder_focus = RecurrentDecoder8x(decoder_ch, self.out_ch, gru=gru)
        self.is_output_fg = is_output_fg
        self.project_mat = Projection(self.out_ch, 4 if is_output_fg else 1)
        self.decoder_glance = FramewiseDecoder8x(decoder_ch, self.out_ch)
        
        self.project_seg = Projection(self.out_ch, 3)

        self.avgpool = AvgPool(3)
        if refiner == 'deep_guided_filter':
            self.refiner = DeepGuidedFilterRefiner()
        else:
            self.refiner = FastGuidedFilterRefiner()
        
    def forward(self,
                query_img: Tensor,
                memory_img: Tensor,
                memory_mask: Tensor,
                rec = [None, None, None, None, None],
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
        
        
        imgs_sm = self.avgpool(qimg_sm)
        feats_q, feats_m = self.backbone(qimg_sm, mimg_sm, mask_sm)
        feats_gl, feats_fc, r3r4rbg, out_bg_mask = self.fuse(imgs_sm, feats_q, feats_m, mask_sm, *rec[2:])

        hid = self.decoder_glance(qimg_sm, *imgs_sm[:2], *feats_gl)
        out_glance = self.project_seg(hid)

        if segmentation_pass:
            return [out_glance, [None, None] + r3r4rbg, out_bg_mask]

        hid, *rec = self.decoder_focus(qimg_sm, *imgs_sm[:2], *feats_fc, *rec[:2])
        r1r2, feats = rec[:-1], rec[-1]
        out_focus = torch.sigmoid(self.project_mat(hid))
        return [out_glance, out_focus, self._collaborate(out_glance, out_focus), r1r2+r3r4rbg, out_bg_mask]

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