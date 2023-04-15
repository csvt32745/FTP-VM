import torch
import math
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from torch import Tensor
from typing import Optional, List
from .basic_block import ResBlock, Projection, GatedConv2d, AvgPool
from . import cbam

class FeatureFusion(nn.Module):
    def __init__(self, indim, outdim, is_resblk=True):
        super().__init__()
        self.block = ResBlock(indim, outdim) if is_resblk \
            else nn.Sequential(
                nn.Conv2d(indim, outdim, 1),
                nn.ReLU(True),
                nn.BatchNorm2d(outdim),
            )
        self.attention = cbam.CBAM(outdim)

    def forward_single_frame(self, x):
        x = self.block(x)
        return x + self.attention(x)
    
    def forward_time_series(self, x):
        B, T = x.shape[:2]
        x = self.forward_single_frame(x.flatten(0, 1)).unflatten(0, (B, T))
        return x
    
    def forward(self, x):
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)

class PSP(nn.Module):
    def __init__(self, features, per_features, out_features, sizes=(1, 2, 4, 8)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, per_features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(per_features * len(sizes) + features, out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, per_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, per_features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def _forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners = True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)
    
    def forward(self, x):
        if x.ndim == 5:
            B, T = x.shape[:2]
            return self._forward(x.flatten(0, 1)).unflatten(0, (B, T))
        else:
            return self._forward(x)

class MemoryReader(nn.Module):
    def __init__(self, affinity='dotproduct'):
        super().__init__()
        self.get_affinity = {
            'l2': self.affinity_l2,
            'dotproduct': self.affinity_dotproduct,
        }[affinity]
 
    def affinity_l2(self, mk, qk):
        # L2 distance
        B, CK, T, H, W = mk.shape
        mk = mk.flatten(start_dim=2)
        qk = qk.flatten(start_dim=2)

        # See supplementary material
        a_sq = mk.pow(2).sum(1).unsqueeze(2)
        ab = mk.transpose(1, 2) @ qk

        affinity = (2*ab-a_sq) / math.sqrt(CK)   # B, THW, HW
        
        # softmax operation; aligned the evaluation style
        maxes = torch.max(affinity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(affinity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        affinity = x_exp / x_exp_sum 

        return affinity

    def affinity_dotproduct(self, mk, qk):
        # Dot product
        B, CK, T, H, W = mk.shape
        mk = mk.flatten(start_dim=2) # b, c, nm
        qk = qk.flatten(start_dim=2) # b, c, nq
        ab = mk.transpose(1, 2) @ qk # b, nm, nq
        affinity = ab / math.sqrt(CK)
        return F.softmax(affinity, dim=1)

    def readout(self, affinity, mv):
        B, CV, T, H, W = mv.shape
        mv = mv.reshape(B, CV, -1) # b, ch_val, nm
        val = torch.bmm(mv, affinity) # b, ch_val, nq
        val = rearrange(val, 'b c (t h w) -> b t c h w', h=H, w=W)
        return val

class TrimapGatedFusion(nn.Module):
    def __init__(self, ch_feats, ch_mask=1):
        super().__init__()
        assert len(ch_feats) == 4
        self.gated_convs = nn.ModuleList([
            nn.Sequential(
                GatedConv2d(ch_feats[i]+ch_mask+(ch_feats[i-1] if i > 0 else 0), ch_feats[i], 3, 1, 1),
                nn.GroupNorm(4, ch_feats[i]),
                nn.Conv2d(ch_feats[i], ch_feats[i], 3, 1, 1),
                nn.LeakyReLU(0.2),
                nn.GroupNorm(4, ch_feats[i]),
                nn.AvgPool2d((2, 2)) if i < 3 else nn.Identity(),
            )
            for i in range(4)
        ])

        self.avgpool = AvgPool(num=4)
        
    def forward(self, img: Tensor, mask: Tensor, feats: List[Tensor]):
        masks_sm = self.avgpool(mask)
        f = feats[0]
        bt = f.shape[:2]
        f = self.gated_convs[0](torch.cat([f, masks_sm[0]], dim=2).flatten(0, 1))
        for i in range(1, 4):
            f = torch.cat([f, feats[i].flatten(0, 1), masks_sm[i].flatten(0, 1)],  dim=1)
            f = self.gated_convs[i](f)
        # f = f.view(*bt, f.shape[1:])
        f = f.unflatten(0, bt)
        return f.transpose(1, 2) # b, t, c, h, w -> b, c, t, h, w

class BottleneckFusion(nn.Module):
    def __init__(self, ch_in, ch_key, ch_value, ch_out, affinity='dotproduct'):
        super().__init__()
        self.project_key = Projection(ch_in, ch_key)
        self.reader = MemoryReader(affinity=affinity)

        self.fuse = FeatureFusion(ch_in+ch_value, ch_out)
        self.bottleneck = PSP(ch_out, ch_out//4, ch_out)
        self.ch_out = ch_out
        
    def forward(self, f16_q, f16_m, value_m):
        f16_m = self.read_value(f16_q, f16_m, value_m)
        out = self.fuse(torch.cat([f16_q, f16_m], dim=2))
        out = self.bottleneck(out)
        return out
    
    def read_value(self, f16_q, f16_m, value_m):
        qk = self.encode_key(f16_q)
        mk = self.encode_key(f16_m)
        A = self.reader.get_affinity(mk, qk)
        return self.reader.readout(A, value_m) # value_m.shape == (b, c, t, h, w)

    def encode_key(self, feat16):
        # b, t, c, h, w -> b, ch_key, t, h, w
        return self.project_key(feat16).transpose(1, 2)