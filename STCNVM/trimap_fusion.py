from functools import lru_cache
from typing import Iterable, List, Optional
import torch
from torch import Tensor
import math
from torch import nn
from torch.nn import functional as F
from torchvision.ops.deform_conv import DeformConv2d
import kornia as K
from einops import rearrange
from functools import reduce

# from .STCN.network import *
# from .STCN.modules import *

from .basic_block import ResBlock, GatedConv2d, AvgPool, FocalModulation
from .recurrent_decoder import ConvGRU
from . import cbam

class TrimapGatedFusion(nn.Module):
    def __init__(self, ch_feats, ch_mask=1):
        super().__init__()
        assert len(ch_feats) == 4
        
        self.gated_convs = nn.ModuleList([
            nn.Sequential(
                GatedConv2d(ch_feats[i]+ch_mask+(ch_feats[i-1] if i > 0 else 0), ch_feats[i], 3, 1, 1),
                nn.Conv2d(ch_feats[i], ch_feats[i], 3, 1, 1),
                nn.LeakyReLU(0.2),
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

class TrimapGatedFusionBN(TrimapGatedFusion):
    def __init__(self, ch_feats, ch_mask=1):
        super().__init__(ch_feats, ch_mask)
        del self.gated_convs
        self.gated_convs = nn.ModuleList([
            nn.Sequential(
                GatedConv2d(ch_feats[i]+ch_mask+(ch_feats[i-1] if i > 0 else 0), ch_feats[i], 3, 1, 1),
                nn.BatchNorm2d(ch_feats[i]),
                nn.Conv2d(ch_feats[i], ch_feats[i], 3, 1, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(ch_feats[i]),
                nn.AvgPool2d((2, 2)) if i < 3 else nn.Identity(),
            )
            for i in range(4)
        ])
        self.avgpool = AvgPool(num=4)

class TrimapGatedFusionFullGate(TrimapGatedFusion):
    def __init__(self, ch_feats, ch_mask=1):
        super().__init__(ch_feats, ch_mask)
        del self.gated_convs
        self.gated_convs = nn.ModuleList([
            nn.Sequential(
                GatedConv2d(ch_feats[i]+ch_mask+(ch_feats[i-1] if i > 0 else 0), ch_feats[i], 3, 1, 1),
                nn.BatchNorm2d(ch_feats[i]),
                GatedConv2d(ch_feats[i], ch_feats[i], 3, 1, 1),
                nn.BatchNorm2d(ch_feats[i]),
                nn.AvgPool2d((2, 2)) if i < 3 else nn.Identity(),
            )
            for i in range(4)
        ])
        self.avgpool = AvgPool(num=4)
        
class TrimapGatedFusionSmall(TrimapGatedFusion):
    def __init__(self, ch_feats, ch_mask=1):
        super().__init__(ch_feats, ch_mask)
        del self.gated_convs
        self.gated_convs = nn.ModuleList([
            nn.Sequential(
                GatedConv2d(ch_feats[i]+ch_mask+(ch_feats[i-1] if i > 0 else 0), ch_feats[i], 3, 1, 1),
                nn.BatchNorm2d(ch_feats[i]),
                nn.AvgPool2d((2, 2)) if i < 3 else nn.Identity(),
            )
            for i in range(4)
        ])
        self.avgpool = AvgPool(num=4)


class TrimapGatedFusionFullRes(nn.Module):
    def __init__(self, ch_feats, ch_mask=1):
        super().__init__()
        assert len(ch_feats) == 4
        ch_outs = [16] + ch_feats
        ch_feats = [3] + ch_feats
        self.gated_convs = nn.ModuleList([
            nn.Sequential(
                GatedConv2d(ch_feats[i]+ch_mask+(ch_outs[i-1] if i > 0 else 0), ch_outs[i], 3, 1, 1),
                nn.Conv2d(ch_outs[i], ch_outs[i], 3, 1, 1),
                nn.LeakyReLU(0.2),
                nn.AvgPool2d((2, 2)) if i < 4 else nn.Identity(),
            )
            for i in range(5)
        ])
        self.avgpool = AvgPool(num=4)
        
    def forward(self, img: Tensor, mask: Tensor, feats: List[Tensor]):
        masks_sm = self.avgpool(mask)
        bt = img.shape[:2]
        f = self.gated_convs[0](torch.cat([img, mask], dim=2).flatten(0, 1))
        for i in range(4):
            f = torch.cat([f, feats[i].flatten(0, 1), masks_sm[i].flatten(0, 1)],  dim=1)
            f = self.gated_convs[i+1](f)
        # f = f.view(*bt, f.shape[1:])
        f = f.unflatten(0, bt)
        return f.transpose(1, 2) # b, t, c, h, w -> b, c, t, h, w

class TrimapNaiveFusion(TrimapGatedFusion):
    def __init__(self, ch_feats, ch_mask=1):
        super().__init__(ch_feats, ch_mask)
        del self.gated_convs
        self.gated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch_feats[i]+ch_mask+(ch_feats[i-1] if i > 0 else 0), ch_feats[i], 3, 1, 1),
                nn.BatchNorm2d(ch_feats[i]),
                nn.Conv2d(ch_feats[i], ch_feats[i], 3, 1, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(ch_feats[i]),
                nn.AvgPool2d((2, 2)) if i < 3 else nn.Identity(),
            )
            for i in range(4)
        ])
        self.avgpool = AvgPool(num=4)

class TrimapGatedFusionInTrimapOnly(nn.Module):
    def __init__(self, ch_feats, ch_mask=1):
        super().__init__()
        self.gated_convs = nn.ModuleList([
            nn.Sequential(
                GatedConv2d(ch_feats[i]+(ch_feats[i-1] if i > 0 else ch_mask), ch_feats[i], 3, 1, 1),
                nn.BatchNorm2d(ch_feats[i]),
                nn.Conv2d(ch_feats[i], ch_feats[i], 3, 1, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(ch_feats[i]),
                nn.AvgPool2d((2, 2)) if i < 3 else nn.Identity(),
            )
            for i in range(4)
        ])
        self.avgpool = AvgPool(num=1)
        
    def forward(self, img: Tensor, mask: Tensor, feats: List[Tensor]):
        masks_sm = self.avgpool(mask)
        f = feats[0]
        bt = f.shape[:2]
        f = self.gated_convs[0](torch.cat([f, masks_sm[0]], dim=2).flatten(0, 1))
        for i in range(1, 4):
            f = torch.cat([f, feats[i].flatten(0, 1)],  dim=1)
            f = self.gated_convs[i](f)
        # f = f.view(*bt, f.shape[1:])
        f = f.unflatten(0, bt)
        return f.transpose(1, 2) # b, t, c, h, w -> b, c, t, h, w

class TrimapGatedFusionInTrimapOnlyFullres(nn.Module):
    def __init__(self, ch_feats, ch_mask=1):
        super().__init__()
        assert len(ch_feats) == 4
        ch_outs = [16] + ch_feats
        ch_feats = [3] + ch_feats
        self.gated_convs = nn.ModuleList([
            nn.Sequential(
                GatedConv2d(ch_feats[i]+(ch_outs[i-1] if i > 0 else ch_mask), ch_outs[i], 3, 1, 1),
                nn.BatchNorm2d(ch_outs[i]),
                nn.Conv2d(ch_outs[i], ch_outs[i], 3, 1, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(ch_outs[i]),
                nn.AvgPool2d((2, 2)) if i < 4 else nn.Identity(),
            )
            for i in range(5)
        ])
        # self.avgpool = AvgPool(num=4)
        
    def forward(self, img: Tensor, mask: Tensor, feats: List[Tensor]):
        # masks_sm = self.avgpool(mask)
        bt = img.shape[:2]
        f = self.gated_convs[0](torch.cat([img, mask], dim=2).flatten(0, 1))
        for i in range(4):
            f = torch.cat([f, feats[i].flatten(0, 1)], dim=1)
            f = self.gated_convs[i+1](f)
        # f = f.view(*bt, f.shape[1:])
        f = f.unflatten(0, bt)
        return f.transpose(1, 2) # b, t, c, h, w -> b, c, t, h, w