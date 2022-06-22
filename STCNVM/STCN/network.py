"""
network.py - The core of the neural network
Defines the structure and memory operations
Modifed from STM: https://github.com/seoungwugoh/STM

The trailing number of a variable usually denote the stride
e.g. f16 -> encoded features with stride 16
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .modules import *


class Decoder(nn.Module):
    def __init__(self, feat_channels, out_channel=1):
        super().__init__()
        assert len(feat_channels) == 4
        ch_bottleneck = 16
        self.up_16_8 = UpsampleBlock(feat_channels[1], feat_channels[0], feat_channels[1]) # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(feat_channels[2], feat_channels[1], feat_channels[2]) # 1/8 -> 1/4
        self.up_4_2 = nn.Sequential(
            # nn.Conv2d(feat_channels[2]+3, ch_bottleneck, 3, 1, 1),
            ResBlock(feat_channels[2]+3, ch_bottleneck*2),
            nn.InstanceNorm2d(ch_bottleneck*2),
            # nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.up_2_1 = nn.Sequential(
            ResBlock(ch_bottleneck*2+3, ch_bottleneck),
            nn.InstanceNorm2d(ch_bottleneck),
            # nn.Conv2d(ch_bottleneck+3, ch_bottleneck, 3, 1, 1),
            # nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.final = ResBlock(ch_bottleneck+3, ch_bottleneck)
        self.pred = nn.Conv2d(ch_bottleneck, out_channel, kernel_size=1)
        self.pred_seg = nn.Conv2d(ch_bottleneck, out_channel, kernel_size=1)
        
        self.avg2d = nn.AvgPool2d((2, 2))
        

    def forward_4d(self, img, f16, f8, f4, segmentation_pass=False):
        # x = self.compress(f16)
        x = f16
        x = self.up_16_8(f8, x)
        x = self.up_8_4(f4, x)

        img2x = self.avg2d(img)
        img4x = self.avg2d(img2x)
        x = self.up_4_2(torch.cat([img4x, x], 1))
        x = self.up_2_1(torch.cat([img2x, x], 1))
        x = self.final(torch.cat([img, x], 1))
        if segmentation_pass:
            return self.pred_seg(x)

        x = self.pred(x)
        return torch.sigmoid(x)
    
    def forward(self, *args):
        if args[0].ndim == 5:
            B, T = args[0].shape[:2]
            args = [rearrange(arg, 'b t c h w -> (b t) c h w') for arg in args[:-1]] + [args[-1]]
            ret = self.forward_4d(*args)
            return rearrange(ret, '(b t) c h w -> b t c h w', b=B, t=T)
        else:
            return self.forward_4d(*args)


class MemoryReader(nn.Module):
    def __init__(self):
        super().__init__()
 
    def _get_affinity(self, mk, qk):
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

    def get_affinity(self, mk, qk):
        # Dot product
        B, CK, T, H, W = mk.shape
        mk = mk.flatten(start_dim=2)
        qk = qk.flatten(start_dim=2)

        ab = mk.transpose(1, 2) @ qk
        affinity = ab / math.sqrt(CK)   # B, THW, HW
        return F.softmax(affinity, dim=1)


    def readout(self, affinity, mv, qv):
        B, CV, T, H, W = mv.shape

        mo = mv.reshape(B, CV, -1) 
        # print(mo.shape, qv.shape, affinity.shape)
        mem = torch.bmm(mo, affinity) # Weighted-sum B, CV, HW
        mem = mem.view(B, -1, CV, H, W) # B, Tq, Cv, H, W
        mem_out = torch.cat([mem, qv], dim=2) # B, Tq, Cv', H, W

        return mem_out