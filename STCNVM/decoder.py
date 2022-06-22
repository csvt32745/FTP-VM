import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .basic_block import *

class FramewiseDecoder(nn.Module):
    def __init__(self, feat_channels, out_channel=1):
        super().__init__()
        assert len(feat_channels) == 4
        ch_bottleneck = 16
        self.up_16_8 = UpsampleBlock(feat_channels[2]+3, feat_channels[3], feat_channels[2]) # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(feat_channels[1]+3, feat_channels[2], feat_channels[1]) # 1/8 -> 1/4
        self.up_4_2 = UpsampleBlock(feat_channels[0]+3, feat_channels[1], feat_channels[0]) # 1/4 -> 1/2
        
        self.up_2_1 = nn.Sequential(
            ResBlock(ch_bottleneck, ch_bottleneck),
            nn.InstanceNorm2d(ch_bottleneck),
            # nn.Conv2d(ch_bottleneck+3, ch_bottleneck, 3, 1, 1),
            # nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.final = ResBlock(ch_bottleneck+3, ch_bottleneck)
        self.pred = nn.Conv2d(ch_bottleneck, out_channel, kernel_size=1)
        self.pred_seg = nn.Conv2d(ch_bottleneck, out_channel, kernel_size=1)
        
        self.avg2d = AvgPool(num=3)
        

    def forward_4d(self, img, f2, f4, f8, f16):
        imgs = self.avg2d(img)
        x = self.up_16_8(torch.cat([f8, imgs[2]], dim=1), f16)
        x = self.up_8_4(torch.cat([f4, imgs[1]], dim=1), x)
        x = self.up_4_2(torch.cat([f2, imgs[0]], dim=1), x)
        x = self.up_2_1(x)
        x = self.final(torch.cat([x, img], 1))
        x = self.pred(x)
        return x
    
    def forward(self, *args):
        if args[0].ndim == 5:
            B, T = args[0].shape[:2]
            ret = self.forward_4d(*[a.flatten(0, 1) for a in args])
            return ret.unflatten(0, (B, T))
        else:
            return self.forward_4d(*args)


class FramewiseDecoder8x(nn.Module):
    def __init__(self, feat_channels, ch_out):
        super().__init__()
        assert len(feat_channels) == 4
        self.up_8_4 = UpsampleBlock(feat_channels[1]+3, feat_channels[2], feat_channels[1]) # 1/8 -> 1/4
        self.up_4_2 = UpsampleBlock(feat_channels[0]+3, feat_channels[1], feat_channels[0]) # 1/4 -> 1/2
        self.up_2_1 = nn.Sequential(
            ResBlock(ch_out, ch_out),
            nn.InstanceNorm2d(ch_out),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.final = ResBlock(ch_out+3, ch_out)
        
    def forward_4d(self, img, img2, img4, f2, f4, x8):
        x = self.up_8_4(torch.cat([f4, img4], dim=1), x8)
        x = self.up_4_2(torch.cat([f2, img2], dim=1), x)
        x = self.up_2_1(x)
        x = self.final(torch.cat([x, img], 1))
        return x
    
    def forward(self, *args):
        if args[0].ndim == 5:
            B, T = args[0].shape[:2]
            ret = self.forward_4d(*[a.flatten(0, 1) for a in args])
            return ret.unflatten(0, (B, T))
        else:
            return self.forward_4d(*args)

class FramewiseDecoder4x(nn.Module):
    def __init__(self, feat_channels, ch_out):
        super().__init__()
        assert len(feat_channels) == 4
        self.up_8_4 = UpsampleBlock(feat_channels[1]+3, feat_channels[2], feat_channels[1]) # 1/8 -> 1/4
        self.up_4_2 = UpsampleBlock(feat_channels[0]+3, feat_channels[1], feat_channels[0]) # 1/4 -> 1/2
        self.up_2_1 = nn.Sequential(
            ResBlock(ch_out, ch_out),
            nn.InstanceNorm2d(ch_out),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.final = ResBlock(ch_out+3, ch_out)
        
    def forward_4d(self, img, img2, img4, f2, f4, x8):
        x = self.up_8_4(torch.cat([f4, img4], dim=1), x8)
        x = self.up_4_2(torch.cat([f2, img2], dim=1), x)
        x = self.up_2_1(x)
        x = self.final(torch.cat([x, img], 1))
        return x
    
    def forward(self, *args):
        if args[0].ndim == 5:
            B, T = args[0].shape[:2]
            ret = self.forward_4d(*[a.flatten(0, 1) for a in args])
            return ret.unflatten(0, (B, T))
        else:
            return self.forward_4d(*args)