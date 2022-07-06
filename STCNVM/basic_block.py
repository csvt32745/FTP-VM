import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class GatedConv2d(nn.Module):
    def __init__(self, ch_in, ch_out, kernel=1, stride=1, padding=0, act=nn.LeakyReLU(0.1)):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out*2, kernel, stride, padding)
        self.ch_out = ch_out
        self.act = act
    
    def forward(self, x):
        if x.ndim == 5:
            b, t = x.shape[:2]
            x, m = self.conv(x.flatten(0, 1)).split(self.ch_out, dim=1)
            return (self.act(x)*torch.sigmoid(m)).unflatten(0, (b, t))
            
        x, m = self.conv(x).split(self.ch_out, dim=1)
        return self.act(x)*torch.sigmoid(m)


class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
 
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)
 
    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
        x = self.skip(x)
        return x + r

# class UpsampleBlock(nn.Module):
#     def __init__(self, skip_c, up_c, out_c, scale_factor=2):
#         super().__init__()
#         self.skip_conv = nn.Conv2d(skip_c, up_c, kernel_size=3, padding=1)
#         self.out_conv = ResBlock(up_c, out_c)
#         self.scale_factor = scale_factor

#     def forward(self, skip_f, up_f):
#         x = self.skip_conv(skip_f)
#         x = x + F.interpolate(up_f, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
#         x = self.out_conv(x)
#         return x

class UpsampleBlock(nn.Module):
    def __init__(self, skip_c, up_c, out_c, scale_factor=2):
        super().__init__()
        # self.skip_conv = nn.Conv2d(skip_c, up_c, kernel_size=3, padding=1)
        self.out_conv = ResBlock(up_c+skip_c, out_c)
        self.scale_factor = scale_factor

    def _forward(self, up_f, skip_f):
        up_f = F.interpolate(up_f, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        x = self.out_conv(torch.cat([up_f, skip_f], dim=1))
        return x
    
    def forward(self, up_f, skip_f):
        if up_f.ndim == 5:
            b, t = up_f.shape[:2]
            return self._forward(up_f.flatten(0, 1), skip_f.flatten(0, 1)).unflatten(0, (b, t))
        return self._forward(up_f, skip_f)


class AvgPool(nn.Module):
    def __init__(self, num=3):
        super().__init__()
        self.avgpool = nn.AvgPool2d(2, 2, count_include_pad=False, ceil_mode=True)
        self.num = num
        
    def forward_single_frame(self, s0):
        ret = []
        for i in range(self.num):
            s0 = self.avgpool(s0)
            ret.append(s0)
        return ret
    
    def forward_time_series(self, s0):
        B, T = s0.shape[:2]
        ret = self.forward_single_frame(s0.flatten(0, 1))
        return [r.unflatten(0, (B, T)) for r in ret]
    
    def forward(self, s0):
        if s0.ndim == 5:
            return self.forward_time_series(s0)
        else:
            return self.forward_single_frame(s0)