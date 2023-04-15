import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional

class Projection(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=1, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding)
    
    def forward_single_frame(self, x):
        return self.conv(x)
    
    def forward_time_series(self, x):
        return self.conv(x.flatten(0, 1)).unflatten(0, x.shape[:2])
        
    def forward(self, x):
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)

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

class UpsampleBlock(nn.Module):
    def __init__(self, skip_c, up_c, out_c, scale_factor=2, bn=False):
        super().__init__()
        # self.skip_conv = nn.Conv2d(skip_c, up_c, kernel_size=3, padding=1)
        self.out_conv = ResBlock(up_c+skip_c, out_c)
        self.scale_factor = scale_factor
        self.norm = nn.BatchNorm2d(out_c) if bn else nn.Identity()

    def _forward(self, up_f, skip_f):
        up_f = F.interpolate(up_f, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        x = self.norm(self.out_conv(torch.cat([up_f, skip_f], dim=1)))
        return x
    
    def forward(self, up_f, skip_f):
        if up_f.ndim == 5:
            b, t = up_f.shape[:2]
            return self._forward(up_f.flatten(0, 1), skip_f.flatten(0, 1)).unflatten(0, (b, t))
        return self._forward(up_f, skip_f)

class ConvGRU(nn.Module):
    def __init__(self,
                 channels: int,
                 kernel_size: int = 3,
                 padding: int = 1):
        super().__init__()
        self.channels = channels
        self.ih = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size, padding=padding),
            nn.Sigmoid()
        )
        self.hh = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size, padding=padding),
            nn.Tanh()
        )
        
    def forward_single_frame(self, x, h):
        r, z = self.ih(torch.cat([x, h], dim=1)).split(self.channels, dim=1)
        c = self.hh(torch.cat([x, r * h], dim=1))
        h = (1 - z) * h + z * c
        return h, h
    
    def forward_time_series(self, x, h):
        o = []
        for xt in x.unbind(dim=1):
            ot, h = self.forward_single_frame(xt, h)
            o.append(ot)
        o = torch.stack(o, dim=1)
        return o, h
        
    def forward(self, x, h: Optional[Tensor]):
        if h is None:
            h = torch.zeros((x.size(0), x.size(-3), x.size(-2), x.size(-1)),
                            device=x.device, dtype=x.dtype)
        
        if x.ndim == 5:
            return self.forward_time_series(x, h)
        else:
            return self.forward_single_frame(x, h)

class GRUBottleneckBlock(nn.Module):
    def __init__(self, channels, gru=ConvGRU):
        super().__init__()
        self.channels = channels
        self.gru = gru(channels // 2)
        
    def forward(self, x, r: Optional[Tensor]):
        a, b = x.split(self.channels // 2, dim=-3)
        b, r = self.gru(b, r)
        x = torch.cat([a, b], dim=-3)
        return x, r

class GRUUpsamplingBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, src_channels, out_channels, gru=ConvGRU):
        super().__init__()
        self.out_channels = out_channels
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels + src_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        self.gru = gru(out_channels // 2)

    def forward_single_frame(self, x, f, s, r: Optional[Tensor]):
        x = self.upsample(x)
        x = x[:, :, :s.size(2), :s.size(3)]
        x = torch.cat([x, f, s], dim=1)
        x = self.conv(x)
        a, b = x.split(self.out_channels // 2, dim=1)
        b, r = self.gru(b, r)
        x = torch.cat([a, b], dim=1)
        return x, r
    
    def forward_time_series(self, x, f, s, r: Optional[Tensor]):
        B, T, _, H, W = s.shape
        x = x.flatten(0, 1)
        f = f.flatten(0, 1)
        s = s.flatten(0, 1)
        x = self.upsample(x)
        x = x[:, :, :H, :W]
        x = torch.cat([x, f, s], dim=1)
        x = self.conv(x)
        x = x.unflatten(0, (B, T))
        a, b = x.split(self.out_channels // 2, dim=2)
        b, r = self.gru(b, r)
        x = torch.cat([a, b], dim=2)
        return x, r
    
    def forward(self, x, f, s, r: Optional[Tensor]):
        if x.ndim == 5:
            return self.forward_time_series(x, f, s, r)
        else:
            return self.forward_single_frame(x, f, s, r)

class GRUUpsamplingBlockWithoutSkip(GRUUpsamplingBlock):
    def __init__(self, in_channels, src_channels, out_channels, gru=ConvGRU):
        super().__init__(in_channels, 0, src_channels, out_channels, gru)

    def forward_single_frame(self, x, s, r: Optional[Tensor]):
        x = self.upsample(x)
        x = x[:, :, :s.size(2), :s.size(3)]
        x = torch.cat([x, s], dim=1)
        x = self.conv(x)
        a, b = x.split(self.out_channels // 2, dim=1)
        b, r = self.gru(b, r)
        x = torch.cat([a, b], dim=1)
        return x, r
    
    def forward_time_series(self, x, s, r: Optional[Tensor]):
        B, T, _, H, W = s.shape
        x = x.flatten(0, 1)
        s = s.flatten(0, 1)
        x = self.upsample(x)
        x = x[:, :, :H, :W]
        x = torch.cat([x, s], dim=1)
        x = self.conv(x)
        x = x.unflatten(0, (B, T))
        a, b = x.split(self.out_channels // 2, dim=2)
        b, r = self.gru(b, r)
        x = torch.cat([a, b], dim=2)
        return x, r
    
    def forward(self, x, s, r: Optional[Tensor]):
        if x.ndim == 5:
            return self.forward_time_series(x, s, r)
        else:
            return self.forward_single_frame(x, s, r)
