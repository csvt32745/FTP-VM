import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Tuple, Optional

from .basic_block import AvgPool, ResBlock, GatedConv2d, UpsampleBlock

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

class RecurrentDecoder(nn.Module):
    def __init__(self, feature_channels, decoder_channels, gru=ConvGRU, no_img_skip=False):
        super().__init__()
        self.avgpool = AvgPool()
        self.decode4 = GRUBottleneckBlock(feature_channels[3])
        upblk = GRUUpsamplingBlockNoImg if no_img_skip else GRUUpsamplingBlock
        self.decode3 = upblk(feature_channels[3], feature_channels[2], 3, decoder_channels[0], gru=gru)
        self.decode2 = upblk(decoder_channels[0], feature_channels[1], 3, decoder_channels[1], gru=gru)
        self.decode1 = upblk(decoder_channels[1], feature_channels[0], 3, decoder_channels[2], gru=gru)
        self.decode0 = OutputBlock(decoder_channels[2], 3, decoder_channels[3])

    def forward(self,
                s0: Tensor, f1: Tensor, f2: Tensor, f3: Tensor, f4: Tensor,
                r1: Optional[Tensor], r2: Optional[Tensor],
                r3: Optional[Tensor], r4: Optional[Tensor]):
        s1, s2, s3 = self.avgpool(s0)
        x4, r4 = self.decode4(f4, r4)
        x3, r3 = self.decode3(x4, f3, s3, r3)
        x2, r2 = self.decode2(x3, f2, s2, r2)
        x1, r1 = self.decode1(x2, f1, s1, r1)
        x0 = self.decode0(x1, s0)
        return x0, r1, r2, r3, r4, [x1, x2, x3, x4]

class RecurrentDecoderTo8x(nn.Module):
    def __init__(self, feature_channels, gru=ConvGRU):
        super().__init__()
        self.avgpool = AvgPool()
        self.decode4 = GRUBottleneckBlock(feature_channels[3], gru=gru)
        self.decode3 = GRUUpsamplingBlock(feature_channels[3], feature_channels[2], 3, feature_channels[2], gru=gru)

    def forward(self,
                img3: Tensor, f3: Tensor, f4: Tensor,
                r3: Optional[Tensor], r4: Optional[Tensor]):
        x4, r4 = self.decode4(f4, r4)
        x3, r3 = self.decode3(x4, f3, img3, r3)
        return x3, x4, r3, r4

class RecurrentDecoder_(nn.Module):
    def __init__(self, feature_channels, decoder_channels):
        super().__init__()
        self.avgpool = AvgPool()
        self.decode4 = GRUBottleneckBlock(feature_channels[2])
        self.decode3 = GRUUpsamplingBlock(feature_channels[2], feature_channels[1], 3, decoder_channels[0])
        self.decode2 = GRUUpsamplingBlock(decoder_channels[0], feature_channels[0], 3, decoder_channels[1])
        self.decode1 = GRUUpsamplingBlockWithoutSkip(decoder_channels[1], 3, decoder_channels[2])
        self.decode0 = OutputBlock(decoder_channels[2], 3, decoder_channels[3])

    def forward(self,
                s0: Tensor, f2: Tensor, f3: Tensor, f4: Tensor,
                r1: Optional[Tensor], r2: Optional[Tensor],
                r3: Optional[Tensor], r4: Optional[Tensor]):
        s1, s2, s3 = self.avgpool(s0)
        x4, r4 = self.decode4(f4, r4)
        x3, r3 = self.decode3(x4, f3, s3, r3)
        x2, r2 = self.decode2(x3, f2, s2, r2)
        x1, r1 = self.decode1(x2, s1, r1)
        x0 = self.decode0(x1, s0)
        return x0, r1, r2, r3, r4

class RecurrentDecoder8x(nn.Module):
    def __init__(self, feature_channels, ch_out, gru=ConvGRU):
        super().__init__()
        self.avgpool = AvgPool()
        self.decode2 = GRUUpsamplingBlock(feature_channels[2], feature_channels[1], 3, feature_channels[1], gru=gru)
        self.decode1 = GRUUpsamplingBlock(feature_channels[1], feature_channels[0], 3, feature_channels[0], gru=gru)
        self.decode0 = OutputBlock(feature_channels[0], 3, ch_out)

    def forward(self,
                img: Tensor, img1: Tensor, img2: Tensor,
                f1: Tensor, f2: Tensor, x3: Tensor,
                r1: Optional[Tensor], r2: Optional[Tensor]):
        x2, r2 = self.decode2(x3, f2, img2, r2)
        x1, r1 = self.decode1(x2, f1, img1, r1)
        x0 = self.decode0(x1, img)
        return x0, r1, r2, [x1, x2]

class RecurrentDecoder4x(nn.Module):
    def __init__(self, feature_channels, ch_skips, ch_out, ch_feat=64, gru=ConvGRU):
        super().__init__()
        # self.avgpool = AvgPool()
        # print(ch_skips)
        self.gated_s4 = GatedConv2d(ch_skips[-1], ch_feat, 1, 1, 0)
        self.gated_s2 = GatedConv2d(ch_skips[2], ch_feat, 1, 1, 0)
        self.gated_s0 = GatedConv2d(ch_skips[0], ch_feat, 1, 1, 0)
        self.decode0 = ResBlock(feature_channels[1]+ch_feat+ch_feat, ch_feat)
        # self.decode1 = UpsampleBlock(feature_channels[0]+3, ch_feat, ch_feat)
        self.decode2 = UpsampleBlock(ch_feat+3, ch_feat, ch_feat)
        self.decode1 = GRUUpsamplingBlock(ch_feat, feature_channels[0], 3, ch_feat, gru=gru)
        # self.decode2 = GRUUpsamplingBlock(ch_feat, ch_feat//2, 3, ch_feat//2, gru=gru)
        self.out = ResBlock(ch_feat, ch_out)

    def forward(self,
                img: Tensor, img1: Tensor,
                f1: Tensor, f2: Tensor, f3: Tensor,
                s0: Tensor, s1: Tensor, s2: Tensor, s3: Tensor, s4: Tensor,
                r1: Optional[Tensor]):
        b, t = img.shape[:2]
        # print(s4.shape, s2.shape, s0.shape)
        s4 = F.interpolate(self.gated_s4(s4), scale_factor=(1, 4, 4)) # 16 -> 4
        s2 = self.gated_s2(s2) # 4
        s0 = self.gated_s0(s0) # 1
        # print(s4.shape, s2.shape, s0.shape)
        # print(f3.shape, f2.shape, f1.shape)
        x2 = self.decode0(torch.cat([f2, s2, s4], dim=2).flatten(0, 1)).unflatten(0, (b, t))
        x1, r1 = self.decode1(x2, f1, img1, r1) # 4 -> 2
        # x1 = self.decode1(x2, torch.cat([f1, img1], dim=2)) # 4 -> 2
        # x0, r0 = self.decode2(x1, s0, img, r0) # 2 -> 1
        x0 = self.decode2(x1, torch.cat([s0, img], dim=2)) # 2 -> 1
        out = self.out(x0.flatten(0, 1)).unflatten(0, (b, t))
        return out, r1, [x0, x1, x2]

class _RecurrentDecoder4x(nn.Module):
    def __init__(self, feature_channels, ch_skips, ch_out, ch_feat=64, gru=ConvGRU):
        super().__init__()
        # self.avgpool = AvgPool()
        # print(ch_skips)
        self.gated_s4 = GatedConv2d(ch_skips[-1], ch_feat, 1, 1, 0)
        self.gated_s2 = nn.Sequential(
            Projection(ch_skips[2]+ch_feat, ch_feat, 3, 1, 1),
            nn.Sigmoid()
        )
        # self.gated_s0 = nn.Sequential(
        #     Projection(ch_skips[0]+ch_feat, ch_feat, 3, 1, 1),
        #     nn.Sigmoid()
        # )
        self.gated_s0 = GatedConv2d(ch_skips[0], ch_feat, 1, 1, 0)
        # self.decode0 = ResBlock(feature_channels[1], ch_feat)
        self.decode0 = ResBlock(feature_channels[1]+ch_feat, ch_feat)
        # self.decode1 = UpsampleBlock(feature_channels[0]+3, ch_feat, ch_feat)
        self.decode1 = GRUUpsamplingBlock(ch_feat, feature_channels[0], 3, ch_feat, gru=gru)
        self.decode2 = UpsampleBlock(ch_feat+3, ch_feat, ch_feat)
        # self.decode2 = GRUUpsamplingBlock(ch_feat, ch_feat//2, 3, ch_feat//2, gru=gru)
        # self.decode2 = GRUUpsamplingBlockWithoutSkip(ch_feat, 3, ch_feat, gru=gru)
        self.out = ResBlock(ch_feat, ch_out)

    def forward(self,
                img: Tensor, img1: Tensor,
                f1: Tensor, f2: Tensor, f3: Tensor,
                s0: Tensor, s1: Tensor, s2: Tensor, s3: Tensor, s4: Tensor,
                r0: Optional[Tensor], r1: Optional[Tensor]):
        b, t = img.shape[:2]
        # print(s4.shape, s2.shape, s0.shape)
        s0 = self.gated_s0(s0) # 1
        s4 = F.interpolate(self.gated_s4(s4), scale_factor=(1, 4, 4)) # 16 -> 4
        # print(s4.shape, s2.shape, s0.shape)
        # print(f3.shape, f2.shape, f1.shape)
        # x2 = self.decode0(f2.flatten(0, 1)).unflatten(0, (b, t))
        x2 = self.decode0(torch.cat([f2, s4], dim=2).flatten(0, 1)).unflatten(0, (b, t))
        # x2 = x2*self.gated_s2(torch.cat([x2, s2], dim=2))

        x1, r1 = self.decode1(x2, f1, img1, r1) # 4 -> 2
        # x1 = self.decode1(x2, torch.cat([f1, img1], dim=2)) # 4 -> 2
        # x0, r0 = self.decode2(x1, img, r0) # 2 -> 1
        # x0 = x0*self.gated_s0(torch.cat([x0, s0], dim=2))
        x0 = self.decode2(x1, torch.cat([s0, img], dim=2)) # 2 -> 1
        out = self.out(x0.flatten(0, 1)).unflatten(0, (b, t))
        return out, r0, r1, [x0, x1, x2]


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
            # ResBlock(out_channels),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(True),
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

class GRUResBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, src_channels, out_channels, gru=ConvGRU):
        super().__init__()
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels + src_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        self.gru = gru(out_channels // 2)

    def forward_single_frame(self, x, f, s, r: Optional[Tensor]):
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

class GRUUpsamplingBlockNoImg(nn.Module):
    def __init__(self, in_channels, skip_channels, src_channels, out_channels, gru=ConvGRU):
        super().__init__()
        self.out_channels = out_channels
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            # ResBlock(out_channels),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(True),
        )
        self.gru = gru(out_channels // 2)

    def forward_single_frame(self, x, f, s, r: Optional[Tensor]):
        x = self.upsample(x)
        # x = x[:, :, :s.size(2), :s.size(3)]
        x = torch.cat([x, f], dim=1)
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
        # x = x[:, :, :H, :W]
        x = torch.cat([x, f], dim=1)
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

class GRUUpsamplingResBlock(GRUUpsamplingBlock):
    def __init__(self, in_channels, skip_channels, src_channels, out_channels, gru=ConvGRU):
        super().__init__(in_channels, skip_channels, src_channels, out_channels, gru)
        del self.conv
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels + skip_channels + src_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                ResBlock(out_channels),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )

class GRUUpsamplingBlockWithoutSkip(nn.Module):
    def __init__(self, in_channels, src_channels, out_channels, gru=ConvGRU):
        super().__init__()
        self.out_channels = out_channels
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + src_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        self.gru = gru(out_channels // 2)

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


class OutputBlock(nn.Module):
    def __init__(self, in_channels, src_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + src_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        
    def forward_single_frame(self, x, s):
        x = self.upsample(x)
        x = x[:, :, :s.size(2), :s.size(3)]
        x = torch.cat([x, s], dim=1)
        x = self.conv(x)
        return x
    
    def forward_time_series(self, x, s):
        B, T, _, H, W = s.shape
        x = x.flatten(0, 1)
        s = s.flatten(0, 1)
        x = self.upsample(x)
        x = x[:, :, :H, :W]
        x = torch.cat([x, s], dim=1)
        x = self.conv(x)
        x = x.unflatten(0, (B, T))
        return x
    
    def forward(self, x, s):
        if x.ndim == 5:
            return self.forward_time_series(x, s)
        else:
            return self.forward_single_frame(x, s)

class Projection(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=1, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding)
    
    def forward_single_frame(self, x):
        return self.conv(x)
    
    def forward_time_series(self, x):
        B, T = x.shape[:2]
        return self.conv(x.flatten(0, 1)).unflatten(0, (B, T))
        
    def forward(self, x):
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)
    