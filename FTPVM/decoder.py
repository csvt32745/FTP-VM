import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_block import *

class SegmentationDecoderTo4x(nn.Module):
    def __init__(self, feature_channels, decoder_channels, gru=ConvGRU):
        super().__init__()
        assert len(feature_channels) == 4
        self.decode4 = GRUBottleneckBlock(feature_channels[3])
        self.decode3 = GRUUpsamplingBlock(feature_channels[3], feature_channels[2], 3, decoder_channels[0], gru=gru)
        # self.decode2 = GRUUpsamplingBlockWithoutSkip(decoder_channels[0], 3, decoder_channels[1], gru=gru)
        self.decode2 = UpsampleBlock(3, decoder_channels[0], decoder_channels[1])
        self.out = ResBlock(decoder_channels[1], decoder_channels[2]) 

        self.default_rec = [None, None]
        self.output_stride = 4
        
    def forward(self,
                img1: Tensor, img4: Tensor, img8: Tensor,
                f8: Tensor, f16: Tensor,
                r8: Optional[Tensor], r16: Optional[Tensor]):
        x16, r16 = self.decode4(f16, r16)
        x8, r8 = self.decode3(x16, f8, img8, r8)
        x4 = self.decode2(x8, img4)
        out = self.out(x4.flatten(0, 1)).unflatten(0, x4.shape[:2])
        return out, r8, r16, [x4, x8, x16]
    

class MattingDecoderFrom4x(nn.Module):
    def __init__(self, feature_channels, ch_skips, ch_decode, gru=ConvGRU):
        super().__init__()
        assert len(ch_decode) == 4
        def get_conv_relu_bn(ch_in, ch_out, kernel, stride=1, padding=0):
            return nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel, stride, padding),
                nn.ReLU(True),
                nn.BatchNorm2d(ch_out),
            )
        self.gated_s16 = get_conv_relu_bn(ch_skips[0], ch_decode[0]//2, 1)
        self.gated_s4 = get_conv_relu_bn(ch_skips[2], ch_decode[0]//2, 1)
        self.decode0 = ResBlock(ch_decode[0]+3, ch_decode[0])
        self.decode0_1 = ResBlock(feature_channels[1]+ch_decode[0], ch_decode[0])
        self.decode1 = GRUUpsamplingBlock(ch_decode[0], feature_channels[0], 3, ch_decode[1], gru=gru)
        self.decode2 = GRUUpsamplingBlockWithoutSkip(ch_decode[1], 3, ch_decode[2], gru=gru)
        self.out = ResBlock(ch_decode[2], ch_decode[3])

        self.default_rec = [None, None]

    def forward(self,
                img1: Tensor, img2: Tensor, img4: Tensor,
                f2: Tensor, f4: Tensor,
                s4: Tensor, s16: Tensor,
                r1: Optional[Tensor], r2: Optional[Tensor]):
        bt = img1.shape[:2]
        s16 = F.interpolate(self.gated_s16(s16.flatten(0, 1)), scale_factor=(4, 4)) # 16 -> 4
        s4 = self.gated_s4(s4.flatten(0, 1))
        
        x4 = self.decode0(torch.cat([s4, s16, img4.flatten(0, 1)], dim=1))
        x4_1 = self.decode0_1(torch.cat([f4.flatten(0, 1), x4], dim=1)).unflatten(0, bt)
        x2, r2 = self.decode1(x4_1, f2, img2, r2) # 4 -> 2
        x1, r1 = self.decode2(x2, img1, r1) # 2 -> 1
        out = self.out(x1.flatten(0, 1)).unflatten(0, bt)
        return out, r1, r2, [x1, x2, x4_1]
