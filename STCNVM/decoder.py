import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_block import *
from .recurrent_decoder import *

class SegmentationDecoderTo4x_bottleneck_gru(nn.Module):
    def __init__(self, feature_channels, decoder_channels, gru=ConvGRU):
        super().__init__()
        assert len(feature_channels) == 4
        self.decode4 = GRUBottleneckBlock(feature_channels[3])
        self.decode3 = GRUUpsamplingBlock(feature_channels[3], feature_channels[2], 3, decoder_channels[0], gru=gru)
        # self.decode2 = GRUUpsamplingBlockWithoutSkip(decoder_channels[0], 3, decoder_channels[1], gru=gru)
        self.decode2 = UpsampleBlock(3, decoder_channels[0], decoder_channels[1])
        self.out = ResBlock(decoder_channels[1], decoder_channels[2]) 

        self.default_rec = [None, None]
        
    def forward(self,
                img1: Tensor, img4: Tensor, img8: Tensor,
                f8: Tensor, f16: Tensor,
                r8: Optional[Tensor], r16: Optional[Tensor]):
        x16, r16 = self.decode4(f16, r16)
        x8, r8 = self.decode3(x16, f8, img8, r8)
        x4 = self.decode2(x8, img4)
        out = self.out(x4.flatten(0, 1)).unflatten(0, x4.shape[:2])
        return out, r8, r16, [x4, x8, x16]

class SegmentationDecoderTo4x(nn.Module):
    def __init__(self, feature_channels, decoder_channels, gru=ConvGRU):
        super().__init__()
        assert len(feature_channels) == 4
        # self.decode4 = GRUBottleneckBlock(feature_channels[3])
        self.decode3 = GRUUpsamplingBlock(feature_channels[3], feature_channels[2], 3, decoder_channels[0], gru=gru)
        # self.decode2 = GRUUpsamplingBlockWithoutSkip(decoder_channels[0], 3, decoder_channels[1], gru=gru)
        self.decode2 = UpsampleBlock(3, decoder_channels[0], decoder_channels[1])
        self.out = ResBlock(decoder_channels[1], decoder_channels[2]) 

        self.default_rec = [None, None]
        
    def forward(self,
                img1: Tensor, img4: Tensor, img8: Tensor,
                f8: Tensor, f16: Tensor,
                r8: Optional[Tensor], r16: Optional[Tensor]):
        # x16, r16 = self.decode4(f16, r16)
        x8, r8 = self.decode3(f16, f8, img8, r8)
        x4 = self.decode2(x8, img4)
        out = self.out(x4.flatten(0, 1)).unflatten(0, x4.shape[:2])
        return out, r8, r16, [x4, x8, f16]

class MattingDecoderFrom4x(nn.Module):
    def __init__(self, feature_channels, ch_skips, ch_feat, ch_out, gru=ConvGRU):
        super().__init__()
        self.gated_s16 = GatedConv2d(ch_skips[0], ch_feat, 1, 1, 0)
        self.gated_s4 = GatedConv2d(ch_skips[2], ch_feat, 1, 1, 0)
        self.decode0 = ResBlock(feature_channels[1]+ch_feat+ch_feat, ch_feat)
        # self.decode1 = UpsampleBlock(feature_channels[0]+3, ch_feat, ch_feat)
        # self.decode2 = UpsampleBlock(3, ch_feat, ch_feat)
        self.decode1 = GRUUpsamplingBlock(ch_feat, feature_channels[0], 3, ch_feat, gru=gru)
        self.decode2 = GRUUpsamplingBlockWithoutSkip(ch_feat, 3, ch_feat, gru=gru)
        self.out = ResBlock(ch_feat, ch_out)

        self.default_rec = [None, None]

    def forward(self,
                img1: Tensor, img2: Tensor,
                f2: Tensor, f4: Tensor,
                s4: Tensor, s16: Tensor,
                r1: Optional[Tensor], r2: Optional[Tensor]):
        bt = img1.shape[:2]
        s16 = F.interpolate(self.gated_s16(s16), scale_factor=(1, 4, 4)) # 16 -> 4
        s4 = self.gated_s4(s4)
        
        x4 = self.decode0(torch.cat([f4, s4, s16], dim=2).flatten(0, 1)).unflatten(0, bt)
        x2, r2 = self.decode1(x4, f2, img2, r2) # 4 -> 2
        x1, r1 = self.decode2(x2, img1, r1) # 2 -> 1
        out = self.out(x1.flatten(0, 1)).unflatten(0, bt)
        return out, r1, r2, [x1, x2, x4]

class MattingDecoderFrom4x_fullres_chattn(nn.Module):
    def __init__(self, feature_channels, ch_skips, ch_out, ch_feat=64, gru=ConvGRU):
        super().__init__()
        # self.avgpool = AvgPool()
        # print(ch_skips)
        self.gated_s16 = nn.Sequential(
            nn.Conv2d(ch_skips[0]+ch_feat, ch_feat, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(ch_feat),
            nn.Conv2d(ch_feat, ch_feat, 1),
            nn.Sigmoid()
        )
        self.gated_s4 = nn.Sequential(
            nn.Conv2d(ch_skips[2]+ch_feat, ch_feat, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(ch_feat),
            nn.Conv2d(ch_feat, ch_feat, 1),
            nn.Sigmoid()
        )
        self.decode1_1 = nn.Sequential(
            ResBlock(ch_feat, ch_feat),
            nn.BatchNorm2d(ch_feat),
        )
        self.decode1_2 = nn.Sequential(
            ResBlock(ch_feat, ch_feat),
            nn.BatchNorm2d(ch_feat),
        )
        # self.decode1 = UpsampleBlock(feature_channels[0]+3, ch_feat, ch_feat)
        self.decode2 = UpsampleBlock(3, ch_feat, ch_feat, bn=True)
        self.decode1 = GRUBlock(feature_channels[0]+feature_channels[1]+3, ch_feat, gru=gru)
        # self.decode2 = GRUUpsamplingBlock(ch_feat, ch_feat//2, 3, ch_feat//2, gru=gru)
        self.out = ResBlock(ch_feat, ch_out)

    def forward(self,
                img1: Tensor, img2: Tensor,
                f2: Tensor, f4: Tensor,
                s4: Tensor, s16: Tensor,
                r1: Optional[Tensor], r2: Optional[Tensor]):
        bt = img1.shape[:2]
        # print(s4.shape, s2.shape, s0.shape)
        f4 = F.interpolate(f4.flatten(0, 1), scale_factor=(2, 2), mode='nearest') # 4 -> 2
        x1 = torch.cat([f2, f4.unflatten(0, bt), img2], dim=2)
        x1, r1 = self.decode1(x1, r1)

        x1 = x1.flatten(0, 1)
        s16 = F.interpolate(s16.flatten(0, 1), scale_factor=(8, 8), mode='nearest') # 16 -> 2
        s16 = 1 + self.gated_s16(torch.cat([x1, s16], dim=1))
        x2 = self.decode1_1(x1*s16)

        s4 = F.interpolate(s4.flatten(0, 1), scale_factor=(2, 2), mode='nearest') # 4 -> 2
        s4 = 1 + self.gated_s4(torch.cat([x2, s4], dim=1))
        x2 = self.decode1_2(x2*s4).unflatten(0, bt)

        x3 = self.decode2(x2, img1) # 2 -> 1
        out = self.out(x3.flatten(0, 1)).unflatten(0, bt)
        return out, r1, r2, [x3, x2, x1]
