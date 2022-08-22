import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_block import *
from .recurrent_decoder import *

# ResBlock = ResBlock2

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

class SegmentationDecoderTo4x_woGRU(nn.Module):
    def __init__(self, feature_channels, decoder_channels, gru=ConvGRU):
        super().__init__()
        assert len(feature_channels) == 4
        self.decode4 = ResBlock(feature_channels[3], feature_channels[3])
        # self.decode3 = GRUUpsamplingBlock(feature_channels[3], feature_channels[2], 3, decoder_channels[0], gru=gru)
        self.decode3 = UpsampleBlock(feature_channels[2]+3, feature_channels[3], decoder_channels[0])
        # self.decode2 = GRUUpsamplingBlockWithoutSkip(decoder_channels[0], 3, decoder_channels[1], gru=gru)
        self.decode2 = UpsampleBlock(3, decoder_channels[0], decoder_channels[1])
        self.out = ResBlock(decoder_channels[1], decoder_channels[2]) 

        self.default_rec = [None, None]
        self.output_stride = 4
        
    def forward(self,
                img1: Tensor, img4: Tensor, img8: Tensor,
                f8: Tensor, f16: Tensor,
                r8: Optional[Tensor], r16: Optional[Tensor]):
        x16 = self.decode4(f16.flatten(0, 1))
        x8 = self.decode3(x16, torch.cat([f8, img8], dim=2).flatten(0, 1))
        x4 = self.decode2(x8, img4.flatten(0, 1))
        bt = img4.shape[:2]
        out = self.out(x4).unflatten(0, bt)
        return out, None, None, [x4.unflatten(0, bt), x8.unflatten(0, bt), x16.unflatten(0, bt)]

class SegmentationDecoderTo4x_2(nn.Module):
    def __init__(self, feature_channels, decoder_channels, gru=ConvGRU):
        super().__init__()
        assert len(feature_channels) == 4
        self.decode4 = GRUBottleneckBlock(feature_channels[3])
        # self.decode3 = GRUUpsamplingBlock(feature_channels[3], feature_channels[2], 3, decoder_channels[0], gru=gru)
        self.decode3 = UpsampleBlock(feature_channels[2]+3, feature_channels[3], decoder_channels[0], bn=True)
        self.decode2 = GRUUpsamplingBlockWithoutSkip(decoder_channels[0], 3, decoder_channels[1], gru=gru)
        # self.decode2 = UpsampleBlock(3, decoder_channels[0], decoder_channels[1])
        self.out = nn.Sequential(
            ResBlock(decoder_channels[1], decoder_channels[2]),
            nn.BatchNorm2d(decoder_channels[2]),
        )

        self.default_rec = [None, None]
        self.output_stride = 4
        
    def forward(self,
                img1: Tensor, img4: Tensor, img8: Tensor,
                f8: Tensor, f16: Tensor,
                r4: Optional[Tensor], r16: Optional[Tensor]):
        x16, r16 = self.decode4(f16, r16)
        # x8, r8 = self.decode3(x16, f8, img8, r8)
        # x4 = self.decode2(x8, img4)
        x8 = self.decode3(x16, torch.cat([f8, img8], dim=2))
        x4, r4 = self.decode2(x8, img4, r4)
        out = self.out(x4.flatten(0, 1)).unflatten(0, x4.shape[:2])
        return out, r4, r16, [x4, x8, x16]

class SegmentationDecoderTo4x_3(nn.Module):
    def __init__(self, feature_channels, decoder_channels, gru=ConvGRU):
        super().__init__()
        assert len(feature_channels) == 4
        self.decode4 = GRUBottleneckBlock(feature_channels[3])
        # self.decode3 = GRUUpsamplingBlock(feature_channels[3], feature_channels[2], 3, decoder_channels[0], gru=gru)
        self.decode3 = UpsampleBlock(feature_channels[2]+3, feature_channels[3], decoder_channels[0])
        self.decode2 = GRUUpsamplingBlockWithoutSkip(decoder_channels[0], 3, decoder_channels[1], gru=gru)
        # self.decode2 = UpsampleBlock(3, decoder_channels[0], decoder_channels[1])
        self.out = ResBlock(decoder_channels[1], decoder_channels[2])

        self.default_rec = [None, None]
        self.output_stride = 4
        
    def forward(self,
                img1: Tensor, img4: Tensor, img8: Tensor,
                f8: Tensor, f16: Tensor,
                r4: Optional[Tensor], r16: Optional[Tensor]):
        x16, r16 = self.decode4(f16, r16)
        # x8, r8 = self.decode3(x16, f8, img8, r8)
        # x4 = self.decode2(x8, img4)
        x8 = self.decode3(x16, torch.cat([f8, img8], dim=2))
        x4, r4 = self.decode2(x8, img4, r4)
        out = self.out(x4.flatten(0, 1)).unflatten(0, x4.shape[:2])
        return out, r4, r16, [x4, x8, x16]

class SegmentationDecoderTo1x(nn.Module):
    def __init__(self, feature_channels, decoder_channels, gru=ConvGRU):
        super().__init__()
        assert len(feature_channels) == 4
        self.decode4 = GRUBottleneckBlock(feature_channels[3])
        self.decode3 = GRUUpsamplingBlock(feature_channels[3], feature_channels[2], 3, decoder_channels[0], gru=gru)
        # self.decode2 = GRUUpsamplingBlockWithoutSkip(decoder_channels[0], 3, decoder_channels[1], gru=gru)
        self.decode2 = UpsampleBlock(3, decoder_channels[0], decoder_channels[1], bn=False)
        self.decode1 = UpsampleBlock(3, decoder_channels[1], decoder_channels[2], scale_factor=4, bn=False)
        self.out = ResBlock(decoder_channels[2], decoder_channels[2]) 

        self.default_rec = [None, None]
        self.output_stride = 1
        
    def forward(self,
                img1: Tensor, img4: Tensor, img8: Tensor,
                f8: Tensor, f16: Tensor,
                r8: Optional[Tensor], r16: Optional[Tensor]):
        x16, r16 = self.decode4(f16, r16)
        x8, r8 = self.decode3(x16, f8, img8, r8)
        x4 = self.decode2(x8, img4)
        x1 = self.decode1(x4, img1)
        out = self.out(x1.flatten(0, 1)).unflatten(0, x1.shape[:2])
        return out, r8, r16, [x4, x8, x16]

class SegmentationDecoderTo1x_full(nn.Module):
    def __init__(self, feature_channels, decoder_channels, gru=ConvGRU):
        super().__init__()
        assert len(feature_channels) == 4
        self.decode4 = GRUBottleneckBlock(feature_channels[3])
        # self.decode3 = GRUUpsamplingBlock(feature_channels[3], feature_channels[2], 3, decoder_channels[0], gru=gru)
        self.decode3 = UpsampleBlock(3+feature_channels[2], feature_channels[3], decoder_channels[0], bn=False)
        self.decode2 = GRUUpsamplingBlockWithoutSkip(decoder_channels[0], 3, decoder_channels[1], gru=gru)
        # self.decode2 = UpsampleBlock(3, decoder_channels[0], decoder_channels[1], bn=False)
        self.decode1 = UpsampleBlock(3, decoder_channels[1], decoder_channels[2], bn=True)
        self.decode0 = UpsampleBlock(3, decoder_channels[2], decoder_channels[3], bn=True)
        # self.out = ResBlock(decoder_channels[2], decoder_channels[2]) 

        self.default_rec = [None, None]
        self.output_stride = 1
        
    def forward(self,
                img1: Tensor, img2: Tensor, img4: Tensor, img8: Tensor,
                f8: Tensor, f16: Tensor,
                r4: Optional[Tensor], r16: Optional[Tensor]):
        x16, r16 = self.decode4(f16, r16)
        x8 = self.decode3(x16, torch.cat([f8, img8], dim=2))
        x4, r4 = self.decode2(x8, img4, r4)
        x2 = self.decode1(x4, img2)
        x1 = self.decode0(x2, img1)
        # out = self.out(x1.flatten(0, 1)).unflatten(0, x1.shape[:2])
        return x1, r4, r16, [x2, x4, x8, x16]


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
                img1: Tensor, img2: Tensor, img4: Tensor,
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

class MattingDecoderFrom4x_2(nn.Module):
    def __init__(self, feature_channels, ch_skips, ch_feat, ch_out, gru=ConvGRU):
        super().__init__()
        self.gated_s16 = GatedConv2d(ch_skips[0], ch_feat, 1, 1, 0)
        self.gated_s4 = GatedConv2d(ch_skips[2], ch_feat, 1, 1, 0)
        self.decode0 = ResBlock(feature_channels[1]+ch_feat+ch_feat+3, ch_feat)
        # self.decode1 = UpsampleBlock(feature_channels[0]+3, ch_feat, ch_feat)
        # self.decode2 = UpsampleBlock(3, ch_feat, ch_feat)
        self.decode1 = GRUUpsamplingBlock(ch_feat, feature_channels[0], 3, ch_feat, gru=gru)
        self.decode2 = GRUUpsamplingBlockWithoutSkip(ch_feat, 3, ch_feat, gru=gru)
        self.out = ResBlock(ch_feat, ch_out)

        self.default_rec = [None, None]

    def forward(self,
                img1: Tensor, img2: Tensor, img4: Tensor,
                f2: Tensor, f4: Tensor,
                s4: Tensor, s16: Tensor,
                r1: Optional[Tensor], r2: Optional[Tensor]):
        bt = img1.shape[:2]
        s16 = F.interpolate(self.gated_s16(s16), scale_factor=(1, 4, 4)) # 16 -> 4
        s4 = self.gated_s4(s4)
        
        x4 = self.decode0(torch.cat([f4, s4, s16, img4], dim=2).flatten(0, 1)).unflatten(0, bt)
        x2, r2 = self.decode1(x4, f2, img2, r2) # 4 -> 2
        x1, r1 = self.decode2(x2, img1, r1) # 2 -> 1
        out = self.out(x1.flatten(0, 1)).unflatten(0, bt)
        return out, r1, r2, [x1, x2, x4]

class MattingDecoderFrom4x_feats(nn.Module):
    def __init__(self, feature_channels, ch_skips, ch_decode, gru=ConvGRU):
        super().__init__()
        assert len(ch_decode) == 4
        self.gated_s16 = GatedConv2d(ch_skips[0], ch_decode[0]//2, 1, 1, 0)
        self.gated_s4 = GatedConv2d(ch_skips[2], ch_decode[0]//2, 1, 1, 0)
        self.decode0 = ResBlock(ch_decode[0]+3, ch_decode[0])
        self.decode0_1 = ResBlock(feature_channels[1]+ch_decode[0], ch_decode[0])
        # self.decode1 = UpsampleBlock(feature_channels[0]+3, ch_feat, ch_feat)
        # self.decode2 = UpsampleBlock(3, ch_feat, ch_feat)
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
        s16 = F.interpolate(self.gated_s16(s16.flatten(0, 1)), scale_factor=(4, 4), mode='bilinear') # 16 -> 4
        s4 = self.gated_s4(s4.flatten(0, 1))
        
        x4 = self.decode0(torch.cat([s4, s16, img4.flatten(0, 1)], dim=1))
        x4_1 = self.decode0_1(torch.cat([f4.flatten(0, 1), x4], dim=1)).unflatten(0, bt)
        x2, r2 = self.decode1(x4_1, f2, img2, r2) # 4 -> 2
        x1, r1 = self.decode2(x2, img1, r1) # 2 -> 1
        out = self.out(x1.flatten(0, 1)).unflatten(0, bt)
        return out, r1, r2, [x1, x2, x4_1]

def get_conv_relu_bn(ch_in, ch_out, kernel, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(ch_in, ch_out, kernel, stride, padding),
        nn.ReLU(True),
        nn.BatchNorm2d(ch_out),
    )

class MattingDecoderFrom4x_feats_naive(nn.Module):
    def __init__(self, feature_channels, ch_skips, ch_decode, gru=ConvGRU):
        super().__init__()
        assert len(ch_decode) == 4
        self.gated_s16 = get_conv_relu_bn(ch_skips[0], ch_decode[0]//2, 1)
        self.gated_s4 = get_conv_relu_bn(ch_skips[2], ch_decode[0]//2, 1)
        self.decode0 = ResBlock(ch_decode[0]+3, ch_decode[0])
        self.decode0_1 = ResBlock(feature_channels[1]+ch_decode[0], ch_decode[0])
        # self.decode1 = UpsampleBlock(feature_channels[0]+3, ch_feat, ch_feat)
        # self.decode2 = UpsampleBlock(3, ch_feat, ch_feat)
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

class MattingDecoderFrom4x_feats_naive_woGRU(nn.Module):
    def __init__(self, feature_channels, ch_skips, ch_decode, gru=ConvGRU):
        super().__init__()
        assert len(ch_decode) == 4
        self.gated_s16 = get_conv_relu_bn(ch_skips[0], ch_decode[0]//2, 1)
        self.gated_s4 = get_conv_relu_bn(ch_skips[2], ch_decode[0]//2, 1)
        self.decode0 = ResBlock(ch_decode[0]+3, ch_decode[0])
        self.decode0_1 = ResBlock(feature_channels[1]+ch_decode[0], ch_decode[0])
        self.decode1 = UpsampleBlock(feature_channels[0]+3, ch_decode[0], ch_decode[1])
        self.decode2 = UpsampleBlock(3, ch_decode[1], ch_decode[2])
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
        x4_1 = self.decode0_1(torch.cat([f4.flatten(0, 1), x4], dim=1))
        x2 = self.decode1(x4_1, torch.cat([f2, img2], dim=2).flatten(0, 1)) # 4 -> 2
        x1 = self.decode2(x2, img1.flatten(0, 1)) # 2 -> 1
        out = self.out(x1).unflatten(0, bt)
        return out, None, None, [x1, x2, x4_1]

class MattingDecoderFrom4x_feats_naive2(nn.Module):
    def __init__(self, feature_channels, ch_skips, ch_decode, gru=ConvGRU):
        super().__init__()
        assert len(ch_decode) == 4
        # self.gated_s16 = get_conv_relu_bn(ch_skips[0], ch_decode[0]//2, 1)
        self.gated_s4 = get_conv_relu_bn(ch_skips[2], ch_decode[0], 1)
        self.decode0 = ResBlock(ch_decode[0]+3, ch_decode[0])
        self.decode0_1 = ResBlock(feature_channels[1]+ch_decode[0], ch_decode[0])
        # self.decode1 = UpsampleBlock(feature_channels[0]+3, ch_feat, ch_feat)
        # self.decode2 = UpsampleBlock(3, ch_feat, ch_feat)
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
        # s16 = F.interpolate(self.gated_s16(s16.flatten(0, 1)), scale_factor=(4, 4)) # 16 -> 4
        s4 = self.gated_s4(s4.flatten(0, 1))
        
        x4 = self.decode0(torch.cat([s4, img4.flatten(0, 1)], dim=1))
        x4_1 = self.decode0_1(torch.cat([f4.flatten(0, 1), x4], dim=1)).unflatten(0, bt)
        x2, r2 = self.decode1(x4_1, f2, img2, r2) # 4 -> 2
        x1, r1 = self.decode2(x2, img1, r1) # 2 -> 1
        out = self.out(x1.flatten(0, 1)).unflatten(0, bt)
        return out, r1, r2, [x1, x2, x4_1]

class CrossSpatialAttentation(nn.Module):
    def __init__(self, ch_in, ch_skip, ch_out=-1, spatial_only=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in+ch_skip, ch_in, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(True),
            nn.Conv2d(ch_in, 1 if spatial_only else ch_in, 1, bias=False),
            nn.BatchNorm2d(1 if spatial_only else ch_in),
            nn.Sigmoid()
        )    
        self.out = nn.Conv2d(ch_in, ch_out if ch_out > 0 else ch_in, 1)
    
    def forward(self, x_in, x_skip):
        w = 1 + self.conv(torch.cat([x_in, x_skip], axis=-3))
        return self.out(x_in * w)

class MattingDecoderFrom4x_feats_2(nn.Module):
    def __init__(self, feature_channels, ch_skips, ch_decode, gru=ConvGRU):
        super().__init__()
        assert len(ch_decode) == 4
        self.proj_s16 = get_conv_relu_bn(ch_skips[0], ch_decode[0], 1)
        self.proj_s4 = get_conv_relu_bn(ch_skips[2], ch_decode[0], 1)
        
        self.attn_s16 = CrossSpatialAttentation(ch_decode[0], ch_decode[0])
        self.attn_s4 = CrossSpatialAttentation(ch_decode[0], ch_decode[0])

        self.decode0 = ResBlock(feature_channels[1], ch_decode[0])
        self.decode0_1 = ResBlock(feature_channels[1]+ch_decode[0], ch_decode[0])
        # self.decode1 = UpsampleBlock(feature_channels[0]+3, ch_feat, ch_feat)
        # self.decode2 = UpsampleBlock(3, ch_feat, ch_feat)
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
        s16 = F.interpolate(self.proj_s16(s16.flatten(0, 1)), scale_factor=(4, 4)) # 16 -> 4
        s4 = self.proj_s4(s4.flatten(0, 1))
        
        x4 = self.decode0(f4.flatten(0, 1))
        x4_1 = self.attn_s16(x4, s16)
        x4_2 = self.attn_s4(x4_1, s4)
        x2, r2 = self.decode1(x4_2.unflatten(0, bt), f2, img2, r2) # 4 -> 2
        x1, r1 = self.decode2(x2, img1, r1) # 2 -> 1
        out = self.out(x1.flatten(0, 1)).unflatten(0, bt)
        return out, r1, r2, [x1, x2, x4_1]

class MattingDecoderFrom4x_feats_3(MattingDecoderFrom4x_feats_2):
    def __init__(self, feature_channels, ch_skips, ch_decode, gru=ConvGRU):
        super().__init__(feature_channels, ch_skips, ch_decode, gru)
        del self.attn_s16
        del self.attn_s4
        self.attn_s16 = CrossSpatialAttentation(ch_decode[0], ch_decode[0], spatial_only=True)
        self.attn_s4 = CrossSpatialAttentation(ch_decode[0], ch_decode[0], spatial_only=True)

class MattingDecoderFrom4x_naive(nn.Module):
    def __init__(self, feature_channels, ch_skips, ch_feat, ch_out, gru=ConvGRU):
        super().__init__()
        self.gated_s16 = nn.Sequential(
            nn.Conv2d(ch_skips[0], ch_feat, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(ch_feat),
        )
        self.gated_s4 = nn.Sequential(
            nn.Conv2d(ch_skips[2]+ch_feat, ch_feat, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(ch_feat),
        )
        self.decode0 = ResBlock(feature_channels[1]+ch_feat+ch_feat+3, ch_feat)
        # self.decode1 = UpsampleBlock(feature_channels[0]+3, ch_feat, ch_feat)
        # self.decode2 = UpsampleBlock(3, ch_feat, ch_feat)
        self.decode1 = GRUUpsamplingBlock(ch_feat, feature_channels[0], 3, ch_feat, gru=gru)
        self.decode2 = GRUUpsamplingBlockWithoutSkip(ch_feat, 3, ch_feat, gru=gru)
        self.out = ResBlock(ch_feat, ch_out)

        self.default_rec = [None, None]

    def forward(self,
                img1: Tensor, img2: Tensor, img4: Tensor,
                f2: Tensor, f4: Tensor,
                s4: Tensor, s16: Tensor,
                r1: Optional[Tensor], r2: Optional[Tensor]):
        bt = img1.shape[:2]
        s16 = F.interpolate(self.gated_s16(s16).flatten(0, 1), scale_factor=(4, 4)).unflatten(0, bt) # 16 -> 4
        s4 = self.gated_s4(s4.flatten(0, 1)).unflatten(0, bt)
        
        x4 = self.decode0(torch.cat([f4, s4, s16, img4], dim=2).flatten(0, 1)).unflatten(0, bt)
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
                img1: Tensor, img2: Tensor, img4: Tensor,
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
