import torch
from torch import Tensor
from typing import Optional
from torch import nn
from torch.nn import functional as F

from .basic_block import ResBlock, GatedConv2d, AvgPool, Projection
from .recurrent_decoder import ConvGRU
from .module import FeatureFusion, PSP, MemoryReader

class BottleneckFusion(nn.Module):
    def __init__(self, ch_in, ch_key, ch_value, ch_out):
        super().__init__()
        self.project_key = Projection(ch_in, ch_key)
        self.reader = MemoryReader()

        self.fuse = FeatureFusion(ch_in+ch_value, ch_out)
        self.bottleneck = PSP(ch_out, ch_out//4, ch_out)
        self.ch_out = ch_out
        
    def forward(self, f16_q, f16_m, value_m, rec_bottleneck: Optional[Tensor]):
        f16_m = self.read_value(f16_q, f16_m, value_m)
        out = self.fuse(torch.cat([f16_q, f16_m], dim=2))
        out = self.bottleneck(out)
        return out, rec_bottleneck
    
    def read_value(self, f16_q, f16_m, value_m):
        qk = self.encode_key(f16_q)
        mk = self.encode_key(f16_m)
        A = self.reader.get_affinity(mk, qk)
        return self.reader.readout(A, value_m) # value_m.shape == (b, c, t, h, w)

    def encode_key(self, feat16):
        # b, t, c, h, w -> b, ch_key, t, h, w
        return self.project_key(feat16).transpose(1, 2)

class BottleneckFusionGRU(BottleneckFusion):
    def __init__(self, ch_in, ch_key, ch_value, ch_out):
        super().__init__(ch_in, ch_key, ch_value, ch_out)
        
        self.ch_gru = ch_in//2
        self.gru = ConvGRU(self.ch_gru)
        
    def forward(self, f16_q, f16_m, value_m, rec_bottleneck: Optional[Tensor]):
        f16_m = self.read_value(f16_q, f16_m, value_m)
        
        a, b = f16_q.split(self.ch_gru, dim=-3)
        b, rec_bottleneck = self.gru(b, rec_bottleneck)
        f16_q = torch.cat([a, b], dim=-3)

        out = self.fuse(torch.cat([f16_q, f16_m], dim=2))
        out = self.bottleneck(out)
        return out, rec_bottleneck