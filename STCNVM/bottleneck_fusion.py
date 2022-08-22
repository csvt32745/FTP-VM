import torch
from torch import Tensor
from typing import Optional
from torch import nn
from torch.nn import functional as F

from .basic_block import ResBlock, ResBlock2, GatedConv2d, AvgPool, Projection
from .recurrent_decoder import ConvGRU
from .module import FeatureFusion, PSP, MemoryReader

class BottleneckFusion(nn.Module):
    def __init__(self, ch_in, ch_key, ch_value, ch_out, affinity='dotproduct'):
        super().__init__()
        self.project_key = Projection(ch_in, ch_key)
        self.reader = MemoryReader(affinity=affinity)

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

class BottleneckFusion_PPM1236(BottleneckFusion):
    def __init__(self, ch_in, ch_key, ch_value, ch_out, affinity='dotproduct'):
        super().__init__(ch_in, ch_key, ch_value, ch_out, affinity)
        self.bottleneck = PSP(ch_out, ch_out//4, ch_out, sizes=(1, 2, 3, 6))
        
class BottleneckFusion_woPPM(BottleneckFusion):
    def __init__(self, ch_in, ch_key, ch_value, ch_out, affinity='dotproduct'):
        super().__init__(ch_in, ch_key, ch_value, ch_out, affinity)
        del self.bottleneck
    
    def forward(self, f16_q, f16_m, value_m, rec_bottleneck: Optional[Tensor]):
        f16_m = self.read_value(f16_q, f16_m, value_m)
        out = self.fuse(torch.cat([f16_q, f16_m], dim=2))
        return out, rec_bottleneck

class BottleneckFusion_woCBAM(BottleneckFusion):
    def __init__(self, ch_in, ch_key, ch_value, ch_out, affinity='dotproduct'):
        super().__init__(ch_in, ch_key, ch_value, ch_out, affinity)
        self.fuse = ResBlock2(ch_in+ch_value, ch_out)
    
    def forward(self, f16_q, f16_m, value_m, rec_bottleneck: Optional[Tensor]):
        f16_m = self.read_value(f16_q, f16_m, value_m)
        out = self.fuse(torch.cat([f16_q, f16_m], dim=2).flatten(0, 1)).unflatten(0, f16_m.shape[:2])
        out = self.bottleneck(out)
        return out, rec_bottleneck

class BottleneckFusion_f16(BottleneckFusion):
    def __init__(self, ch_in, ch_key, ch_value, ch_out, affinity='dotproduct'):
        super().__init__(ch_in, ch_key, ch_value, ch_out, affinity)
        self.fuse = FeatureFusion(ch_in*2+ch_value, ch_out)
    
    def read_value(self, f16_q, f16_m, value_m):
        qk = self.encode_key(f16_q)
        mk = self.encode_key(f16_m)
        value_m = torch.cat([f16_m.transpose(1, 2), value_m], dim=1)
        A = self.reader.get_affinity(mk, qk)
        return self.reader.readout(A, value_m) # value_m.shape == (b, c, t, h, w)

    def encode_key(self, feat16):
        # b, t, c, h, w -> b, ch_key, t, h, w
        return self.project_key(feat16).transpose(1, 2)

class BottleneckFusion_gate(BottleneckFusion):
    def __init__(self, ch_in, ch_key, ch_value, ch_out, affinity='dotproduct'):
        super().__init__(ch_in, ch_key, ch_value, ch_out, affinity)
        self.entropy_mask = nn.Sequential(
            Projection(1, 1, 3, 1, 1),
            nn.Sigmoid()
        )
    
    def read_value(self, f16_q, f16_m, value_m):
        qk = self.encode_key(f16_q)
        mk = self.encode_key(f16_m)
        A = self.reader.get_affinity(mk, qk)
        entropy = (torch.log(A+1e-5)*A).sum(dim=1) # b, nq
        # print(entropy.shape)
        b, _, t, h, w = qk.shape
        entropy = entropy.view(b, t, 1, h, w)
        # print(entropy.shape)
        mask = self.entropy_mask(entropy)
        readout = self.reader.readout(A, value_m) # value_m.shape == (b, c, t, h, w)
        return readout*mask

class BottleneckFusionGRU(BottleneckFusion):
    def __init__(self, ch_in, ch_key, ch_value, ch_out, affinity='dotproduct'):
        super().__init__(ch_in, ch_key, ch_value, ch_out, affinity)
        
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