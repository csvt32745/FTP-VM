from functools import lru_cache
from typing import Optional
import torch
from torch import Tensor
import math
from torch import nn
from torch.nn import functional as F
from torchvision.ops.deform_conv import DeformConv2d
import kornia as K
from einops import rearrange
from functools import reduce

# from .STCN.network import *
# from .STCN.modules import *

from .basic_block import ResBlock, GatedConv2d, AvgPool, FocalModulation
from .recurrent_decoder import ConvGRU
from . import cbam

class FeatureFusion2(nn.Module):
    def __init__(self, indim, outdim):
        super().__init__()
        self.attention = cbam.CBAM(indim)
        self.block2 = ResBlock(indim, outdim)

    def forward_single_frame(self, x):
        x = self.attention(x)
        x = self.block2(x)
        return x
    
    def forward_time_series(self, x):
        B, T = x.shape[:2]
        x = self.forward_single_frame(x.flatten(0, 1)).unflatten(0, (B, T))
        return x
    
    def forward(self, x):
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)


class FeatureFusion(nn.Module):
    def __init__(self, indim, outdim, is_resblk=True):
        super().__init__()
        self.block = ResBlock(indim, outdim) if is_resblk \
            else nn.Sequential(
                nn.Conv2d(indim, outdim, 1),
                nn.ReLU(True),
                nn.BatchNorm2d(outdim),
            )
        self.attention = cbam.CBAM(outdim)

    def forward_single_frame(self, x):
        x = self.block(x)
        return x + self.attention(x)
    
    def forward_time_series(self, x):
        B, T = x.shape[:2]
        x = self.forward_single_frame(x.flatten(0, 1)).unflatten(0, (B, T))
        return x
    
    def forward(self, x):
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)

FeatureFusion3 = FeatureFusion

class SingleDeformConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, offset_group=1):
        super().__init__()
        offset_channels = 2 * kernel_size * kernel_size
        self.conv_offset = nn.Conv2d(
            in_channels,
            offset_channels * offset_group,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation,
        )
        self.DCN_V1 = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation,
            groups = groups,
            bias = False
        )
    def forward(self, x):
        offset = self.conv_offset(x)
        return self.DCN_V1(x, offset)

class AlignDeformConv2d(nn.Module):
    def __init__(self, mem_channels, que_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, offset_group=1):
        super().__init__()
        n_k = kernel_size * kernel_size * offset_group
        offset_channels = 3 * n_k
        self.split = [n_k*2, n_k]
        self.n_k = n_k
        self.conv_offset = nn.Conv2d(
            mem_channels+que_channels,
            offset_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation,
        )
        self.dfconv2d = DeformConv2d(
            mem_channels,
            out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation,
            groups = groups,
            bias = False
        )

    def forward_single_frame(self, mem, que):
        offset, mask = self.conv_offset(torch.cat([mem, que], dim=1)).split(self.split, dim=1)
        return self.dfconv2d(mem, offset, mask)
    
    def forward_time_series(self, mem, que):
        B, T = mem.shape[:2]
        x = self.forward_single_frame(mem.flatten(0, 1), que.flatten(0, 1)).unflatten(0, (B, T))
        return x
    
    def forward(self, *args):
        if args[0].ndim == 5:
            return self.forward_time_series(*args)
        else:
            return self.forward_single_frame(*args)

class ConvSelfAttention(nn.Module):
    def __init__(self, 
        dim=32, attn_dim=32, head=2, 
        qkv_bias=False, patch_size=16, 
        drop_p=0., same_qkconv=False):
        super().__init__()
        # (b*t, 256, H/4, W/4)
        
        def check_size(patch_sz):
            patch_sz = patch_sz
            length = 0
            while patch_sz > 1:
                assert (patch_sz&1) == 0, 'patch_size is required to be 2^N'
                patch_sz = patch_sz >> 1
                length += 1
            return length
        length = check_size(int(patch_size))
        
        def get_convstem(ch_in, ch_out, length):
            chs = [ch_in] + [ch_out]*length
            # net = [Deform_Conv_V1(ch_in, ch_in, 3, 1, 1)]
            net = []
            for i in range(length):
                net += [
                    nn.Conv2d(chs[i], chs[i+1], 3, 1, 1),
                    nn.MaxPool2d(2, 2),
                    nn.ReLU(True),
                ]
            net.append(nn.Conv2d(chs[-1], ch_out, 1))
            return nn.Sequential(*net) # abort the last act
        
        self.kernel, self.stride, self.padding = (patch_size, patch_size, 0)
        self.conv_q = get_convstem(dim, attn_dim*head, length)
        self.conv_k = self.conv_q if same_qkconv else get_convstem(dim, attn_dim*head, length)
        # self.conv_k = self.conv_q
        # (b*t, qkv, H', W')
        self.is_proj_v = head > 1
        self.conv_v = nn.Conv2d(dim, dim*head, 1, bias=qkv_bias) if self.is_proj_v else nn.Identity()
        self.head = head
        self.ch_qkv = attn_dim
        self.merge_v = nn.Conv2d(dim*self.head, dim, 1) if self.is_proj_v else nn.Identity()
        self.qk_scale = dim ** -0.5
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        # print(self.unfold)
        self.drop_attn = nn.Dropout(p=drop_p)
        # self.drop_proj = nn.Dropout(p=drop_p)

    
    @staticmethod
    @lru_cache(maxsize=None)
    def get_diag_id(length: int):
        return list(range(length))

    def forward(self, x_query, x_key=None, x_value=None):
        
        b, t, c, h, w = x_query.shape
        if x_key is None:
            x_key = x_query
        # t_kv = x_kv.size(1)

        x_query = self.drop_attn(x_query.flatten(0, 1))
        x_key = self.drop_attn(x_key.flatten(0, 1))
        x_value = x_key if x_value is None else x_value.flatten(0, 1)
        
        v = self.conv_v(x_value) # ((b t), ...)
        v = self.unfold(v) # ((b t) P*P*c*m h'*w')
        v = rearrange(v, '(b t) (m c) w -> b m (t w) c', b=b, m=self.head)

        q = rearrange(self.conv_q(x_query), "(b t) (m c) h w -> b m c (t h w)", b=b, m=self.head, c=self.ch_qkv)
        k = rearrange(self.conv_k(x_key), "(b t) (m c) h w -> b m c (t h w)", b=b, m=self.head, c=self.ch_qkv)

        A = q.transpose(-2, -1) @ k
        # exclude self
        # i = self.get_diag_id(A.size(-1)) 
        # A[..., i, i] = -torch.inf
        A = A.softmax(dim=-1) # b m (t hq wq) (t hk wk)

        out = A @ v  # b m (t hq wq) c
        out = rearrange(out, 'b m (t w) c -> (b t m) c w', t=t)
        out = F.fold(out, (h, w), kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        out = rearrange(out, '(b t m) c h w -> (b t) (c m) h w', b=b, m=self.head)
        out = self.merge_v(out)
        out = rearrange(out, '(b t) c h w -> b t c h w', b=b)
        return out, A

class AlignedSelfAttention(ConvSelfAttention):
    def __init__(self, dim=32, attn_dim=32, head=2, qkv_bias=False, patch_size=16, drop_p=0.25):
        super().__init__(dim, attn_dim, head, qkv_bias, patch_size, drop_p)
    
    def forward(self, x_query, x_memory=None, mask=None):
        b, t, c, h, w = x_query.shape
        x_kv = x_query if x_memory is None else x_memory
        # t_kv = x_kv.size(1)

        x_kv = x_kv.flatten(0, 1)
        x_query = x_query.flatten(0, 1)
        v = self.conv_v(x_kv) # ((b t), ...)
        v = self.unfold(v) # ((b t) P*P*c*m h'*w')
        v = rearrange(v, '(b t) (m c) w -> b m (t w) c', b=b, m=self.head)

        q = rearrange(self.conv_q(x_query), "(b t) (m c) h w -> b m c (t h w)", b=b, m=self.head, c=self.ch_qkv)
        k = rearrange(self.conv_k(x_kv), "(b t) (m c) h w -> b m c (t h w)", b=b, m=self.head, c=self.ch_qkv)

        A = q.transpose(-2, -1) @ k
        # exclude self
        # i = self.get_diag_id(A.size(-1)) 
        # A[..., i, i] = -torch.inf
        A = A.softmax(dim=-1) # b m (t hq wq) (t hk wk)

        out = A @ v  # b m (t hq wq) c
        out = rearrange(out, 'b m (t w) c -> (b t m) c w', t=t)
        out = F.fold(out, (h, w), kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        out = rearrange(out, '(b t m) c h w -> (b t) (c m) h w', b=b, m=self.head)
        out = self.merge_v(out)
        out = rearrange(out, '(b t) c h w -> b t c h w', b=b)
        return out, A

class DeformableConvGRU(ConvGRU):
    def __init__(self, channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__(channels, kernel_size, padding)
        self.dfconv = AlignDeformConv2d(channels, channels, channels)

    def forward_single_frame(self, x, h):
        h = self.dfconv(h, x)
        return super().forward_single_frame(x, h)
    
    def forward_time_series(self, x, h):
        o = []
        for xt in x.unbind(dim=1):
            ot, h = self.forward_single_frame(xt, h)
            o.append(ot)
        o = torch.stack(o, dim=1)
        return o, h
        
    def forward(self, x, h: Optional[torch.Tensor]):
        if h is None:
            h = torch.zeros((x.size(0), x.size(-3), x.size(-2), x.size(-1)),
                            device=x.device, dtype=x.dtype)
        
        if x.ndim == 5:
            return self.forward_time_series(x, h)
        else:
            return self.forward_single_frame(x, h)
    
class DeformableFrameAlign(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dfconv = AlignDeformConv2d(channels, channels, channels)
        self.fuse = FeatureFusion2(channels*2, channels)

    def forward_single_frame(self, x, h):
        return torch.cat([x, self.dfconv(x, x)], dim=1), h
        # return self.fuse(torch.cat([x, self.dfconv(x, x)], dim=1)), h
    
    def forward_time_series(self, x, h):
        if h is None:
            h = x[:, [0]]
        xx = torch.cat([h, x[:, :-1]], dim=1) # b t c h w
        return self.fuse(torch.cat([x, self.dfconv(xx, x)], dim=2)), x[:, [-1]]
        
    def forward(self, x, h: Optional[torch.Tensor]):
        # if h is None:
        #     h = torch.zeros((x.size(0), x.size(-3), x.size(-2), x.size(-1)),
        #                     device=x.device, dtype=x.dtype)
                            
        if x.ndim == 5:
            return self.forward_time_series(x, h)
        else:
            return self.forward_single_frame(x, h)

class ChannelAttention(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_out, bias=False),
            # nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, feat, take_mean=False):
        if take_mean:
            size = feat.shape[:-3]
            return self.fc(feat.mean(dim=(-2, -1))).view(*size, -1, 1, 1)
        return self.fc(feat)

class LRASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.aspp2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.Sigmoid()
        )
        
    def forward_single_frame(self, x):
        return self.aspp1(x) * self.aspp2(x)
    
    def forward_time_series(self, x):
        B, T = x.shape[:2]
        x = self.forward_single_frame(x.flatten(0, 1)).unflatten(0, (B, T))
        return x
    
    def forward(self, x):
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)

class PSP(nn.Module):
    def __init__(self, features, per_features, out_features, sizes=(1, 2, 4, 8)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, per_features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(per_features * len(sizes) + features, out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, per_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, per_features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def _forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners = True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)
    
    def forward(self, x):
        if x.ndim == 5:
            B, T = x.shape[:2]
            return self._forward(x.flatten(0, 1)).unflatten(0, (B, T))
        else:
            return self._forward(x)

class AttnGRU(nn.Module):
    def __init__(self,
                 channels: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 hidden = 16,
                 patch_size=9,
                 head=1,
                 dropout=0):
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
        self.attn = SoftCrossAttention(channels, hidden, head, patch_size=patch_size, is_proj_v=True, dropout=dropout)
        
    def forward_single_frame(self, x, h):
        h = torch.tanh(self.attn(x, h))
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

AttnGRU2 = AttnGRU

class AttnGRU2_big(AttnGRU2):
    def __init__(self, 
        channels: int, 
        kernel_size: int = 3, 
        padding: int = 1, 
        hidden=16, 
        patch_size=15, 
        head=1
    ):
        super().__init__(channels, kernel_size, padding, hidden, patch_size, head)

class FocalGRU(ConvGRU):
    def __init__(self, channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__(channels, kernel_size, padding)
        
        # self.focal = FocalModulation(channels, 3, 1)
        self.focal = FocalModulation(channels, 5, 4)
        
    def forward_single_frame(self, x, h):
        h = torch.tanh(self.focal(h, x))
        r, z = self.ih(torch.cat([x, h], dim=1)).split(self.channels, dim=1)
        c = self.hh(torch.cat([x, r * h], dim=1))
        h = (1 - z) * h + z * c
        return h, h


class SoftSplit(nn.Module):
    def __init__(self, channel, hidden, kernel_size, stride, padding, dropout=0.):
        super(SoftSplit, self).__init__()
        self.kernel_size = kernel_size
        self.t2t = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding)
        c_in = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(c_in, hidden) if hidden > 0 else nn.Identity()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # B, C, H, W
        feat = self.t2t(x) # B, C*K*K, P
        feat = feat.permute(0, 2, 1)
        feat = self.embedding(feat) # B, P, C'
        feat = self.dropout(feat)
        # B, P, C'
        return feat

class SoftComp(nn.Module):
    def __init__(self, channel, hidden, kernel_size, stride, padding):
        super(SoftComp, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        c_out = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(hidden, c_out) if hidden > 0 else nn.Identity()
        # self.bias = nn.Parameter(torch.zeros((channel, h, w), dtype=torch.float32), requires_grad=True)

    def forward(self, x, out_size):
        # B, P, C'
        feat = self.embedding(x) # B, P, C*K*K
        feat = feat.permute(0, 2, 1)
        feat = F.fold(feat, output_size=out_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        # B, C, H, W
        return feat

class SoftCrossAttention(nn.Module):
    def __init__(self, 
            dim=32, 
            hidden=32,
            head=2,
            patch_size=9,
            is_proj_v=False,
            dropout = 0.0
        ):
        super().__init__()
        self.head = head
        self.is_proj_v = head > 1 or is_proj_v
        hidden_v = hidden if self.is_proj_v else -1
        self.kernel, self.stride, self.padding = [(i, i) for i in [patch_size, patch_size//2, patch_size//2]]
        self.ss_k = SoftSplit(dim, hidden, self.kernel, self.stride, self.padding, dropout=dropout)
        self.ss_v = SoftSplit(dim, hidden_v, self.kernel, self.stride, self.padding, dropout=dropout)
        self.sc = SoftComp(dim, hidden_v, self.kernel, self.stride, self.padding)
        
        self.proj_k = nn.Linear(hidden, hidden*head)
        self.proj_v = nn.Linear(hidden, hidden*head) if self.is_proj_v else nn.Identity()
        if self.is_proj_v:
            if dropout > 0:
                self.proj_out = nn.Sequential(nn.Linear(hidden*head, hidden), nn.Dropout(p=dropout))
            else:
                self.proj_out = nn.Linear(hidden*head, hidden)
        else:
            self.proj_out = nn.Identity()

        self.patch_size = patch_size

    def _forward(self, x_query, x_key, x_value=None):
        size = x_query.shape[-2:]
        q = self.ss_k(x_query) # b*t, p, c
        k = self.ss_k(x_key)
        v = self.ss_v(x_value)
        
        # b*t, m, p, c
        q = rearrange(self.proj_k(q), 'b p (m c) -> b m p c', m=self.head)
        k = rearrange(self.proj_k(k), 'b p (m c) -> b m p c', m=self.head)
        v = rearrange(self.proj_v(v), 'b p (m c) -> b m p c', m=self.head)
        
        A = q @ k.transpose(-2, -1) # b*t, m, p, p
        A = A.softmax(dim=-1)
        out = A @ v  # b*t, m, p, c
        out = self.proj_out(rearrange(out, 'b m p c -> b p (m c)'))
        out = self.sc(out, size) # b*t, c, h, w

        return out
    
    def forward(self, x_query, x_key, x_value=None):
        if x_query.ndim == 5:
            b, t = x_query.shape[:2]
            x_query = x_query.flatten(0, 1)
            x_key = x_key.flatten(0, 1)
            x_value = x_key if x_value is None else x_value.flatten(0, 1)
            return self._forward(x_query, x_key, x_value).unflatten(0, (b, t))
        if x_value is None:
            x_value = x_key
        return self._forward(x_query, x_key, x_value)


class MemoryReader(nn.Module):
    def __init__(self, affinity='dotproduct'):
        super().__init__()
        self.get_affinity = {
            'l1': self.affinity_l2,
            'dotproduct': self.affinity_dotproduct,
        }[affinity]
 
    def affinity_l2(self, mk, qk):
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

    def affinity_dotproduct(self, mk, qk):
        # Dot product
        B, CK, T, H, W = mk.shape
        mk = mk.flatten(start_dim=2) # b, c, nm
        qk = qk.flatten(start_dim=2) # b, c, nq
        ab = mk.transpose(1, 2) @ qk # b, nm, nq
        affinity = ab / math.sqrt(CK)
        return F.softmax(affinity, dim=1)

    def readout(self, affinity, mv):
        B, CV, T, H, W = mv.shape
        mv = mv.reshape(B, CV, -1) # b, ch_val, nm
        val = torch.bmm(mv, affinity) # b, ch_val, nq
        val = rearrange(val, 'b c (t h w) -> b t c h w', h=H, w=W)
        return val

class GatedUnit(nn.Module):
    def __init__(self,
                 channels: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 stride: int = 1):
        super().__init__()
        self.channels = channels
        self.ih = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size, padding=padding, stride=stride),
            nn.Sigmoid()
        )
        self.hh = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size, padding=padding, stride=stride),
            nn.Tanh()
        )
        
    def forward_(self, x, h):
        r, z = self.ih(torch.cat([x, h], dim=1)).split(self.channels, dim=1)
        c = self.hh(torch.cat([x, r * h], dim=1))
        h = (1 - z) * h + z * c
        return h
    
    def forward(self, x, h):
        if x.ndim==5:
            return self.forward_(x.flatten(0, 1), h.flatten(0, 1)).unflatten(0, x.shape[:2])
        return self.forward_(x, h)

class GFMGatedFuse(nn.Module):
    def __init__(self, ch_feats, ch_out, dropout=0., ch_mask=1):
        super().__init__()
        self.fuse = FeatureFusion(ch_feats[-1]*2, ch_out)
        self.bottleneck = PSP(ch_out, ch_out//4, ch_out)
        self.sa = ConvSelfAttention(ch_feats[-1], head=2, patch_size=1, drop_p=dropout)
        # self.sa = SoftCrossAttention(ch_feats[-1], hidden=16 head=2, patch_size=1, drop_p=0)
        self.avgpool = AvgPool(num=4)
        self.avg = nn.AvgPool2d((2, 2))
        self.ch_feats_out = list(ch_feats)
        self.ch_feats_out[-1] = ch_out
        self.gated_convs = nn.ModuleList([
            nn.Sequential(
                GatedConv2d(ch_feats[i]+ch_mask+(ch_feats[i-1] if i > 0 else 0), ch_feats[i], 3, 1, 1),
                nn.Conv2d(ch_feats[i], ch_feats[i], 3, 1, 1),
                nn.LeakyReLU(0.2),
            )
            for i in range(4)
        ])
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, feats_q, feats_m, masks_m):
        masks = self.avgpool(masks_m)

        f4_m = feats_m[0]
        b, t = f4_m.shape[:2]
        f4_m = self.gated_convs[0](torch.cat([f4_m, masks[0]], dim=2).flatten(0, 1))
        for i in range(1, 4):
            f4_m = torch.cat([self.avg(f4_m), feats_m[i].flatten(0, 1), masks[i].flatten(0, 1)],  dim=1)
            f4_m = self.gated_convs[i](f4_m)
        f4_m = f4_m.unflatten(0, (b, t))

        f4_q = feats_q[3]
        f4_m, A = self.sa(f4_q, feats_m[3], f4_m)
        f4_m = self.dropout(f4_m)
        # TODO
        # f4 = self.fuse(torch.cat([f4_q, torch.zeros_like(f4_m)], dim=2))
        f4 = self.fuse(torch.cat([f4_q, f4_m], dim=2))
        f4 = self.bottleneck(f4)

        feats = list(feats_q)
        feats[3] = f4
        return feats, feats

class GFMNaiveFuse(GFMGatedFuse):
    def __init__(self, ch_feats, ch_out, dropout=0, ch_mask=1):
        super().__init__(ch_feats, ch_out, dropout, ch_mask)
        del self.gated_convs
        self.gated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch_feats[i]+ch_mask+(ch_feats[i-1] if i > 0 else 0), ch_feats[i], 3, 1, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(ch_feats[i], ch_feats[i], 3, 1, 1),
                nn.LeakyReLU(0.2),
            )
            for i in range(4)
        ])

class GFMNaiveFuse_h1sqk(GFMGatedFuse):
    def __init__(self, ch_feats, ch_out, dropout=0, ch_mask=1):
        super().__init__(ch_feats, ch_out, dropout, ch_mask)
        del self.gated_convs
        del self.sa
        
        self.sa = ConvSelfAttention(ch_feats[-1], head=1, patch_size=1, drop_p=0, same_qkconv=True)
        self.gated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch_feats[i]+ch_mask+(ch_feats[i-1] if i > 0 else 0), ch_feats[i], 3, 1, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(ch_feats[i], ch_feats[i], 3, 1, 1),
                nn.LeakyReLU(0.2),
            )
            for i in range(4)
        ])

class GFMGatedFuse_fullres(GFMGatedFuse):
    def __init__(self, ch_feats, ch_out, dropout=0, ch_mask=1):
        super().__init__(ch_feats, ch_out, dropout, ch_mask)
        del self.sa
        self.sa = ConvSelfAttention(ch_feats[-1], head=1, patch_size=1, drop_p=0, same_qkconv=True)

        del self.gated_convs
        ch_gate = [3] + ch_feats
        self.gated_convs = nn.ModuleList([
            nn.Sequential(
                GatedConv2d(ch_gate[i]+ch_mask+(ch_gate[i-1] if i > 0 else 0), ch_gate[i], 3, 1, 1),
                nn.Conv2d(ch_gate[i], ch_gate[i], 3, 1, 1),
                nn.LeakyReLU(0.2),
            )
            for i in range(5)
        ])
    
    def forward(self, feats_q, feats_m, imgs_m, masks_m):
        masks = self.avgpool(masks_m)

        b, t = imgs_m.shape[:2]
        f4_m = self.gated_convs[0](torch.cat([imgs_m, masks_m], dim=2).flatten(0, 1))
        for i in range(4):
            f4_m = torch.cat([self.avg(f4_m), feats_m[i].flatten(0, 1), masks[i].flatten(0, 1)],  dim=1)
            f4_m = self.gated_convs[i+1](f4_m)
        f4_m = f4_m.unflatten(0, (b, t))

        f4_q = feats_q[3]
        f4_m, A = self.sa(f4_q, feats_m[3], f4_m)
        f4_m = self.dropout(f4_m)
        f4 = self.fuse(torch.cat([f4_q, f4_m], dim=2))
        f4 = self.bottleneck(f4)

        feats = list(feats_q)
        feats[3] = f4
        return feats, feats

class GFMGatedFuse_inputmaskonly(GFMGatedFuse):
    def __init__(self, ch_feats, ch_out, dropout=0, ch_mask=1):
        super().__init__(ch_feats, ch_out, dropout, ch_mask)
        del self.gated_convs
        self.gated_convs = nn.ModuleList([
            nn.Sequential(
                GatedConv2d(ch_feats[i]+(ch_feats[i-1] if i > 0 else ch_mask), ch_feats[i], 3, 1, 1),
                nn.Conv2d(ch_feats[i], ch_feats[i], 3, 1, 1),
                nn.LeakyReLU(0.2),
            )
            for i in range(4)
        ])
    
    def forward(self, feats_q, feats_m, masks_m):
        masks = self.avgpool(masks_m)

        f4_m = feats_m[0]
        b, t = f4_m.shape[:2]
        f4_m = self.gated_convs[0](torch.cat([f4_m, masks[0]], dim=2).flatten(0, 1))
        for i in range(1, 4):
            f4_m = torch.cat([self.avg(f4_m), feats_m[i].flatten(0, 1)],  dim=1)
            f4_m = self.gated_convs[i](f4_m)
        f4_m = f4_m.unflatten(0, (b, t))

        f4_q = feats_q[3]
        f4_m, A = self.sa(f4_q, feats_m[3], f4_m)
        f4_m = self.dropout(f4_m)
        f4 = self.fuse(torch.cat([f4_q, f4_m], dim=2))
        f4 = self.bottleneck(f4)

        feats = list(feats_q)
        feats[3] = f4
        return feats, feats

class GFMGatedFuse_bn(GFMGatedFuse):
    def __init__(self, ch_feats, ch_out, dropout=0, ch_mask=1):
        super().__init__(ch_feats, ch_out, dropout, ch_mask)
        self.gated_convs = nn.ModuleList([
            nn.Sequential(
                GatedConv2d(ch_feats[i]+ch_mask+(ch_feats[i-1] if i > 0 else 0), ch_feats[i], 3, 1, 1),
                # nn.Conv2d(ch_feats[i], ch_feats[i], 3, 1, 1),
                # nn.LeakyReLU(0.2),
                nn.BatchNorm2d(ch_feats[i])
            )
            for i in range(4)
        ])
        

class GFMGatedFuse_head1(GFMGatedFuse):
    def __init__(self, ch_feats, ch_out, dropout=0.):
        super().__init__(ch_feats, ch_out, dropout=dropout)
        del self.sa
        self.sa = ConvSelfAttention(ch_feats[-1], head=1, patch_size=1, drop_p=0)

class GFMGatedFuse_sameqk(GFMGatedFuse):
    def __init__(self, ch_feats, ch_out, dropout=0.):
        super().__init__(ch_feats, ch_out, dropout=dropout)
        del self.sa
        self.sa = ConvSelfAttention(ch_feats[-1], head=2, patch_size=1, drop_p=0, same_qkconv=True)

class GFMGatedFuse_sameqk_head1(GFMGatedFuse):
    def __init__(self, ch_feats, ch_out, dropout=0.):
        super().__init__(ch_feats, ch_out, dropout=dropout)
        del self.sa
        self.sa = ConvSelfAttention(ch_feats[-1], head=1, patch_size=1, drop_p=0, same_qkconv=True)

class GFMGatedFuse_h1sqk_ff2(GFMGatedFuse):
    def __init__(self, ch_feats, ch_out, dropout=0.):
        super().__init__(ch_feats, ch_out, dropout=dropout)
        del self.sa
        self.sa = ConvSelfAttention(ch_feats[-1], head=1, patch_size=1, drop_p=0, same_qkconv=True)

        del self.fuse
        self.fuse = FeatureFusion(ch_feats[-1]*2, ch_out, is_resblk=False)

class GFMGatedFuse_grufuse(GFMGatedFuse):
    def __init__(self, ch_feats, ch_out, dropout=0.):
        super().__init__(ch_feats, ch_out, dropout=dropout)
        del self.sa
        self.sa = ConvSelfAttention(ch_feats[-1], head=1, patch_size=1, drop_p=0, same_qkconv=True)
        
        del self.fuse
        self.gate = GatedUnit(ch_feats[-1])
        self.fuse = FeatureFusion(ch_feats[-1], ch_out, is_resblk=False)
    
    def forward(self, feats_q, feats_m, masks_m):
        masks = self.avgpool(masks_m)

        f4_m = feats_m[0]
        b, t = f4_m.shape[:2]
        f4_m = self.gated_convs[0](torch.cat([f4_m, masks[0]], dim=2).flatten(0, 1))
        for i in range(1, 4):
            f4_m = torch.cat([self.avg(f4_m), feats_m[i].flatten(0, 1), masks[i].flatten(0, 1)],  dim=1)
            f4_m = self.gated_convs[i](f4_m)
        f4_m = f4_m.unflatten(0, (b, t))

        f4_q = feats_q[3]
        f4_m, A = self.sa(f4_q, feats_m[3], f4_m)
        # TODO
        # f4 = self.fuse(torch.cat([f4_q, torch.zeros_like(f4_m)], dim=2))
        f4 = self.gate(f4_q, f4_m)
        f4 = self.fuse(f4)
        f4 = self.bottleneck(f4)

        feats = list(feats_q)
        feats[3] = f4
        return feats, feats

class GFMGatedFuse_samekv(GFMGatedFuse):
    def __init__(self, ch_feats, ch_out, dropout=0.):
        super().__init__(ch_feats, ch_out, dropout=dropout)
        
    def forward(self, feats_q, feats_m, masks_m):
        masks = self.avgpool(masks_m)
        f4_m = feats_m[0]
        b, t = f4_m.shape[:2]
        f4_m = self.gated_convs[0](torch.cat([f4_m, masks[0]], dim=2).flatten(0, 1))
        for i in range(1, 4):
            f4_m = torch.cat([self.avg(f4_m), feats_m[i].flatten(0, 1), masks[i].flatten(0, 1)],  dim=1)
            f4_m = self.gated_convs[i](f4_m)
        f4_m = f4_m.unflatten(0, (b, t))

        f4_q = feats_q[3]
        f4_m, A = self.sa(f4_q, f4_m)

        f4 = self.fuse(torch.cat([f4_q, f4_m], dim=2))
        f4 = self.bottleneck(f4)

        feats = list(feats_q)
        feats[3] = f4
        return feats, feats

class GFMGatedFuse_splitstage(nn.Module):
    def __init__(self, ch_feats, ch_out):
        super().__init__()
        self.fuse = FeatureFusion(ch_feats[-1]*2, ch_out)
        self.bottleneck = PSP(ch_out, ch_out//4, ch_out)
        self.sa = ConvSelfAttention(ch_feats[-1], head=2, patch_size=1, drop_p=0)
        # self.sa = SoftCrossAttention(ch_feats[-1], hidden=16 head=2, patch_size=1, drop_p=0)
        self.avgpool = AvgPool(num=4)
        self.ch_feats_out = list(ch_feats)
        self.ch_feats_out[-1] = ch_out
        avg_size = [int(2**i) for i in range(4)][::-1]
        self.gated_convs = nn.ModuleList([
            nn.Sequential(
                GatedConv2d(ch_feats[i]+1, ch_feats[i], 3, 1, 1),
                nn.Conv2d(ch_feats[i], ch_feats[i], 3, 1, 1),
                nn.LeakyReLU(0.2),
                nn.AvgPool2d((avg_size[i], avg_size[i]))
            )
            for i in range(4)
        ])
        self.gate_fuse = nn.Conv2d(sum(ch_feats), ch_feats[-1], 3, 1, 1)
        
        
    def forward(self, feats_q, feats_m, masks_m):
        masks = self.avgpool(masks_m)

        b, t = feats_m[0].shape[:2]
        gated_feats = [
            self.gated_convs[i](
                torch.cat([feats_m[i], masks[i]], dim=2).flatten(0, 1)
            ) 
            for i in range(4)
        ]
        f4_m = self.gate_fuse(torch.cat(gated_feats, dim=1))
        f4_m = f4_m.unflatten(0, (b, t))

        f4_q = feats_q[3]
        f4_m, A = self.sa(f4_q, feats_m[3], f4_m)

        f4 = self.fuse(torch.cat([f4_q, f4_m], dim=2))
        f4 = self.bottleneck(f4)

        feats = list(feats_q)
        feats[3] = f4
        return feats, feats

class GFMGatedFuse_splitfeatfuse(GFMGatedFuse_splitstage):
    def __init__(self, ch_feats, ch_out):
        super().__init__(ch_feats, ch_out)
        # self.fuse = FeatureFusion(ch_feats[-1]*2, ch_out)
        # self.bottleneck = PSP(ch_out, ch_out//4, ch_out)
        # self.sa = ConvSelfAttention(ch_feats[-1], head=2, patch_size=1, drop_p=0)
        # # self.sa = SoftCrossAttention(ch_feats[-1], hidden=16 head=2, patch_size=1, drop_p=0)
        # self.avgpool = AvgPool(num=4)
        # self.ch_feats_out = list(ch_feats)
        # self.ch_feats_out[-1] = ch_out
        # avg_size = [int(2**i) for i in range(4)][::-1]
        # self.gated_convs = nn.ModuleList([
        #     nn.Sequential(
        #         GatedConv2d(ch_feats[i]+1, ch_feats[i], 3, 1, 1),
        #         nn.Conv2d(ch_feats[i], ch_feats[i], 3, 1, 1),
        #         nn.LeakyReLU(0.2),
        #         nn.AvgPool2d((avg_size[i], avg_size[i]))
        #     )
        #     for i in range(4)
        # ])
        self.gate_fuse = nn.Sequential(
            cbam.ChannelGate(sum(ch_feats)),
            nn.Conv2d(sum(ch_feats), ch_feats[-1], 3, 1, 1),
        )
        
        
        
    def forward(self, feats_q, feats_m, masks_m):
        masks = self.avgpool(masks_m)

        b, t = feats_m[0].shape[:2]
        gated_feats = [
            self.gated_convs[i](
                torch.cat([feats_m[i], masks[i]], dim=2).flatten(0, 1)
            ) 
            for i in range(4)
        ]
        f4_m = self.gate_fuse(torch.cat(gated_feats, dim=1))
        f4_m = f4_m.unflatten(0, (b, t))

        f4_q = feats_q[3]
        f4_m, A = self.sa(f4_q, feats_m[3], f4_m)

        f4 = self.fuse(torch.cat([f4_q, f4_m], dim=2))
        f4 = self.bottleneck(f4)

        feats = list(feats_q)
        feats[3] = f4
        return feats, feats

class GFMGatedFuse2(nn.Module):
    def __init__(self, ch_feats, ch_out, dropout=0., ch_mask=1):
        super().__init__()
        self.fuse = FeatureFusion(ch_feats[-1]*2, ch_out)
        self.bottleneck = PSP(ch_out, ch_out//4, ch_out)
        self.sa = ConvSelfAttention(ch_feats[-1], head=2, patch_size=1, drop_p=dropout)
        # self.sa = SoftCrossAttention(ch_feats[-1], hidden=16 head=2, patch_size=1, drop_p=0)
        self.avgpool = AvgPool(num=4)
        self.avg = nn.AvgPool2d((2, 2))
        self.ch_feats_out = list(ch_feats)
        self.ch_feats_out[-1] = ch_out
        self.gated_convs = nn.ModuleList([
            nn.Sequential(
                GatedConv2d(ch_feats[i]+ch_mask+(ch_feats[i-1] if i > 0 else 0), ch_feats[i], 3, 1, 1),
                nn.Conv2d(ch_feats[i], ch_feats[i], 3, 1, 1),
                nn.LeakyReLU(0.2),
            )
            for i in range(4)
        ])
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, feats_q, feats_m, masks_m):
        masks = self.avgpool(masks_m)

        f4_m = feats_m[0]
        b, t = f4_m.shape[:2]
        f4_m = self.gated_convs[0](torch.cat([f4_m, masks[0]], dim=2).flatten(0, 1))
        for i in range(1, 4):
            f4_m = torch.cat([self.avg(f4_m), feats_m[i].flatten(0, 1), masks[i].flatten(0, 1)],  dim=1)
            f4_m = self.gated_convs[i](f4_m)
        f4_m = f4_m.unflatten(0, (b, t))

        f4_q = feats_q[3]
        f4_m, A = self.sa(f4_q, feats_m[3], f4_m)
        f4_m = self.dropout(f4_m)
        f4 = self.fuse(torch.cat([f4_q, f4_m], dim=2))
        f4 = self.bottleneck(f4)

        feats = list(feats_q)
        feats[3] = f4
        return feats, feats