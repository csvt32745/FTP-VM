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
from .basic_block import ResBlock
from . import cbam

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

class MemoryReader(nn.Module):
    def __init__(self, affinity='dotproduct'):
        super().__init__()
        self.get_affinity = {
            'l2': self.affinity_l2,
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