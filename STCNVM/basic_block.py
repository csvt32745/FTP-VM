import torch
import torch.nn as nn
import torch.nn.functional as F

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

class FocalModulation(nn.Module):
    def __init__(self, dim, focal_window, focal_level, focal_factor=2, bias=True, proj_drop=0., use_postln=False):
        super().__init__()

        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.use_postln = use_postln

        # self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # self.f = nn.Conv2d(dim, dim + (self.focal_level+1), kernel_size=1, bias=bias)
        self.q = nn.Linear(dim, dim, bias=bias)
        self.f = nn.Linear(dim, dim + (self.focal_level+1), bias=bias)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=bias)

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        # self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.focal_layers = nn.ModuleList()
                
        self.kernel_sizes = []
        for k in range(self.focal_level):
            kernel_size = self.focal_factor*k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, 
                    groups=dim, padding=kernel_size//2, bias=False),
                    nn.GELU(),
                    )
                )              
            self.kernel_sizes.append(kernel_size)          
        if self.use_postln:
            self.ln = nn.LayerNorm(dim)

    def forward(self, x, v=None):
        """
        Args:
            x: input features with shape of (B, C, H, W)
        """
        C = x.size(1)
        x = x.permute(0, 2, 3, 1)
        v = x if v is None else v.permute(0, 2, 3, 1)
        
        # pre linear projection
        v = self.f(v).permute(0, 3, 1, 2).contiguous() # for feat aggr
        q = self.q(x).permute(0, 3, 1, 2).contiguous()
        ctx, self.gates = torch.split(v, (C, self.focal_level+1), 1)
        
        # context aggreation
        ctx_all = 0 
        for l in range(self.focal_level):         
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx*self.gates[:, l:l+1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global*self.gates[:,self.focal_level:]

        # focal modulation
        modulator = self.h(ctx_all)
        out = q*modulator
        out = out.permute(0, 2, 3, 1).contiguous()
        if self.use_postln:
            out = self.ln(out)
        
        # post linear porjection
        out = self.proj(out)
        out = self.proj_drop(out).permute(0, 3, 1, 2)
        return out
