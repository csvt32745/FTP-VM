import torch
from torch import nn
from torch.nn import functional as F
import timm

class Backbone(nn.Module):
    def __init__(self, backbone_arch, backbone_pretrained, out_indices, in_chans=3):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_arch, pretrained=backbone_pretrained,
            features_only=True, out_indices=out_indices, in_chans=in_chans)
        self.channels = self.backbone.feature_info.channels()
    
    def forward_single_frame(self, x):
        return self.backbone(x)
    
    def forward_time_series(self, x):
        B, T = x.shape[:2]
        return [f.unflatten(0, (B, T)) for f in self.backbone(x.flatten(0, 1))]
        
    def forward(self, x):
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)

class RGBEncoder(nn.Module):
    def __init__(
        self, 
        backbone_arch, 
        backbone_pretrained, 
        out_indices):
        super().__init__()
        self.backbone = Backbone(backbone_arch, backbone_pretrained, out_indices, in_chans=3)
        self.channels = self.backbone.channels

    def forward(self, qimg, mimg, mmask):
        # ..., C, H, W
        q = qimg.size(1)
        m = mimg.size(1)
        feats = self.backbone(torch.cat([qimg, mimg], dim=1))
        return zip(*[f.split([q, m], dim=1) for f in feats])

class RGBMaskEncoder(nn.Module):
    def __init__(
        self, 
        backbone_arch, 
        backbone_pretrained, 
        out_indices):
        super().__init__()
        self.backbone = Backbone(backbone_arch, backbone_pretrained, out_indices, in_chans=4)
        self.channels = self.backbone.channels

    def forward(self, qimg, mimg, mmask):
        # ..., C, H, W
        q = qimg.size(1)
        m = mimg.size(1)
        shape = list(qimg.shape)
        shape[2] = mmask.size(2)
        qimg = torch.cat([qimg, torch.zeros(shape, device=qimg.device)], dim=2)
        mimg = torch.cat([mimg, mmask], dim=2)
        feats = self.backbone(torch.cat([qimg, mimg], dim=1))
        return zip(*[f.split([q, m], dim=1) for f in feats])

class DualEncoder(nn.Module):
    def __init__(
        self, 
        backbone_arch, 
        backbone_pretrained, 
        out_indices):
        super().__init__()

        self.backbone_q  = Backbone(backbone_arch, backbone_pretrained, out_indices, in_chans=3)
        self.backbone_m  = Backbone(backbone_arch, backbone_pretrained, out_indices, in_chans=4)
        self.channels = self.backbone_q.channels

    def forward(self, qimg, mimg, mmask):
        # ..., C, H, W
        return self.backbone_q(qimg), self.backbone_m(torch.cat([mimg, mmask], dim=-3))


if __name__ == '__main__':
    q = torch.rand((2, 3, 3, 128, 128))
    m = torch.rand((2, 1, 3, 128, 128))
    msk = torch.rand((2, 1, 1, 128, 128))
    backbone_arch = 'mobilenetv3_large_100'
    models = [
        RGBEncoder,
        RGBMaskEncoder,
        DualEncoder
    ]

    for model in models:
        model = model(backbone_arch, False, list(range(4)))
        fq, fm = model(q, m, msk)
        print(model.channels)
        print(len(fq), len(fm), fq[-1].shape, fm[-1].shape)
    
    print("Test OK!")