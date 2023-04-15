from torch import nn
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