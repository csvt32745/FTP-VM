import torch
from torch import nn
from torch.nn import functional as F

"""
Adopted from <https://github.com/wuhuikai/DeepGuidedFilter/>
"""

class FastGuidedFilterRefiner(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.guilded_filter = FastGuidedFilter(1)
    
    def forward_single_frame(self, fine_src, base_src, base_pha):
        return self.guilded_filter(base_src.mean(1, keepdim=True), base_pha, fine_src.mean(1, keepdim=True))
    
    def forward_time_series(self, fine_src, base_src, base_pha):
        return self.forward_single_frame(
            fine_src.flatten(0, 1),
            base_src.flatten(0, 1),
            base_pha.flatten(0, 1)).unflatten(0, fine_src.shape[:2])
    
    def forward(self, fine_src, base_src, base_pha):
        if fine_src.ndim == 5:
            return self.forward_time_series(fine_src, base_src, base_pha)
        else:
            return self.forward_single_frame(fine_src, base_src, base_pha)


class FastGuidedFilter(nn.Module):
    def __init__(self, r: int, eps: float = 1e-5):
        super().__init__()
        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)

    def forward(self, lr_x, lr_y, hr_x):
        mean_x = self.boxfilter(lr_x)
        mean_y = self.boxfilter(lr_y)
        cov_xy = self.boxfilter(lr_x * lr_y) - mean_x * mean_y
        var_x = self.boxfilter(lr_x * lr_x) - mean_x * mean_x
        A = cov_xy / (var_x + self.eps)
        b = mean_y - A * mean_x
        A = F.interpolate(A, hr_x.shape[2:], mode='bilinear', align_corners=False)
        b = F.interpolate(b, hr_x.shape[2:], mode='bilinear', align_corners=False)
        return A * hr_x + b


class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()
        self.r = r

    def forward(self, x):
        # Note: The original implementation at <https://github.com/wuhuikai/DeepGuidedFilter/>
        #       uses faster box blur. However, it may not be friendly for ONNX export.
        #       We are switching to use simple convolution for box blur.
        kernel_size = 2 * self.r + 1
        kernel_x = torch.full((x.data.shape[1], 1, 1, kernel_size), 1 / kernel_size, device=x.device, dtype=x.dtype)
        kernel_y = torch.full((x.data.shape[1], 1, kernel_size, 1), 1 / kernel_size, device=x.device, dtype=x.dtype)
        x = F.conv2d(x, kernel_x, padding=(0, self.r), groups=x.data.shape[1])
        x = F.conv2d(x, kernel_y, padding=(self.r, 0), groups=x.data.shape[1])
        return x