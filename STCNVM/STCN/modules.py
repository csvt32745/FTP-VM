"""
modules.py - This file stores the rathering boring network blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm

from . import mod_resnet
from . import cbam


class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
 
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)
 
    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
        x = self.downsample(x)
        return x + r


class FeatureFusionBlock(nn.Module):
    def __init__(self, indim, outdim):
        super().__init__()

        self.block1 = ResBlock(indim, outdim)
        self.attention = cbam.CBAM(outdim)
        self.block2 = ResBlock(outdim, outdim)

    def forward(self, x, f16):
        x = torch.cat([x, f16], 1)
        x = self.block1(x)
        r = self.attention(x)
        x = self.block2(x + r)
        return x
        

class ValueEncoder(nn.Module):
    def __init__(self, ch_fuse, ch_out, backbone_arch, backbone_pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_arch, pretrained=backbone_pretrained,
            in_chans=4, features_only=True, out_indices=(3,))
        self.channels = self.backbone.feature_info.channels()
        # resnet = mod_resnet.resnet18(pretrained=True, extra_chan=1)
        # self.conv1 = resnet.conv1
        # self.bn1 = resnet.bn1
        # self.relu = resnet.relu  # 1/2, 64
        # self.maxpool = resnet.maxpool

        # self.layer1 = resnet.layer1 # 1/4, 64
        # self.layer2 = resnet.layer2 # 1/8, 128
        # self.layer3 = resnet.layer3 # 1/16, 256

        self.fuser = FeatureFusionBlock(self.channels[-1] + ch_fuse, ch_out)

    def forward(self, image, key_f16, mask):
        # key_f16 is the feature from the key encoder
        # print(image.shape, mask.shape, key_f16.shape)
        f = torch.cat([image, mask], 1)

        # x = self.conv1(f)
        # x = self.bn1(x)
        # x = self.relu(x)   # 1/2, 64
        # x = self.maxpool(x)  # 1/4, 64
        # x = self.layer1(x)   # 1/4, 64
        # x = self.layer2(x) # 1/8, 128
        # x = self.layer3(x) # 1/16, 256
        x = self.backbone(f)[0]
        # print(x.shape, key_f16.shape)
        x = self.fuser(x, key_f16)

        return x

 
class KeyEncoder(nn.Module):
    def __init__(self, backbone_arch, backbone_pretrained=True, out_indices=(1, 2, 3), in_chans=3):
        super().__init__()
        # resnet = models.resnet50(pretrained=True)
        # self.conv1 = resnet.conv1
        # self.bn1 = resnet.bn1
        # self.relu = resnet.relu  # 1/2, 64
        # self.maxpool = resnet.maxpool

        # self.res2 = resnet.layer1 # 1/4, 256
        # self.layer2 = resnet.layer2 # 1/8, 512
        # self.layer3 = resnet.layer3 # 1/16, 1024
        self.backbone = timm.create_model(
            backbone_arch, pretrained=backbone_pretrained,
            features_only=True, out_indices=out_indices, in_chans=in_chans)
        self.channels = self.backbone.feature_info.channels()

    def forward(self, f):
        # x = self.conv1(f) 
        # x = self.bn1(x)
        # x = self.relu(x)   # 1/2, 64
        # x = self.maxpool(x)  # 1/4, 64
        # f4 = self.res2(x)   # 1/4, 256
        # f8 = self.layer2(f4) # 1/8, 512
        # f16 = self.layer3(f8) # 1/16, 1024

        return self.backbone(f)


class UpsampleBlock(nn.Module):
    def __init__(self, skip_c, up_c, out_c, scale_factor=2):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_c, up_c, kernel_size=3, padding=1)
        self.out_conv = ResBlock(up_c, out_c)
        self.scale_factor = scale_factor

    def forward(self, skip_f, up_f):
        x = self.skip_conv(skip_f)
        x = x + F.interpolate(up_f, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        x = self.out_conv(x)
        return x


class KeyProjection(nn.Module):
    def __init__(self, indim, keydim):
        super().__init__()
        self.key_proj = nn.Conv2d(indim, keydim, kernel_size=3, padding=1)

        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)
    
    def forward(self, x):
        return self.key_proj(x)
