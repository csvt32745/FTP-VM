import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from .backbone import *
from .fast_guided_filter import FastGuidedFilterRefiner
from .basic_block import *
from .decoder import *
from .util import *
from .module import *
from util.tensor_util import pad_divide_by, unpad

class FastTrimapPropagationVideoMatting(nn.Module):
    def __init__(
        self, 
        backbone_arch='mobilenetv3_large_100', 
        backbone_pretrained=True, 
        ch_bottleneck=128,
        ch_key=32,
        ch_seg=[96, 48, 16],
        ch_mat=[64, 32, 16, 8],
        ch_mask=1,
    ):

        super().__init__()

        # Encoder
        self.backbone = Backbone(backbone_arch, backbone_pretrained, (0, 1, 2, 3), in_chans=3)
        self.trimap_fuse = TrimapGatedFusion(self.backbone.channels, ch_mask=ch_mask)
        self.bottleneck_fuse = BottleneckFusion(
            self.backbone.channels[-1], ch_key, self.backbone.channels[-1], ch_bottleneck)

        self.feat_channels = list(self.backbone.channels)
        self.feat_channels[-1] = ch_bottleneck

        self.avgpool = AvgPool(3)
        
        # Decoder
        self.ch_seg = ch_seg # 8, 4, out
        self.seg_decoder = SegmentationDecoderTo4x(self.feat_channels, self.ch_seg)

        self.seg_project = Projection(self.ch_seg[-1], 3)
        self.seg_upsample = self.seg_decoder.output_stride > 1
        self.seg_stride = [self.seg_decoder.output_stride]*2

        self.ch_mat = ch_mat
        self.mat_decoder = MattingDecoderFrom4x(
            self.feat_channels, 
            [ch_bottleneck] + self.ch_seg, 
            self.ch_mat, gru=ConvGRU
        )
        self.mat_project = Projection(self.ch_mat[-1], 1)
        self.default_rec = [self.seg_decoder.default_rec, self.mat_decoder.default_rec]

        self.refiner = FastGuidedFilterRefiner(self.ch_mat)
        

    def forward(self, 
        qimgs: Tensor, mimgs: Tensor, masks: Tensor,
        rec_seg = None,
        rec_mat = None,
        downsample_ratio: float = 1,
        segmentation_pass: bool = False,
        replace_given_seg: bool = False,
    ):
        """
        `qimgs`: query frames (b, t, 3, h, w),\n
        `mimgs`: memory frames (b, 1, 3, h, w),\n
        `masks`: memory masks (b, 1, 1, h, w),\n
        `rec_seg`: RNN memory of trimap decoder, which is the output of decoder, default = `None` ,\n
        `rec_mat`: RNN memory of matting decoder, which is the output of decoder, default = `None`,\n
        `downsample_ratio`: downsample to process high-res frames, and recovered by Fast Guided Filter, default = `1`,\n
        `segmentation_pass`: output segmentation only, default = `False`,\n
        `replace_given_seg`: use the memory trimap as the result trimap in the first frame, default = `False`
        """
        if rec_seg is None:
            rec_seg, rec_mat = self.default_rec
        is_refine, qimg_sm, mimg_sm, mask_sm = self._smaller_input(qimgs, mimgs, masks, downsample_ratio)

        # Encode
        q = qimg_sm.size(1)
        feats = self.backbone(torch.cat([qimg_sm, mimg_sm], dim=1))
        # zip func is unavailable in torch.jit
        feats_q = [f[:, :q] for f in feats] # b, t_q, ch_feat_i, h, w
        feats_m = [f[:, q:] for f in feats] # b, t_q, ch_feat_i, h, w

        value_m = self.trimap_fuse(mimg_sm, mask_sm, feats_m) # b, c, t, h, w
        feats_q[-1] = self.bottleneck_fuse(feats_q[-1], feats_m[-1], value_m)
        replace_seg = mask_sm if replace_given_seg else None
        return self.decode(qimgs, qimg_sm, feats_q, segmentation_pass, is_refine, rec_seg, rec_mat, replace_seg=replace_seg)

    def forward_with_memory(self, 
        qimgs: Tensor, m_feat16: Tensor, m_value: Tensor,
        rec_seg = None,
        rec_mat = None,
        downsample_ratio: float = 1,
        segmentation_pass: bool = False,
    ):
        """
        `qimgs`: query frames (b, t, 3, h, w),
        `m_feat16`: memory key, output of `encode_imgs_to_value`, can be used without encode memory trimap & frames repeatedly
        `m_value`: memory value, output of `encode_imgs_to_value`, can be used without encode memory trimap & frames repeatedly
        `rec_seg`: RNN memory of trimap decoder, which is the output of decoder, default = None ,
        `rec_mat`: RNN memory of matting decoder, which is the output of decoder, default = None,
        `downsample_ratio`: downsample to process high-res frames, and recovered by Fast Guided Filter, default = 1,
        `segmentation_pass`: output segmentation only, default = False,
        """
        if rec_mat is None:
            rec_seg, rec_mat = self.default_rec
        
        if is_refine := (downsample_ratio != 1):
            qimg_sm = self._interpolate(qimgs, scale_factor=downsample_ratio)
        else:
            qimg_sm = qimgs

        # Encode
        feats_q = self.backbone(qimg_sm)
        feats_q[-1] = self.bottleneck_fuse(feats_q[-1], m_feat16, m_value)
    
        return self.decode(qimgs, qimg_sm, feats_q, segmentation_pass, is_refine, rec_seg, rec_mat)

    def decode(self, 
        qimgs, qimg_sm, feats_q, 
        segmentation_pass,
        is_refine,
        rec_seg = None,
        rec_mat = None,
        replace_seg = None,
    ):
        """
        Decode query & fused features to trimaps & mattes\n
        (and upsample the resulting mattes back to the original resolution).\n
        return\n
        `out_seg`: output trimaps (logits) (b, t, 3, h, w)\n
        `out_mat`: output boundary mattes (b, t, 1, h, w)\n
        `out_collab`, output complete mattes (b, t, 1, h, w)\n
        [`rec_seg`, `rec_mat`]: RNN memories
        """
        # Decode
        qimg_sm_avg = self.avgpool(qimg_sm) # 2, 4, 8
        # Segmentation
        hid, *remain = self.seg_decoder(
            qimg_sm, qimg_sm_avg[1], qimg_sm_avg[2],
            *feats_q[2:], 
            *rec_seg)
        # hid, rec1, rec2 ..., inter-feats
        rec_seg, feat_seg = remain[:-1], remain[-1]
        out_seg = self.seg_project(hid)
        if self.seg_upsample:
            out_seg = F.interpolate(out_seg.flatten(0, 1), scale_factor=self.seg_stride, mode='bilinear', align_corners=False).unflatten(0, out_seg.shape[:2])
        if segmentation_pass:
            return [out_seg, rec_seg]

        # Matting
        hid, *remain = self.mat_decoder(
            qimg_sm, *qimg_sm_avg[:2], 
            *feats_q[:2],
            feat_seg[0], feat_seg[2],
            *rec_mat
        )
        rec_mat, feats = remain[:-1], remain[-1]
        out_mat = torch.sigmoid(self.mat_project(hid))
        if replace_seg is None:
            out_collab = collaborate_fuse(out_seg, out_mat)
        else:
            t = replace_seg.size(1)
            out_collab = torch.zeros_like(out_mat)
            out_collab[:, :t] = collaborate_fuse_trimap(replace_seg, out_mat[:, :t])
            if out_mat.size(1) > t:
                out_collab[:, t:] = collaborate_fuse(out_seg[:, t:], out_mat[:, t:])


        if is_refine:
            out_collab = self.refiner(qimgs, qimg_sm, out_collab)

        return [out_seg, out_mat, out_collab, [rec_seg, rec_mat]]
        # TODO: Remove testing features
        # return [out_seg, out_mat, out_collab, [rec_seg, rec_mat], feats]

    def _smaller_input(self,
        query_img: Tensor,
        memory_img: Tensor,
        memory_mask: Tensor,
        downsample_ratio = 1
    ):
        if is_refine := (downsample_ratio != 1):
            qimg_sm = self._interpolate(query_img, scale_factor=downsample_ratio)
            mimg_sm = self._interpolate(memory_img, scale_factor=downsample_ratio)
            mask_sm = self._interpolate(memory_mask, scale_factor=downsample_ratio)
        else:
            qimg_sm = query_img
            mimg_sm = memory_img
            mask_sm = memory_mask
        return is_refine, qimg_sm, mimg_sm, mask_sm

    def encode_key(self, imgs):
        """ encode to feats and query key """
        feats = self.backbone(imgs)
        key = self.trimap_fuse(feats[3]).transpose(1, 2)
        return feats, key

    def encode_imgs_to_value(self, imgs, masks, downsample_ratio = 1):
        """ encode to memory feats & values """
        if downsample_ratio != 1:
            imgs = self._interpolate(imgs, scale_factor=downsample_ratio)
            masks = self._interpolate(masks, scale_factor=downsample_ratio)
        feat = self.backbone(imgs)
        values = self.trimap_fuse(imgs, masks, feat) # b, c, t, h, w
        return feat[-1], values

    def _interpolate(self, x: Tensor, scale_factor: float):
        if x.ndim == 5:
            B, T = x.shape[:2]
            x = F.interpolate(x.flatten(0, 1), scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
            x, _ = pad_divide_by(x, 16)
            x = x.unflatten(0, (B, T))
        else:
            x = F.interpolate(x, scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
            x, _ = pad_divide_by(x, 16)
        return x