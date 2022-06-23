import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops.focal_loss import sigmoid_focal_loss
from typing import Optional
import numpy as np
import kornia as K
import einops

from collections import defaultdict

from STCNVM.module import PRM


def get_iou_hook(values):
    return 'iou/iou', (values['hide_iou/i']+1)/(values['hide_iou/u']+1)

def get_sec_iou_hook(values):
    return 'iou/sec_iou', (values['hide_iou/sec_i']+1)/(values['hide_iou/sec_u']+1)

iou_hooks_so = [
    # get_iou_hook,
]

iou_hooks_mo = [
    # get_iou_hook,
    # get_sec_iou_hook,
]

# https://stackoverflow.com/questions/63735255/how-do-i-compute-bootstrapped-cross-entropy-loss-in-pytorch
class BootstrappedCE(nn.Module):
    def __init__(self, start_warm=5000, end_warm=60000, top_p=0.15):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p

    def forward(self, input, target, it):
        if it < self.start_warm:
            return F.cross_entropy(input, target)

        raw_loss = F.cross_entropy(input, target, reduction='none').view(-1)
        num_pixels = raw_loss.numel()

        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1-self.top_p)*((self.end_warm-it)/(self.end_warm-self.start_warm))
        # loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        loss, _ = raw_loss.sort(descending=True)[:int(num_pixels * this_p)]
        return loss.mean()

class FocalLoss(nn.Module):
    # https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/focal_loss.py
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.
    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 2.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor):
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

def get_onehot_from_trimap(trimap, eps=1e-5):
    fg = trimap > 1-eps
    bg = trimap < eps
    return torch.cat([bg, ~(bg|fg), fg], dim=2).float()

class SegLossComputer:
    def __init__(self, para):
        super().__init__()
        self.para = para
        self.bce = nn.BCEWithLogitsLoss()
        self.bsce = BootstrappedCE()
        # self.ce = nn.CrossEntropyLoss()
        self.ce = FocalLoss()
        self.avg2d = nn.AvgPool3d((1, 2, 2))
        self.avg2d_bg = nn.AvgPool3d((1, 4, 4))
        self.prm = PRM()

    def compute(self, data, it):
         # logit
        losses = {}
        logits = data['logits']
        mask = data['mask']
        gt = data['gt_query']
        
        if (size:= logits.size(2)) == 3:
            # GFM
            losses['seg_bce'] = self.gfm_loss(it, logits, data['trimap_query'])
            losses['seg_tv'] = tv_loss(logits)
            # losses['total_loss'] = sum(losses.values())
            # return data, losses
        elif size == 1:
            losses['seg_bce'] = self.bce(logits, gt)

        # BG predict
        if (bg_out:=get_extra_outs(data, 'extra_outs', [3, 4])) is not None:
            # weight = 1-self.avg2d_bg(gt)
            # losses['bg_l1'] = L1_mask(bg_out, self.avg2d_bg(data['rgb'][:, 1:]), mask=weight)*0.2
            if bg_out.size(2) == 4:
                bg_out, coarse_mask = bg_out.split([3, 1], dim=2)
                losses['coarse_mask_l1'] = L1_mask(coarse_mask, self.avg2d_bg(gt))*0.5
                data['coarse_mask'] = coarse_mask
            data['pred_bg'] = bg_out
        
        losses['total_loss'] = sum(losses.values())
        return data, losses

    def gfm_loss(self, it, logits, mask):
        # return self.fce(logits, mask)
        label = torch.zeros_like(mask, dtype=torch.long)
        # (bg, tran, fg)
        label[mask >= 1e-5] = 1 # tran
        label[mask >= 1-1e-5] = 2 #

        # label = torch.zeros_like(mask, dtype=torch.long)
        # cond = mask > 0.5
        # label[cond] = 2 # fg

        # return self.bsce(logits.flatten(0, 1), label.flatten(0, 1)[:, 0], it)
        return self.ce(logits.flatten(0, 1), label.flatten(0, 1)[:, 0])

def tv_loss(logits):
    prob = torch.sigmoid(logits) # B, T, C, H, W
    loss_h = F.l1_loss(prob[:, :, :, :-1], prob[:, :, :, 1:])
    loss_w = F.l1_loss(prob[..., :-1], prob[..., 1:])
    # loss_t = F.l1_loss(prob[:, :-1], prob[:, 1:])
    return loss_h + loss_w# + loss_t

def get_extra_outs(data: dict, key, expect_ch=1):
    if type(expect_ch) in [list, tuple]:
        cmp = lambda x, y: x in y
    else:
        cmp = lambda x, y: x == y
    if ((extra_outs := data.get(key, None)) is not None):
        if (isinstance(extra_outs, list) and cmp(extra_outs[0].size(2), expect_ch)) \
            or cmp(extra_outs.size(2), expect_ch):
            return extra_outs
    return None

class MatLossComputer:
    def __init__(self, para):
        super().__init__()
        self.para = para
        self.lapla_loss = LapLoss(max_levels=4).cuda()
        # self.sce = nn.CrossEntropyLoss()
        self.ce = FocalLoss()
        self.bsce = BootstrappedCE()
        self.spatial_grad = K.filters.SpatialGradient()
        self.avg2d = nn.AvgPool3d((1, 2, 2))
        self.prm  = PRM()
        self.avg2d_bg = nn.AvgPool3d((1, 4, 4))

    @staticmethod
    def unpack_data_with_bgnum(data: dict, key):
        if (bg_num := data.get('bg_num', -1)) > 1 and ((feats := data.get(key, None)) is not None):
            feats = [einops.rearrange(f, '(b bg_num) t c h w -> bg_num b t c h w', bg_num=bg_num) for f in feats]
            return feats
        return None

    def compute(self, data, it):
        mask = data['mask']
        gt_mask = data['gt_query']
        # trimap = data['trimap']
        losses = {}
        
        if (bg_out:=get_extra_outs(data, 'extra_outs', [3, 4])) is not None:
            gt_mask_x4 = self.avg2d_bg(gt_mask)
            if bg_out.size(2) == 4:
                bg_out, coarse_mask = bg_out.split([3, 1], dim=2)
                losses['coarse_mask_l1'] = L1_mask(coarse_mask, gt_mask_x4)*0.5
                data['coarse_mask'] = coarse_mask
            bg_pha = torch.cumsum(1-self.avg2d_bg(data['bgr_pha']), dim=0).clamp(0, 1)
            weight = (1+gt_mask_x4[:, 1:])*bg_pha
            losses['bg_l1'] = L1_mask(bg_out[:, 1:], self.avg2d_bg(data['bg'][:, 1:]), mask=weight)*0.05
            data['pred_bg'] = bg_out

        if 'collab' in data:
            losses.update(self.gfm_loss(it, data, gt_mask))
            losses['total_loss'] = sum(losses.values())
            return data, losses

        if 'pred_fg' in data:
            # losses.update(
            #     self.matting_loss(mask, gt_mask, data['pred_fg'], data['fg_query'], 
            #         feats=self.unpack_data_with_bgnum(data, 'feats'))
            # )# if it >= 60000 else None) # TODO
            losses.update(
                self.matting_loss(it, mask, gt_mask, data['pred_fg'], data['fg_query']
                    )
                    # feats=self.unpack_data_with_bgnum(data, 'feats'))
            )# if it >= 60000 else None) # TODO
        else:
            losses.update(self.matting_loss(it, mask, gt_mask))#, data['fg'], data['bg'], data['rgb'])
        
        losses['total_loss'] = sum(losses.values())
        return data, losses

    def prm_loss(self, it, pred_pha, prm_phas, true_pha):
        prm4, prm8 = prm_phas
        dilate_width = np.random.randint(1, 16)
        if it >= 20000:
        # if True:
            prm4 = self.prm(prm8, prm4, dilate_width)
            pred_pha = self.prm(prm4, pred_pha, dilate_width)
        prm_gt = self.avg2d(self.avg2d(true_pha))
        prm_loss = L1_mask(prm4, prm_gt)

        prm_gt = self.avg2d(prm_gt)
        prm_loss += L1_mask(prm8, prm_gt)
        return {'prm_l1': prm_loss*0.5}, pred_pha

    def gfm_loss(self, it, data, gt_mask):
        loss = {}
        logits = data['glance']
        trimap = data['trimap']
        label = torch.zeros_like(trimap, dtype=torch.long)
        label[trimap >= 1e-5] = 1 # tran
        label[trimap >= 1-1e-5] = 2 # fg
        # loss['seg_bce'] = self.bsce(logits.flatten(0, 1), label.flatten(0, 1)[:, 0], it)
        loss['seg_bce'] = self.ce(logits.flatten(0, 1), label.flatten(0, 1)[:, 0])
        # loss['seg_bce'], target = self.fce(logits, trimap, True) # return target = True
        loss['seg_tv'] = tv_loss(logits)

        # for k, v in self.alpha_loss(data['focus'], gt_mask, mask=target[:, :, [1]]).items():
        for k, v in self.alpha_loss(data['focus'], gt_mask, mask=(trimap==1)).items():
            loss['focus_'+k] = v
        for k, v in self.alpha_loss(data['collab'], gt_mask).items():
            loss['collab_'+k] = v
        return loss

    def alpha_loss(self, pred_pha, true_pha, mask=None):
        loss = {}

        # Alpha losses
        loss['pha_l1'] = L1_mask(pred_pha, true_pha, mask)
        # loss['pha_l1_l2'] = L1L2_split_loss(pred_pha_, true_pha_)
        # loss['pha_laplacian'] = laplacian_loss(pred_pha.flatten(0, 1), true_pha.flatten(0, 1))
        # loss['pha_grad'] = F.l1_loss(K.filters.sobel(pred_pha), K.filters.sobel(true_pha))
        if mask is not None:
            pred_pha = pred_pha*mask
            true_pha = true_pha*mask
        pred_pha_ = pred_pha.flatten(0, 1)
        true_pha_ = true_pha.flatten(0, 1)
        loss['pha_laplacian'] = self.lapla_loss(pred_pha_, true_pha_)
        loss['pha_coherence'] = F.mse_loss(pred_pha[:, 1:] - pred_pha[:, :-1],
                                           true_pha[:, 1:] - true_pha[:, :-1]) * 5

        # pred_grad = self.spatial_grad(pred_pha)
        # true_grad = self.spatial_grad(true_pha)
        # loss['pha_grad'] = F.l1_loss(pred_grad, true_grad)
        # loss['pha_grad_punish'] = 0.001 * torch.abs(pred_grad).mean()
        return loss

    def matting_loss(self, it, pred_pha, true_pha, pred_fgr=None, true_fgr=None, feats=None):
        """
        Args:
            pred_fgr: Shape(B, T, 3, H, W)
            pred_pha: Shape(B, T, 1, H, W)
            true_fgr: Shape(B, T, 3, H, W)
            true_pha: Shape(B, T, 1, H, W)
        """
        loss = dict()
        loss.update(self.alpha_loss(pred_pha, true_pha))
        
        # ================================================

        # Foreground losses
        true_msk = true_pha.gt(0) # > 0
        if pred_fgr is not None:
            # composited = fg*pred_pha + bg*(1-pred_pha)
            # loss['fgr_l1'] = F.l1_loss(pred_fgr, true_fgr)
            loss['fgr_l1'] = L1_mask(pred_fgr, true_fgr, true_msk)
            pred_fgr = pred_fgr * true_msk
            true_fgr = true_fgr * true_msk
            loss['fgr_coherence'] = F.mse_loss(pred_fgr[:, 1:] - pred_fgr[:, :-1],
                                               true_fgr[:, 1:] - true_fgr[:, :-1]) * 5
        
        # Feature losses
        if feats is not None:
            # avg_msk = einops.rearrange(true_msk, '(b bg_num) t c h w -> bg_num b t c h w', bg_num=feats[0].size(0))[0].float()
            floss = 0
            for f in feats:
                # avg_msk = self.avg2d(avg_msk) # b, t, c/1, h/2, w/2
                f = f.flatten(0, 1)
                # num, b, t, c, h, w
                floss = floss + L1_mask(f[0], f[1])
            loss['bgwise_feat_l1'] = floss*0.001

        return loss

# ----------------------------------------------------------------------------- Laplacian Loss

def L1L2_split_loss(x, y, epsilon=1.001e-5):
    mask = ((y > 1-epsilon) | (y < epsilon)).float() # FG & BG
    dif = x - y
    l1 = torch.abs(dif)
    l2 = torch.square(dif)
    b,c,h,w = mask.shape
    
    mask_sum = mask.sum()
    res = torch.sum(l2 * mask)/(mask_sum+1e-5) + torch.sum(l1 * (1-mask))/(b*h*w-mask_sum+1e-5)
    return res

def L2_mask(x, y, mask=None, epsilon=1.001e-5, normalize=True):
    res = torch.abs(x - y)
    # b,c,h,w = y.shape
    if mask is not None:
        res = res * mask
        if normalize:
            _safe = torch.sum((mask > epsilon).float()).clamp(epsilon, np.prod(y.shape)+1)
            return torch.sum(res) / _safe
        else:
            return torch.sum(res)
    if normalize:
        return torch.mean(res)
    else:
        return torch.sum(res) 

def L1_mask(x, y, mask=None, epsilon=1.001e-5, normalize=True):
    res = torch.abs(x - y)
    # b,c,h,w = y.shape
    if mask is not None:
        res = res * mask
        if normalize:
            _safe = torch.sum((mask > epsilon).float()).clamp(epsilon, np.prod(y.shape)+1)
            return torch.sum(res) / _safe
        else:
            return torch.sum(res)
    if normalize:
        return torch.mean(res)
    else:
        return torch.sum(res)

'''
Borrowed from https://gist.github.com/alper111/b9c6d80e2dba1ee0bfac15eb7dad09c8
It directly follows OpenCV's image pyramid implementation pyrDown() and pyrUp().
Reference: https://docs.opencv.org/4.4.0/d4/d86/group__imgproc__filter.html#gaf9bba239dfca11654cb7f50f889fc2ff
'''
class LapLoss(torch.nn.Module):
    def __init__(self, max_levels=5):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                    [4., 16., 24., 16., 4.],
                    [6., 24., 36., 24., 6.],
                    [4., 16., 24., 16., 4.],
                    [1., 4., 6., 4., 1.]])
        kernel /= 256.
        self.register_buffer('KERNEL', kernel.float())
        # self.L1 = nn.L1Loss()

    def downsample(self, x):
        # rejecting even rows and columns
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        # Padding zeros interleaved in x (similar to unpooling where indices are always at top-left corner)
        # Original code only works when x.shape[2] == x.shape[3] because it uses the wrong indice order
        # after the first permute
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3])
        cc = cc.permute(0,1,3,2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2]*2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3]*2, x.shape[2]*2)
        x_up = cc.permute(0,1,3,2)
        return self.conv_gauss(x_up, 4*self.KERNEL.repeat(x.shape[1], 1, 1, 1))

    def conv_gauss(self, img, kernel):
        img = F.pad(img, (2, 2, 2, 2), mode='reflect')
        out = F.conv2d(img, kernel, groups=img.shape[1])
        return out

    def laplacian_pyramid(self, img):
        current = img
        pyr = []
        for level in range(self.max_levels):
            filtered = self.conv_gauss(current, \
                self.KERNEL.repeat(img.shape[1], 1, 1, 1))
            down = self.downsample(filtered)
            up = self.upsample(down)
            diff = current-up
            pyr.append(diff)
            current = down
        return pyr

    
    def forward(self, img, tgt, mask=None, normalize=True):
        pyr_input  = self.laplacian_pyramid(img)
        pyr_target = self.laplacian_pyramid(tgt)
        loss = sum((2 ** level) * L1_mask(ab[0], ab[1], mask=mask, normalize=False) \
                    for level, ab in enumerate(zip(pyr_input, pyr_target)))
        if normalize:
            b,c,h,w = tgt.shape
            if mask is not None:
                _safe = torch.sum((mask > 1e-6).float()).clamp(1e-6, b*c*h*w+1)
            else:
                _safe = b*c*h*w
            return loss / _safe
        return loss

