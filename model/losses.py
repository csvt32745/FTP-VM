import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
import numpy as np
import kornia as K
import einops
from functools import lru_cache

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
            y = y.reshape(-1)

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
        celoss_type=para['celoss_type']
        self.ce = {
            'focal': FocalLoss,
            'normal': nn.CrossEntropyLoss,
            'normal_weight': lambda: nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 3, 1])).cuda(),
            'focal_weight': lambda: FocalLoss(alpha=torch.FloatTensor([1, 3, 1])).cuda(),
            'focal_gamma1': lambda: FocalLoss(gamma=1).cuda(),
            'focal_gamma5': lambda: FocalLoss(gamma=5).cuda(),
            'focal_gamma0.5': lambda: FocalLoss(gamma=0.5).cuda(),
        }[celoss_type]()
        self.avg2d = nn.AvgPool3d((1, 2, 2))
        self.tvloss = TotalVariationLoss(para['tvloss_type']).cuda()
        self.lambda_tvloss = para['lambda_segtv']
        self.start_tvloss = para['start_segtv']

    def compute(self, data, it):
        losses = {}
        logits = data['logits']
        # mask = data['mask']
        
        if (size:= logits.size(2)) == 3:
            # GFM
            label = get_label_from_trimap(data['trimap_query'])
            losses['seg_bce'] = self.gfm_loss(it, logits, label)
            if it >= self.start_tvloss:
                losses['seg_tv'] = self.tvloss(logits, label)*self.lambda_tvloss
            
            losses['total_loss'] = sum(losses.values())
            return data, losses

        elif size == 1:
            gt = data['gt']
            losses['seg_bce'] = self.bce(logits, gt)
        
        losses['total_loss'] = sum(losses.values())
        return data, losses

    def gfm_loss(self, it, logits, label):
        return self.ce(logits.flatten(0, 1), label.flatten(0, 1))


class MatLossComputer:
    def __init__(self, para):
        super().__init__()
        self.para = para
        self.lapla_loss = LapLoss(max_levels=5).cuda()
        # assert (celoss_type:=para['celoss_type']) in ['focal', 'normal', 'normal_weight', 'focal_weight']
        celoss_type=para['celoss_type']
        self.ce = {
            'focal': FocalLoss,
            'normal': nn.CrossEntropyLoss,
            'normal_weight': lambda: nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 3, 1])).cuda(),
            'focal_weight': lambda: FocalLoss(alpha=torch.FloatTensor([1, 3, 1])).cuda(),
            'focal_gamma1': lambda: FocalLoss(gamma=1).cuda(),
            'focal_gamma5': lambda: FocalLoss(gamma=5).cuda(),
            'focal_gamma0.5': lambda: FocalLoss(gamma=0.5).cuda(),
        }[celoss_type]()
        self.spatial_grad = K.filters.SpatialGradient()
        self.avg2d = nn.AvgPool3d((1, 2, 2))
        self.tvloss = TotalVariationLoss(para['tvloss_type']).cuda()
        self.lambda_tvloss = para['lambda_segtv']
        self.start_tvloss = para['start_segtv']
        self.full_matte = para['full_matte']
        self.lambda_tcloss = para['lambda_tc']

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
        

        if 'collab' in data:
            if self.full_matte:
                losses.update(self.full_matte_loss(it, data, gt_mask))
            else:
                losses.update(self.gfm_loss(it, data, gt_mask))
            losses['total_loss'] = sum(losses.values())
            return data, losses

        losses.update(self.matting_loss(it, mask, gt_mask))
        
        losses['total_loss'] = sum(losses.values())
        return data, losses

    def gfm_loss(self, it, data, gt_mask):
        loss = {}
        logits = data['glance']
        trimap = data['trimap_query']
        label = get_label_from_trimap(trimap)
        loss['seg_bce'] = self.ce(logits.flatten(0, 1), label.flatten(0, 1))
        if it >= self.start_tvloss:
            loss['seg_tv'] = self.tvloss(logits, label)*self.lambda_tvloss

        for k, v in self.alpha_loss(data['focus'], gt_mask, mask=(label==1).unsqueeze(2)).items():
            loss['focus_'+k] = v
        for k, v in self.alpha_loss(data['collab'], gt_mask).items():
            loss['collab_'+k] = v
        return loss
    
    def full_matte_loss(self, it, data, gt_mask):
        loss = {}
        logits = data['glance']
        trimap = data['trimap_query']
        label = get_label_from_trimap(trimap)
        loss['seg_bce'] = self.ce(logits.flatten(0, 1), label.flatten(0, 1))
        if it >= self.start_tvloss:
            loss['seg_tv'] = self.tvloss(logits, label)*self.lambda_tvloss

        for k, v in self.alpha_loss(data['focus'], gt_mask).items():
            loss['focus_'+k] = v
        return loss

    def alpha_loss(self, pred_pha, true_pha, mask=None):
        loss = {}

        # Alpha losses
        loss['pha_l1'] = L1_mask(pred_pha, true_pha, mask)
        if mask is not None:
            pred_pha = pred_pha*mask
            true_pha = true_pha*mask
        pred_pha_ = pred_pha.flatten(0, 1)
        true_pha_ = true_pha.flatten(0, 1)
        loss['pha_laplacian'] = self.lapla_loss(pred_pha_, true_pha_)
        loss['pha_coherence'] = F.mse_loss(pred_pha[:, 1:] - pred_pha[:, :-1],
                                           true_pha[:, 1:] - true_pha[:, :-1]) * self.lambda_tcloss

        return loss

    def matting_loss(self, it, pred_pha, true_pha, pred_fgr=None, true_fgr=None):
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
            loss['fgr_l1'] = L1_mask(pred_fgr, true_fgr, true_msk)
            pred_fgr = pred_fgr * true_msk
            true_fgr = true_fgr * true_msk
            loss['fgr_coherence'] = F.mse_loss(pred_fgr[:, 1:] - pred_fgr[:, :-1],
                                               true_fgr[:, 1:] - true_fgr[:, :-1]) * 5

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

def get_label_from_trimap(mask):
    label = torch.zeros_like(mask, dtype=torch.long)
    # (bg, tran, fg)
    label[mask >= 1e-5] = 1 # tran
    label[mask >= 1-1e-5] = 2 #
    return label.squeeze(2)

class TotalVariationLoss(nn.Module):
    def __init__(self, loss_type='3d_seg'):
        super().__init__()
        self.kernel = nn.Parameter(self.get_inconsis_kernel())
        
        self.loss_func = {
            'disabled': lambda p, t: 0.,
            
            '2dtv': lambda p, t: self.tv2d(p),
            '3dtv': lambda p, t: self.tv3d(p),
            '2dtv+temp_seg': lambda p, t: self.tv2d(p)+self.seg_inconsistency_temp(p, t),

            'temp_seg': lambda p, t: self.seg_inconsistency_temp(p, t),
            'temp_seg_allclass': lambda p, t: self.seg_inconsistency_temp_all_class(p, t, self.mask_weighted_avg),
            'temp_seg_allclass_mean': lambda p, t: self.seg_inconsistency_temp_all_class(p, t, self.mean_weighted_avg),
            'temp_seg_allclass_weight': lambda p, t: self.seg_inconsistency_temp_all_class_with_weight(p, t),
            'temp_seg_allclass_weight_0.33': lambda p, t: self.seg_inconsistency_temp_all_class_with_weight(p, t
                , trueclass_lambda=0.33, otherclass_lambda=0.33),
            'temp_seg_allclass_weight_0.8': lambda p, t: self.seg_inconsistency_temp_all_class_with_weight(p, t
                , trueclass_lambda=0.8, otherclass_lambda=0.1),
            
            'temp_seg_allclass_weight_l2': lambda p, t: self.seg_inconsistency_temp_all_class_with_weight_l2(p, t),
            
            '3d_seg': lambda p, t: self.seg_inconsistency_3d(p, t),
            '3d_seg_allclass': lambda p, t: self.seg_inconsistency_3d_all_class(p, t, self.mask_weighted_avg),
            '3d_seg_allclass_mean': lambda p, t: self.seg_inconsistency_3d_all_class(p, t, self.mean_weighted_avg),
            '3d_seg_allclass_weight': lambda p, t: self.seg_inconsistency_3d_all_class_with_weight(p, t),
            
        }[loss_type]
    
    def forward(self, logits, target):
        return self.loss_func(torch.sigmoid(logits), target)

    @staticmethod
    def get_inconsis_kernel():
        kernel = torch.ones((3, 3))
        kernel[0, 0] = 0
        kernel[0, -1] = 0
        kernel[-1, 0] = 0
        kernel[-1, -1] = 0
        return kernel
    
    @staticmethod
    def tv2d(prob):
        loss_h = F.l1_loss(prob[:, :, :, :-1], prob[:, :, :, 1:])
        loss_w = F.l1_loss(prob[..., :-1], prob[..., 1:])
        return loss_h + loss_w

    def tv3d(self, prob):
        loss_t = F.l1_loss(prob[:, :-1], prob[:, 1:])
        return self.tv2d(prob) + loss_t

    def seg_inconsistency_temp(self, output, target, consistency_function='abs_diff_true'):
        # https://github.com/mrebol/f2f-consistent-semantic-segmentation/blob/master/model/loss.py#L53
            
        output = output.transpose(0, 1) # t, b, ...
        target = target.transpose(0, 1) # t, b, ...
        
        pred = torch.argmax(output, dim=2).to(dtype=target.dtype)
        target_select = target.unsqueeze(2)

        gt1 = target[:-1]
        gt2 = target[1:]
            
        if consistency_function == 'argmax_pred':
            pred1 = pred[:-1]
            pred2 = pred[1:]
            diff_pred_valid = (pred1 != pred2).to(output.dtype)
        elif consistency_function == 'abs_diff':
            diff_pred_valid = (torch.abs(output[:-1] - output[1:])).sum(dim=2)
        elif consistency_function == 'sq_diff':
            diff_pred_valid = (torch.pow(output[:-1] - output[1:], 2)).sum(dim=2)
        elif consistency_function == 'abs_diff_true':
            pred1 = pred[:-1]
            pred2 = pred[1:]
            right_pred_mask = (pred1 == gt1) | (pred2 == gt2)
            diff_pred = torch.abs(output[:-1] - output[1:])
            diff_pred_true = torch.gather(diff_pred, dim=2, index=target_select[:-1]).squeeze(dim=2)
            diff_pred_valid = diff_pred_true * (right_pred_mask).to(dtype=output.dtype)
        elif consistency_function == 'sq_diff_true':
            pred1 = pred[:-1]
            pred2 = pred[1:]
            right_pred_mask = (pred1 == gt1) | (pred2 == gt2)
            diff_pred = torch.pow(output[:-1] - output[1:], 2)
            diff_pred_true = torch.gather(diff_pred, dim=2, index=target_select[:-1]).squeeze(dim=2)
            diff_pred_valid = diff_pred_true * (right_pred_mask).to(dtype=output.dtype)
        elif consistency_function == 'sq_diff_true_XOR':
            pred1 = pred[:-1]
            pred2 = pred[1:]
            right_pred_mask = (pred1 == gt1) ^ (pred2 == gt2)
            diff_pred = torch.pow(output[:-1] - output[1:], 2)
            diff_pred_true = torch.gather(diff_pred, dim=2, index=target_select[:-1]).squeeze(dim=2)
            diff_pred_valid = diff_pred_true * (right_pred_mask).to(dtype=output.dtype)
        elif consistency_function == 'abs_diff_th20':
            output1 = output[:-1]
            output2 = output[1:]
            th_mask = (output1 > 0.2) & (output2 > 0.2)
            diff_pred_valid = (torch.abs((output1 - output2) * th_mask.to(dtype=output.dtype))).sum(
                dim=2)
        
        diff_gt_valid = (gt1 != gt2)  # torch.uint8
        diff_gt_valid_dil = self.dilation(diff_gt_valid)
        inconsistencies = diff_pred_valid * (diff_gt_valid_dil)
        return inconsistencies.mean()


    def seg_inconsistency_3d(self, output, target):
        # output = output.transpose(0, 1) # t, b, ...
        # target = target.transpose(0, 1) # t, b, ...
        

        def weighted_avg(x, m):
            return (x*m).sum() / (m.sum() + 1e-5)

        pred = torch.argmax(output, dim=2).to(dtype=target.dtype)
        target_select = target.unsqueeze(2)
        right_pred = (target == pred)


        # T
        right_pred_mask = right_pred[:, :-1] | right_pred[:, 1:]
        diff_pred = torch.abs(output[:, :-1] - output[:, 1:]) 
        diff_pred_true = torch.gather(diff_pred, dim=2, index=target_select[:, :-1]).squeeze(dim=2)
        mask = right_pred_mask.to(dtype=output.dtype) * self.dilation(target[:, :-1] != target[:, 1:])
        # t = diff_pred_true * mask
        t = weighted_avg(diff_pred_true, mask)

        # H
        right_pred_mask = right_pred[..., :-1, :] | right_pred[..., 1:, :]
        diff_pred = torch.abs(output[..., :-1, :] - output[..., 1:, :]) 
        diff_pred_true = torch.gather(diff_pred, dim=2, index=target_select[..., :-1, :]).squeeze(dim=2)
        mask = right_pred_mask.to(dtype=output.dtype) * self.dilation(target[..., :-1, :] != target[..., 1:, :])
        # w = diff_pred_true * mask
        w = weighted_avg(diff_pred_true, mask)

        # W
        right_pred_mask = right_pred[..., :-1] | right_pred[..., 1:]
        diff_pred = torch.abs(output[..., :-1] - output[..., 1:]) 
        diff_pred_true = torch.gather(diff_pred, dim=2, index=target_select[..., :-1]).squeeze(dim=2)
        # print(diff_pred_true.shape, right_pred_mask.shape, target[..., :-1].shape)
        mask = right_pred_mask.to(dtype=output.dtype) * self.dilation(target[..., :-1] != target[..., 1:])
        # h = diff_pred_true * mask
        h = weighted_avg(diff_pred_true, mask)

        # print(t.shape, h.shape, w.shape)
        # return (t.mean()+h.mean()+w.mean())
        return t+h+w
    
    @staticmethod
    def mask_weighted_avg(x, m):
        return (x*m).sum() / (m.sum()*x.size(2) + 1e-5)

    @staticmethod
    def mean_weighted_avg(x, m):
        return (x*m).mean()

    def dilation(self, mask):
            ret = K.morphology.dilation(mask.float(), kernel=self.kernel, border_value=0)
            return K.morphology.dilation(ret, kernel=self.kernel, border_value=0)<1e-5

    def seg_inconsistency_3d_all_class(self, output, target, weighted_avg):
        # output = output.transpose(0, 1) # t, b, ...
        # target = target.transpose(0, 1) # t, b, ...
        


        pred = torch.argmax(output, dim=2).to(dtype=target.dtype)
        # target_select = target.unsqueeze(2)
        right_pred = (target == pred)

        # T
        right_pred_mask = right_pred[:, :-1] | right_pred[:, 1:]
        diff_pred = torch.abs(output[:, :-1] - output[:, 1:]) 
        # diff_pred_true = torch.gather(diff_pred, dim=2, index=target_select[:, :-1]).unsqueeze(dim=2)
        mask = right_pred_mask.to(dtype=output.dtype) * self.dilation(target[:, :-1] != target[:, 1:])
        # t = diff_pred_true * mask
        t = weighted_avg(diff_pred, mask.unsqueeze(dim=2))

        # H
        right_pred_mask = right_pred[..., :-1, :] | right_pred[..., 1:, :]
        diff_pred = torch.abs(output[..., :-1, :] - output[..., 1:, :]) 
        mask = right_pred_mask.to(dtype=output.dtype) * self.dilation(target[..., :-1, :] != target[..., 1:, :])
        # w = diff_pred_true * mask
        w = weighted_avg(diff_pred, mask.unsqueeze(dim=2))


        # W
        right_pred_mask = right_pred[..., :-1] | right_pred[..., 1:]
        diff_pred = torch.abs(output[..., :-1] - output[..., 1:]) 
        # diff_pred_true = torch.gather(diff_pred, dim=2, index=target_select[..., :-1]).squeeze(dim=2)
        # print(diff_pred_true.shape, right_pred_mask.shape, target[..., :-1].shape)
        mask = right_pred_mask.to(dtype=output.dtype) * self.dilation(target[..., :-1] != target[..., 1:])
        # h = diff_pred_true * mask
        h = weighted_avg(diff_pred, mask.unsqueeze(dim=2))

        # print(t.shape, h.shape, w.shape)
        # return (t.mean()+h.mean()+w.mean())
        return t+h+w

    def seg_inconsistency_temp_all_class(self, output, target, weighted_avg):
        # output = output.transpose(0, 1) # t, b, ...
        # target = target.transpose(0, 1) # t, b, ...

        pred = torch.argmax(output, dim=2).to(dtype=target.dtype)
        # target_select = target.unsqueeze(2)
        right_pred = (target == pred)

        # T
        right_pred_mask = right_pred[:, :-1] | right_pred[:, 1:]
        diff_pred = torch.abs(output[:, :-1] - output[:, 1:]) 
        # diff_pred_true = torch.gather(diff_pred, dim=2, index=target_select[:, :-1]).unsqueeze(dim=2)
        mask = right_pred_mask.to(dtype=output.dtype) * self.dilation(target[:, :-1] != target[:, 1:])
        # t = diff_pred_true * mask
        return weighted_avg(diff_pred, mask.unsqueeze(dim=2))

    def seg_inconsistency_temp_all_class_with_weight(self, output, target, trueclass_lambda=0.5, otherclass_lambda=0.25):

        pred = torch.argmax(output, dim=2).to(dtype=target.dtype)
        right_pred = (target == pred)
        gt_onehot = F.one_hot(target, num_classes=3) # b, t, h, w, 3
        gt_onehot = gt_onehot.permute(0, 1, 4, 2, 3)# b, t, 3, h, w
        trueclass_lambda = trueclass_lambda-otherclass_lambda

        def weighted_avg(x, m, g):
            m = (g*trueclass_lambda + otherclass_lambda)*m
            return (x*m).sum() / (m.sum() + 1e-5)

        # T
        right_pred_mask = right_pred[:, :-1] | right_pred[:, 1:]
        diff_pred = torch.abs(output[:, :-1] - output[:, 1:]) 
        # diff_pred_true = torch.gather(diff_pred, dim=2, index=target_select[:, :-1]).unsqueeze(dim=2)
        mask = right_pred_mask.to(dtype=output.dtype) * self.dilation(target[:, :-1] != target[:, 1:])
        # t = diff_pred_true * mask
        return weighted_avg(diff_pred, mask.unsqueeze(dim=2), gt_onehot[:, :-1])

    def seg_inconsistency_temp_all_class_with_weight_thre20(self, output, target, trueclass_lambda=0.5, otherclass_lambda=0.25):

        pred = torch.argmax(output, dim=2).to(dtype=target.dtype)
        right_pred = (target == pred)
        gt_onehot = F.one_hot(target, num_classes=3) # b, t, h, w, 3
        gt_onehot = gt_onehot.permute(0, 1, 4, 2, 3)# b, t, 3, h, w
        trueclass_lambda = trueclass_lambda-otherclass_lambda

        def weighted_avg(x, m, g):
            m = (g*trueclass_lambda + otherclass_lambda)*m
            return (x*m).sum() / (m.sum() + 1e-5)

        # T
        right_pred_mask = right_pred[:, :-1] | right_pred[:, 1:]
        thre_mask = (output[:, :-1] > 0.2) | right_pred[:, 1:]
        diff_pred = torch.abs(output[:, :-1] - output[:, 1:]) 
        # diff_pred_true = torch.gather(diff_pred, dim=2, index=target_select[:, :-1]).unsqueeze(dim=2)
        mask = right_pred_mask.to(dtype=output.dtype) * self.dilation(target[:, :-1] != target[:, 1:])
        # t = diff_pred_true * mask
        return weighted_avg(diff_pred, mask.unsqueeze(dim=2), gt_onehot[:, :-1])

    def seg_inconsistency_temp_all_class_with_weight_l2(self, output, target, trueclass_lambda=0.5, otherclass_lambda=0.25):

        pred = torch.argmax(output, dim=2).to(dtype=target.dtype)
        right_pred = (target == pred)
        gt_onehot = F.one_hot(target, num_classes=3) # b, t, h, w, 3
        gt_onehot = gt_onehot.permute(0, 1, 4, 2, 3)# b, t, 3, h, w
        trueclass_lambda = trueclass_lambda-otherclass_lambda

        def weighted_avg(x, m, g):
            m = (g*trueclass_lambda + otherclass_lambda)*m
            return (x*m).sum() / (m.sum() + 1e-5)

        # T
        right_pred_mask = right_pred[:, :-1] | right_pred[:, 1:]
        diff_pred = torch.square(output[:, :-1] - output[:, 1:]) 
        mask = right_pred_mask.to(dtype=output.dtype) * self.dilation(target[:, :-1] != target[:, 1:])
        # t = diff_pred_true * mask
        return weighted_avg(diff_pred, mask.unsqueeze(dim=2), gt_onehot[:, :-1])

    def seg_inconsistency_3d_all_class_with_weight(self, output, target, trueclass_lambda=0.5, otherclass_lambda=0.25):
        # output = output.transpose(0, 1) # t, b, ...
        # target = target.transpose(0, 1) # t, b, ...


        pred = torch.argmax(output, dim=2).to(dtype=target.dtype)
        # target_select = target.unsqueeze(2)
        right_pred = (target == pred)
        gt_onehot = F.one_hot(target, num_classes=3) # b, t, h, w, 3
        gt_onehot = gt_onehot.permute(0, 1, 4, 2, 3)# b, t, 3, h, w
        trueclass_lambda = trueclass_lambda-otherclass_lambda
        
        def weighted_avg(x, m, g):
            m = (g*trueclass_lambda + otherclass_lambda)*m
            return (x*m).sum() / (m.sum() + 1e-5)

        # T
        right_pred_mask = right_pred[:, :-1] | right_pred[:, 1:]
        diff_pred = torch.abs(output[:, :-1] - output[:, 1:]) 
        # diff_pred_true = torch.gather(diff_pred, dim=2, index=target_select[:, :-1]).unsqueeze(dim=2)
        mask = right_pred_mask.to(dtype=output.dtype) * self.dilation(target[:, :-1] != target[:, 1:])
        # t = diff_pred_true * mask
        t = weighted_avg(diff_pred, mask.unsqueeze(dim=2), gt_onehot[:, :-1])

        # H
        right_pred_mask = right_pred[..., :-1, :] | right_pred[..., 1:, :]
        diff_pred = torch.abs(output[..., :-1, :] - output[..., 1:, :]) 
        mask = right_pred_mask.to(dtype=output.dtype) * self.dilation(target[..., :-1, :] != target[..., 1:, :])
        # w = diff_pred_true * mask
        w = weighted_avg(diff_pred, mask.unsqueeze(dim=2), gt_onehot[..., :-1, :])

        # W
        right_pred_mask = right_pred[..., :-1] | right_pred[..., 1:]
        diff_pred = torch.abs(output[..., :-1] - output[..., 1:]) 
        # diff_pred_true = torch.gather(diff_pred, dim=2, index=target_select[..., :-1]).squeeze(dim=2)
        # print(diff_pred_true.shape, right_pred_mask.shape, target[..., :-1].shape)
        mask = right_pred_mask.to(dtype=output.dtype) * self.dilation(target[..., :-1] != target[..., 1:])
        # h = diff_pred_true * mask
        h = weighted_avg(diff_pred, mask.unsqueeze(dim=2), gt_onehot[..., :-1])

        # print(t.shape, h.shape, w.shape)
        # return (t.mean()+h.mean()+w.mean())
        return t+h+w
    