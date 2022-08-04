"""
model.py - warpper and utility functions for network training
Compute loss, back-prop, update parameters, logging, etc.
"""


import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2

import random
from torch.nn import functional as F
from torchvision.transforms.functional import center_crop
from torchinfo import summary
import einops

from model.which_model import get_model_by_string
# from model.losses import LossComputer, iou_hooks_mo, iou_hooks_so
from model.losses import MatLossComputer, SegLossComputer
from util.log_integrator import Integrator
from util.image_saver import pool_pairs

from warmup_scheduler import GradualWarmupScheduler

class PropagationModel:
    def __init__(self, para, logger=None, save_path=None, local_rank=0, world_size=1):
        self.para = para
        print("Using model: ", para['which_model'])
        # "which_model=which_module"
        self.PNet = get_model_by_string(para['which_model'])().cuda()
        
        print("Net Parameters: ", summary(self.PNet, verbose=0).total_params)
        self.logger = logger
        self.save_path = save_path
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.para.save(os.path.join(os.path.dirname(self.save_path), 'config.json'))
        self.compose_multiobj = para['compose_multiobj']
        self.split_trimap = para['split_trimap']
        self.random_memtrimap = para['random_memtrimap']

        if logger is not None:
            self.last_time = time.time()
        self.train_integrator = Integrator(self.logger, distributed=False)

        self.loss_computer = MatLossComputer(para)
        self.seg_loss_computer = SegLossComputer(para)

        self.train()
        self.optimizer = optim.Adam(filter(
            lambda p: p.requires_grad, self.PNet.parameters()), lr=para['lr'], weight_decay=1e-7)
        # self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, para['steps'], para['gamma'])
        self._scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, para['iterations'], eta_min=1e-7, last_epoch=-1, verbose=False)
        self.scheduler = GradualWarmupScheduler(self.optimizer, 1.0, 5000, self._scheduler)
        
        # Logging info
        self.report_interval = 100
        self.save_im_interval = 800
        self.save_model_interval = 20000
        if para['debug']:
            self.report_interval = self.save_im_interval = 1
        
        self.seg_pass = self.far_seg_pass
        self.mat_pass = self.far_mat_pass

    def random_crop(self, *imgs):
        h, w = imgs[0].shape[-2:]
        w = random.choice(range(w // 2, w))
        h = random.choice(range(h // 2, h))
        results = []
        for img in imgs:
            B, T = img.shape[:2]
            img = img.flatten(0, 1)
            img = F.interpolate(img, (max(h, w), max(h, w)), mode='bilinear', align_corners=False)
            img = center_crop(img, (h, w))
            img = img.reshape(B, T, *img.shape[1:])
            results.append(img)
        return results

    @staticmethod
    def seg_to_trimap(logit):
        val, idx = torch.sigmoid(logit).max(dim=2, keepdim=True) # ch
        # (bg, t, fg)
        tran_mask = idx == 1
        fg_mask = idx == 2
        return tran_mask*0.5 + fg_mask

    @staticmethod
    def trimap_to_3chmask(trimap):
        # b, t, 1, h, w -> b, t, 3, h, w
        fg = trimap > (1-1e-3)
        bg = trimap < 1e-3
        mask = torch.cat([fg, ~(fg|bg), bg], dim=2).float()
        return mask

    def far_seg_pass(self, data):
        trimap = data['trimap']
        # gt = data['gt']
        rgb = data['rgb']

        # B, T, C, H, W
        T = rgb.size(1)
        # rgb_m, rgb_q = rgb.split([1, T-1], dim=1)
        # gt_m, gt_q = gt.split([1, T-1], dim=1)
        
        mem_tri = data['mem_trimap'] if self.random_memtrimap else trimap[:, [0]]
        if self.split_trimap:
            mem_tri = self.trimap_to_3chmask(mem_tri)
        ret = self.PNet(rgb, rgb[:, [0]], mem_tri, segmentation_pass=True)
        logits = ret[0]
        out = {
            'logits': logits
        }

        if len(ret) >= 3:
            out['extra_outs'] = ret[-1]

        if logits.size(1) == T:
            # data['gt_query'] = gt
            data['rgb_query'] = rgb
        else:
            # data['gt_query'] = gt[:, 1:]
            data['rgb_query'] = rgb[:, 1:]

        if logits.size(2) == 3:
            # GFM            
            out['mask'] = self.seg_to_trimap(logits)
            data['trimap_query'] = trimap[:, 1:]
        else:
            out['mask'] = torch.sigmoid(logits)

        return data, out

    @staticmethod
    def compose_multiobj_data(fg, bg, gt):
        # B, T, C, H, W
        other_gt = gt[:-1]
        bg[1:] = bg[1:] * (1-other_gt) + fg[:-1] * other_gt
        other_gt = gt[-1]
        bg[0] = bg[0] * (1-other_gt) + fg[-1] * other_gt
        rgb = bg * (1-gt) + fg * gt
        return rgb, bg

    def far_mat_pass(self, data):
        if (bg := data['bg']).ndim == 6:
            # for fgr with multiple bgrs
            # fg.shape = B, T, C, H, W
            # bg.shape = B, bg_num, T, C, H, W
            data['bg_num'] = bg_num = bg.size(1)
            data['fg'] = fg = einops.repeat(data['fg'], 'b t c h w -> (b bg) t c h w', bg=bg_num)
            data['gt'] = gt = einops.repeat(data['gt'], 'b t c h w -> (b bg) t c h w', bg=bg_num)
            data['bg'] = bg = einops.rearrange(bg, 'b bg t c h w -> (b bg) t c h w')
        else:
            data['bg_num'] = 1
            fg = data['fg']
            gt = data['gt']

        # trimap = data['trimap']
        if self.compose_multiobj and random.random() < 0.5:
            rgb, bg = self.compose_multiobj_data(fg, bg, gt)
            data['rgb'] = rgb
            data['bg'] = bg
        else:
            rgb = data['rgb'] = fg*gt + bg*(1-gt)
        # B, T, C, H, W

        mem_tri = data['mem_trimap'] if self.random_memtrimap else data['trimap'][:, [0]]
        if self.split_trimap:
            mem_tri = self.trimap_to_3chmask(mem_tri)
        ret = self.PNet(rgb, rgb[:, [0]], mem_tri, segmentation_pass=False)
        out = {}
        
        if (isinstance(ret[-1], list) and (ret[-1][0].size(2) == 1)) \
            or (isinstance(ret[-1], torch.Tensor) and (ret[-1].size(2) in [3, 4])):
            out['extra_outs'] = ret[-1]
        
        if len(ret) == 5:
            # GFM [glance, focus, collab, rec, extra]
            data['glance'] = ret[0]
            out['glance_out'] = self.seg_to_trimap(ret[0])
            data['focus'] = ret[1]
            data['collab'] = out['mask'] = ret[2]
        else:
            out['mask'] = ret[0]

        data['gt_query'] = gt
        data['rgb_query'] = rgb
        # data['fg_query'] = fg
        return data, out

    @staticmethod
    def cur_to_prev(cur):
        return cur[:, [0] + list(range(cur.size(1)-1))]
        # return torch.cat([cur[:, [0]], cur[:, :-1]], dim=1)


    def do_pass(self, data, it=0, segmentation_pass=False):
        # No need to store the gradient outside training
        torch.set_grad_enabled(self._is_train)
        
        for k, v in data.items():
            if type(v) != list and type(v) != dict and type(v) != int:
                data[k] = v.cuda(non_blocking=True)

        data, out = self.seg_pass(data) if segmentation_pass else self.mat_pass(data)

        if self._do_log or self._is_train:
            data, losses = (self.seg_loss_computer if segmentation_pass else self.loss_computer)\
                .compute({**data, **out}, it)

            # Logging
            if self._do_log:
                self.integrator.add_dict(losses)
                if self._is_train:
                    if it % self.save_im_interval == 0 and it != 0:
                        if self.logger is not None:
                            images = data
                            size = (256, 256)
                            self.logger.log_cv2('train/pairs', pool_pairs(images, size, True), it)

        if self._is_train:
            if (it) % self.report_interval == 0 and it != 0:
                if self.logger is not None:
                    self.logger.log_scalar('train/lr', self.scheduler.get_last_lr()[0], it)
                    self.logger.log_metrics('train', 'time', (time.time()-self.last_time)/self.report_interval, it)
                self.last_time = time.time()
                self.train_integrator.finalize('train', it)
                self.train_integrator.reset_except_hooks()

            if it % self.save_model_interval == 0 and it != 0:
                if self.logger is not None:
                    self.save(it)

            # Backward pass
            for param_group in self.optimizer.param_groups:
                for p in param_group['params']:
                    p.grad = None
            losses['total_loss'].backward()
            self.optimizer.step()
            self.scheduler.step()

    def save(self, it, ckpt_dict={}):
        if self.save_path is None:
            print('Saving has been disabled.')
            return
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        model_path = self.save_path + ('_%s.pth' % it)
        torch.save(self.PNet.state_dict(), model_path)
        print('Model saved to %s.' % model_path)

        self.save_checkpoint(it, ckpt_dict)

    def save_checkpoint(self, it, ckpt_dict={}):
        if self.save_path is None:
            print('Saving has been disabled.')
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        checkpoint_path = self.save_path + '_checkpoint.pth'
        checkpoint = { 
            'it': it,
            'network': self.PNet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()}
        checkpoint.update(ckpt_dict)
        torch.save(checkpoint, checkpoint_path)

        print('Checkpoint saved to %s.' % checkpoint_path)

    def load_model(self, path, extra_keys=[]):
        checkpoint: dict = torch.load(path)

        it = checkpoint['it']
        network = checkpoint['network']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']

        self.check_and_load_model_dict(self.PNet, network)
        self.optimizer.load_state_dict(optimizer)
        self.scheduler.load_state_dict(scheduler)
        extra_dict = { k: checkpoint.get(k, None) for k in extra_keys }

        print('Model loaded.')

        return it, extra_dict

    def load_network(self, path):
        self.check_and_load_model_dict(self.PNet, torch.load(path))
        
        
        print('Network weight loaded:', path)
    
    @staticmethod
    def check_and_load_model_dict(model: nn.Module, state_dict: dict):
        s = 'attn.proj_out.0.'
        for k in set(state_dict.keys()) - set(model.state_dict().keys()):
            if s in k:
                # new_k = 'decoder_glance.decode3.gru.attn.proj_out.0'
                idx = k.find(s)+len(s)-2
                new_k = k[:idx] + k[idx+2:]
                print("rename weight:")
                print(k, new_k)
                state_dict[new_k] = state_dict[k]
                state_dict.pop(k)
        model.load_state_dict(state_dict)

    def train(self):
        self._is_train = True
        self._do_log = True
        self.integrator = self.train_integrator
        # Shall be in eval() mode to freeze BN parameters
        # self.PNet.eval()
        return self

    def val(self):
        self._is_train = False
        self._do_log = True
        self.PNet.eval()
        return self

    def test(self):
        self._is_train = False
        self._do_log = False
        self.PNet.eval()
        return self
