import os
import torch

from torch import Tensor
from torch import nn
from torch.nn import functional as F

from .model import *
from .memory_bank import MemoryBank
from dataset.vm108_dataset import VM108ValidationDataset

from util.tensor_util import pad_divide_by, unpad
from torch.utils.data import DataLoader
from time import time
from tqdm import tqdm, trange
import mediapy as media

class InferenceCore:
    def __init__(self, 
        model:STCN_Full, dataset:VM108ValidationDataset, loader_iter, pad=16, last_data=None
    ):

        self.model = model
        self.dataset = dataset
        self.loader_iter = loader_iter
        # self.batch_size = dataloader.batch_size
        self.name, images, gts, fgs, bgs = self.request_data_from_loader(last_data)
        self.total_frames = dataset.get_num_frames(self.name)

        # True dimensions
        t = images.shape[1]
        h, w = images.shape[-2:]

        # Pad each side to multiple of 16
        self.pad_size = pad
        images, self.pad = pad_divide_by(images, self.pad_size)
        gts = self.pad_imgs(gts)
        gt_fgs = self.pad_imgs(fgs)
        gt_bgs = self.pad_imgs(bgs)
        self.images = images # T, C, H, W
        self.gts = gts
        self.gt_fgs = gt_fgs
        self.gt_bgs = gt_bgs
        self.masks = None
        self.fgs = None
        self.bgs = None
        
        # Padded dimensions
        nh, nw = images.shape[-2:]
        self.h, self.w = h, w
        self.nh, self.nw = nh, nw
        self.kh = self.nh//16
        self.kw = self.nw//16

        self.device = 'cuda'

        self.last_data = None
        self.is_vid_overload = False

        self.save_start_idx = 0
        print('Process video %s with %d frames' % (self.name.replace('/', '_'), self.total_frames))

    def pad_imgs(self, imgs):
        return pad_divide_by(imgs, self.pad_size)[0]

    def unpad_imgs(self, imgs):
        return unpad(imgs, self.pad)

    def request_data_from_loader(self, last_data):
        # return vid_name, rgb, gt
        data = next(self.loader_iter) if last_data is None else last_data
        return data['info']['name'][0], data['rgb'][0], data['gt'][0], data['fg'][0], data['bg'][0]

    def add_images_from_loader(self):
        # With PAD
        # return if loading success, current data (None if ended)
        data = next(self.loader_iter, None)

        if data is None or data['info']['name'][0] != self.name:
            # Video finish
            self.last_data = data
            self.is_vid_overload = True
            return

        rgbs = self.pad_imgs(data['rgb'][0])
        gts = self.pad_imgs(data['gt'][0])
        fgs = self.pad_imgs(data['fg'][0])
        bgs = self.pad_imgs(data['bg'][0])
        self.images = torch.cat([self.images, rgbs])
        self.gts = torch.cat([self.gts, gts])
        self.gt_fgs = torch.cat([self.gt_fgs, fgs])
        self.gt_bgs = torch.cat([self.gt_bgs, bgs])

    @staticmethod
    def get_frame_stamps(start, end, step):
        this_range = list(range(start, end, step))
        this_range.append(end)
        # [idx, idx+b, idx+b*2, ..., end_idx]
        return this_range

    def propagate(self):
        raise NotImplementedError

    def save_video(self, path):
        if self.save_start_idx > 0:
            return
        name = self.name.replace('/', '_')
        # path = os.path.join(path, vm108.mode)
        print(f"Save video: {path}, {name} ")
        # T, 3, H, W
        T = self.masks.size(0)
        masks = torch.repeat_interleave(self.masks, 3, dim=1)
        gts = torch.repeat_interleave(self.gts[:T], 3, dim=1)
        vid_arr = [self.images[:T], masks, gts]
        vid = torch.cat(vid_arr, axis=3).permute(0, 2, 3, 1) # T, H, 3W, 3
        if self.bgs is not None:
            bgs = F.interpolate(self.bgs, masks.shape[-2:], mode='bilinear')
            gt_bgs = self.gt_bgs[:T]
            blank = torch.zeros_like(bgs)
            vid2 = torch.cat([blank, bgs, gt_bgs], axis=3).permute(0, 2, 3, 1) # T, H, 3W, 3
            vid = torch.cat([vid, vid2], axis=1) # T, 2H, 3W, 3
        if False:
        # if self.fgs is not None:
            fgs = self.fgs*self.masks
            gt_fgs = self.gt_fgs[:T]*self.gts[:T]
            blank = torch.zeros_like(fgs)
            vid2 = torch.cat([blank, fgs, gt_fgs], axis=3).permute(0, 2, 3, 1) # T, H, 3W, 3
            vid = torch.cat([vid, vid2], axis=1) # T, 2H, 3W, 3
        # print(vid.shape)
        os.makedirs(path, exist_ok=True)
        media.write_video(os.path.join(path, f'{name}.mp4'), vid.numpy(), fps=15)

    def save_imgs(self, path):
        name = self.name.replace('/', '_')
        print(f"Save imgs: {path}, {name} ")
        # save masks
        pha_path = os.path.join(path, name, 'pha')
        os.makedirs(pha_path, exist_ok=True)
        masks = self.masks[:, 0].numpy() # T, H, W
        for i in range(masks.shape[0]):
            media.write_image(os.path.join(pha_path, f'{i:04d}.png'), masks[i])
            
        return
        # TODO: save FGs
        # if self.fgs is not None:
        fgr_path = os.path.join(path, name, 'fgr')
        os.makedirs(fgr_path, exist_ok=True)
        fgrs = self.images if self.fgs is None else self.fgs
        fgrs = fgrs.permute(0, 2, 3, 1).numpy() # T, H, W, 3
        for i in trange(fgrs.shape[0]):
            media.write_image(os.path.join(fgr_path, f'{i:04d}.png'), fgrs[i])

    def save_gt(self, path, is_fix_fgr=True):
        name = self.name.replace('/', '_')
        if is_fix_fgr:
            name = name.split('-')[0]
        print(f"Save gt: {path}, {name} ")
        # save masks

        pha_path = os.path.join(path, name, 'pha')
        if os.path.isdir(pha_path):
            return

        os.makedirs(pha_path, exist_ok=True)
        gts = self.gts[:, 0].numpy() # T, H, W
        for i in range(gts.shape[0]):
            media.write_image(os.path.join(pha_path, f'{i:04d}.png'), gts[i])

        # save fgs
        # if self.fgs is not None:
            # Not need to save gt fg if there's no fg output
        fgr_path = os.path.join(path, name, 'fgr')  
        os.makedirs(fgr_path, exist_ok=True)
        fgrs = self.gt_fgs.permute(0, 2, 3, 1).numpy() # T, H, W, 3
        for i in trange(fgrs.shape[0]):
            media.write_image(os.path.join(fgr_path, f'{i:04d}.png'), fgrs[i])

    def clear(self):
        del self.gts
        del self.gt_fgs
        del self.images
        del self.fgs
        del self.masks

class InferenceCoreSTCN(InferenceCore):
    def __init__(self, 
        model: STCN_Full, 
        dataset: VM108ValidationDataset, loader_iter, 
        num_objects=1, top_k=40, mem_every=5, include_last=False,
        pad=16, last_data=None
    ):
        super().__init__(model, dataset, loader_iter, pad, last_data)
    
        self.mem_every = mem_every
        self.include_last = include_last

        self.k = num_objects
        self.mem_bank = MemoryBank(k=self.k, top_k=top_k)

        self.cur_idx = 0

    def encode_key(self, rgb):
        k16, qv16, qf16, qf8, qf4 = self.model.encode_key(rgb) # 1 T C H W
        return rgb, qf8, qf4, k16, qv16, qf16

    def decode_with_query(self, rgb, qf8, qf4, k16, qv16, qf16):
        # feed the result into decoder
        return self.model.decode_with_query(self.mem_bank, rgb, qf8, qf4, k16, qv16) # 1 T 1 H W

    def encode_value(self, rgb, qf8, qf4, k16, qv16, qf16, mask, idx=-1):
        if idx > 0:
            return k16[:, :, [idx]], self.model.encode_value(rgb[:, [idx]], rgb[:, [idx]], rgb[:, [idx]])
        return k16, self.model.encode_value(rgb, qf16, mask)

    def do_pass(self, key_k, key_v, idx, end_idx):
        self.mem_bank.add_memory(key_k, key_v)
        closest_ti = end_idx

        # Note that we never reach closest_ti, just the frame before it
        # this_range = self.get_frame_stamps(idx, closest_ti, 1)
        this_range = self.get_frame_stamps(idx, closest_ti, self.mem_every)
        final_idx = closest_ti - 1

        for i in tqdm(range(len(this_range)-1)):
            start, end = this_range[i:i+2]
            while self.images.size(0) < end:
                self.add_images_from_loader()
            rgb = self.images[start:end].unsqueeze(0).cuda() # 1 T 3 H W
            
            query_feats = self.encode_key(rgb)
            out_mask = self.decode_with_query(*query_feats)

            self.masks = torch.cat([self.masks, out_mask[0].cpu()], 0)
            self.cur_idx = end
            if start != final_idx: # and start % self.mem_every == 0:
                prev_key, prev_value = self.encode_value(*query_feats, out_mask, idx=0)
                self.mem_bank.add_memory(prev_key, prev_value)
                self.mem_bank.memory_pruning()

        return closest_ti
    
    def propagate(self, frame_idx=0, end_idx=-1, mask=None):
        # propagate the frame with input mask
        while self.images.size(0) < frame_idx:
            self.add_images_from_loader()
        
        # take GT if input mask is not given
        mask = self.pad_imgs(mask) if mask is not None else self.gts[[frame_idx]]
        if self.masks == None:
            self.masks = mask
        else:
            self.masks = torch.cat([self.masks, mask], 0)

        mask = mask.unsqueeze(0).cuda() # 1, 1, 1, H, W
        # KV pair for the interacting frame
        rgb = self.images[frame_idx].unsqueeze(0).unsqueeze(0).cuda() # 1, 1, C, H, W
        feats = self.encode_key(rgb)
        key_k, key_v = self.encode_value(*feats, mask.cuda())

        # Propagate
        self.do_pass(key_k, key_v, frame_idx+1, (end_idx if end_idx > 0 else self.total_frames))

class InferenceCoreMemoryRecurrent(InferenceCoreSTCN):
    def __init__(self, 
        model: STCN_VM, 
        dataset: VM108ValidationDataset, loader_iter, 
        num_objects=1, top_k=40, mem_every=10, include_last=False, 
        pad=16, last_data=None
    ):
        super().__init__(model, dataset, loader_iter, num_objects, top_k, mem_every, include_last, pad, last_data)
        self.model = model
        self.gru_mems = [None] * 4
    
    def decode_with_query(self, rgb, qf8, qf4, k16, qv16, qf16):
        # feed the result into decoder
        out, self.gru_mems = self.model.decode_with_query(self.mem_bank, self.gru_mems, rgb, qf8, qf4, k16, qv16)
        return out

    def propagate(self, frame_idx=0, end_idx=-1, mask=None):
        return super().propagate(frame_idx, end_idx, mask)

class InferenceCoreRecurrent(InferenceCore):
    def __init__(self, 
        model: DualMattingNetwork, 
        dataset: VM108ValidationDataset, 
        loader_iter, 
        pad=16, 
        last_data=None,
        memory_gt=False,
        memory_iter=-1
    ):
        super().__init__(model, dataset, loader_iter, pad, last_data)
    
        self.model = model
        self.gru_mems = [None] * 4
        self.clip_size = dataset.frames_per_item
        self.is_output_fg = self.model.is_output_fg
        self.forward = self._forward_fg if self.is_output_fg else self._forward
        self.memory_gt = memory_gt
        assert (memory_iter <= 0) or (memory_iter%dataset.frames_per_item == 0)
        self.memory_iter = memory_iter if memory_iter > 1 \
            else dataset.frames_per_item if memory_iter == 1 \
            else 1e7
        self.save_bg = False
    
    def _forward(self, query_imgs, memory_img, memory_mask):
        ret = self.model.forward(query_imgs, memory_img, memory_mask, *self.gru_mems)
        pha, self.gru_mems = ret[:2]
        bg = ret[-1]
        self.save_bg = self.save_bg or (isinstance(bg, torch.Tensor) and (bg.size(2) in [3, 4]))
        if pha.size(2) > 1:
            pha = pha[:, :, [0]]
        if self.masks is None:
            self.masks = pha[0].cpu() 
            if self.save_bg:
                self.bgs = bg[0, :, :3].cpu()
        else:
            self.masks = torch.cat([self.masks, pha[0].cpu()], 0)
            if self.save_bg:
                self.bgs = torch.cat([self.bgs, bg[0, :, :3].cpu()], 0)
        return pha

    def _forward_fg(self, query_imgs, memory_img, memory_mask):
        ret = self.model.forward(query_imgs, memory_img, memory_mask, *self.gru_mems)
        pha, fgr, self.gru_mems = ret[:3]
        bg = ret[-1]
        self.save_bg = self.save_bg or (isinstance(bg, torch.Tensor) and (bg.size(2) in [3, 4]))
        if pha.size(2) > 1:
            pha = pha[:, :, [0]]
        if self.masks is None:
            self.masks = pha[0].cpu()
            self.fgs = fgr[0].cpu()
        else:
            self.masks = torch.cat([self.masks, pha[0].cpu()], 0)
            self.fgs = torch.cat([self.fgs, fgr[0].cpu()], 0)
        return pha, fgr

    def propagate(self, frame_idx=0, end_idx=-1, mask=None, mask_idx=None):
        if end_idx < 0:
            end_idx = self.total_frames
        
        # skip the first frame which is mem mask
        # this_range = self.get_frame_stamps(frame_idx+1, end_idx, self.clip_size)
        # still output the frame with mem mask since there's output FG
        this_range = self.get_frame_stamps(frame_idx, end_idx, self.clip_size)

        # take GT if input mask is not given
        while self.images.size(0) <= frame_idx:
                self.add_images_from_loader()
        if mask_idx is None:
            mask_idx = frame_idx
        mask = self.pad_imgs(mask) if mask is not None else self.gts[[mask_idx]]

        # input mask as result
        # if self.masks is None:
        #     self.masks = mask
        # else:
        #     self.masks = torch.cat([self.masks, mask], 0)

        # mem_mask = torch.zeros_like(mask).unsqueeze(0).cuda() # 1, 1, 1, H, W
        mem_mask = mask.unsqueeze(0).cuda() # 1, 1, 1, H, W
        mem_rgb = self.images[mask_idx].unsqueeze(0).unsqueeze(0).cuda() # 1, 1, C, H, W
        frame_count = 0
        total_time = 0
        # for i in tqdm(range(10)):
        for i in tqdm(range(len(this_range)-1)):
            start, end = this_range[i:i+2]
            while self.images.size(0) < end:
                self.add_images_from_loader()
            rgb = self.images[start:end].unsqueeze(0).cuda() # 1 T 3 H W
            time_start = time()
            out = self.forward(rgb, mem_rgb, mem_mask)
            total_time += time()-time_start

            frame_count += (end-start)
            if frame_count >= self.memory_iter:
                mem_rgb = self.images[end-1].unsqueeze(0).unsqueeze(0).cuda()
                mem_mask = self.gts[end-1] if self.memory_gt else self.masks[end-1]
                mem_mask = mem_mask.unsqueeze(0).unsqueeze(0).cuda()
                frame_count = 0
            self.cur_idx = end
            # break
        
        return (end_idx-frame_idx)/total_time

class InferenceCoreRecurrentNeighbor(InferenceCoreRecurrent):
    def __init__(self, model: DualMattingNetwork, dataset: VM108ValidationDataset, loader_iter, pad=16, last_data=None, memory_gt=False, memory_iter=False):
        super().__init__(model, dataset, loader_iter, pad, last_data, memory_gt, memory_iter)
    
    def _forward(self, query_imgs, memory_img, memory_mask):
        pha, self.gru_mems = self.model.forward(query_imgs, memory_img, memory_mask, *self.gru_mems)[:2]
        if pha.size(2) > 1:
            pha = pha[:, :, [0]]
        if self.masks is None:
            self.masks = pha[0].cpu() 
        else:
            self.masks = torch.cat([self.masks, pha[0].cpu()], 0)
        return pha

    def _forward_fg(self, query_imgs, memory_img, memory_mask):
        pha, fgr, self.gru_mems = self.model.forward(query_imgs, memory_img, memory_mask, *self.gru_mems)[:3]
        if pha.size(2) > 1:
            pha = pha[:, :, [0]]
        if self.masks is None:
            self.masks = pha[0].cpu()
            self.fgs = fgr[0].cpu()
        else:
            self.masks = torch.cat([self.masks, pha[0].cpu()], 0)
            self.fgs = torch.cat([self.fgs, fgr[0].cpu()], 0)
        return pha, fgr

    def propagate(self, frame_idx=0, end_idx=-1, mask=None):
        if end_idx < 0:
            end_idx = self.total_frames
        
        # skip the first frame which is mem mask
        # this_range = self.get_frame_stamps(frame_idx+1, end_idx, self.clip_size)
        # still output the frame with mem mask since there's output FG
        this_range = self.get_frame_stamps(frame_idx, end_idx, self.clip_size)

        # take GT if input mask is not given
        mask = self.pad_imgs(mask) if mask is not None else self.gts[[frame_idx]]

        # input mask as result
        # if self.masks is None:
        #     self.masks = mask
        # else:
        #     self.masks = torch.cat([self.masks, mask], 0)

        # mem_mask = torch.zeros_like(mask).unsqueeze(0).cuda() # 1, 1, 1, H, W
        mem_mask = mask.unsqueeze(0).cuda() # 1, 1, 1, H, W
        mem_rgb = self.images[frame_idx].unsqueeze(0).unsqueeze(0).cuda() # 1, 1, C, H, W

        # for i in tqdm(range(10)):
        for i in tqdm(range(len(this_range)-1)):
            start, end = this_range[i:i+2]
            while self.images.size(0) < end:
                self.add_images_from_loader()
            rgb = self.images[start:end].unsqueeze(0).cuda() # 1 T 3 H W
            
            out = self.forward(rgb, mem_rgb, mem_mask)
            mem_rgb = self.images[end-1].unsqueeze(0).unsqueeze(0).cuda()
            mem_mask = self.gts[end-1].unsqueeze(0).unsqueeze(0).cuda()
            self.cur_idx = end
        
        return True

class InferenceCoreRecurrentGFM(InferenceCoreRecurrent):
    def __init__(self, model: DualMattingNetwork, dataset: VM108ValidationDataset, loader_iter, pad=16, last_data=None, memory_gt=False, memory_iter=False):
        super().__init__(model, dataset, loader_iter, pad, last_data, memory_gt, memory_iter)

        self.glance_outs = None
        self.focus_outs = None
        self.save_bg = False

    def _forward(self, query_imgs, memory_img, memory_mask):
        glance, focus, pha, self.gru_mems, bg = self.model.forward(query_imgs, memory_img, memory_mask, self.gru_mems)
        glance = self.seg_to_trimap(glance)
        self.save_bg = self.save_bg or (isinstance(bg, torch.Tensor) and (bg.size(2) in [3, 4]))
        if self.masks is None:
            self.masks = pha[0].cpu() 
            self.glance_outs = glance[0].cpu()
            self.focus_outs = focus[0].cpu()
            if self.save_bg:
                self.bgs = bg[0, :, :3].cpu()
        else:
            self.masks = torch.cat([self.masks, pha[0].cpu()], 0)
            self.glance_outs = torch.cat([self.glance_outs, glance[0].cpu()], 0)
            self.focus_outs = torch.cat([self.focus_outs, focus[0].cpu()], 0)
            if self.save_bg:
                self.bgs = torch.cat([self.bgs, bg[0, :, :3].cpu()], 0)
        return pha

    def _forward_fg(self, query_imgs, memory_img, memory_mask):
        raise NotImplementedError
    
    @staticmethod
    def seg_to_trimap(logit):
        val, idx = torch.sigmoid(logit).max(dim=2, keepdim=True) # ch
        # (bg, t, fg)
        tran_mask = idx == 1
        fg_mask = idx == 2
        return tran_mask*0.5 + fg_mask

class InferenceCoreDoubleRecurrentGFM(InferenceCoreRecurrentGFM):
    def __init__(self, model: DualMattingNetwork, dataset: VM108ValidationDataset, loader_iter, pad=16, last_data=None, memory_gt=False, memory_iter=False):
        super().__init__(model, dataset, loader_iter, pad, last_data, memory_gt, memory_iter)
        self.gru_mems = [[None]*4]*2

class InferenceCoreRecurrentGFMBG(InferenceCoreRecurrentGFM):
    def __init__(self, model: DualMattingNetwork, dataset: VM108ValidationDataset, loader_iter, pad=16, last_data=None, memory_gt=False, memory_iter=False):
        super().__init__(model, dataset, loader_iter, pad, last_data, memory_gt, memory_iter)
        self.gru_mems = [None]*5