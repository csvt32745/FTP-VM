import os
from matplotlib.pyplot import annotate
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
        model, dataset:VM108ValidationDataset, loader_iter, pad=16, last_data=None, downsample_ratio=1.
    ):

        self.model = model.eval()
        self.dataset = dataset
        self.loader_iter = loader_iter
        self.downsample_ratio = downsample_ratio
        # self.batch_size = dataloader.batch_size
        info, images, gts, fgs, bgs, trimaps = self.request_data_from_loader(last_data)
        self.name = info['name'][0]
        self.total_frames = dataset.get_num_frames(self.name)
        self.partial_annot = ('annotated' in info)
        if self.partial_annot:
            print('Partial annotation!')
            self.annotated = info['annotated'][0].tolist()
        else:
            self.annotated = []
        # True dimensions
        h, w = images.shape[-2:]

        # Pad each side to multiple of 16
        self.pad_size = pad
        images, self.pad = pad_divide_by(images, self.pad_size)
        
        # Warning: gts and so on might have shape = (0, ) 
        # when partial annotation & the first frame gt is not given
        gts = self.pad_imgs(gts)
        gt_fgs = self.pad_imgs(fgs) if fgs is not None else None
        gt_bgs = self.pad_imgs(bgs) if bgs is not None else None
        trimaps = self.pad_imgs(trimaps) if trimaps is not None else None
        
        self.images = self.tensor_aloc(images) # T, C, H, W
        self.gts = self.tensor_aloc(gts)
        self.gt_fgs = self.tensor_aloc(gt_fgs)
        self.gt_bgs = self.tensor_aloc(gt_bgs)
        self.trimaps = self.tensor_aloc(trimaps)
        self.masks = None
        self.fgs = None
        self.bgs = None
        self.current_t = images.size(0)
        self.current_out_t = 0
        
        # Padded dimensions
        nh, nw = images.shape[-2:]
        self.h, self.w = h, w
        self.nh, self.nw = nh, nw
        self.kh = self.nh//pad
        self.kw = self.nw//pad

        self.device = 'cuda'

        self.last_data = None
        self.is_vid_overload = False

        self.save_start_idx = 0
        print('Process video %s with %d frames' % (self.name.replace('/', '_'), self.total_frames))

    def tensor_aloc(self, tensor: torch.Tensor, start=0):
        if tensor is None:
            return None
        shape = list(tensor.shape)
        shape[0] = self.total_frames
        return self.tensor_cat(torch.zeros(shape, dtype=tensor.dtype, device=tensor.device), tensor, start)

    def tensor_cat(self, tensor: torch.Tensor, target: torch.Tensor, start=0):
        if target is None:
            return None
        if tensor is None:
            return self.tensor_aloc(target, start)
        tensor[start:start+target.size(0)] = target
        return tensor

    @staticmethod
    def tensor_insert(tensor: torch.Tensor, target: torch.Tensor, indices: torch.LongTensor, start=0):
        if target is None:
            return None
        tensor[indices+start] = target
        return tensor

    def tensor_repeat_indices(self, tensor: torch.Tensor,):
        assert self.partial_annot
        idx = torch.LongTensor(self.annotated + [tensor.shape[0]])
        return torch.repeat_interleave(tensor[self.annotated], idx[1:]-idx[:-1], dim=0)

    def pad_imgs(self, imgs):
        return pad_divide_by(imgs, self.pad_size)[0]

    def unpad_imgs(self, imgs):
        return unpad(imgs, self.pad)

    def request_data_from_loader(self, last_data):
        # return vid_name, rgb, gt
        data = next(self.loader_iter) if last_data is None else last_data
        ret = [data['info'], data['rgb'][0], data['gt'][0]]
        ret.extend([data.get(k, [None])[0] for k in ['fg', 'bg', 'trimap']])
        return ret

    def add_images_from_loader(self):
        # With PAD
        # return if loading success, current data (None if ended)
        data = next(self.loader_iter, None)

        if data is None or data['info']['name'][0] != self.name:
            # Video finish
            self.last_data = data
            self.is_vid_overload = True
            return

        rgbs = data['rgb'][0]
        self.images = self.tensor_cat(self.images, self.pad_imgs(rgbs), self.current_t)
        gts = data['gt'][0]
        fgs = data.get('fg', [None])[0]
        bgs = data.get('bg', [None])[0]
        trimaps = data.get('trimap', [None])[0]
        if gts.shape[0] != rgbs.shape[0]:
            # partial annotation
            annot = data['info']['annotated']
            if len(annot) > 0:    
                annot = annot[0]
                self.gts = self.tensor_insert(self.gts, self.pad_imgs(gts), annot, self.current_t)
                if trimaps is not None: self.trimaps = self.tensor_insert(self.trimaps, self.pad_imgs(trimaps), annot, self.current_t)
                if fgs is not None: self.gt_fgs = self.tensor_insert(self.gt_fgs, self.pad_imgs(fgs), annot, self.current_t)
                if bgs is not None: self.gt_bgs = self.tensor_insert(self.gt_bgs, self.pad_imgs(bgs), annot, self.current_t)
                self.annotated.extend((annot+self.current_t).tolist())
        else:
            self.gts = self.tensor_cat(self.gts, self.pad_imgs(gts), self.current_t)
            if trimaps is not None: self.trimaps = self.tensor_cat(self.trimaps, self.pad_imgs(trimaps), self.current_t)
            if fgs is not None: self.gt_fgs = self.tensor_cat(self.gt_fgs, self.pad_imgs(fgs), self.current_t)
            if bgs is not None: self.gt_bgs = self.tensor_cat(self.gt_bgs, self.pad_imgs(bgs), self.current_t)
        self.current_t += rgbs.size(0)

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
        print(f"Save video: {path}, {name} ")
        # T, 3, H, W
        T = self.current_out_t
        masks = torch.repeat_interleave(self.masks[:T], 3, dim=1)
        gts = self.gts[:T]
        if self.partial_annot:
            gts = self.tensor_repeat_indices(self.gts)
        gts = torch.repeat_interleave(gts, 3, dim=1)
        vid_arr = [self.unpad_downsample(i) for i in [self.images[:T], masks, gts]]
        vid = torch.cat(vid_arr, axis=3).permute(0, 2, 3, 1) # T, H, 3W, 3
        os.makedirs(path, exist_ok=True)
        media.write_video(os.path.join(path, f'{name}.mp4'), vid.numpy(), fps=15)

    def unpad_downsample(self, imgs, target_width=512):
        # t, c, h, w
        imgs = self.unpad_imgs(imgs)
        # h, w = imgs.shape[-2:]
        h, w = self.h, self.w
        if self.downsample_ratio == 1 and w < target_width:
            return imgs
        h = int(h*target_width/w)
        return F.interpolate(imgs, size=(h, target_width), mode='bilinear')


    def save_naive_upsampled_imgs(self, path, imgs, start, end):
        assert imgs.ndim == 4 # t c h w
        name = self.name.replace('/', '_')
        # print(f"Save naive upsampled imgs: {path}, {name} ")
        # save masks
        pha_path = os.path.join(path, name, 'pha')
        os.makedirs(pha_path, exist_ok=True)
        imgs = self.unpad_imgs(imgs[:, 0]).numpy() # T, H, W
        for i, n in enumerate(range(start, end)):
            media.write_image(os.path.join(pha_path, f'{n:04d}.png'), imgs[i])
            
        return

    def save_imgs(self, path):
        name = self.name.replace('/', '_')
        print(f"Save imgs: {path}, {name} ")
        # save masks
        pha_path = os.path.join(path, name, 'pha')
        os.makedirs(pha_path, exist_ok=True)
        masks = self.unpad_imgs(self.masks[:, 0]).numpy() # T, H, W
        for i in range(self.current_out_t):
            media.write_image(os.path.join(pha_path, f'{i:04d}.png'), masks[i])
            
        return
        # TODO: save FGs
        # if self.fgs is not None:
        fgr_path = os.path.join(path, name, 'fgr')
        os.makedirs(fgr_path, exist_ok=True)
        fgrs = self.images if self.fgs is None else self.fgs
        fgrs = self.unpad_imgs(fgrs.permute(0, 2, 3, 1)).numpy() # T, H, W, 3
        for i in trange(fgrs.shape[0]):
            media.write_image(os.path.join(fgr_path, f'{i:04d}.png'), fgrs[i])

    def save_gt(self, path):#, is_fix_fgr=False):
        name = self.name.replace('/', '_')
        # if is_fix_fgr:
        #     name = name.split('-')[0]
        print(f"Save gt: {path}, {name} ")
        # save masks

        
        tri_path = os.path.join(path, name, 'trimap')
        if os.path.isdir(tri_path):
            return
        os.makedirs(tri_path, exist_ok=True)
        tris = self.unpad_imgs(self.trimaps[:, 0]).numpy()
        indices = self.annotated if self.partial_annot else range(self.current_t)
        for i in indices:
            media.write_image(os.path.join(tri_path, f'{i:04d}.png'), tris[i])


        pha_path = os.path.join(path, name, 'pha')
        os.makedirs(pha_path, exist_ok=True)
        gts = self.unpad_imgs(self.gts[:, 0]).numpy() # T, H, W
        for i in indices:
            media.write_image(os.path.join(pha_path, f'{i:04d}.png'), gts[i])

        # save fgs
        if self.gt_fgs is not None:
            # Not need to save gt fg if there's no fg output
            fgr_path = os.path.join(path, name, 'fgr')  
            os.makedirs(fgr_path, exist_ok=True)
            fgrs = self.unpad_imgs(self.gt_fgs).permute(0, 2, 3, 1).numpy() # T, H, W, 3
            for i in indices:
                media.write_image(os.path.join(fgr_path, f'{i:04d}.png'), fgrs[i])

    def clear(self):
        del self.gts
        del self.gt_fgs
        del self.gt_bgs
        del self.images
        del self.fgs
        del self.bgs
        del self.masks
        del self.trimaps

class InferenceCoreSTCN(InferenceCore):
    def __init__(self, 
        model, 
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

            self.masks = self.tensor_cat(self.masks, out_mask[0].cpu(), self.current_out_t)
            self.current_out_t = end
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
        self.masks = self.tensor_cat([self.masks, mask], self.current_out_t)

        mask = mask.unsqueeze(0).cuda() # 1, 1, 1, H, W
        # KV pair for the interacting frame
        rgb = self.images[frame_idx].unsqueeze(0).unsqueeze(0).cuda() # 1, 1, C, H, W
        feats = self.encode_key(rgb)
        key_k, key_v = self.encode_value(*feats, mask.cuda())

        # Propagate
        self.do_pass(key_k, key_v, frame_idx+1, (end_idx if end_idx > 0 else self.total_frames))

class InferenceCoreMemoryRecurrent(InferenceCoreSTCN):
    def __init__(self, 
        model, 
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
        model: STCNFuseMatting, 
        dataset: VM108ValidationDataset, 
        loader_iter, 
        pad=16, 
        last_data=None,
        memory_gt=False,
        memory_iter=-1,
        disable_recurrent=False,
        memory_bg=False,
        downsample_ratio=1,
        memory_save_iter=-1,
        memory_bank_size=5,
        replace_by_given_tri=False,
    ):
        super().__init__(model, dataset, loader_iter, pad, last_data, downsample_ratio=downsample_ratio)
        self.disable_recurrent = disable_recurrent
        self.model = model
        self.gru_mems = model.default_rec if 'default_rec' in dir(model) else [None] * 4

        self.memory_bank = MemoryBank(memory_bank_size)
        self.memory_save_iter = memory_save_iter
        self.clip_size = dataset.frames_per_item
        self.is_output_fg = self.model.is_output_fg if 'is_output_fg' in dir(self.model) else False
        self.forward = self._forward_fg if self.is_output_fg else self._forward
        self.memory_gt = memory_gt
        self.memory_bg = memory_bg
        assert (memory_iter <= 0) or (memory_iter > 0 and (memory_iter%dataset.frames_per_item == 0))
        self.memory_iter = memory_iter if memory_iter > 1 or memory_iter == 0 \
            else dataset.frames_per_item if memory_iter == 1 \
            else 1e7
        self.save_bg = False
        self.downsample_ratio = downsample_ratio
        self.mem_idx = -1
        self.replace_by_given_tri = replace_by_given_tri
    
    # def _forward(self, query_imgs, memory_img, memory_mask):
    #     ret = self.model.forward(query_imgs, memory_img, memory_mask, *self.gru_mems, downsample_ratio=self.downsample_ratio)
    #     if self.disable_recurrent:
    #         pha, _ = ret[:2]
    #     else:
    #         pha, self.gru_mems = ret[:2]

    #     bg = ret[-1]
    #     self.save_bg = self.save_bg or (isinstance(bg, torch.Tensor) and (bg.size(2) in [3, 4]))
    #     if pha.size(2) > 1:
    #         pha = pha[:, :, [0]]
    #     if self.masks is None:
    #         self.masks = pha[0].cpu() 
    #         if self.save_bg:
    #             self.bgs = bg[0, :, :3].cpu()
    #     else:
    #         self.masks = torch.cat([self.masks, pha[0].cpu()], 0)
    #         if self.save_bg:
    #             self.bgs = torch.cat([self.bgs, bg[0, :, :3].cpu()], 0)
    #     return pha
    
    def _forward(self, query_imgs, memory_img, memory_mask, replace_tri=False):
        # ret = self.model.forward(query_imgs, memory_img, memory_mask, *self.gru_mems, downsample_ratio=self.downsample_ratio)
        ret = self.model.forward_with_memory(query_imgs, *self.memory_bank.get_memory(), *self.gru_mems, self.downsample_ratio)
        if self.disable_recurrent:
            pha, _ = ret[:2]
        else:
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

    def propagate(self, frame_idx=0, end_idx=-1, tmp_save_root=''):
        if end_idx < 0:
            end_idx = self.total_frames
        
        this_range = self.get_frame_stamps(frame_idx, end_idx, self.clip_size)

        # take GT if input mask is not given
        while self.current_t <= frame_idx:
                self.add_images_from_loader()
        mask_idx = frame_idx

        mask = self.get_memory_mask(mask_idx).unsqueeze(0)
        # =====================

        # TODO
        # mem_mask = torch.zeros_like(mask, device='cuda').unsqueeze(0) # 1, 1, 1, H, W
        mem_mask = mask.unsqueeze(0).cuda() # 1, 1, 1, H, W
        mem_rgb = self.get_memory_img(mask_idx).unsqueeze(0).unsqueeze(0).cuda() # 1, 1, C, H, W
        frame_count = 0
        frame_count_savemem = 0
        total_time = 0
        self.memory_bank.add_gt_memory(*self.model.encode_imgs_to_value(mem_rgb, mem_mask, self.downsample_ratio))
        # for i in tqdm(range(10)):
        for i in tqdm(range(len(this_range)-1)):
            start, end = this_range[i:i+2]
            while self.current_t < end:
                self.add_images_from_loader()
            rgb = self.images[start:end].unsqueeze(0).cuda() # 1 T 3 H W
            replace_tri = False

            if self.memory_iter >= 0 and frame_count >= self.memory_iter:
                mem_rgb = self.get_memory_img(start).unsqueeze(0).unsqueeze(0).cuda()
                mem_mask = self.get_memory_mask(start).unsqueeze(0).unsqueeze(0).cuda()
                self.memory_bank.add_gt_memory(*self.model.encode_imgs_to_value(mem_rgb, mem_mask, self.downsample_ratio))
                frame_count = 0
                replace_tri = True
            
            if self.memory_save_iter > 0 and frame_count_savemem >= self.memory_save_iter:
                frame_count_savemem = frame_count_savemem-self.memory_save_iter
                self.add_memory_bank(start-frame_count_savemem-1)
            time_start = time()
            out = self.forward(rgb, mem_rgb, mem_mask, replace_tri=(replace_tri and self.replace_by_given_tri))
            total_time += time()-time_start
            
            if self.downsample_ratio != 1 and tmp_save_root != '':
                self.save_naive_upsampled_imgs(tmp_save_root, self.model.tmp_out_collab[0].cpu(), start, end)

            dt = (end-start)
            frame_count += dt
            frame_count_savemem += dt
            self.current_out_t = end
            
            # if end >= 60:
            #     break
        
        return (end_idx-frame_idx)/total_time

    def get_memory_mask(self, idx):
        return torch.zeros_like(self.trimaps[0]) if self.memory_bg else self.trimaps[idx]

    def get_memory_img(self, idx):
        return self.gt_bgs[idx] if self.memory_bg else self.images[idx]

    def add_memory_bank(self, idx):
        raise NotImplementedError

class InferenceCoreRecurrentGFM(InferenceCoreRecurrent):
    def __init__(self, model: STCNFuseMatting, dataset: VM108ValidationDataset, loader_iter, pad=16, last_data=None, memory_gt=False, memory_iter=False, memory_bg=False, downsample_ratio=1., memory_save_iter=-1, memory_bank_size=5, replace_by_given_tri=False,):
        super().__init__(model, dataset, loader_iter, pad, last_data, memory_gt, memory_iter, memory_bg=memory_bg, downsample_ratio=downsample_ratio, memory_save_iter=memory_save_iter, memory_bank_size=memory_bank_size, replace_by_given_tri=replace_by_given_tri)

        self.glance_outs = None
        self.focus_outs = None
        self.save_bg = False

    def add_memory_bank(self, idx):
        # print("Add mem: ", idx)
        self.mem_idx = idx
        rgb = self.images[idx].unsqueeze(0).unsqueeze(0).cuda()
        # tri = self.trimaps[idx].unsqueeze(0).unsqueeze(0).cuda()
        tri = self.glance_outs[idx].unsqueeze(0).unsqueeze(0).cuda()
        self.memory_bank.add_memory(*self.model.encode_imgs_to_value(rgb, tri), self.downsample_ratio)

    def _forward(self, query_imgs, memory_img, memory_mask, replace_tri=False):
        if self.memory_save_iter < 0 and replace_tri:
            glance, focus, pha, gru_mems, bg = self.model.forward(query_imgs, memory_img, memory_mask, *self.gru_mems, downsample_ratio=self.downsample_ratio, replace_given_seg=replace_tri)
        else:
            glance, focus, pha, gru_mems, bg = self.model.forward_with_memory(query_imgs, *self.memory_bank.get_memory(), *self.gru_mems, downsample_ratio=self.downsample_ratio)

        if not self.disable_recurrent:
            self.gru_mems = gru_mems
        # print(self.gru_mems)
        glance = self.seg_to_trimap(glance)
        
        self.save_bg = self.save_bg or (isinstance(bg, torch.Tensor) and (bg.size(2) in [3, 4]))
        
        self.masks = self.tensor_cat(self.masks, pha[0].cpu(), self.current_out_t)
        self.glance_outs = self.tensor_cat(self.glance_outs, glance[0].cpu(), self.current_out_t)
        self.focus_outs = self.tensor_cat(self.focus_outs, focus[0].cpu(), self.current_out_t)
        if self.save_bg:
            self.bgs = self.tensor_cat(self.bgs, bg[0, :, :3].cpu(), self.current_out_t)

        # if self.masks is None:
        #     self.masks = pha[0].cpu() 
        #     self.glance_outs = glance[0].cpu()
        #     self.focus_outs = focus[0].cpu()
        #     if self.save_bg:
        #         self.bgs = bg[0, :, :3].cpu()
        # else:
        #     self.masks = torch.cat([self.masks, pha[0].cpu()], 0)
        #     self.glance_outs = torch.cat([self.glance_outs, glance[0].cpu()], 0)
        #     self.focus_outs = torch.cat([self.focus_outs, focus[0].cpu()], 0)
        #     if self.save_bg:
        #         self.bgs = torch.cat([self.bgs, bg[0, :, :3].cpu()], 0)
        return pha

    def _forward_fg(self, query_imgs, memory_img, memory_mask):
        raise NotImplementedError
    
    def get_memomry_mask(self, idx):
        if self.memory_bg:
            return torch.zeros_like(self.trimaps[0])
        return self.trimaps[idx] if self.memory_gt else self.glance_outs[idx]

    @staticmethod
    def seg_to_trimap(logit):
        val, idx = torch.sigmoid(logit).max(dim=2, keepdim=True) # ch
        # (bg, t, fg)
        tran_mask = idx == 1
        fg_mask = idx == 2
        return tran_mask*0.5 + fg_mask

    def clear(self):
        super().clear()
        del self.glance_outs
        del self.focus_outs

    # def save_imgs(self, path):
    #     name = self.name.replace('/', '_')
    #     print(f"Save imgs: {path}, {name} ")
    #     # save masks
    #     pha_path = os.path.join(path, name, 'pha')
    #     os.makedirs(pha_path, exist_ok=True)
    #     masks = self.unpad_imgs(self.masks[:, 0]).numpy() # T, H, W
    #     tris = self.unpad_imgs(self.glance_outs[:, 0]).numpy() # T, H, W

    #     # save_trimap too
    #     tri_path = os.path.join(path, name, 'trimap25')
    #     os.makedirs(tri_path, exist_ok=True)

    #     for i in range(self.current_out_t):
    #         media.write_image(os.path.join(pha_path, f'{i:04d}.png'), masks[i])
    #         media.write_image(os.path.join(tri_path, f'{i:04d}_trimap.png'), tris[i])
    #     return

    def save_video(self, path):
        if self.save_start_idx > 0:
            return
        name = self.name.replace('/', '_')
        print(f"Save video: {path}, {name} ")
        # T, 3, H, W
        T = self.current_out_t
        masks = torch.repeat_interleave(self.masks[:T], 3, dim=1)
        gts = self.gts[:T]
        if self.partial_annot:
            gts = self.tensor_repeat_indices(gts)
        gts = torch.repeat_interleave(gts, 3, dim=1)
        vid_arr = [self.unpad_downsample(i) for i in [self.images[:T], masks, gts]]
        vid = torch.cat(vid_arr, axis=3).permute(0, 2, 3, 1) # T, H, 3W, 3
        
        if self.glance_outs is not None:
            glance = torch.repeat_interleave(self.glance_outs[:T], 3, dim=1)
            focus = torch.repeat_interleave(self.focus_outs[:T], 3, dim=1)
            trimap = self.trimaps[:T]
            if self.partial_annot:
                trimap = self.tensor_repeat_indices(trimap)
            trimap = torch.repeat_interleave(trimap, 3, dim=1)
            vid_arr = [self.unpad_downsample(i) for i in [focus, glance, trimap]]
            vid2 = torch.cat(vid_arr, axis=3).permute(0, 2, 3, 1) # T, H, 3W, 3
            vid = torch.cat([vid, vid2], axis=1) # T, 2H, 3W, 3
        
        
        os.makedirs(path, exist_ok=True)
        media.write_video(os.path.join(path, f'{name}.mp4'), vid.numpy(), fps=15)

class InferenceCoreDoubleRecurrentGFM(InferenceCoreRecurrentGFM):
    def __init__(self, model, dataset: VM108ValidationDataset, loader_iter, pad=16, last_data=None, memory_gt=False, memory_iter=False, memory_bg=False,):
        super().__init__(model, dataset, loader_iter, pad, last_data, memory_gt, memory_iter, memory_bg=False,)
        self.gru_mems = [[None]*4]*2

class InferenceCoreRecurrentGFMBG(InferenceCoreRecurrentGFM):
    def __init__(self, model, dataset: VM108ValidationDataset, loader_iter, pad=16, last_data=None, memory_gt=False, memory_iter=False, memory_bg=False,):
        super().__init__(model, dataset, loader_iter, pad, last_data, memory_gt, memory_iter, memory_bg=False,)
        self.gru_mems = [None]*5

class InferenceCoreRecurrentBG(InferenceCoreRecurrent):
    def __init__(self, model, dataset: VM108ValidationDataset, loader_iter, pad=16, last_data=None, memory_gt=False, memory_iter=False, memory_bg=False,):
        super().__init__(model, dataset, loader_iter, pad, last_data, memory_gt, memory_iter, memory_bg=False,)
        self.gru_mems = [None]*4

class InferenceCoreRecurrent3chTrimap(InferenceCoreRecurrentGFM):
    def __init__(self, model, dataset: VM108ValidationDataset, loader_iter, pad=16, last_data=None, memory_gt=False, memory_iter=False, memory_bg=False, downsample_ratio=1):
        super().__init__(model, dataset, loader_iter, pad, last_data, memory_gt, memory_iter, memory_bg, downsample_ratio)

    def get_memory_mask(self, idx):
        trimap = super().get_memory_mask(idx)
        # print(trimap.shape, self.trimap_to_3chmask(trimap).shape)
        return self.trimap_to_3chmask(trimap)

    @staticmethod
    def trimap_to_3chmask(trimap):
        fg = trimap > (1-1e-3)
        bg = trimap < 1e-3
        mask = torch.cat([fg, ~(fg|bg), bg], dim=-3).float()
        return mask

class InferenceCoreRecurrentMemAlpha(InferenceCoreRecurrentGFM):
    def __init__(self, model: STCNFuseMatting, dataset: VM108ValidationDataset, loader_iter, pad=16, last_data=None, memory_gt=False, memory_iter=False, memory_bg=False, downsample_ratio=1., memory_save_iter=-1, memory_bank_size=5):
        super().__init__(model, dataset, loader_iter, pad, last_data, memory_gt, memory_iter, memory_bg=memory_bg, downsample_ratio=downsample_ratio, memory_save_iter=memory_save_iter, memory_bank_size=memory_bank_size)

    def get_memory_mask(self, idx):
        trimap = super().get_memory_mask(idx).unsqueeze(0).unsqueeze(0).cuda()
        img = self.get_memory_img(idx).unsqueeze(0).unsqueeze(0).cuda()
        alpha = self.model.forward(img, img, trimap.repeat_interleave(2, dim=2), *self.gru_mems, downsample_ratio=self.downsample_ratio)[2]
        # print(trimap.shape, self.trimap_to_3chmask(trimap).shape)
        return torch.cat([trimap, alpha], dim=2)[0, 0].cpu()

    def add_memory_bank(self, idx):
        # print("Add mem: ", idx)
        rgb = self.images[idx].unsqueeze(0).unsqueeze(0).cuda()
        # tri = self.trimaps[idx].unsqueeze(0).unsqueeze(0).cuda()
        tri = self.glance_outs[idx].unsqueeze(0).unsqueeze(0).cuda()
        pha = self.masks[idx].unsqueeze(0).unsqueeze(0).cuda()
        self.memory_bank.add_memory(*self.model.encode_imgs_to_value(rgb, torch.cat([tri, pha], dim=2)), self.downsample_ratio)