from functools import lru_cache
import os
import random
import numpy as np
import cv2
# cv2.setNumThreads(0)
import torch
from torch.utils.data import Dataset
from PIL import Image


from .augmentation import MotionAugmentation
from .util import get_dilated_trimaps, get_perturb_masks

class ImageMatteDataset(Dataset):
    def __init__(self,
                 imagematte_dir,
                 background_image_dir,
                 background_video_dir,
                 size,
                 seq_length,
                 seq_sampler,
                 transform: MotionAugmentation,
                 bg_num=1,
                 get_bgr_phas=False):
        self.get_bgr_phas = get_bgr_phas
        self.imagematte_dir = imagematte_dir
        self.imagematte_files = os.listdir(os.path.join(imagematte_dir, 'FG'))
        self.background_image_dir = background_image_dir
        self.background_image_files = os.listdir(background_image_dir)
        self.bg_num = bg_num
        self.background_video_dir = background_video_dir
        self.background_video_clips = sorted([
            d for d in os.listdir(background_video_dir) 
            if os.path.isdir(os.path.join(background_video_dir, d))
        ])
        self.background_video_frames = [
            [
                os.path.join(background_video_dir, clip, f)
                for f in sorted(os.listdir(os.path.join(background_video_dir, clip)))
            ]
            for clip in self.background_video_clips
        ]

        self.seq_length = seq_length
        self.seq_sampler = seq_sampler
        self.size = size
        self.transform = transform
        
    def __len__(self):
        return max(len(self.imagematte_files), len(self.background_image_files) + len(self.background_video_clips))
    
    def __getitem__(self, idx):
        bgr_clips = []
        for i in range(self.bg_num):
            if random.random() < 0.2:
                bgr_clips.append(self._get_random_image_background())
            else:
                bgr_clips.append(self._get_random_video_background())
        
        fgrs, phas = self._get_imagematte(idx)
        
        if self.transform is not None:
            ret = self.transform(fgrs, phas, bgr_clips[0])
            if self.get_bgr_phas:
                fgrs, phas, bgr_clips[0], bgr_phas = ret
                bgr_phas = bgr_phas[1:]
            else:
                fgrs, phas, bgr_clips[0] = ret
            for i in range(1, self.bg_num):
                bgr_clips[i] = self.transform.bgr_augmentation(bgr_clips[i])
        
        if random.random() < 0.1:
            # random non fgr for memory frame
            fgrs[0].zero_()
            phas[0].zero_()
            if self.get_bgr_phas:
                bgr_phas[0].zero_()
            
        # return fgrs, phas, bgrs
        data = {
            'fg': fgrs,
            'bg': torch.stack(bgr_clips, 0) if self.bg_num > 1 else bgr_clips[0],
            # 'rgb': fgrs*phas + bgrs*(1-phas),
            'gt': phas,  
            'trimap': get_dilated_trimaps(phas, 17, random_kernel=False),
            'mem_trimap': get_dilated_trimaps(phas[[0]], np.random.randint(1, self.size//16)*2+1, random_kernel=True)
        }
        if self.get_bgr_phas:
            data['bgr_pha'] = bgr_phas
        
        return data

    @lru_cache(maxsize=128)
    def _read_fg_gt(self, name):
        fg = Image.open(os.path.join(self.imagematte_dir, 'FG', name)).convert('RGB').copy()
        gt = Image.open(os.path.join(self.imagematte_dir, 'GT', name)).convert('L').copy()
        return fg, gt

    def _get_imagematte(self, idx):
        # with Image.open(os.path.join(self.imagematte_dir, 'FG', self.imagematte_files[idx % len(self.imagematte_files)])) as fgr, \
        #      Image.open(os.path.join(self.imagematte_dir, 'GT', self.imagematte_files[idx % len(self.imagematte_files)])) as pha:
        name = self.imagematte_files[idx % len(self.imagematte_files)]

        fg_name = name[:-4].rsplit('_', maxsplit=1)
        fg_name[1] = '0'
        fgr, pha = self._read_fg_gt(fg_name[0]+'_'+fg_name[1]+name[-4:])

        fgr = self._downsample_if_needed(fgr)
        pha = self._downsample_if_needed(pha)
        fgrs = [fgr] * self.seq_length
        phas = [pha] * self.seq_length
        return fgrs, phas
    
    def _get_random_image_background(self):
        with Image.open(os.path.join(self.background_image_dir, self.background_image_files[random.choice(range(len(self.background_image_files)))])) as bgr:
            bgr = self._downsample_if_needed(bgr.convert('RGB'))
        bgrs = [bgr] * self.seq_length
        return bgrs

    def _get_random_video_background(self):
        clip_idx = random.choice(range(len(self.background_video_clips)))
        frame_count = len(self.background_video_frames[clip_idx])
        frame_idx = random.choice(range(max(1, frame_count - self.seq_length)))
        # clip = self.background_video_clips[clip_idx]
        bgrs = []
        for i in self.seq_sampler(self.seq_length):
            frame_idx_t = frame_idx + i
            frame = self.background_video_frames[clip_idx][frame_idx_t % frame_count]
            with Image.open(frame) as bgr:
                bgr = self._downsample_if_needed(bgr.convert('RGB'))
            bgrs.append(bgr)
        return bgrs
    
    def _downsample_if_needed(self, img):
        w, h = img.size
        if min(w, h) > self.size:
            scale = self.size / min(w, h)
            w = int(scale * w)
            h = int(scale * h)
            img = img.resize((w, h))
        return img

class ImageMatteAugmentation(MotionAugmentation):
    def __init__(self, size, get_bgr_pha=False):
        super().__init__(
            size=size,
            prob_fgr_affine=0.95,
            prob_bgr_affine=0.3,
            prob_noise=0.05,
            prob_color_jitter=0.3,
            prob_grayscale=0.03,
            prob_sharpness=0.05,
            prob_blur=0.02,
            prob_hflip=0.5,
            prob_pause=0.03,
            get_bgr_pha=get_bgr_pha,
            prob_pha_scale=0.1
        )
