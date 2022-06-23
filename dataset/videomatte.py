from functools import lru_cache
import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
# cv2.setNumThreads(0)
import numpy as np
from .augmentation import MotionAugmentation
from .util import get_dilated_trimaps, get_perturb_masks

# import pyarrow as pa
from multiprocessing import Manager


class VideoMatteDataset(Dataset):
    def __init__(self,
                 videomatte_dir,
                 background_image_dir,
                 background_video_dir,
                 size,
                 seq_length,
                 seq_sampler,
                 transform:MotionAugmentation=None,
                 is_VM108=True,
                 mode='train',
                 bg_num=1,
                 get_bgr_phas=False):
        assert mode in ['train', 'test']
        self.bg_num = bg_num
        self.get_bgr_phas = get_bgr_phas
        # self.manager = Manager()
        self.background_image_dir = background_image_dir
        self.background_image_files = os.listdir(background_image_dir)
        # self.background_image_files = self.manager.list(self.background_image_files)

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
        # self.background_video_frames = self.manager.list(self.background_video_frames)
        
        self.is_VM108 = is_VM108
        self.videomatte_dir = videomatte_dir
        if is_VM108:
            with open(os.path.join(videomatte_dir, f'{mode}_videos.txt'), 'r') as f:
                self.videomatte_clips = [l.strip() for l in f.readlines()]
            vm_subdir = os.path.join(videomatte_dir, 'FG_done')
            self.videomatte_frames = [
                [
                    os.path.join(vm_subdir, clip, f)
                    for f in sorted(os.listdir(os.path.join(vm_subdir, clip)))
                ]
                for clip in self.videomatte_clips
            ]
            # self.videomatte_frames = self.manager.list(self.videomatte_frames)
            # raise
        else:
            # VM240k
            raise
        
        self.videomatte_idx = [(clip_idx, frame_idx) 
                               for clip_idx in range(len(self.videomatte_clips)) 
                               for frame_idx in range(0, len(self.videomatte_frames[clip_idx]), seq_length)]
                               
        # self.videomatte_idx = self.manager.list(self.videomatte_idx)

        self.size = size
        self.seq_length = seq_length
        self.seq_sampler = seq_sampler
        self.transform = transform
        
        print("VideoMatteDataset Loaded ====================")
        print("BG imgs: ", len(self.background_image_files))
        print("BG clips & frames: %d, %d" % (len(self.background_video_clips), sum([len(l) for l in self.background_video_frames])))
        print("FG clips & frames: %d, %d" % (len(self.videomatte_clips), len(self.videomatte_idx)))

    def __len__(self):
        return len(self.videomatte_idx)
    
    def __getitem__(self, idx):
        bgr_clips = []
        for i in range(self.bg_num):
            if random.random() < 0.5:
                bgr_clips.append(self._get_random_image_background())
            else:
                bgr_clips.append(self._get_random_video_background())
        
        fgrs, phas = self._get_videomatte(idx)
        
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
            'trimap': get_dilated_trimaps(phas, np.random.randint(2, self.size//24)*2+1, random_kernel=True),
        }
        if self.get_bgr_phas:
            data['bgr_pha'] = bgr_phas
        
        return data

    def _get_random_image_background(self):
        with Image.open(os.path.join(self.background_image_dir, random.choice(self.background_image_files))) as bgr:
            bgr = self._downsample_if_needed(bgr.convert('RGB'))
        bgrs = [bgr] * self.seq_length
        return bgrs
    
    def _get_random_video_background(self):
        clip_idx = random.choice(range(len(self.background_video_clips)))
        frame_count = len(self.background_video_frames[clip_idx])
        frame_idx = random.choice(range(max(1, frame_count - self.seq_length)))
        clip = self.background_video_clips[clip_idx]
        bgrs = []
        for i in self.seq_sampler(self.seq_length):
            frame_idx_t = frame_idx + i
            frame = self.background_video_frames[clip_idx][frame_idx_t % frame_count]
            with Image.open(frame) as bgr:
                bgr = self._downsample_if_needed(bgr.convert('RGB'))
            bgrs.append(bgr)
        return bgrs
    
    def _get_videomatte(self, idx):
        clip_idx, frame_idx = self.videomatte_idx[idx]
        clip = self.videomatte_clips[clip_idx]
        frame_count = len(self.videomatte_frames[clip_idx])
        fgrs, phas = [], []
        for i in self.seq_sampler(self.seq_length):
            frame = self.videomatte_frames[clip_idx][(frame_idx + i) % frame_count]
            if self.is_VM108:
                # fg_gt = cv2.imread(frame, cv2.IMREAD_UNCHANGED)
                fg_gt = np.array(Image.open(frame).convert('RGBA'))
                fgr = fg_gt[..., :3] # [R, G, B, A] -> [R, G, B])
                # fgr = fg_gt[..., -2::-1] # [B, G, R, A] -> [R, G, B])
                fgr = Image.fromarray(fgr).convert('RGB')
                pha = fg_gt[..., -1]
                pha = Image.fromarray(pha)
                
            else:
                fgr = Image.open(os.path.join(self.videomatte_dir, 'fgr', clip, frame)).copy().convert('RGB')
                pha = Image.open(os.path.join(self.videomatte_dir, 'pha', clip, frame)).copy().convert('L')
            fgr = self._downsample_if_needed(fgr)
            pha = self._downsample_if_needed(pha)
            fgrs.append(fgr)
            phas.append(pha)
        return fgrs, phas
    
    def _downsample_if_needed(self, img):
        w, h = img.size
        if min(w, h) > self.size:
            scale = self.size / min(w, h)
            w = int(scale * w)
            h = int(scale * h)
            img = img.resize((w, h))
        return img

class VideoMatteTrainAugmentation(MotionAugmentation):
    def __init__(self, size, get_bgr_pha=False):
        super().__init__(
            size=size,
            prob_fgr_affine=0.3,
            prob_bgr_affine=0.3,
            prob_noise=0.1,
            prob_color_jitter=0.3,
            prob_grayscale=0.02,
            prob_sharpness=0.1,
            prob_blur=0.02,
            prob_hflip=0.5,
            prob_pause=0.03,
            get_bgr_pha=get_bgr_pha
        )

class VideoMatteValidAugmentation(MotionAugmentation):
    def __init__(self, size):
        super().__init__(
            size=size,
            prob_fgr_affine=0,
            prob_bgr_affine=0,
            prob_noise=0,
            prob_color_jitter=0,
            prob_grayscale=0,
            prob_sharpness=0,
            prob_blur=0,
            prob_hflip=0,
            prob_pause=0,
        )
