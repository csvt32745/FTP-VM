import torch
import os
import json
import numpy as np
import cv2
# cv2.setNumThreads(0)
import random
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F
from collections import defaultdict
import random
from tqdm import tqdm

from .util import get_dilated_trimaps, get_perturb_masks

class YouTubeVISDataset(Dataset):
    def __init__(self, videodir, annfile, size, seq_length, seq_sampler, transform=None, debug_data=None, is_trimap=False):
        self.is_trimap = is_trimap
        self.videodir = videodir
        self.size = size
        self.seq_length = seq_length
        self.seq_sampler = seq_sampler
        self.transform = transform

        print("Loading Youtube VIS annotation ...")
        if debug_data is None:
            with open(annfile) as f:
                data = json.load(f)
        else:
            data = debug_data


        self.masks = defaultdict(list)
        for ann in tqdm(data['annotations']):
            # Just add RLE
            self.masks[ann['video_id']].append(ann['segmentations'])
            
            # Preprocessing maksks
            # H, W = ann['height'], ann['width']
            # segs = []
            # for rle in ann['segmentations']:
            #     seg = np.zeros((H, W), dtype=np.uint8) if rle is None else self._decode_rle(rle)
            #     segs.append(self._downsample_if_needed(Image.fromarray(seg), Image.NEAREST))
            # self.masks[ann['video_id']].append(segs)

        # masks =
        # {
        #   video_id: [[seq of mask1], [seq of mask2], ...]
        # }

        self.videos = {}
        for video in data['videos']:
            video_id = video['id']
            if video_id in self.masks:
                self.videos[video_id] = video
        
        self.index = []
        for video_id in self.videos.keys():
            for frame in range(len(self.videos[video_id]['file_names'])):
                self.index.append((video_id, frame))
                
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        video_id, frame_id = self.index[idx]
        video = self.videos[video_id]
        frame_count = len(self.videos[video_id]['file_names'])
        H, W = video['height'], video['width']
        
        imgs, segs = [], []
        masks = random.choice(self.masks[video_id])
        for t in self.seq_sampler(self.seq_length):
            frame = (frame_id + t) % frame_count

            filename = video['file_names'][frame]
            mask = masks[frame]
            # masks = self.masks[video_id][frame]
        
            with Image.open(os.path.join(self.videodir, filename)) as img:
                imgs.append(self._downsample_if_needed(img.convert('RGB'), Image.BILINEAR))
        
            seg = np.zeros((H, W), dtype=np.uint8) if mask is None else self._decode_rle(mask)
            segs.append(self._downsample_if_needed(Image.fromarray(seg), Image.NEAREST))
            
        if self.transform is not None:
            imgs, segs = self.transform(imgs, segs)
        
        data = {
            'rgb': imgs,
            'gt': segs,
            
        }
        if self.is_trimap:
            data['trimap'] = get_dilated_trimaps(segs, np.random.randint(2, 15)*2+1)
        return data
    

    def _decode_rle(self, rle):
        H, W = rle['size']
        msk = np.zeros(H * W, dtype=np.uint8)
        encoding = rle['counts']
        skip = 0
        for i in range(0, len(encoding) - 1, 2):
            skip += encoding[i]
            draw = encoding[i + 1]
            msk[skip : skip + draw] = 255
            skip += draw
        return msk.reshape(W, H).transpose()
    
    def _downsample_if_needed(self, img, resample):
        w, h = img.size
        if min(w, h) > self.size:
            scale = self.size / min(w, h)
            w = int(scale * w)
            h = int(scale * h)
            img = img.resize((w, h), resample)
        return img


class YouTubeVISAugmentation:
    def __init__(self, size):
        self.size = size
        self.jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.15)
    
    def __call__(self, imgs, segs):
        
        # To tensor
        imgs = torch.stack([F.to_tensor(img) for img in imgs])
        segs = torch.stack([F.to_tensor(seg) for seg in segs])
        
        # Resize
        params = transforms.RandomResizedCrop.get_params(imgs, scale=(0.8, 1), ratio=(0.9, 1.1))
        imgs = F.resized_crop(imgs, *params, (self.size, self.size), interpolation=F.InterpolationMode.BILINEAR)
        segs = F.resized_crop(segs, *params, (self.size, self.size), interpolation=F.InterpolationMode.BILINEAR)
        
        # Color jitter
        imgs = self.jitter(imgs)
        
        # Grayscale
        if random.random() < 0.05:
            imgs = F.rgb_to_grayscale(imgs, num_output_channels=3)
        
        # Horizontal flip
        if random.random() < 0.5:
            imgs = F.hflip(imgs)
            segs = F.hflip(segs)
        
        return imgs, segs
