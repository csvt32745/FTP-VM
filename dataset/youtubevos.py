import torch
import os
import json
import numpy as np
import cv2
cv2.setNumThreads(0)
import random
from torch.utils.data import Dataset
from PIL import Image
import random
from .util import get_dilated_trimaps

class YouTubeVOSDataset(Dataset):
    def __init__(self, root, size, seq_length, seq_sampler, transform=None, debug_data=None, random_memtrimap=False):
        self.root = root
        self.random_memtrimap = random_memtrimap
        self.size = size
        self.seq_length = seq_length
        self.seq_sampler = seq_sampler
        self.transform = transform

        print("Loading Youtube VOS ...")
        if debug_data is None:
            with open(os.path.join(root, 'meta.json')) as f:
                data = json.load(f)
        else:
            data = debug_data
        data = data['videos']
        self.videos = list(data.keys())
        self.objects = [data[vid]['objects'] for vid in self.videos]
        self.frames = [
            [f[:-4] for f in sorted(os.listdir(os.path.join(root, 'JPEGImages', vid))) if f[-3:] in ['jpg', 'jpeg']]
            for vid in self.videos
        ]
        # video name, 
        self.index = [
            (i, vid, f)
            for i, vid in enumerate(self.videos)
            for f in range(len(self.frames[i]))
        ]
        print("Total frames: ", len(self.index))
                
    def __len__(self):
        return len(self.index)
    
    def read_images(self, vid, frame):
        with Image.open(os.path.join(self.root, 'JPEGImages', vid, frame+".jpg")) as f:
            rgb = f.convert('RGB')
        with Image.open(os.path.join(self.root, 'Annotations', vid, frame+".png")) as f:
            gt = f.convert('P')
        return rgb, gt

    def __getitem__(self, idx):
        vid_id, video, frame_id = self.index[idx]
        # video = self.videos[idx]
        objects = self.objects[vid_id]
        total_frames = self.frames[vid_id]

        target_mask = int(random.sample(objects.keys(), 1)[0])
        frame_count = len(total_frames)
        # frame_id = int(random.choice(objects[target_mask]['frames'])[:-4])
        # frame_id = np.searchsorted(total_frames, frame_id)

        imgs, gts = [], []
        for t in self.seq_sampler(self.seq_length):
            frame = frame_id + t # reflection pad
            i = 0
            while i<5: # 
                i+=1
                if frame < 0:
                    frame *= -1
                elif frame >= frame_count:
                    frame = 2*frame_count - frame - 1 # reflection pad
                else:
                    break
            frame = frame % frame_count
            
            rgb, gt = self.read_images(video, total_frames[frame])
            imgs.append(self._downsample_if_needed(rgb, Image.BILINEAR))
            gt = self._downsample_if_needed(gt, Image.NEAREST)
            gt = Image.fromarray((np.array(gt) == target_mask).astype(np.uint8)*255)
            gts.append(gt)
            
        if self.transform is not None:
            imgs, gts = self.transform(imgs, gts)
        
        data = {
            'rgb': imgs,
            'gt': gts,
        }
        if self.random_memtrimap:
            data['trimap'] = get_dilated_trimaps(gts, 17, random_kernel=False)
            data['mem_trimap'] = get_dilated_trimaps(gts[[0]], np.random.randint(3, 16)*2+1, random_kernel=True)
        else:
            data['trimap'] = get_dilated_trimaps(gts, np.random.randint(3, 16)*2+1, random_kernel=True)
        return data

    
    def _downsample_if_needed(self, img, resample):
        w, h = img.size
        if min(w, h) > self.size:
            scale = self.size / min(w, h)
            w = int(scale * w)
            h = int(scale * h)
            img = img.resize((w, h), resample)
        return img