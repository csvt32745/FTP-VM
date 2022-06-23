import os
from os import path

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import json
import random
import itertools

from dataset.range_transform import im_normalization, im_mean
# from dataset.mask_perturb import perturb_mask
# from dataset.gen_scribble import get_scribble
from dataset.reseed import reseed
from dataset.util import get_dilated_trimaps

class VM108ValidationDataset(Dataset):
    FG_FOLDER = 'FG_done'
    BG_FOLDER = 'BG_done'
    def __init__(self,
        root='../dataset_mat/VideoMatting108', 
        size=512, frames_per_item=0, 
        mode='train', is_subset=False, video_list_path=None, video_list=None, trimap_width=25
    ):
        assert mode in ['train', 'val']
        self.trimap_width = trimap_width
        self.root = root
        self.mode = mode
        self.size = size
        self.frames_per_item = frames_per_item
        self.is_subset = is_subset
        assert video_list is None or video_list_path is None, 'only one of them should be given'
        self.video_list_path = video_list_path
        self.video_list = video_list
        self.prepare_frame_list()

        #self.samples = self.samples[:240]
        self.dataset_length = len(self.idx_to_vid_and_chunk)
        print('%d videos accepted in %s.' % (len(self.videos), self.root))

        # These set of transform is the same for im/gt pairs, but different among the 3 sampled frames
        interp_mode = transforms.InterpolationMode.BILINEAR

        self.crop = transforms.Compose([
            transforms.Resize((size, size), interpolation=interp_mode),
        ])

        # Final transform without randomness
        self.final_im_transform = transforms.Compose([
            transforms.ToTensor(),
            # im_normalization,
        ])

        self.to_tensor = transforms.ToTensor()

    def prepare_frame_list(self):
        with open(os.path.join(self.root, 'frame_corr.json'), 'r') as f:
            self.frame_corr = json.load(f)

        self.videos = []
        self.idx_to_vid_and_chunk = []
        self.num_frames_of_video = {}
        if self.video_list is not None:
            video_list = self.video_list
        else:
            path = self.video_list_path if self.video_list_path is not None \
                else os.path.join(self.root, f'{self.mode}_videos' + ('_subset' if self.is_subset else '') + '.txt')
            with open(path, 'r') as f:
                video_list = f.readlines()

        total_frame_list = sorted(self.frame_corr.keys())
        for vid in video_list:
            vid = vid.strip()
            frames = [k for k in total_frame_list if os.path.dirname(k) == vid]
            # if len(frames) < 3:
            #     continue
            self.num_frames_of_video[vid] = len(frames)
            if self.frames_per_item > 0:
                split_size = len(frames)//self.frames_per_item + (len(frames) % self.frames_per_item>0)
                frames = [arr.tolist() for arr in np.array_split(frames, split_size)]
            else:
                frames = [frames]
            # self.videos_num_chunks.append(len(frames))
            self.idx_to_vid_and_chunk.extend(list(zip([vid]*len(frames), frames)))
            self.videos.append(vid)

    def get_num_frames(self, video):
        return self.num_frames_of_video[video]

    def read_fg_gt(self, name_fg):
        fg_gt = np.array(Image.open(path.join(self.root, self.FG_FOLDER, name_fg)).copy().convert('RGBA'))
        # fg = fg_gt[..., -2::-1] # [B, G, R, A] -> [R, G, B])
        fg = fg_gt[..., :3] # [R, G, B, A] -> [R, G, B])
        fg = Image.fromarray(fg).convert('RGB')
        gt = fg_gt[..., -1]
        gt = Image.fromarray(gt)
        return fg, gt

    def read_bg(self, name_bg):
        path_bg = path.join(self.root, self.BG_FOLDER, name_bg)
        if not os.path.exists(path_bg):
            path_bg = os.path.splitext(path_bg)[0]+'.png'
        bg = Image.open(path_bg).convert('RGB')
        # bg = np.float32(cv2.imread(bgp, cv2.IMREAD_COLOR))
        return bg

    def read_imgs(self, name):
        fg, gt = self.read_fg_gt(name)
        bg = self.read_bg(self.frame_corr[name])
        return fg, gt, bg

    def __getitem__(self, idx):
        # video = self.videos[idx]
        video, frames = self.idx_to_vid_and_chunk[idx]
        info = {}
        info['name'] = video
        # frames = self.frames[video]

        # sample frames from a video
        info['frames'] = [] # Appended with actual frames

        sequence_seed = np.random.randint(2147483647)
        fgs = []
        bgs = []
        gts = []
        for name in frames:
            # img I/O
            fg, gt, bg = self.read_imgs(name)

            reseed(sequence_seed)                
            fg = self.crop(fg)
            reseed(sequence_seed)                
            bg = self.crop(bg)
            reseed(sequence_seed)
            gt = self.crop(gt)

            fg = self.final_im_transform(fg)
            bg = self.final_im_transform(bg)
            gt = self.to_tensor(gt)
            fgs.append(fg)
            bgs.append(bg)
            gts.append(gt)

        fgs = torch.stack(fgs, 0)
        bgs = torch.stack(bgs, 0)
        gts = torch.stack(gts, 0)
        imgs = fgs*gts + bgs*(1-gts)

        # print("vm108: ", full_img.shape, fg.shape, bg.shape, gt.shape)
        data = {
            'rgb': imgs,
            'fg': fgs,
            'bg': bgs,
            'gt': gts,
            'trimap': get_dilated_trimaps(gts, self.trimap_width),
            'info': info
        }
        return data


    def __len__(self):
        return self.dataset_length

class VM108ValidationDatasetFixFG(VM108ValidationDataset):
    def __init__(self, 
        root='../dataset_mat/VideoMatting108', 
        fg_list_path='', bg_list_path='',
        size=512, frames_per_item=0,
    ):
        self.fg_list_path = fg_list_path
        self.bg_list_path = bg_list_path
        
        super().__init__(root, size, frames_per_item)

    def read_imgs(self, names):
        # names == [fg_name, bg_name]
        fg, gt = self.read_fg_gt(names[0])
        bg = self.read_bg(names[1])
        return fg, gt, bg

    def prepare_frame_list(self):
        self.videos = []
        self.idx_to_vid_and_chunk = []
        self.num_frames_of_video = {}
        
        with open(self.fg_list_path, 'r') as f:
            self.fg_clips = [l.strip() for l in f.readlines()]
        with open(self.bg_list_path, 'r') as f:
            self.bg_clips = [l.strip() for l in f.readlines()]

        for fg_path, bg_path in itertools.product(self.fg_clips, self.bg_clips):
            
            fg_frames = [os.path.join(fg_path, p) for p in sorted(os.listdir(os.path.join(self.root, self.FG_FOLDER, fg_path)))]
            bg_frames = [os.path.join(bg_path, p) for p in sorted(os.listdir(os.path.join(self.root, self.BG_FOLDER, bg_path)))]
            vid_name = '%s-%s' % (fg_path.replace('-', '_'), bg_path.replace('-', '_'))
            vid_name = vid_name.replace('/', '_')
            
            self.num_frames_of_video[vid_name] = len(fg_frames)
            # make bg has the same length as fg
            if (fg_len := len(fg_frames)) > (bg_len := len(bg_frames)):
                bg_pad = list(bg_frames) # reflection padding
                bg_pad.reverse()
                bg_frames = ((bg_frames + bg_pad) * (fg_len//bg_len))
            bg_frames = bg_frames[:fg_len]
            assert (len_frames := len(fg_frames)) == len(bg_frames), f'FG: {len(fg_frames)}, BG: {len(bg_frames)}'
            frames = list(zip(fg_frames, bg_frames))
            if self.frames_per_item > 0:
                split_size = len_frames // self.frames_per_item + (len_frames % self.frames_per_item>0)
                frames = [arr.tolist() for arr in np.array_split(frames, split_size)]
            else:
                frames = [frames]
            self.idx_to_vid_and_chunk.extend(list(zip([vid_name]*len_frames, frames)))
            self.videos.append(vid_name)