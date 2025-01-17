import os
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import json
import itertools
import random
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
        self.size = (size, size) if type(size) == int else size
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
        if self.size[0] > 0:
            self.crop = transforms.Compose([
                transforms.Resize(self.size, interpolation=interp_mode),
            ])
        else:
            self.crop = lambda x: x

        # Final transform without randomness
        self.final_im_transform = transforms.Compose([
            transforms.ToTensor(),
            # im_normalization,
        ])

        self.to_tensor = transforms.ToTensor()

    def set_frames_per_item(self, frames_per_item):
        if 'frames_per_item' in dir(self):
            print("Set dataset batch from %d to %d" % (self.frames_per_item, frames_per_item))
            if frames_per_item == self.frames_per_item:
                return
        self.frames_per_item = frames_per_item
        self.prepare_frame_list()

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
            frames = split_frames(frames, self.frames_per_item)
            self.idx_to_vid_and_chunk.extend(list(zip([vid]*len(frames), frames)))
            self.videos.append(vid)

    def get_num_frames(self, video):
        return self.num_frames_of_video[video]

    def read_fg_gt(self, name_fg):
        fg_gt = np.array(Image.open(os.path.join(self.root, self.FG_FOLDER, name_fg)).copy().convert('RGBA'))
        # fg = fg_gt[..., -2::-1] # [B, G, R, A] -> [R, G, B])
        fg = fg_gt[..., :3] # [R, G, B, A] -> [R, G, B])
        fg = Image.fromarray(fg).convert('RGB')
        gt = fg_gt[..., -1]
        gt = Image.fromarray(gt)
        return fg, gt

    def read_bg(self, name_bg):
        path_bg = os.path.join(self.root, self.BG_FOLDER, name_bg)
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
        # info['frames'] = [] # Appended with actual frames

        fgs = []
        bgs = []
        gts = []
        for name in frames:
            # img I/O
            fg, gt, bg = self.read_imgs(name)

            fg = self.crop(fg)
            bg = self.crop(bg)
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

class ValidationDataset(Dataset):
    """ Just read the imgs, can be used in any dataset """
    def __init__(self, 
        root='../dataset_mat/videomatte_motion_sd', 
        size=-1, frames_per_item=0, trimap_width=25, get_bgr=False
    ):
        super().__init__()
        self.trimap_width = trimap_width
        self.root = root
        self.size = (size, size) if type(size) == int else size
        self.set_frames_per_item(frames_per_item)
        self.resize_mode = F.InterpolationMode.BILINEAR
        self.resize = self.size[0] > 0
        self.get_bgr = get_bgr

        # if self.size[0] <= 0:
        #     self.crop = lambda x: x # placeholder
        # else:
            # interp_mode = transforms.InterpolationMode.BILINEAR
        #     self.crop = transforms.Compose([
        #         transforms.Resize(self.size, interpolation=interp_mode)
        #     ])

        # self.final_im_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     # im_normalization,
        # ])

        # self.to_tensor = transforms.ToTensor()

    def set_frames_per_item(self, frames_per_item):
        if 'frames_per_item' in dir(self):
            print("Set dataset batch from %d to %d" % (self.frames_per_item, frames_per_item))
            if frames_per_item == self.frames_per_item:
                return
        self.frames_per_item = frames_per_item
        self.prepare_frame_list()

    def prepare_frame_list(self):
        self.idx_to_vid_and_chunk = []
        self.num_frames_of_video = {}
        self.videos = sorted(os.listdir(self.root))

        for vid in self.videos:
            # vid/pha/0000.jpg -> 0000
            frames = [os.path.splitext(f)[0] for f in sorted(os.listdir(os.path.join(self.root, vid, 'pha')))]
            if len(frames) == 0:
                print(f"{vid} doesn't have frames! ({len(frames)})")
                continue
            self.num_frames_of_video[vid] = len(frames)

            frames = split_frames(frames, self.frames_per_item)
            self.idx_to_vid_and_chunk.extend(list(zip([vid]*len(frames), frames)))
            # (vid: str, frames: [...])

        self.dataset_length = len(self.idx_to_vid_and_chunk)
        print('%d videos accepted in %s.' % (len(self.videos), self.root))

    def __getitem__(self, idx):
        # video = self.videos[idx]
        video, frames = self.idx_to_vid_and_chunk[idx]
        info = {}
        info['name'] = video
        # frames = self.frames[video]

        # sample frames from a video
        # info['frames'] = [] # Appended with actual frames

        rgbs = []
        gts = []
        for name in frames:
            # img I/O
            rgb = Image.open(os.path.join(self.root, video, 'rgb', name+"_rgb.png")).copy().convert('RGB')
            gt = Image.open(os.path.join(self.root, video, 'pha', name+".png")).copy().convert('L')
            rgbs.append(F.to_tensor(rgb))
            gts.append(F.to_tensor(gt))

        if os.path.isdir(tri_dir := os.path.join(self.root, video, f'trimap_{self.trimap_width}')):    
            trimaps = []
            for name in frames:
                trimap = Image.open(os.path.join(tri_dir, name+"_trimap.png")).copy().convert('L')
                trimaps.append(F.to_tensor(trimap))
            trimaps = torch.stack(trimaps)
        else:
            trimaps = None
        
        if self.get_bgr:
            bgs = [F.to_tensor(Image.open(os.path.join(self.root, video, 'bgr', name+".png")).copy().convert('RGB')) for name in frames]
            bgs = torch.stack(bgs)
        else:
            bgs = None

        rgbs = torch.stack(rgbs)
        gts = torch.stack(gts)

        if self.resize:
            rgbs = F.resize(rgbs, self.size, interpolation=self.resize_mode)
            gts = F.resize(gts, self.size, interpolation=self.resize_mode)

            if trimaps is not None:
                trimaps = F.resize(trimaps, self.size, interpolation=self.resize_mode)
            if self.get_bgr:
                bgs = F.resize(bgs, self.size, interpolation=self.resize_mode)

        # print("vm108: ", full_img.shape, fg.shape, bg.shape, gt.shape)
        data = {
            'rgb': rgbs,
            'gt': gts,
            'trimap': get_dilated_trimaps(gts, self.trimap_width) if trimaps is None else trimaps,
            'info': info
        }
        if self.get_bgr:
            data['bg'] = bgs
            
        return data

    def __len__(self):
        return self.dataset_length
    
    def get_num_frames(self, video):
        return self.num_frames_of_video[video]

class ClipShuffleValidationDataset(ValidationDataset):
    """
    Shuffle the split clips in the video with given clip-length
    Simulate the scene change
    """
    def __init__(self, 
            root='../dataset_mat/vm108_1024',
            remap_path = '',
            size=-1, 
            frames_per_item=0, 
            trimap_width=25, 
            get_bgr=False,
        ):
        self.remap_path = remap_path
        super().__init__(root, size, frames_per_item, trimap_width, get_bgr)
        
    def set_frames_per_item(self, frames_per_item):
        if 'frames_per_item' in dir(self):
            print("Set dataset batch from %d to %d" % (self.frames_per_item, frames_per_item))
            if frames_per_item == self.frames_per_item:
                return
        self.frames_per_item = frames_per_item
        self.prepare_frame_list()

    
    def prepare_frame_list(self):
        self.idx_to_vid_and_chunk = []
        self.num_frames_of_video = {}
        self.videos = sorted(os.listdir(self.root))
        frame_corr = json.load(open(self.remap_path))
        random.seed(52728)

        for vid in self.videos:
            # vid/pha/0000.jpg -> 0000
            # frames = [os.path.splitext(f)[0] for f in sorted(os.listdir(os.path.join(self.root, vid, 'pha')))]
            frames = [os.path.splitext(f)[0] for f in frame_corr[vid]]
            if len(frames) == 0:
                print(f"{vid} doesn't have frames! ({len(frames)})")
                continue
            
            self.num_frames_of_video[vid] = len(frames)
            frames = split_frames(frames, self.frames_per_item)
            self.idx_to_vid_and_chunk.extend(list(zip([vid]*len(frames), frames)))
            # (vid: str, frames: [...])

        self.dataset_length = len(self.idx_to_vid_and_chunk)
        print('%d videos accepted in %s.' % (len(self.videos), self.root))

class RealhumanDataset(Dataset):
    def __init__(self, 
        root='../dataset_mat/real_human', 
        size=-1, frames_per_item=0
    ):
        super().__init__()
        self.root = root
        self.size = (size, size) if type(size) == int else size
        self.set_frames_per_item(frames_per_item)
        self.resize_mode = F.InterpolationMode.BILINEAR
        self.resize = self.size[0] > 0

    def set_frames_per_item(self, frames_per_item):
        self.frames_per_item = frames_per_item
        self.prepare_frame_list()

    def prepare_frame_list(self):
        self.idx_to_vid_and_chunk = []
        self.num_frames_of_video = {}
        self.videos = sorted(os.listdir(os.path.join(self.root, 'alpha')))

        for vid in self.videos:
            # alpha/vid/0000.png -> 0000.png
            frames = sorted(os.listdir(os.path.join(self.root, 'alpha', vid)))
            if len(frames) == 0:
                print(f"{vid} doesn't have frames! ({len(frames)})")
                continue
            self.num_frames_of_video[vid] = len(frames)

            frames = split_frames(frames, self.frames_per_item)
            self.idx_to_vid_and_chunk.extend(list(zip([vid]*len(frames), frames)))
            # (vid: str, frames: [...])

        self.dataset_length = len(self.idx_to_vid_and_chunk)
        print('%d videos accepted in %s.' % (len(self.videos), self.root))

    def __getitem__(self, idx):
        # video = self.videos[idx]
        video, frames = self.idx_to_vid_and_chunk[idx]
        info = {}
        info['name'] = video
        # frames = self.frames[video]

        # sample frames from a video
        # info['frames'] = [] # Appended with actual frames

        rgbs = []
        gts = []
        trimaps = []
        for name in frames:
            # img I/O
            rgb = Image.open(os.path.join(self.root, 'image', video, name)).copy().convert('RGB')
            gt = Image.open(os.path.join(self.root, 'alpha', video, name)).copy().convert('L')
            trimap = Image.open(os.path.join(self.root, 'trimap', video, name)).copy().convert('L')
            
            rgbs.append(F.to_tensor(rgb))
            gts.append(F.to_tensor(gt))
            trimaps.append(F.to_tensor(trimap))

        rgbs = torch.stack(rgbs)
        gts = torch.stack(gts)
        trimaps = torch.stack(trimaps)

        if self.resize:
            rgbs = F.resize(rgbs, self.size, interpolation=self.resize_mode)
            gts = F.resize(gts, self.size, interpolation=self.resize_mode)
            trimaps = F.resize(trimaps, self.size, interpolation=self.resize_mode)

        data = {
            'rgb': rgbs,
            'gt': gts,
            'trimap': trimaps,
            'info': info
        }

        return data

    def __len__(self):
        return self.dataset_length
    
    def get_num_frames(self, video):
        return self.num_frames_of_video[video]

class RealhumanDataset_AllFrames(RealhumanDataset):
    def __init__(self, root='../dataset_mat/real_human', size=-1, frames_per_item=0):
        super().__init__(root, size, frames_per_item)
    
    def prepare_frame_list(self):
        self.idx_to_vid_and_chunk = []
        self.num_frames_of_video = {}
        self.videos = sorted(os.listdir(os.path.join(self.root, 'alpha')))
        self.annotated_list = {}

        for vid in self.videos:
            # alpha/vid/0000.png -> 0000.png
            frames = sorted(os.listdir(os.path.join(self.root, 'image_allframe', vid)))
            self.annotated_list[vid] = sorted(os.listdir(os.path.join(self.root, 'alpha', vid)))
            if len(frames) == 0:
                print(f"{vid} doesn't have frames! ({len(frames)})")
                continue
            self.num_frames_of_video[vid] = len(frames)

            frames = split_frames(frames, self.frames_per_item)
            self.idx_to_vid_and_chunk.extend(list(zip([vid]*len(frames), frames)))
            # (vid: str, frames: [...])

        self.dataset_length = len(self.idx_to_vid_and_chunk)
        print('%d videos accepted in %s.' % (len(self.videos), self.root))

    def __getitem__(self, idx):
        # video = self.videos[idx]
        video, frames = self.idx_to_vid_and_chunk[idx]
        info = {}
        info['name'] = video
        annotated_list = self.annotated_list[video]
        # frames = self.frames[video]

        # sample frames from a video
        # info['frames'] = [] # Appended with actual frames

        rgbs = []
        gts = []
        trimaps = []
        annot_idx = []
        for i, name in enumerate(frames):
            # img I/O
            rgb = Image.open(os.path.join(self.root, 'image_allframe', video, name)).copy().convert('RGB')
            rgbs.append(F.to_tensor(rgb))
            
            if name in annotated_list:
                annot_idx.append(i)
                gt = Image.open(os.path.join(self.root, 'alpha', video, name)).copy().convert('L')
                trimap = Image.open(os.path.join(self.root, 'trimap', video, name)).copy().convert('L')
                gts.append(F.to_tensor(gt))
                trimaps.append(F.to_tensor(trimap))

        rgbs = torch.stack(rgbs)
        if len(gts) > 0:
            gts = torch.stack(gts)
            trimaps = torch.stack(trimaps)

        if self.resize:
            rgbs = F.resize(rgbs, self.size, interpolation=self.resize_mode)
            if len(gts) > 0:
                gts = F.resize(gts, self.size, interpolation=self.resize_mode)
                trimaps = F.resize(trimaps, self.size, interpolation=self.resize_mode)
        info['annotated'] = annot_idx
        data = {
            'rgb': rgbs,
            'info': info
        }
        if len(gts) > 0:
            data['gt'] = gts
            data['trimap'] = trimaps
        else:
            data['gt'] = data['trimap'] = torch.Tensor([])
            
        return data


def stretch_bg_frames(bg_frames, fg_len):
    # make bg has the same length as fg
    if fg_len > (bg_len := len(bg_frames)):
        bg_pad = list(bg_frames) # reflection padding
        bg_pad.reverse()
        bg_frames = ((bg_frames + bg_pad) * (fg_len//bg_len))
    bg_frames = bg_frames[:fg_len]
    assert fg_len == len(bg_frames), f'FG: {fg_len}, BG: {len(bg_frames)}'
    return bg_frames

def split_frames(frames, frames_per_item):
    # split frames into batches (items)
    if frames_per_item > 0:
        len_frames = len(frames)
        split_size = len_frames // frames_per_item + (len_frames % frames_per_item>0)
        frames = [arr.tolist() for arr in np.array_split(frames, split_size)]
    else:
        frames = [frames]
    return frames
