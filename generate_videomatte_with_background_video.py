# Modified from https://github.com/PeterL1n/RobustVideoMatting/blob/master/evaluation/generate_videomatte_with_background_video.py
""" python generate_videomatte_with_background_video.py \
    --videomatte-dir ../dataset_sc/VideoMatte240K_JPEG_HD/test \
    --background-dir ../dataset_mat/dvm_bg_test \
    --out-dir ../dataset_mat/videomatte_motion_1024 \
    --resize 1024 576 \
    --trimap_width 25
 """
import argparse
import os
import pims
import numpy as np
import cv2
# cv2.setNumThreads(0)
import random
from functools import lru_cache
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--videomatte-dir', type=str, required=True)
parser.add_argument('--background-dir', type=str, required=True)
parser.add_argument('--num-samples', type=int, default=20)
parser.add_argument('--num-frames', type=int, default=100)
parser.add_argument('--resize', type=int, default=None, nargs=2)
parser.add_argument('--out-dir', type=str, required=True)
parser.add_argument('--trimap_width', type=int, default=25)
args = parser.parse_args()

# Hand selected a list of videos
background_filenames = [
    "0000",
    "0007",
    "0008",
    "0010",
    "0013",
    "0015",
    "0016",
    "0018",
    "0021",
    "0029",
    "0033",
    "0035",
    "0039",
    "0050",
    "0052",
    "0055",
    "0060",
    "0063",
    "0087",
    "0086",
    "0090",
    "0101",
    "0110",
    "0117",
    "0120",
    "0122",
    "0123",
    "0125",
    "0128",
    "0131",
    "0172",
    "0176",
    "0181",
    "0187",
    "0193",
    "0198",
    "0220",
    "0221",
    "0224",
    "0229",
    "0233",
    "0238",
    "0241",
    "0245",
    "0246"
]

random.seed(10)
    
videomatte_filenames = [(clipname, sorted(os.listdir(os.path.join(args.videomatte_dir, 'fgr', clipname)))) 
                        for clipname in sorted(os.listdir(os.path.join(args.videomatte_dir, 'fgr')))]

random.shuffle(background_filenames)

@lru_cache(32)
def _get_kernel(kernel_size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

def get_dilated_trimaps_np_uint8(pha, kernel_size):
    # H, W
    kernel = _get_kernel(kernel_size)
    fg_and_unknown = (pha > 1).astype(np.uint8)
    fg = (pha > 254).astype(np.uint8)
    dilate = cv2.dilate(fg_and_unknown, kernel).astype(np.float32)
    erode = cv2.erode(fg, kernel).astype(np.float32)
    trimap = erode * 1. + (dilate-erode) * 0.5
    trimap = np.clip(trimap*255, 0, 255).astype(np.uint8)
    return trimap

p = os.path.join(args.background_dir, background_filenames[0])
if os.path.isdir(p):
    bg_img_format = True
elif os.path.exists(p+".mp4"):
    bg_img_format = False
else:
    raise RuntimeError(f'{p} not found!')

for i in range(args.num_samples):
    if bg_img_format:
        bgrs = os.path.join(args.background_dir, background_filenames[i % len(background_filenames)])
    else:
        bgrs = pims.PyAVVideoReader(os.path.join(args.background_dir, background_filenames[i % len(background_filenames)]))

    clipname, framenames = videomatte_filenames[i % len(videomatte_filenames)]
    
    out_path = os.path.join(args.out_dir, str(i).zfill(4))
    os.makedirs(os.path.join(out_path, 'fgr'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'pha'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'rgb'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'bgr'), exist_ok=True)
    os.makedirs(os.path.join(out_path, f'trimap_{args.trimap_width}'), exist_ok=True)
    
    base_t = random.choice(range(len(framenames) - args.num_frames))
    
    for t in tqdm(range(args.num_frames), desc=str(i).zfill(4)):
        with Image.open(os.path.join(args.videomatte_dir, 'fgr', clipname, framenames[base_t + t])) as fgr, \
             Image.open(os.path.join(args.videomatte_dir, 'pha', clipname, framenames[base_t + t])) as pha:
            fgr = fgr.convert('RGB')
            pha = pha.convert('L')
            
            if args.resize is not None:
                fgr = fgr.resize(args.resize, Image.BILINEAR)
                pha = pha.resize(args.resize, Image.BILINEAR)
                
            
            if i // len(videomatte_filenames) % 2 == 1:
                fgr = fgr.transpose(Image.FLIP_LEFT_RIGHT)
                pha = pha.transpose(Image.FLIP_LEFT_RIGHT)
            
            trimap = get_dilated_trimaps_np_uint8(np.asarray(pha), kernel_size=args.trimap_width)
            Image.fromarray(trimap).save(os.path.join(out_path, f'trimap_{args.trimap_width}', str(t).zfill(4) + '_trimap.png'))
            fgr.save(os.path.join(out_path, 'fgr', str(t).zfill(4) + '.png'))
            pha.save(os.path.join(out_path, 'pha', str(t).zfill(4) + '.png'))
        
        if bg_img_format:
            bgr = Image.open(os.path.join(bgrs, f'{t:04d}.jpg'))
        else:
            bgr = Image.fromarray(bgrs[t])
        bgr = bgr.resize(fgr.size, Image.BILINEAR)
        bgr.save(os.path.join(out_path, 'bgr', str(t).zfill(4) + '.png'))
        
        pha = np.asarray(pha).astype(float)[:, :, None] / 255
        com = Image.fromarray(np.uint8(np.asarray(fgr) * pha + np.asarray(bgr) * (1 - pha)))
        com.save(os.path.join(out_path, 'rgb', str(t).zfill(4) + '_rgb.png'))
