from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--root', help='input video root', required=True, type=str)
parser.add_argument('--out_root', help='output video root', required=True, type=str)
parser.add_argument('--gpu', help='gpu id', default=0, type=int)
parser.add_argument('--target_size', help='downsample the video by ratio of the larger width to target_size, and upsampled back by FGF', default=1024, type=int)
parser.add_argument('--seq_chunk', help='the frames to process in a batch', default=4, type=int)

args = parser.parse_args()

from inference_model_list import inference_model_list
from model.which_model import get_model_by_string
import torch
import os
from inference_test import convert_video
os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)

root = args.root
outroot = args.out_root
model_name = 'FTPVM'
os.makedirs(outroot, exist_ok=True)
model_attr = inference_model_list[model_name]
model = get_model_by_string(model_attr[1])().to(device='cuda')
model.load_state_dict(torch.load(model_attr[3]))
# check_and_load_model_dict(model, torch.load(model_attr[3]))


files = os.listdir(root)
for vid in files:
    name, ext = os.path.splitext(vid)
    if '.mp4' != ext:
        continue
    for tri in [i for i in files if i[:len(name)+1]==(name+"_") and 'trimap' in i]:
        # print(name, tri)
        suffix = tri.split('trimap')[-1].split('.')[0]
        output_name = name+"_"+suffix
        # if os.path.isfile(os.path.join(outroot, output_name+"_com.mp4")):
        #     print('Already finished, skip: ', name)
        #     continue
        
        mem_img = os.path.join(root, name+"_thumbnail.png")
        mem_mask = os.path.join(root, tri)
        # if not os.path.isfile(mem_img):
        #     print('Memory image not found, skip: ', mem_img)
        # if not os.path.isfile(mem_mask):
        #     print('Memory mask not found, skip: ', mem_mask)


        convert_video(
            model,
            input_source=os.path.join(root, vid),
            memory_img=mem_img,
            memory_mask=mem_mask,
            output_type='video',
            output_composition = os.path.join(outroot, output_name+"_com.mp4"),
            output_alpha = os.path.join(outroot, output_name+"_pha.mp4"),
            output_foreground = os.path.join(outroot, output_name+"_fgr.mp4"),
            output_video_mbps=8,
            seq_chunk=args.seq_chunk,
            num_workers=1,
            target_size=args.target_size,
        )