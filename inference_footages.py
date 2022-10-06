from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--root', help='input video root', required=True, type=str)
parser.add_argument('--out_root', help='output video root', default='sd', type=str)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--target_size', default=1024, type=int)
# parser.add_argument('--downsample_ratio', default=1, type=float)

args = parser.parse_args()

from inference_model_list import inference_model_list
from model.which_model import get_model_by_string
import torch
import os
from inference_test import convert_video
os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)

# root = '/home/csvt32745/matte/dataset_mat/footage'
# input_resize=(1920, 1080)
root = args.root
# root = '/home/csvt32745/matte/test_vid'
input_resize=(1920, 1080)
# input_resize=(1280, 720)

outroot = args.out_root
# outroot = '/home/csvt32745/matte/footage_out'
# model_name = 'STCNFuseMatting_fullres_matnaive_none_temp_seg'
# model_name = 'STCNFuseMatting_fullres_matnaive_wodata_seg_d646'
model_name = 'STCNFuseMatting_fullres_matnaive'
# model_name = 'STCNFuseMatting_fullres_matnaive_same_memque0.5'
# downsample_ratio=0.5
os.makedirs(outroot, exist_ok=True)
def check_and_load_model_dict(model, state_dict: dict):
    for k in set(state_dict.keys()) - set(model.state_dict().keys()):
        if 'refiner' in k:
            print('remove refiner', k)
            state_dict.pop(k)
    model.load_state_dict(state_dict)

model_attr = inference_model_list[model_name]
model = get_model_by_string(model_attr[1])()#.to(device=device)
check_and_load_model_dict(model, torch.load(model_attr[3]))


files = os.listdir(root)
for vid in files:
    name, ext = os.path.splitext(vid)
    if '.mp4' != ext:
        continue
    for tri in [i for i in files if i[:len(name)+1]==(name+"_") and 'trimap' in i]:
        # print(name, tri)
        suffix = tri.split('trimap')[-1].split('.')[0]
        output_name = name+"_"+suffix
        if os.path.isfile(os.path.join(outroot, output_name+"_com.mp4")):
            print('Already finished, skip: ', name)
            continue
        
        mem_img = os.path.join(root, name+"_thumbnail.png")
        mem_mask = os.path.join(root, tri)
        if not os.path.isfile(mem_img):
            print('Memory image not found, skip: ', mem_img)
        if not os.path.isfile(mem_mask):
            print('Memory mask not found, skip: ', mem_mask)


        convert_video(
            model,
            input_source = os.path.join(root, vid),
            # input_source = '/home/csvt32745/matte/OTVM/tmp_demo/holosum/frames',
            # input_resize = input_resize,
            memory_img = mem_img,
            memory_mask = mem_mask,
            # downsample_ratio=downsample_ratio,
            output_type='video',
            output_composition = os.path.join(outroot, output_name+"_com.mp4"),
            output_alpha = os.path.join(outroot, output_name+"_pha.mp4"),
            output_foreground = os.path.join(outroot, output_name+"_fgr.mp4"),

            # output_type='png_sequence',
            # output_composition = os.path.join(outroot, 'test'),

            output_video_mbps=8,
            seq_chunk=1,
            num_workers=1,
            # seq_chunk=4,
            # num_workers=8,
            target_size=args.target_size,
        )



#     # print(model)
#     # converter = Converter(args.variant, args.checkpoint, args.device)