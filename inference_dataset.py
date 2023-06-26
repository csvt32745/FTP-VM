"""
Validate various model on VM108, VM240k and RealHuman Dataset
"""
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--size', help='eval video size: sd, 1024, hd, 4k', default='1024', type=str)
parser.add_argument('--batch_size', help='frames in a batch', default=8, type=int)
parser.add_argument('--n_workers', help='num workers', default=8, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--trimap_width', default=25, type=int)
parser.add_argument('--disable_video', help='Without savinig videos', action='store_true')
parser.add_argument('--downsample_ratio', default=1, type=float)
parser.add_argument('--out_root', default=".", type=str)
parser.add_argument('--dataset_root', default="../dataset_mat", type=str)
parser.add_argument('--disable_vm108', help='Without VM108', action='store_true')
parser.add_argument('--disable_realhuman', help='Without RealHuman', action='store_true')
parser.add_argument('--disable_vm240k', help='Without VM240k', action='store_true')

args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)

from torch.utils.data import DataLoader

from dataset.imagematte import *
from dataset.videomatte import *
from dataset.augmentation import *
from dataset.vm108_dataset import *

from inference_func import *
from model.which_model import get_model_by_string
from FTPVM.model import *
from FTPVM.inference_model import *
from inference_model_list import inference_model_list

# (experiment id, model_name)
# or (model_name)
model_list = [
    'FTPVM',
]

print(model_list)
model_list = [
    [i[0]]+list(inference_model_list[i[1]][1:]) if type(i) in [tuple, list] else inference_model_list[i] 
    for i in model_list
]


print(args)
assert args.size in ['sd', '1024', 'hd', '4k']

def get_size_name(size):
    if type(size) == str:
        return size
    return str(size) if type(size) == int else f'{size[1]}x{size[0]}'
    
dataset_list = []
frames_per_item = args.batch_size
trimap_width = args.trimap_width
downsample_ratio = args.downsample_ratio
out_root = args.out_root
disable_vm108 = args.disable_vm108
disable_realhuman = args.disable_realhuman
disable_vm240k = args.disable_vm240k

size = {
    'sd': [144, 256],
    '1024': [576, 1024],
    'hd': [1080, 1920],
    '4k': [2160, 3840],
}[args.size]


# =========================
# vm108 dataset
if not disable_vm108 and args.size != '4k':
    flg = True
    dataset = VM108ValidationDataset(
        root = os.path.join(args.dataset_root, 'VideoMatting108'),
        size=size, frames_per_item=frames_per_item, mode='val', trimap_width=trimap_width
    )

    # For pre-composed data
    # if args.size == 'sd':
    #     dataset = ValidationDataset(
    #         root='../dataset_mat/vm108_256',
    #         frames_per_item=frames_per_item, trimap_width=trimap_width, size=-1)
    # elif args.size == '1024':
    #     dataset = ValidationDataset(
    #         root='../dataset_mat/vm108_1024',
    #         frames_per_item=frames_per_item, trimap_width=trimap_width, size=-1)
    # elif args.size == 'hd':
    #     dataset = ValidationDataset(
    #         root='../dataset_mat/vm108_hd',
    #         frames_per_item=frames_per_item, trimap_width=trimap_width, size=-1)
    # else:
    #     flg = False

    if flg:
        dataset_name='vm108'
        root = dataset_name+f'_val_tri{trimap_width}_'+get_size_name(size)
        root = os.path.join(out_root, root)
        dataset_list.append((root, dataset_name, dataset))

# =========================
# vm240k dataset
if not disable_vm240k:
    flg = True
    dataset = ValidationDataset(
        root=os.path.join(args.dataset_root, 'videomatte_motion_4k'),
        frames_per_item=frames_per_item, trimap_width=trimap_width, size=size)
    
    # For pre-composed data
    # if args.size == '1024':
    #     size = [576, 1024] # H, W
    #     dataset = ValidationDataset(
    #         root='../dataset_mat/videomatte_motion_1024',
    #         frames_per_item=frames_per_item, trimap_width=trimap_width, size=-1)
    # elif args.size == 'hd':
    #     size = 'hd' # H, W
    #     dataset = ValidationDataset(
    #         root='../dataset_mat/videomatte_motion_hd',
    #         frames_per_item=frames_per_item, trimap_width=trimap_width, size=-1)
    # elif args.size == '4k':
    #     size = '4k' # H, W
    #     dataset = ValidationDataset(
    #         root='../dataset_mat/videomatte_motion_4k',
    #         frames_per_item=frames_per_item, trimap_width=trimap_width, size=-1)
    # else:
    #     flg = False

    if flg:
        dataset_name='vm240k'
        root = dataset_name+f'_val_tri{trimap_width}_'+get_size_name(size)
        root = os.path.join(out_root, root)
        dataset_list.append((root, dataset_name, dataset))


# =========================
# realhuman dataset
# Its trimap is fixed by dataset
if not disable_realhuman and args.size != '4k':
    for dataset_name, realhuman in [
        ('realhuman_allframe', RealhumanDataset_AllFrames),
        # ('realhuman', RealhumanDataset),
    ]:
        flg = True
        dataset = realhuman(
                root=os.path.join(args.dataset_root, 'real_human'),
                frames_per_item=frames_per_item, size=size)
        
        # For pre-composed data
        # if args.size == 'sd':
        #     size = 256
        #     dataset = realhuman(
        #         root='../dataset/real_human_256',
        #         frames_per_item=frames_per_item, size=-1)
        # elif args.size == '1024':
        #     size = [576, 1024] # H, W
        #     dataset = realhuman(
        #         root='../dataset/real_human_1024',
        #         frames_per_item=frames_per_item, size=-1)
        # elif args.size == 'hd':
        #     size = 'hd' # H, W
        #     dataset = realhuman(
        #         root='../dataset/real_human',
        #         frames_per_item=frames_per_item, size=-1)
        # else:
        #     flg = False

        if flg:
            root = dataset_name+f'_val_'+get_size_name(size)
            root = os.path.join(out_root, root)
            dataset_list.append((root, dataset_name, dataset))

# =========================

def get_dataloader(dataset):
    loader = DataLoader(dataset, batch_size=1, num_workers=args.n_workers, shuffle=False, pin_memory=True)
    return loader

gt_name = 'GT'
print([d[1] for d in dataset_list])
for root, dataset_name, dataset in dataset_list:
    loader = get_dataloader(dataset)

    for model_name, model_func, inference_core, model_path in model_list:
        if type(model_func) == str:
            model_func = get_model_by_string(model_func)
        if downsample_ratio != 1:
            print('Downsample: ', downsample_ratio)
            model_name = model_name + f'_ds_{downsample_ratio:.4f}'
        if trimap_width != 25:
            model_name = model_name + f"_width{trimap_width}"
        run_evaluation(
            root=root, 
            model_name=model_name, model_func=model_func, model_path=model_path, 
            inference_core_func=inference_core,
            dataset_name=dataset_name, dataset=dataset, dataloader=loader, gt_name=gt_name,
            downsample_ratio=downsample_ratio, save_video=not args.disable_video
            )
