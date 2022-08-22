from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--size', help='eval video size: sd, 1024', default='sd', type=str)
parser.add_argument('--frames_per_item', help='frames in a batch', default=10, type=int)
parser.add_argument('--n_workers', help='num workers', default=8, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--disable_video', help='Without savinig videos', action='store_true')
# parser.add_argument('--memory_freq', help='update memory in n frames, 0 for every frames', default=-1, type=int)
args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)

from torch.utils.data import DataLoader
from fastai.data.load import DataLoader as FAIDataLoader

from dataset.imagematte import *
from dataset.videomatte import *
from dataset.augmentation import *
from dataset.vm108_dataset import *

from inference_func import *
from model.model import get_model_by_string
from STCNVM.model import *
from STCNVM.module import *
from STCNVM.inference_model import *
from inference_model_list import inference_model_list

model_list = [
    # 'STCNFuseMatting_fullres_480_temp_seg_allclass_weight_x1',
	# 'STCNFuseMatting_fullres_matnaive',
    # 'STCNFuseMatting_fullres_matnaive_480_temp_seg',
    # 'STCNFuseMatting_fullres_matnaive_seg2_480_temp_seg_allclass',
    # 'STCNFuseMatting_fuse=naive_480',
    # 'STCNFuseMatting_fullres_480_none_temp_seg'
	'STCNFuseMatting_fullres_matnaive_backbonefuse',
    'STCNFuseMatting_fullres_matnaive_naivefuse',
	'STCNFuseMatting_fullres_matnaive_none_temp_seg',
    'STCNFuseMatting_SingleDec',
    'STCNFuseMatting_SameDec_480'
]
print(model_list)
model_list = [inference_model_list[i] for i in model_list]


print(args)
assert args.size in ['sd', '1024']

# memory_freqs = [120, 240, 480]
memory_freqs = [1]
# memory_freqs = [30, 60, 120, 240, 480]
frames_per_item = args.frames_per_item
trimap_width = 25

# memory would be updated (or not) after finishing each batch (item)
print("Memory updated frequencies:",  memory_freqs)
for freq in memory_freqs:
    assert freq <= 1 or freq % frames_per_item == 0


def get_size_name(size):
    return str(size) if type(size) == int else f'{size[1]}x{size[0]}'
dataset_list = []


# =========================
# vm108 dataset

# dataset = VM108ValidationDataset(
#     frames_per_item=frames_per_item, mode='val', is_subset=False, 
#     trimap_width=25, size=size)

if args.size == 'sd':
    size = 256
    dataset = VM240KValidationDataset(
        root='../dataset_mat/vm108_256',
        frames_per_item=frames_per_item, trimap_width=trimap_width, size=-1)
elif args.size == '1024':
    size = [576, 1024] # H, W
    dataset = VM240KValidationDataset(
        root='../dataset_mat/vm108_1024',
        frames_per_item=frames_per_item, trimap_width=trimap_width, size=-1)

dataset_name='vm108'
root = dataset_name+'_val_midtri_'+get_size_name(size)
dataset_list.append((root, dataset_name, dataset))

# =========================
# realhuman dataset

for dataset_name, realhuman in [
    ('realhuman_allframe', RealhumanDataset_AllFrames),
    # ('realhuman', RealhumanDataset),
]:
    # flg = True
    # if args.size == 'sd':
    #     size = 256
    #     dataset = realhuman(
    #         root='../dataset_mat/real_human_256',
    #         frames_per_item=frames_per_item, size=-1)
    if args.size == '1024':
        size = [576, 1024] # H, W
        dataset = realhuman(
            root='../dataset_mat/real_human_1024',
            frames_per_item=frames_per_item, size=-1)
        
        root = dataset_name+"_"+get_size_name(size)
        dataset_list.append((root, dataset_name, dataset))

# =========================

print([d[1] for d in dataset_list])

def get_dataloader(dataset):
    loader = DataLoader(dataset, batch_size=1, num_workers=8, shuffle=False, pin_memory=True)
    # loader = FAIDataLoader(
    #         dataset = vm108,
    #         batch_size=1,
    #         num_workers=8,
    #         shuffle=False, 
    #         drop_last=False,
    #         timeout=60,
    #     )
    return loader

gt_name = 'GT'

# flg = False
for mem_freq in memory_freqs:
        # assert flg == False, 'memory update freq <= 1 should at the last element'
    for root, dataset_name, dataset in dataset_list:
        if mem_freq <= 1:
            print("memory update freq <= 1, set dataset frames per item = 1")
            dataset.set_frames_per_item(1)
            # flg = True
        else:
            dataset.set_frames_per_item(frames_per_item)
        loader = get_dataloader(dataset)
        for model_name, model_func, inference_core, model_path in model_list:
            if type(model_func) == str:
                model_func = get_model_by_string(model_func)
                
            model_name_freq = f"{model_name}_mem{mem_freq}f"
            run_evaluation(
                root=root, 
                model_name=model_name_freq, model_func=model_func, model_path=model_path,
                inference_core_func=inference_core,
                dataset_name=dataset_name, dataset=dataset, dataloader=loader, 
                memory_freq=mem_freq, memory_gt=True, gt_name=gt_name, save_video=not args.disable_video)