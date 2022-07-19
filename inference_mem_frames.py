from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--size', help='eval video size: sd, 1024, hd, 4k', default='sd', type=str)
parser.add_argument('--frames_per_item', help='frames in a batch', default=10, type=int)
parser.add_argument('--n_workers', help='num workers', default=8, type=int)
parser.add_argument('--gpu', default=0, type=int)
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
    'GFM_GatedFuseVM_4xfoucs_randmemtrimap',
    'GFM_GatedFuseVM_4xfoucs',
]
model_list = [inference_model_list[i] for i in model_list]


print(args)
assert args.size in ['sd', '1024']#, 'hd', '4k']

# memory_freqs = [120, 240, 480]
memory_freqs = [1]
# memory_freqs = [30, 60, 120, 240, 480]
frames_per_item = args.frames_per_item
trimap_width = 25

# move 0 to the final element
for i, freq in enumerate(memory_freqs[:-1]):
    if freq in [0, 1]:
        memory_freqs[-1] = freq
        break

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
# vm240k dataset
# size = [288, 512] # H, W
if args.size == '1024':
    size = [576, 1024] # H, W
    dataset = VM240KValidationDataset(
        root='../dataset_mat/videomatte_motion_1024',
        # root='../dataset_mat/videomatte_motion_sd',
        frames_per_item=frames_per_item, trimap_width=trimap_width, size=-1)
    dataset_name='vm240k'
    root = dataset_name+'_val_midtri_'+get_size_name(size)
    dataset_list.append((root, dataset_name, dataset))

# =========================


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

for root, dataset_name, dataset in dataset_list:
    loader = get_dataloader(dataset)
    for model_name, model_func, inference_core, model_path in model_list:
        if type(model_func) == str:
            model_func = get_model_by_string(model_func)
        flg = False
        for mem_freq in memory_freqs:
            if mem_freq <= 1:
                print("memory update freq <= 1, set dataset frames per item = 1")
                dataset.set_frames_per_item(1)
                flg = True
            else:
                assert flg == False, 'memory update freq <= 1 should at the last element'
                
            model_name_freq = f"{model_name}_mem{mem_freq}f"
            run_evaluation(
                root=root, 
                model_name=model_name_freq, model_func=model_func, model_path=model_path,
                inference_core_func=inference_core,
                dataset_name=dataset_name, dataset=dataset, dataloader=loader, 
                memory_freq=mem_freq, memory_gt=True, gt_name=gt_name)