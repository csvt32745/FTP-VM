"""
Validate various model on VM108 Dataset
with various memory update period
"""
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--size', help='eval video size: sd, 1024', default='1024', type=str)
parser.add_argument('--frames_per_item', help='frames in a batch', default=10, type=int)
parser.add_argument('--n_workers', help='num workers', default=8, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--out_root', default=".", type=str)
parser.add_argument('--dataset_root', default="../dataset_mat", type=str)
parser.add_argument('--disable_video', help='Without savinig videos', action='store_true')
parser.add_argument('--replace_tri', help='Replace the output seg by mem trimap', action='store_true')
parser.add_argument('--trimap_width', default=25, type=int)
parser.add_argument('--memory_freq', help='update memory in n frames, 1 for every frames', nargs='+', type=int,
    default=[30, 60, 120, 240, 480, 1])
args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)

from torch.utils.data import DataLoader
from fastai.data.load import DataLoader as FAIDataLoader

from dataset.vm108_dataset import *

from inference_func import *
from model.model import get_model_by_string
from FTPVM.model import *
from FTPVM.module import *
from FTPVM.inference_model import *
from inference_model_list import inference_model_list

# (experiment id, model id)
# or (model id)
model_list = [
    "FTPVM"
]
print(model_list)
model_list = [inference_model_list[i] for i in model_list]


print(args)

memory_freqs = args.memory_freq#[30, 60, 120, 240, 480, 1]
frames_per_item = args.frames_per_item
trimap_width = args.trimap_width

# memory would be updated (or not) after finishing each batch (item)
print("Memory updated frequencies:",  memory_freqs)
for freq in memory_freqs:
    assert freq <= 1 or freq % frames_per_item == 0


def get_size_name(size):
    return str(size) if type(size) == int else f'{size[1]}x{size[0]}'
dataset_list = []

size = {
    'sd': [144, 256],
    '1024': [576, 1024],
    'hd': [1080, 1920]
}[args.size]

# =========================
# vm108 dataset

dataset = VM108ValidationDataset(
        root=os.path.join(args.dataset_root, 'VideoMatting108'),
        size=size, frames_per_item=frames_per_item, mode='val', trimap_width=trimap_width
    )

dataset_name='vm108'
root = dataset_name+f'_val_tri{trimap_width}_'+get_size_name(size)
root = os.path.join(args.out_root, root)
dataset_list.append((root, dataset_name, dataset))

print([d[1] for d in dataset_list])

def get_dataloader(dataset):
    loader = DataLoader(dataset, batch_size=1, num_workers=8, shuffle=False, pin_memory=True)
    return loader

gt_name = 'GT'

for mem_freq in memory_freqs:
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

            if args.replace_tri:
                model_name = model_name+'_replace-tri'
            model_name_freq = f"{model_name}_mem{mem_freq}f"
            
            run_evaluation(
                root=root, 
                model_name=model_name_freq, model_func=model_func, model_path=model_path,
                inference_core_func=inference_core,
                dataset_name=dataset_name, dataset=dataset, dataloader=loader, 
                memory_freq=mem_freq, memory_gt=True, gt_name=gt_name, save_video=not args.disable_video, replace_by_given_tri=args.replace_tri)