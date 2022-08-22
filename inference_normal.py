from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--size', help='eval video size: sd, 1024, hd, 4k', default='sd', type=str)
parser.add_argument('--batch_size', help='frames in a batch', default=8, type=int)
parser.add_argument('--n_workers', help='num workers', default=8, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--memory_bg', help='Given bg & zero-trimap as memory input', action='store_true')
parser.add_argument('--disable_video', help='Without savinig videos', action='store_true')
parser.add_argument('--downsample_ratio', default=1, type=float)
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
   
    # 'STCNFuseMatting_fuse=naive',
	# 'STCNFuseMatting_480',
	# 'STCNFuseMatting',
	# 'STCNFuseMatting_gru_before_fuse',
	# 'STCNFuseMatting_fuse=fullres',
	# 'STCNFuseMatting_randtri',

	# 'STCNFuseMatting_fuse=fullgate',
    # 'STCNFuseMatting_fuse=naive_480',
	# 'STCNFuseMatting_fuse=intrimap_only',
    # 'STCNFuseMatting_480_normalce',
	# 'STCNFuseMatting_fuse=small',
	# 'STCNFuseMatting_big',
	# 'STCNFuseMatting_ytvos',
	# 'STCNFuseMatting_fullres_mat',
    # 'STCNFuseMatting_fuse=bn',

    # 'STCNFuseMatting_SameDec_480',
	# 'STCNFuseMatting_fuse=intrimap_only_fullres',
	# 'STCNFuseMatting_fuse=bn_480',
	# 'STCNFuseMatting_fuse=fullres',

	# 'STCNFuseMatting_fuse=naive_480',
	# 'STCNFuseMatting_fuse=bn_wo_consis_480',
    # 'STCNFuseMatting_1xseg_4x2mat',
	# 'STCNFuseMatting_fuse=bn_seg_consis_correctonly_480',
	# 'STCNFuseMatting_fullresseg',
	# 'STCNFuseMatting_fuse=gn_480',
	# 'STCNFuseMatting_fuse=bn2_480',
	# 'STCNFuseMatting_fullres_mat3_480',
	# 'STCNFuseMatting_fullres_480_none_temp_seg',
	# 'STCNFuseMatting_fullres_480_temp_seg_allclass_weight_x1',
	# 'STCNFuseMatting_fullres_matnaive',
    # 'STCNFuseMatting_fullres_matnaive_80k',
    # 'STCNFuseMatting_fullres_matnaive_480_temp_seg',
    # 'STCNFuseMatting_fullres_matnaive_seg2_480_temp_seg_allclass',
	# 'STCNFuseMatting_fullres_gn_3chmask',
	# 'STCNFuseMatting_SingleDec',
	# 'STCNFuseMatting_SingleDec_big',
	# 'STCNFuseMatting_fullres_matnaive_seg2_480_temp_seg_allclass_weight',
	# 'STCNFuseMatting_fullres_matnaive_temp_seg_allclass_weight',
    # 'RVM',
	# 'STCNFuseMatting_fullres_matnaive_l2attn',
	# 'STCNFuseMatting_fullres_matnaive2_seg3',
    # 'STCNFuseMatting_fullres_matnaive_naivefuse',
	# 'STCNFuseMatting_fullres_matnaive_l2gate',
	# 'STCNFuseMatting_fullres_matnaive_backbonefuse',
	# 'STCNFuseMatting_fullres_matnaive_multiobj',
	# 'STCNFuseMatting_fullres_matnaive_none_temp_seg',
    # 'STCNFuseMatting_fullres_matnaive_wodata_seg_d646',
	# 'STCNFuseMatting_fullres_matnaive_memalpha',
	# 'STCNFuseMatting_fullres_matnaive_ytvos',
	# 'STCNFuseMatting_fullres_matnaive_ppm1236',
	# '2stage',
    'STCNFuseMatting_fullres_matnaive_woPPM',
	'STCNFuseMatting_fullres_matnaive_woCBAM',

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
trimap_width = 25
downsample_ratio = args.downsample_ratio

# =========================
# vm108 dataset

flg = True
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
elif args.size == 'hd':
    size = 'hd'
    dataset = VM240KValidationDataset(
        root='../dataset_mat/vm108_hd',
        frames_per_item=frames_per_item, trimap_width=trimap_width, size=-1)
else:
    flg = False

if flg:
    dataset_name='vm108'
    root = dataset_name+'_val_midtri_'+get_size_name(size)
    dataset_list.append((root, dataset_name, dataset))

# =========================
# vm240k dataset

flg = True
if args.size == '1024':
    size = [576, 1024] # H, W
    dataset = VM240KValidationDataset(
        root='../dataset_mat/videomatte_motion_1024',
        frames_per_item=frames_per_item, trimap_width=trimap_width, size=-1)
elif args.size == 'hd':
    size = 'hd' # H, W
    dataset = VM240KValidationDataset(
        root='../dataset_mat/videomatte_motion_hd',
        frames_per_item=frames_per_item, trimap_width=trimap_width, size=-1)
else:
    flg = False

if flg:
    dataset_name='vm240k'
    root = dataset_name+'_val_midtri_'+get_size_name(size)
    dataset_list.append((root, dataset_name, dataset))


# =========================
# realhuman dataset

for dataset_name, realhuman in [
    ('realhuman_allframe', RealhumanDataset_AllFrames),
    # ('realhuman', RealhumanDataset),
]:
    flg = True
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
    elif args.size == 'hd':
        size = 'hd' # H, W
        dataset = realhuman(
            root='../dataset_mat/real_human',
            frames_per_item=frames_per_item, size=-1)
    else:
        flg = False

    if flg:
        root = dataset_name+"_"+get_size_name(size)
        dataset_list.append((root, dataset_name, dataset))

# =========================

def get_dataloader(dataset):
    loader = DataLoader(dataset, batch_size=1, num_workers=args.n_workers, shuffle=False, pin_memory=True)
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
print([d[1] for d in dataset_list])
for root, dataset_name, dataset in dataset_list:
    loader = get_dataloader(dataset)

    for model_name, model_func, inference_core, model_path in model_list:
        if type(model_func) == str:
            model_func = get_model_by_string(model_func)
        if downsample_ratio != 1:
            print('Downsample: ', downsample_ratio)
            model_name = model_name + f'_ds_{downsample_ratio:.4f}'
        run_evaluation(
            root=root, 
            model_name=model_name, model_func=model_func, model_path=model_path, 
            inference_core_func=inference_core,
            dataset_name=dataset_name, dataset=dataset, dataloader=loader, gt_name=gt_name,
            memory_bg=args.memory_bg, downsample_ratio=downsample_ratio, save_video=not args.disable_video
            )
