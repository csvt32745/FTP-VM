from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--size', help='eval video size: sd, 1024, hd, 4k', default='sd', type=str)
parser.add_argument('--frames_per_item', help='frames in a batch', default=8, type=int)
parser.add_argument('--n_workers', help='num workers', default=8, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--memory_bg', help='Given bg & zero-trimap as memory input', action='store_true')
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
    # 'GFM_GatedFuseVM_fuse=splitfuse',
    # 'GFM_GatedFuseVM_fuse=sameqk_head1'
    # ('GFM_GatedFuseVM_4xfoucs_feedzero', 'GFM_GatedFuseVM_4xfoucs')
    # 'GFM_GatedFuseVM_2dtv_tempcons_weightce_512',
    # 'GFM_GatedFuseVM_3dtvloss_weightce_512',
    # 'GFM_GatedFuseVM_normal_celoss_480',
    # 'GFM_GatedFuseVM_3dtvloss_480',
    # 'GFM_GatedFuseVM_4xfoucs_focal_sameqk_head1',
    # 'GFM_GatedFuseVM_to4xglance_4xfocus_focal_sameqk_head1',
    # 'GFM_GatedFuseVM_to4xglance_4xfocus_3',
    # ('GFM_GatedFuseVM_to4xglance_4xfocus_3_bgmem', 'GFM_GatedFuseVM_to4xglance_4xfocus_3'),
    # 'GFM_GatedFuseVM_to4xglance_4xfocus_3_naivefuse',
    # 'GFM_GatedFuseVM_to4xglance_4xfocus_3_focal',
    # 'GFM_GatedFuseVM_to4xglance_4xfocus_3_focal_sameqk_head1',
    # 'GFM_GatedFuseVM_to4xglance_4xfocus_3_fixtrimap',
    # 'GFM_GatedFuseVM_to4xglance_4xfocus_4',
    # 'GFM_GatedFuseVM_to4xglance_4xfocus_3_inputmaskonly',
	# 'GFM_GatedFuseVM_to4xglance_4xfocus_3_random_trimap',
	# 'GFM_GatedFuseVM_to4xglance_4xfocus_3_sameqk_head1',
    # 'GFM_FuseVM',
	# 'GFM_GatedFuseVM_to4xglance_4xfocus_3_convgru',
	# 'GFM_GatedFuseVM_to4xglance_4xfocus_2',
    # 'GFM_GatedFuseVM_to4xglance_4xfocus_3_fusefeature'
    # 'GFM_GatedFuseVM_to4xglance_4xfocus_fixtrimap',
	# 'GFM_GatedFuseVM_to4xglance_4xfocus_3_bn',
	# 'GFM_GatedFuseVM_to4xglance_4xfocus_3_fixtrimap',
	# 'GFM_GatedFuseVM_to4xglance_4xfocus_5',
    # 'GFM_GatedFuseVM_to4xglance_4xfocus_6_sameqk_head1',
	# 'GFM_GatedFuseVM_to4xglance_4xfocus_5_sameqk_head1',
	# 'GFM_GatedFuseVM_to4xglance_4xfocus_3_sameqk_head1',
    # 'GFM_GatedFuseVM_to4xglance_4xfocus_7',
	# 'GFM_GatedFuseVM_to4xglance_4xfocus_3_sameqk_head1_convgru',
    # 'GFM_GatedFuseVM_to4xglance_4xfocus_3_fixtrimap_firstmat',
    # 'GFM_GatedFuseVM_to4xglance_4xfocus_3=grufuse',
	# 'GFM_GatedFuseVM_to4xglance_4xfocus_3_fullresgate',
	# 'GFM_GatedFuseVM_to4xglance_4xfocus_3=naive_h1sqk',
    # 'GFM_GatedFuseVM_to4xglance_4xfocus_3_fixtrimap_ytvos',
	# 'GFM_GatedFuseVM_to4xglance_4xfocus_3_fuse_convgru=ff2',
	# 'GFM_GatedFuseVM_to4xglance_4xfocus_3_randtrimap',
	# 'GFM_GatedFuseVM_to4xglance_4xfocus_3_multiobj',
	# 'GFM_GatedFuseVM_to4xglance_4xfocus_3_sameqk_head1_convgru',
    # 'STCNFuseMatting_fuse=naive',
	'STCNFuseMatting_480',
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
	'STCNFuseMatting_fuse=bn_480',

]

print(model_list)
model_list = [
    [i[0]]+list(inference_model_list[i[1]][1:]) if type(i) in [tuple, list] else inference_model_list[i] 
    for i in model_list
]


print(args)
assert args.size in ['sd', '1024', 'hd', '4k']

def get_size_name(size):
    return str(size) if type(size) == int else f'{size[1]}x{size[0]}'
    
dataset_list = []
frames_per_item = args.frames_per_item
trimap_width = 25

# =========================
# vm108 dataset

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
# realhuman dataset

for dataset_name, realhuman in [
    ('realhuman_allframe', RealhumanDataset_AllFrames),
    # ('realhuman', RealhumanDataset),
]:
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
        size = [1080, 1920] # H, W
        dataset = realhuman(
            root='../dataset_mat/real_human',
            frames_per_item=frames_per_item, size=-1)

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
            
        run_evaluation(
            root=root, 
            model_name=model_name, model_func=model_func, model_path=model_path, 
            inference_core_func=inference_core,
            dataset_name=dataset_name, dataset=dataset, dataloader=loader, gt_name=gt_name,
            memory_bg=args.memory_bg
            )
