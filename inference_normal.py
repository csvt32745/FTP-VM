import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
from torch.utils.data import DataLoader
from fastai.data.load import DataLoader as FAIDataLoader

from dataset.imagematte import *
from dataset.videomatte import *
from dataset.augmentation import *
from dataset.vm108_dataset import *

from inference_func import *
from STCNVM.model import *
from STCNVM.module import *
from STCNVM.inference_model import *


model_list = [
   
    # ('DualVM_FG', lambda: DualMattingNetwork(is_output_fg=True), InferenceCoreRecurrent, 
    # '/home/csvt32745/matte/MaskPropagation/saves/Jun01_10.41.32_DualVM_FG_fixFgCor/Jun01_10.41.32_DualVM_FG_fixFgCor_100000.pth'),
    # ('DualVM', lambda: DualMattingNetwork(is_output_fg=False), InferenceCoreRecurrent, 
    # '/home/csvt32745/matte/MaskPropagation/saves/Jul01_23.59.45_DualVM/Jul01_23.59.45_DualVM_120000.pth'),
    
    # ('GFM_VM', GFM_VM, InferenceCoreRecurrentGFM, 
    # '/home/csvt32745/matte/MaskPropagation/saves/Jun15_23.10.51_GFMVM/Jun15_23.10.51_GFMVM_120000.pth'),
  
    # ('DualVM_fm', lambda: DualMattingNetwork(gru=FocalGRU, is_output_fg=False), InferenceCoreRecurrent, 
    # '/home/csvt32745/matte/MaskPropagation/saves/Jun18_14.03.42_DualVM_fm/Jun18_14.03.42_DualVM_fm_120000.pth'),
    # ('GFM_FuseVM_mem_5f', GFM_FuseVM, InferenceCoreRecurrentGFM, 
    # '/home/csvt32745/matte/MaskPropagation/saves/Jun18_13.59.23_GFM_FuseVM_256_bsce/Jun18_13.59.23_GFM_FuseVM_256_bsce_120000.pth'),
    # ('GFM_FuseVM2', lambda: GFM_FuseVM(fuse='GFMFuse2'), InferenceCoreRecurrentGFM, 
    # '/home/csvt32745/matte/MaskPropagation/saves/Jun19_02.59.20_GFM_FuseVM2_256/Jun19_02.59.20_GFM_FuseVM2_256_120000.pth'),
    # ('GFM_FuseVM2_focalloss', lambda: GFM_FuseVM(fuse='GFMFuse2'), InferenceCoreDoubleRecurrentGFM, 
    # '/home/csvt32745/matte/MaskPropagation/saves/Jun22_18.27.32_GFM_FuseVM/Jun22_18.27.32_GFM_FuseVM_120000.pth'),
    # ('GFM_FuseVM2_focalGRUFix', lambda: GFM_FuseVM(fuse='GFMFuse2', gru=FocalGRUFix), InferenceCoreRecurrentGFM, 
    # '/home/csvt32745/matte/MaskPropagation/saves/Jun21_14.20.18_GFM_FuseVM2_FocalGRUFix/Jun21_14.20.18_GFM_FuseVM2_FocalGRUFix_120000.pth'),
    # ('DualVM_attngru', lambda: DualMattingNetwork(gru=AttnGRU, is_output_fg=False), InferenceCoreRecurrent, 
    # '/home/csvt32745/matte/MaskPropagation/saves/Jun21_14.19.39_DualVM_attngru/Jun21_14.19.39_DualVM_attngru_120000.pth'),
    # ('GFM_GatedFuseVM_2dtv_tempcons_weightce_512', GFM_GatedFuseVM, InferenceCoreDoubleRecurrentGFM,
    # '/home/csvt32745/matte/MaskPropagation/saves/Jul01_20.11.40_GFM_GatedFuseVM_tempcon_2dtv_weightce_512/Jul01_20.11.40_GFM_GatedFuseVM_tempcon_2dtv_weightce_512_120000.pth'),
    # ('GFM_GatedFuseVM_3dtvloss_weightce_512', GFM_GatedFuseVM, InferenceCoreDoubleRecurrentGFM,
    # '/home/csvt32745/matte/MaskPropagation/saves/Jul01_20.12.09_GFM_GatedFuseVM_3dtvloss_weightce_512/Jul01_20.12.09_GFM_GatedFuseVM_3dtvloss_weightce_512_120000.pth'),
    # ('GFM_GatedFuseVM_temp_cons', GFM_GatedFuseVM, InferenceCoreDoubleRecurrentGFM,
    # '/home/csvt32745/matte/MaskPropagation/saves/Jun27_14.43.02_GFM_GatedFuseVM_temp_cons/Jun27_14.43.02_GFM_GatedFuseVM_temp_cons_120000.pth'),
    # ('GatedSTCN_GFM_VM2_full_loss', lambda: GatedSTCN_GFM_VM(encoder=2), InferenceCoreDoubleRecurrentGFM,
    # '/home/csvt32745/matte/MaskPropagation/saves/Jun29_18.45.42_GatedSTCN_GFM_VM2_full_loss/Jun29_18.45.42_GatedSTCN_GFM_VM2_full_loss_120000.pth'),
    # ('GFM_GatedFuseVM_wo_decode_img_480', lambda: GFM_GatedFuseVM(decoder_wo_img=True), InferenceCoreDoubleRecurrentGFM,
    # '/home/csvt32745/matte/MaskPropagation/saves/Jul04_10.44.37_GFM_GatedFuseVM_wo_decode_img_480/Jul04_10.44.37_GFM_GatedFuseVM_wo_decode_img_480_120000.pth'),
    # ('GFM_GatedFuseVM_convgru', lambda: GFM_GatedFuseVM(gru=ConvGRU), InferenceCoreDoubleRecurrentGFM,
    # '/home/csvt32745/matte/MaskPropagation/saves/Jul04_10.47.39_GFM_GatedFuseVM_convgru/Jul04_10.47.39_GFM_GatedFuseVM_convgru_120000.pth'),
    ('GFM_GatedFuseVM_AttnGRU', lambda: GFM_GatedFuseVM(gru=AttnGRU), InferenceCoreDoubleRecurrentGFM,
    '/home/csvt32745/matte/MaskPropagation/saves/Jul05_03.14.15_GFM_GatedFuseVM_attngru/Jul05_03.14.15_GFM_GatedFuseVM_attngru_120000.pth'),
    # ('GFM_GatedFuseVM_AttnGRU', lambda: GFM_GatedFuseVM(gru=AttnGRU2), InferenceCoreDoubleRecurrentGFM,
    # '/home/csvt32745/matte/MaskPropagation/saves/Jul04_10.46.48_GFM_GatedFuseVM_attngru2/Jul04_10.46.48_GFM_GatedFuseVM_attngru2_120000.pth'),
    # ('GFM_GatedFuseVM_FocalGRU', lambda: GFM_GatedFuseVM(gru=FocalGRU), InferenceCoreDoubleRecurrentGFM,
    # '/home/csvt32745/matte/MaskPropagation/saves/Jul05_00.51.44_GFM_GatedFuseVM_focalgru/Jul05_00.51.44_GFM_GatedFuseVM_focalgru_120000.pth'),
    # ('GFM_FuseVM2_AttnGRU_new', lambda: GFM_FuseVM(fuse='GFMFuse2', gru=AttnGRU), InferenceCoreDoubleRecurrentGFM, 
    # '/home/csvt32745/matte/MaskPropagation/saves/Jun26_13.46.07_GFM_FuseVM_AttnGRU_wo_tvloss/Jun26_13.46.07_GFM_FuseVM_AttnGRU_wo_tvloss_120000.pth'),
    # ('DualVM_new', lambda: DualMattingNetwork(is_output_fg=False), InferenceCoreRecurrent, 
    # '/home/csvt32745/matte/MaskPropagation/saves/Jun22_18.27.31_DualVM_tri/Jun22_18.27.31_DualVM_tri_120000.pth'),
    # ('GatedVM', lambda: GatedVM(is_output_fg=False), InferenceCoreRecurrent, 
    # '/home/csvt32745/matte/MaskPropagation/saves/Jun27_21.03.07_GatedVM/Jun27_21.03.07_GatedVM_120000.pth'),
    # ('BGVM2_big_aug_motion_pha', lambda: BGVM(bg_inpainting=lambda *args: BGInpaintingGRU(*args, gru=lambda *args2: AttnGRU2(*args2, patch_size=17, hidden=32))), InferenceCoreRecurrentBG,
    # '/home/csvt32745/matte/MaskPropagation/saves/Jun28_15.36.19_BGVM2_big_aug_motion_pha/Jun28_15.36.19_BGVM2_big_aug_motion_pha_120000.pth'),
    # ('STCN_GFM_VM_new', STCN_GFM_VM, InferenceCoreDoubleRecurrentGFM,
    # '/home/csvt32745/matte/MaskPropagation/saves/Jun26_13.46.03_STCN_GFM_VM/Jun26_13.46.03_STCN_GFM_VM_120000.pth'),
    # ('DualVM_attngru_effb3a', lambda: DualMattingNetwork(backbone_arch='efficientnet_b3a', gru=AttnGRU, is_output_fg=False), InferenceCoreRecurrent,
    # '/home/csvt32745/matte/MaskPropagation/saves/Jun24_23.38.08_DualVM_attngru_effb3a/Jun24_23.38.08_DualVM_attngru_effb3a_120000.pth'),
    # ('GFM_GatedFuseVM_4xfoucs', GFM_GatedFuseVM_4xfoucs, InferenceCoreRecurrentGFM, 
    # '/home/csvt32745/matte/MaskPropagation/saves/Jul04_15.37.35_GFM_GatedFuseVM_4xfoucs/Jul04_15.37.35_GFM_GatedFuseVM_4xfoucs_120000.pth'),
]


def get_size_name(size):
    return str(size) if type(size) == int else f'{size[1]}x{size[0]}'
    
dataset_list = []

# size = [576, 1024] # H, W
size = 256
dataset = VM108ValidationDataset(
    frames_per_item=8, mode='val', is_subset=False, 
    trimap_width=25, size=size)
dataset_name='vm108'
# size = [288, 512] # H, W
root = dataset_name+'_val_midtri_'+get_size_name(size)
dataset_list.append((root, dataset_name, dataset))


size = [288, 512] # H, W
# size = [576, 1024] # H, W
dataset = VM240KValidationDataset(
    # root='../dataset_mat/videomatte_motion_1024',
    root='../dataset_mat/videomatte_motion_sd',
    frames_per_item=8, trimap_width=25, size=-1)
dataset_name='vm240k'
root = dataset_name+'_val_midtri_'+get_size_name(size)
dataset_list.append((root, dataset_name, dataset))


def get_dataloader(dataset):
    loader = DataLoader(dataset, batch_size=1, num_workers=16, shuffle=False, pin_memory=True)
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
        run_evaluation(
            root=root, 
            model_name=model_name, model_func=model_func, model_path=model_path, 
            inference_core_func=inference_core,
            dataset_name=dataset_name, dataset=dataset, dataloader=loader, gt_name=gt_name)
