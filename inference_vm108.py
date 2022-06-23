# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import gc
import mediapy as media
import numpy as np
from time import time

import torch
torch.set_grad_enabled(False)
from torch.utils.data import DataLoader
from fastai.data.load import DataLoader as FAIDataLoader

from dataset.imagematte import *
from dataset.videomatte import *
from dataset.augmentation import *
from dataset.vm108_dataset import *

from STCNVM.model import *
from STCNVM.module import *
from STCNVM.inference_model import *
from evalutation.evaluate_lr import Evaluator


# %%


# %%
# from evalutation.evaluate_lr import Evaluator
# Evaluator(
#     pred_dir='./predictions/DualVM',
#     true_dir='./predictions/GT',
#     num_workers=2
# )

# %%
vm108 = VM108ValidationDataset(frames_per_item=8, mode='val', is_subset=False)
dataset_name='vm108_subval_512x512'
# vm108 = VM108ValidationDatasetFixFG(
#     fg_list_path='/home/csvt32745/matte/dataset_mat/VideoMatting108/exp_fgrs.rxt',
#     bg_list_path='/home/csvt32745/matte/dataset_mat/VideoMatting108/exp_bgrs.rxt',
#     frames_per_item=4,
#     size=512
# )
# dataset_name='fix_fgr'
loader = DataLoader(vm108, batch_size=1, num_workers=8, shuffle=False, pin_memory=True)
# loader = FAIDataLoader(
#         dataset = vm108,
#         batch_size=1,
#         num_workers=8,
#         shuffle=False, 
#         drop_last=False,
#         timeout=60,
#     )

# %%
model_list = [
    # ('STCN_VM', STCN_VM, InferenceCoreMemoryRecurrent, 
    # '/home/csvt32745/matte/MaskPropagation/saves/May22_05.06.46_STCN_VM/May22_05.06.46_STCN_VM_120000.pth'),
    # ('STCN', STCN_Full, InferenceCoreSTCN, 
    # '/home/csvt32745/matte/MaskPropagation/saves/May21_00.36.27_STCN_with_seg/May21_00.36.27_STCN_with_seg_120000.pth'),
    # ('STCN_RecDecoder', STCN_RecDecoder, InferenceCoreMemoryRecurrent, '/home/csvt32745/matte/MaskPropagation/saves/May23_03.58.11_STCN_RecDecoder/May23_03.58.11_STCN_RecDecoder_120000.pth'),
    
    # ('DualVM_FG', lambda: DualMattingNetwork(is_output_fg=True), InferenceCoreRecurrent, 
    # '/home/csvt32745/matte/MaskPropagation/saves/Jun01_10.41.32_DualVM_FG_fixFgCor/Jun01_10.41.32_DualVM_FG_fixFgCor_100000.pth'),
    # ('DualVM', lambda: DualMattingNetwork(is_output_fg=False), InferenceCoreRecurrent, 
    # '/home/csvt32745/matte/MaskPropagation/saves/May22_05.36.30_DualVM/May22_05.36.30_DualVM_120000.pth'),
    
    ('GFM_VM', GFM_VM, InferenceCoreRecurrentGFM, 
    '/home/csvt32745/matte/MaskPropagation/saves/Jun15_23.10.51_GFMVM/Jun15_23.10.51_GFMVM_120000.pth'),
    # ('DualVM_FG_mem20f', lambda: DualMattingNetwork(is_output_fg=True), InferenceCoreRecurrent, 
    # '/home/csvt32745/matte/MaskPropagation/saves/May31_13.21.21_DualVM_FG_fixFgCor/May31_13.21.21_DualVM_FG_fixFgCor_120000.pth'),
    
    # ('DualVM_fm', lambda: DualMattingNetwork(gru=FocalGRU, is_output_fg=False), InferenceCoreRecurrent, 
    # '/home/csvt32745/matte/MaskPropagation/saves/Jun18_14.03.42_DualVM_fm/Jun18_14.03.42_DualVM_fm_120000.pth'),
    # ('GFM_FuseVM_mem_5f', GFM_FuseVM, InferenceCoreRecurrentGFM, 
    # '/home/csvt32745/matte/MaskPropagation/saves/Jun18_13.59.23_GFM_FuseVM_256_bsce/Jun18_13.59.23_GFM_FuseVM_256_bsce_120000.pth'),
    # ('GFM_FuseVM2', lambda: GFM_FuseVM(fuse='GFMFuse2'), InferenceCoreRecurrentGFM, 
    # '/home/csvt32745/matte/MaskPropagation/saves/Jun19_02.59.20_GFM_FuseVM2_256/Jun19_02.59.20_GFM_FuseVM2_256_120000.pth'),
    ('GFM_FuseVM2_focalloss', lambda: GFM_FuseVM(fuse='GFMFuse2'), InferenceCoreDoubleRecurrentGFM, 
    '/home/csvt32745/matte/MaskPropagation/saves/Jun18_21.19.27_GFM_FuseVM2_256_focal/Jun18_21.19.27_GFM_FuseVM2_256_focal_120000.pth'),
    # ('GFM_FuseVM2_focalGRUFix', lambda: GFM_FuseVM(fuse='GFMFuse2', gru=FocalGRUFix), InferenceCoreRecurrentGFM, 
    # '/home/csvt32745/matte/MaskPropagation/saves/Jun21_14.20.18_GFM_FuseVM2_FocalGRUFix/Jun21_14.20.18_GFM_FuseVM2_FocalGRUFix_120000.pth'),
    ('DualVM_attngru', lambda: DualMattingNetwork(gru=AttnGRU, is_output_fg=False), InferenceCoreRecurrent, 
    '/home/csvt32745/matte/MaskPropagation/saves/Jun21_14.19.39_DualVM_attngru/Jun21_14.19.39_DualVM_attngru_120000.pth'),
    ('DualVM_new', lambda: DualMattingNetwork(is_output_fg=False), InferenceCoreRecurrent, 
    '/home/csvt32745/matte/MaskPropagation/saves/Jun20_10.18.39_DualVM_new/Jun20_10.18.39_DualVM_new_120000.pth'),
    
    ('GFM_BGFuse8xVM', GFM_BGFuse8xVM, InferenceCoreRecurrentGFMBG,
    '/home/csvt32745/matte/MaskPropagation/saves/Jun22_00.33.09_GFM_BGFuse8xVM/Jun22_00.33.09_GFM_BGFuse8xVM_120000.pth'),
    ('GFM_Fuse8xVM', GFM_Fuse8xVM, InferenceCoreRecurrentGFM,
    '/home/csvt32745/matte/MaskPropagation/saves/Jun21_22.12.01_GFM_Fuse8xVM/Jun21_22.12.01_GFM_Fuse8xVM_120000.pth'),
    ('BGVM_attngru', lambda: BGVM(gru=AttnGRU, is_output_fg=False), InferenceCoreRecurrent,
    '/home/csvt32745/matte/MaskPropagation/saves/Jun21_20.47.38_BGVM_attngru/Jun21_20.47.38_BGVM_attngru_120000.pth'),
]

# %%
pred = 'predictions'
# pred = 'pred_fixfgr'
# pred = 'pred_fixfgr256'
gt_root = 'GT'

# %%
class TimeStamp:
    def __init__(self):
        self.last = time()
    
    def count(self):
        cur = time()
        dif = cur-self.last
        self.last = cur
        return dif

# %%
def inference(inference_core: InferenceCore, root, dataset):
    fps = inference_core.propagate()
    # root = 'DualVM'
    # dataset = 'vm108_512x512'
    inference_core.save_imgs(os.path.join(pred, root, dataset))
    inference_core.save_gt(os.path.join(pred, gt_root, dataset))
    inference_core.save_video(os.path.join(pred, root, dataset))
    return fps

# %%
for root, model_func, inference_core_func, model_path in model_list:
    print(f"=" * 30)
    print(f"[ Current model: {root} ]")
    model = model_func()
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    inference_core: InferenceCore = None
    last_data = None
    loader_iter = iter(loader)
    debug = 0
    ts = TimeStamp()
    # while False:
    fps = []
    while True:

        if True:
        # if inference_core_func in [InferenceCoreRecurrent, InferenceCoreDoubleRecurrentGFM, InferenceCoreRecurrentGFM]:
            inference_core = inference_core_func(
                model, vm108, loader_iter, last_data=last_data,
                memory_iter=-1, memory_gt=True)
        else:
            inference_core = inference_core_func(
                model, vm108, loader_iter, last_data=last_data, top_k=40)
        fps.append(inference(inference_core, root, dataset_name))
        if inference_core.is_vid_overload:
            last_data = inference_core.last_data
        else:
            last_data = next(loader_iter, None)
        
        if last_data is None:
            break

        # debug += 1
        # if debug > 0:
        #     break

        if inference_core is not None:
            inference_core.clear()
            del inference_core
            gc.collect()

    print(f"[ Inference time: {ts.count()} ]")
    
    Evaluator(
        pred_dir=os.path.join(pred, root),
        true_dir=os.path.join(pred, gt_root),
        num_workers=8, is_eval_fgr=False, is_fix_fgr=False
        )
    print(f"[ Computer score time: {ts.count()} ]")
    print(f"[ Inference GPU FPS: {np.mean(fps)} ]")

