import gc
from time import time

import numpy as np
import torch
torch.set_grad_enabled(False)

from STCNVM.inference_model import *
from evalutation.evaluate_lr import Evaluator


class TimeStamp:
    def __init__(self):
        self.last = time()
    
    def count(self):
        cur = time()
        dif = cur-self.last
        self.last = cur
        return dif

def run_inference(inference_core: InferenceCore, pred_path, gt_path, dataset):
    fps = inference_core.propagate()
    
    inference_core.save_imgs(os.path.join(pred_path, dataset))
    inference_core.save_gt(os.path.join(gt_path, dataset))
    inference_core.save_video(os.path.join(pred_path, dataset))
    return fps

def run_evaluation(
    root,
    model_name, model_func, model_path,
    inference_core_func, 
    dataset_name, dataset, dataloader,
    memory_freq=-1, memory_gt=True,
    gt_name='GT'
    ):
    print(f"=" * 30)
    print(f"[ Current model: {model_name} ]")

    pred_path = os.path.join(root, model_name)
    gt_path = os.path.join(root, gt_name)

    model = model_func()
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    
    inference_core: InferenceCore = None
    last_data = None
    loader_iter = iter(dataloader)
    debug = 0
    ts = TimeStamp()
    fps = []
    while True:
        inference_core = inference_core_func(
            model, dataset, loader_iter, last_data=last_data,
            memory_iter=memory_freq, memory_gt=memory_gt)

        fps.append(run_inference(inference_core, pred_path, gt_path, dataset_name))

        # load next batch
        if inference_core.is_vid_overload:
            last_data = inference_core.last_data
        else:
            last_data = next(loader_iter, None)
        
        if last_data is None:
            break

        # debug += 1
        # if debug > 0:
        #     break

        # clear memory
        if inference_core is not None:
            inference_core.clear()
            del inference_core
            gc.collect()

    print(f"[ Inference time: {ts.count()} ]")
    
    Evaluator(
        pred_dir=pred_path,
        true_dir=gt_path,
        num_workers=8, is_eval_fgr=False, is_fix_fgr=False
    )
    print(f"[ Computer score time: {ts.count()} ]")
    print(f"[ Inference GPU FPS: {np.mean(fps)} ]")

