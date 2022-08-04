# import cv2
# cv2.setNumThreads(0)

import datetime
import time
from os import path
import math

import random
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from fastai.data.load import DataLoader as FAIDataLoader

from model.model import PropagationModel
from dataset.youtubevis import *
from dataset.youtubevos import *
from dataset.imagematte import *
from dataset.videomatte import *
from dataset.augmentation import *

from util.logger import TensorboardLogger
from util.hyper_para import HyperParameters


"""
Initial setup
"""
torch.manual_seed(14159265)
np.random.seed(14159265)
random.seed(14159265)


# Parse command line arguments
para = HyperParameters()
para.parse()

if para['benchmark']:
    torch.backends.cudnn.benchmark = True

"""
Model related
"""

long_id = '%s_%s' % (datetime.datetime.now().strftime('%b%d_%H.%M.%S'), para['id'])
logger = TensorboardLogger(para['id'], long_id)
logger.log_string('hyperpara', str(para))

# Construct the rank 0 model
model = PropagationModel(para, logger=logger, 
                save_path=path.join('saves', long_id, long_id) if long_id is not None else None).train()

# Automatically search the latest model path
def search_checkpoint(save_path, target):
    save_models = [
        (i, datetime.datetime.strptime(split[0], '%b%d_%H.%M.%S')) 
        for i in os.listdir(save_path) 
        if (target in i) and ((split := i.split("_"+target, maxsplit=1))[1] == '')
    ]
    
    save_models = sorted(save_models, key=lambda x: x[1], reverse=True)
    for s, t in save_models:
        p = os.path.join(save_path, s)
        ckpt_name = s+"_checkpoint.pth"
        if ckpt_name in os.listdir(p):
            ckpt_name = os.path.join(p, ckpt_name)
            return ckpt_name
    return None

# Load pertrained model if needed
seg_count = (1 - para['seg_start']) # delay start
if seg_count < 0:
    seg_count += para['seg_cd']
seg_iter = 0 if para['seg_start'] == 0 else 1e5 # 0 for seg initially, 1e5 for delay

if (model_path := para['load_model']) is not None or para['resume']:
    if para['resume']:
        print('Search model: ', para['id'])
        model_path = search_checkpoint('saves', para['id'])
        assert model_path is not None, 'No last model checkpoint exists!'
        print("Latest model ckpt: ", model_path)
    
    total_iter, extra_dict = model.load_model(model_path, ['seg_count', 'seg_iter'])
    print('Previously trained model loaded!')
    if extra_dict['seg_count'] is not None:
        seg_count = extra_dict['seg_count']
    if extra_dict['seg_iter'] is not None:
        seg_iter = extra_dict['seg_iter']
else:
    total_iter = 0

print('seg_count: %d, seg_iter: %d'%(seg_count, seg_iter))

if para['load_network'] is not None:
    model.load_network(para['load_network'])
    print('Previously trained network loaded!')

"""
Dataloader related
"""
# To re-seed the randomness everytime we start a worker
def worker_init_fn(worker_id): 
    return np.random.seed(torch.initial_seed()%(2**31))
    # return np.random.seed(torch.initial_seed()%(2**31) + worker_id + local_rank*100)

def construct_loader(dataset, batch_size=para['batch_size']):
    # train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, rank=local_rank, shuffle=True)
    # train_loader = DataLoader(dataset, para['batch_size'], sampler=train_sampler, num_workers=para['num_worker'],
    train_loader = DataLoader(dataset, batch_size, num_workers=para['num_worker'],
                                shuffle=True, drop_last=True, pin_memory=True)
                            # worker_init_fn=worker_init_fn, drop_last=True, pin_memory=True)
    # train_loader = FAIDataLoader(
    #     dataset = dataset,
    #     batch_size=batch_size,
    #     num_workers=para['num_worker'],
    #     pin_memory=True, 
    #     shuffle=True, 
    #     drop_last=True,
    #     timeout=60,
    # )
    return train_loader

def renew_vm108_loader(long_seq=True, nb_frame_only=False):
    # size=512
    size=para['size']
    train_dataset = VideoMatteDataset(
        '../dataset_sc/VideoMatting108_512',
        '../dataset_sc/BG20k_512/BG-20k/train',
        '../dataset_sc/VideoMatting108_512/BG_done',
        size=size,
        # seq_length=12 if long_seq else 5,
        seq_length=6 if long_seq else 8,
        seq_sampler=TrainFrameSampler() if nb_frame_only else TrainFrameSamplerAddFarFrame(),
        transform=VideoMatteTrainAugmentation(size, get_bgr_pha=para['get_bgr_pha']),
        is_VM108=True,
        bg_num=1,
        get_bgr_phas=para['get_bgr_pha'],
        random_memtrimap=para['random_memtrimap'],
    )
    print('VM108 dataset size: ', len(train_dataset))

    # return construct_loader(train_dataset, batch_size=2 if long_seq else 2)
    return construct_loader(train_dataset, batch_size=4 if long_seq else 4)

def renew_d646_loader(long_seq=True, nb_frame_only=False):
    # size=512
    size=para['size']
    train_dataset = ImageMatteDataset(
        '../dataset_sc/Distinctions646_512/Train',
        '../dataset_sc/BG20k_512/BG-20k/train',
        '../dataset_sc/VideoMatting108_512/BG_done',
        size=size,
        # seq_length=6 if long_seq else 3,
        seq_length=3 if long_seq else 4,
        seq_sampler=TrainFrameSampler() if nb_frame_only else TrainFrameSamplerAddFarFrame(),
        transform=ImageMatteAugmentation(size, get_bgr_pha=para['get_bgr_pha']),
        bg_num=1,
        get_bgr_phas=para['get_bgr_pha'],
        random_memtrimap=para['random_memtrimap'],
    )
    print('D646 dataset size: ', len(train_dataset))

    # return construct_loader(train_dataset, batch_size=8 if long_seq else 4)
    return construct_loader(train_dataset, batch_size=12 if long_seq else 10)

def renew_ytvis_loader(long_seq=True, nb_frame_only=False):
    size = min(352, para['size'])
    # size = para['size']
    speed = [0.5, 1, 2]
    if para['ytvos']:
        train_dataset = YouTubeVOSDataset(
            '../dataset_sc/YoutubeVOS/train',
            size, 
            8 if long_seq else 8,
            TrainFrameSampler(speed) if nb_frame_only else TrainFrameSamplerAddFarFrame(speed),
            YouTubeVISAugmentation(size),
            random_memtrimap=para['random_memtrimap'],
        )
    else:
        train_dataset = YouTubeVISDataset(
            '../dataset_sc/YoutubeVIS/train/JPEGImages',
            '../dataset_sc/YoutubeVIS/train/instances.json',
            size, 
            8 if long_seq else 8,
            TrainFrameSampler(speed) if nb_frame_only else TrainFrameSamplerAddFarFrame(speed),
            YouTubeVISAugmentation(size),
            random_memtrimap=para['random_memtrimap'],
        )
    print('YTVis dataset size: ', len(train_dataset))

    # return construct_loader(train_dataset, batch_size=12 if long_seq else 6)
    return construct_loader(train_dataset, batch_size=8 if long_seq else 8)

"""
Dataset related
"""
print('Use longer sequence: ', para['long_seq'])
d646_loader = renew_d646_loader(long_seq=para['long_seq'], nb_frame_only=para['nb_frame_only'])
vm108_loader = renew_vm108_loader(long_seq=para['long_seq'], nb_frame_only=para['nb_frame_only'])
train_loader = d646_loader
    
if total_iter < para['seg_stop']:
    seg_loader = renew_ytvis_loader(long_seq=para['long_seq'], nb_frame_only=para['nb_frame_only'])

"""
Determine current/max epoch
"""
iter_base = min(len(d646_loader), len(vm108_loader), para['seg_iter'])
total_epoch = math.ceil(para['iterations']/iter_base)
current_epoch = total_iter // max(len(d646_loader), len(vm108_loader), para['seg_iter'])

"""
Starts training
"""
# Need this to select random bases in different workers
np.random.seed(np.random.randint(2**30-1))# + local_rank*100)
if is_dataset_switched := (total_iter >= para['iter_switch_dataset']):
    print('Switch to video dataset!')
    train_loader = vm108_loader

def get_extra_dict():
    return {
        'seg_iter': seg_iter,
        'seg_count': seg_count,
    }

try:
    for e in range(current_epoch, total_epoch): 
        torch.cuda.empty_cache()
        time.sleep(2)
        if total_iter >= para['iterations']:
            break
        print('Epoch %d/%d' % (e, total_epoch))

        # Train loop
        model.train()

        # Segmentation pass
        if total_iter < para['seg_stop'] and ((seg_count == 0) or (seg_iter < para['seg_iter'])):
            print("Segmentation Training: ")
            for data in seg_loader:
                model.do_pass(data, total_iter, segmentation_pass=True)
                total_iter += 1
                seg_iter += 1
                
                if total_iter >= para['iterations']:
                    break

                if seg_iter >= para['seg_iter']:
                    print("Segmentation Stop.")
                    break
            seg_count = 1
        # Matting pass
        else:    
            for data in train_loader:
                model.do_pass(data, total_iter)
                total_iter += 1
                seg_count += 1

                if total_iter >= para['iterations']:
                    break

                if not is_dataset_switched and total_iter >= para['iter_switch_dataset']:
                    print('Switch to video dataset!')
                    train_loader = vm108_loader
                    is_dataset_switched = True
                    break
                
                if seg_count >= para['seg_cd']:
                    print("Segmentation CD finisih.")
                    seg_count = 0
                    seg_iter = 0
                    break
                    
finally:
    if not para['debug'] and model.logger is not None and total_iter>5000:
        model.save(total_iter, ckpt_dict=get_extra_dict())
