import cv2
cv2.setNumThreads(0)
import numpy as np
import torch
from functools import lru_cache
from .mask_perturb import perturb_mask

def all_to_onehot(masks, labels):
    Ms = np.zeros((len(labels), masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
    for k, l in enumerate(labels):
        Ms[k] = (masks == l).astype(np.uint8)
    return Ms

@lru_cache(32)
def _get_kernel(kernel_size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

def get_dilated_trimaps(phas, kernel_size):
    trimaps = []
    kernel = _get_kernel(kernel_size)
    for pha in (phas[:, 0].numpy()*255).astype(np.uint8): # N, H, W
        fg_and_unknown = np.array(np.not_equal(pha, 0).astype(np.float32))
        fg = np.array(np.equal(pha, 255).astype(np.float32))
        dilate = cv2.dilate(fg_and_unknown, kernel, iterations=1)
        erode = cv2.erode(fg, kernel, iterations=1)
        trimap = erode * 1 + (dilate-erode) * 0.5
        trimaps.append(trimap)
    return torch.from_numpy(np.stack(trimaps)).unsqueeze(1) # T, 1, H, W


def get_perturb_masks(phas):
    ret = []
    mae_min = 10
    mae_max = 40
    mae_target = np.random.rand()*(mae_max-mae_min) + mae_min
    for pha in (phas[:, 0].numpy()*255).astype(np.uint8): # N, H, W
        pha = perturb_mask(pha, mae_target)
        ret.append(torch.from_numpy(pha))
    return torch.stack(ret).unsqueeze(1)/255.