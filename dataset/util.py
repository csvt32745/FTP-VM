import cv2
cv2.setNumThreads(0)
import numpy as np
import torch
from functools import lru_cache

def all_to_onehot(masks, labels):
    Ms = np.zeros((len(labels), masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
    for k, l in enumerate(labels):
        Ms[k] = (masks == l).astype(np.uint8)
    return Ms

@lru_cache(32)
def _get_kernel(kernel_size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

def get_dilated_trimaps(phas, kernel_size, eps=5e-3, random_kernel=False):
    trimaps = []
    kernel = _get_random_kernel(kernel_size, np.random.randint(1, 5)) if random_kernel else _get_kernel(kernel_size)
    phas = phas[:, 0].clamp(0, 1).numpy() # N, H, W
    fg_and_unknowns = (phas > eps).astype(np.uint8)
    fgs = (phas > (1-eps)).astype(np.uint8)
    for i in range(len(fgs)): # N, H, W
        dilate = cv2.dilate(fg_and_unknowns[i], kernel, iterations=1).astype(np.float32)
        erode = cv2.erode(fgs[i], kernel, iterations=1).astype(np.float32)
        trimap = erode * 1. + (dilate-erode) * 0.5
        trimaps.append(trimap)
    return torch.from_numpy(np.stack(trimaps)).unsqueeze(1) # T, 1, H, W

def get_dilated_trimaps_np_uint8(pha, kernel_size):
    # H, W
    kernel = _get_kernel(kernel_size)
    fg_and_unknown = (pha > 1).astype(np.uint8)
    fg = (pha > 254).astype(np.uint8)
    dilate = cv2.dilate(fg_and_unknown, kernel).astype(np.float32)
    erode = cv2.erode(fg, kernel).astype(np.float32)
    trimap = erode * 1. + (dilate-erode) * 0.5
    trimap = np.clip(trimap*255, 0, 255).astype(np.uint8)
    return trimap

@lru_cache(128)
def _get_random_kernel(size, choice):
    # choice = np.random.randint(1, 5)
    if choice == 1:
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    elif choice == 2:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    elif choice == 3:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size//2))
    elif choice == 4:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size//2, size))