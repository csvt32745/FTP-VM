import cv2
import numpy as np

import torch
from dataset.range_transform import inv_im_trans
from collections import defaultdict

def tensor_to_numpy(image):
    image_np = (image.numpy() * 255).astype('uint8')
    return image_np

def tensor_to_np_float(image):
    image_np = image.numpy().astype('float32')
    return image_np

def detach_to_cpu(x):
    return x.detach().cpu()

def transpose_np(x):
    return np.transpose(x, [1,2,0])

def tensor_to_gray_im(x):
    x = detach_to_cpu(x)
    x = tensor_to_numpy(x)
    x = transpose_np(x)
    return x

def tensor_to_im(x):
    x = detach_to_cpu(x)
    x = inv_im_trans(x).clamp(0, 1)
    x = tensor_to_numpy(x)
    x = transpose_np(x)
    return x

# Predefined key <-> caption dict
key_captions = {
    'im': 'Image', 
    'gt': 'GT', 
}

"""
Return an image array with captions
keys in dictionary will be used as caption if not provided
values should contain lists of cv2 images
"""
def get_image_array(images, grid_shape, captions={}):
    h, w = grid_shape
    cate_counts = len(images)
    rows_counts = len(next(iter(images.values())))

    font = cv2.FONT_HERSHEY_SIMPLEX

    output_image = np.zeros([w*cate_counts, h*(rows_counts+1), 3], dtype=np.uint8)
    col_cnt = 0
    for k, v in images.items():

        # Default as key value itself
        caption = captions.get(k, k)

        # Handles new line character
        dy = 40
        for i, line in enumerate(caption.split('\n')):
            cv2.putText(output_image, line, (10, col_cnt*w+100+i*dy),
                     font, 0.8, (255,255,255), 2, cv2.LINE_AA)

        # Put images
        for row_cnt, img in enumerate(v):
            im_shape = img.shape
            if len(im_shape) == 2:
                img = img[..., np.newaxis]

            img = (img * 255).astype('uint8')

            output_image[(col_cnt+0)*w:(col_cnt+1)*w,
                         (row_cnt+1)*h:(row_cnt+2)*h, :] = img
            
        col_cnt += 1

    return output_image

def base_transform(im, size):
        im = tensor_to_np_float(im)
        if len(im.shape) == 3:
            im = im.transpose((1, 2, 0))
        else:
            im = im[:, :, None]

        # Resize
        if im.shape[1] != size:
            im = cv2.resize(im, size, interpolation=cv2.INTER_NEAREST)

        return im.clip(0, 1)

def im_transform(im, size):
    # TODO: disable inv im transform
        return base_transform(detach_to_cpu(im), size=size)
        # return base_transform(inv_im_trans(detach_to_cpu(im)), size=size)

def mask_transform(mask, size):
    return base_transform(detach_to_cpu(mask), size=size)

def out_transform(mask, size):
    return base_transform(detach_to_cpu(torch.sigmoid(mask)), size=size)

def pool_pairs(images, size, so):
    #TODO: adapt to mine
    req_images = defaultdict(list)

    b, s, _, _, _ = images['gt'].shape

    GT_name = 'GT'
    # for b_idx in range(b):
    #     GT_name += ' %s\n' % images['info']['name'][b_idx]
    is_fg = 'pred_fg' in images
    is_bg = 'pred_bg' in images
    is_coarse_mask = 'coarse_mask' in images
    is_out_mem = images['rgb'].size(1) == images['rgb_query'].size(1)
    is_trimap = ((logits := images.get('logits', None)) is None) or (logits.size(2) != 3)
    for b_idx in range(min(2, b)):
        # for b_idx in range(b):
        if not is_out_mem:
            req_images['RGB'].append(im_transform(images['rgb'][b_idx, 0], size))
            req_images['Mask'].append(np.zeros((size[1], size[0], 3)))
            gt_mask = images['gt'][b_idx, 0]
            req_images[GT_name].append(mask_transform(gt_mask, size))
            # if is_fg:
            #     req_images['Pred_FG'].append(np.zeros((size[1], size[0], 3)))
            #     req_images['GT_FG'].append(im_transform(images['fg'][b_idx, 0]*gt_mask, size))
            if is_bg:
                req_images['Pred_BG'].append(np.zeros((size[1], size[0], 3)))
                if (bg := images.get('bg', None)) is not None:
                    req_images['GT_BG'].append(im_transform(bg[b_idx, 0], size))
            if is_coarse_mask:
                req_images['Coarse_Mask'].append(np.zeros((size[1], size[0], 3)))
            if is_trimap:
                req_images['Trimap'].append(mask_transform(images['trimap'][b_idx, 0], size))
        srange = range(min(4, s-1))
        for s_idx in srange:
            req_images['RGB'].append(im_transform(images['rgb_query'][b_idx, s_idx], size))
            # if s_idx == 0:
            #     req_images['Mask'].append(np.zeros((size[1], size[0], 3)))
            # else:
            req_images['Mask'].append(mask_transform(images['mask'][b_idx, s_idx], size))
            gt_mask = images['gt_query'][b_idx, s_idx]
            req_images[GT_name].append(mask_transform(gt_mask, size))
            # if is_fg:
            #     req_images['Pred_FG'].append(im_transform(images['pred_fg'][b_idx, s_idx]*gt_mask, size))
            #     req_images['GT_FG'].append(im_transform(images['fg_query'][b_idx, s_idx]*gt_mask, size))
            if is_bg:
                req_images['Pred_BG'].append(im_transform(images['pred_bg'][b_idx, s_idx], size))
                if (bg := images.get('bg', None)) is not None:
                    req_images['GT_BG'].append(im_transform(bg[b_idx, 1+s_idx], size))
            if is_coarse_mask:
                req_images['Coarse_Mask'].append(mask_transform(images['coarse_mask'][b_idx, s_idx], size))
            if is_trimap:
                req_images['Trimap'].append(mask_transform(images['trimap'][b_idx, s_idx], size))

    return get_image_array(req_images, size, key_captions)