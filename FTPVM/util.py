import torch

def collaborate_fuse(out_glance, out_focus):
    val, idx = torch.sigmoid(out_glance).max(dim=2, keepdim=True) # ch
    # (bg, t, fg)
    tran_mask = idx.clone() == 1
    fg_mask = idx.clone() == 2
    return out_focus*tran_mask + fg_mask

def get_tran_fg_mask_from_logits(logits):
    val, idx = torch.sigmoid(logits).max(dim=2, keepdim=True) # ch
    # (bg, t, fg)
    tran_mask = idx.clone() == 1
    fg_mask = idx.clone() == 2
    return tran_mask, fg_mask

def get_tran_fg_mask_from_trimap(trimap):
    fg_mask = trimap > (1-1e-5)
    tran_mask = (~fg_mask) & (trimap > 1e-5) # ~fg & ~bg
    return tran_mask, fg_mask


def collaborate_fuse_trimap(trimap, out_focus):
    fg_mask = trimap > (1-1e-2)
    tran_mask = (~fg_mask) & (trimap > 1e-2) # ~fg & ~bg
    return out_focus*tran_mask + fg_mask