import torch

def collaborate_fuse(out_glance, out_focus):
    val, idx = torch.sigmoid(out_glance).max(dim=2, keepdim=True) # ch
    # (bg, t, fg)
    tran_mask = idx.clone() == 1
    fg_mask = idx.clone() == 2
    return out_focus*tran_mask + fg_mask
