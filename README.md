# FTP-VM (CVPR2023)
Implementation and videos of **End-to-End Video Matting With Trimap Propagation** in CVPR2023.

[[CVPR OpenAccess]](https://openaccess.thecvf.com/content/CVPR2023/html/Huang_End-to-End_Video_Matting_With_Trimap_Propagation_CVPR_2023_paper.html) [Paper PDF] [Presentation Video] [Supplementary Video]

# To do
- Clean the training / data processing code
- Readme / Manual
- Reference github repo
- Upload more supplementary videos
- (Possibly) Collaborate with SegmentAnyThing


# Installation
(To be updated more)
For using FTP-VM only, please install the following package via pip.
```
torch
timm
```

# Model Usage
For those who want  use the model via code directly.
```python
import torch
from FTPVM.model import FastTrimapPropagationVideoMatting as FTPVM
model = FTPVM()
model.load_state_dict(torch.load('saves/ftpvm.pth'))
```
Usage
```python
# Images are in [0, 1] with size of (batch, time, channel, height, width)
# Memory has 1 frame per batch, and trimap (mask) has 1 channel.
query_imgs = torch.rand((2, 4, 3, 256, 256))
memory_imgs = torch.rand((2, 1, 3, 256, 256))
memory_trimaps = torch.rand((2, 1, 1, 256, 256))
# General forward
trimaps, boundary_mattes, full_mattes, recurrent_mems = model(query_imgs, memory_imgs, memory_trimaps)
# Forward with RNN memory
trimaps, boundary_mattes, full_mattes, recurrent_mems = model(query_imgs, memory_imgs, memory_trimaps, *recurrent_mems)
# Preserve memory key & values in Memory matching, which is useful in application
memory_key_val = model.encode_imgs_to_value(memory_imgs, memory_trimaps)
trimaps, boundary_mattes, full_mattes, recurrent_mems = model.forward_with_memory(query_imgs, *memory_key_val, *recurrent_mems)
```

# Inference

## Dataset
## Webcam
still not robust enough to webcam frames :(
## Raw video


# Training (TBD)
## Dataset




# Citation
```bibtex
@InProceedings{Huang_2023_CVPR,
    author    = {Huang, Wei-Lun and Lee, Ming-Sui},
    title     = {End-to-End Video Matting With Trimap Propagation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {14337-14347}
}
```
## Useful Repositories
(TBD)
RVM
MiVOS
STCN
