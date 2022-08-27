"""
python inference.py \
    --variant mobilenetv3 \
    --checkpoint "CHECKPOINT" \
    --device cuda \
    --input-source "input.mp4" \
    --output-type video \
    --output-composition "composition.mp4" \
    --output-alpha "alpha.mp4" \
    --output-foreground "foreground.mp4" \
    --output-video-mbps 4 \
    --seq-chunk 1
"""

import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Optional, Tuple
from tqdm.auto import tqdm
from PIL import Image

from inference_io import VideoReader, VideoWriter, ImageSequenceReader, ImageSequenceWriter
from inference_model_list import inference_model_list
from model.which_model import get_model_by_string
from torch.nn import functional as F

def convert_video(model,
                  input_source: str,
                  memory_img: str,
                  memory_mask: Optional[str] = None,
                  input_resize: Optional[Tuple[int, int]] = None,
                  downsample_ratio: Optional[float] = None,
                  output_type: str = 'video',
                  output_composition: Optional[str] = None,
                  output_alpha: Optional[str] = None,
                  output_foreground: Optional[str] = None,
                  output_video_mbps: Optional[float] = None,
                  seq_chunk: int = 1,
                  num_workers: int = 0,
                  progress: bool = True,
                  device: Optional[str] = None,
                  dtype: Optional[torch.dtype] = torch.float32,
                  target_size: int = 1024):
    
    """
    Args:
        input_source:A video file, or an image sequence directory. Images must be sorted in accending order, support png and jpg.
        input_resize: If provided, the input are first resized to (w, h).
        downsample_ratio: The model's downsample_ratio hyperparameter. If not provided, model automatically set one.
        output_type: Options: ["video", "png_sequence"].
        output_composition:
            The composition output path. File path if output_type == 'video'. Directory path if output_type == 'png_sequence'.
            If output_type == 'video', the composition has green screen background.
            If output_type == 'png_sequence'. the composition is RGBA png images.
        output_alpha: The alpha output from the model.
        output_foreground: The foreground output from the model.
        seq_chunk: Number of frames to process at once. Increase it for better parallelism.
        num_workers: PyTorch's DataLoader workers. Only use >0 for image input.
        progress: Show progress bar.
        device: Only need to manually provide if model is a TorchScript freezed model.
        dtype: Only need to manually provide if model is a TorchScript freezed model.
    """
    
    assert downsample_ratio is None or (downsample_ratio > 0 and downsample_ratio <= 1), 'Downsample ratio must be between 0 (exclusive) and 1 (inclusive).'
    assert any([output_composition, output_alpha, output_foreground]), 'Must provide at least one output.'
    assert output_type in ['video', 'png_sequence'], 'Only support "video" and "png_sequence" output modes.'
    assert seq_chunk >= 1, 'Sequence chunk must be >= 1'
    assert num_workers >= 0, 'Number of workers must be >= 0'
    
    # Initialize transform
    if input_resize is not None:
        transform = transforms.Compose([
            transforms.Resize(input_resize[::-1]),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.ToTensor()

    # Initialize reader
    if os.path.isfile(input_source):
        source = VideoReader(input_source, transform)
    else:
        source = ImageSequenceReader(input_source, transform)
    
    reader = DataLoader(source, batch_size=seq_chunk, pin_memory=True, num_workers=num_workers)
    

    # Initialize writers
    if output_type == 'video':
        frame_rate = source.frame_rate if isinstance(source, VideoReader) else 30
        output_video_mbps = 1 if output_video_mbps is None else output_video_mbps
        if output_composition is not None:
            writer_com = VideoWriter(
                path=output_composition,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000))
        if output_alpha is not None:
            writer_pha = VideoWriter(
                path=output_alpha,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000))
        if output_foreground is not None:
            writer_fgr = VideoWriter(
                path=output_foreground,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000))
    else:
        if output_composition is not None:
            writer_com = ImageSequenceWriter(output_composition, 'png')
        if output_alpha is not None:
            writer_pha = ImageSequenceWriter(output_alpha, 'png')
        if output_foreground is not None:
            writer_fgr = ImageSequenceWriter(output_foreground, 'png')

    # Inference
    model = model.eval()
    if device is None or dtype is None:
        param = next(model.parameters())
        dtype = param.dtype
        device = param.device
    
    m_img = transform(Image.open(memory_img)).unsqueeze(0).unsqueeze(0).to(device)
    if memory_mask is not None and memory_mask != '':
        m_mask = transform(Image.open(memory_mask).convert(mode='L')).unsqueeze(0).unsqueeze(0).to(device)
    else:
        print("Memory frame is background!")
        shape = list(m_img.shape) # b t c h w
        shape[2] = 1
        m_mask = torch.zeros(shape, dtype=m_img.dtype, device=m_img.device)
    print(m_img.shape, m_mask.shape)
    # if (output_composition is not None) and (output_type == 'video'):
    bgr = torch.tensor([120, 255, 155], device=device, dtype=dtype).div(255).view(1, 1, 3, 1, 1)
    
    # print(downsample_ratio)
    try:
        with torch.no_grad():
            bar = tqdm(total=len(source), disable=not progress, dynamic_ncols=True)
            rec = model.default_rec
            memory = None
            for src in reader:

                if downsample_ratio is None:
                    downsample_ratio = auto_downsample_ratio(*src.shape[2:], target=target_size)
                    print(downsample_ratio)
                if memory is None:
                    memory = model.encode_imgs_to_value(m_img, m_mask, downsample_ratio=downsample_ratio)

                src = src.to(device, dtype, non_blocking=True).unsqueeze(0) # [B, T, C, H, W]
                trimap, matte, pha, rec, _ = model.forward_with_memory(src, *memory, *rec, downsample_ratio=downsample_ratio)
                pha = pha.clamp(0, 1)
                trimap = seg_to_trimap(trimap)

                fgr = src * pha + bgr * (1 - pha)
                
                if output_foreground is not None:
                    writer_fgr.write(fgr[0])
                if output_alpha is not None:
                    writer_pha.write(pha[0])

                if output_composition is not None:
                    # t, c, h, w
                    target_height = 540
                    rgb = torch.cat([src[0], fgr[0]], dim=3)
                    ratio = target_height / rgb.size(2)
                    rgb = F.interpolate(rgb, scale_factor=(ratio, ratio))
                    pha = F.interpolate(pha[0], scale_factor=(ratio, ratio))

                    ratio = target_height / trimap.size(3)
                    trimap = F.interpolate(trimap[0], scale_factor=(ratio, ratio))
                    
                    mask = torch.repeat_interleave(torch.cat([trimap, pha], dim=3), 3, dim=1)
                    writer_com.write(torch.cat([rgb, mask], dim=2))
                
                bar.update(src.size(1))

    finally:
        # Clean up
        if output_composition is not None:
            writer_com.close()
        if output_alpha is not None:
            writer_pha.close()
        if output_foreground is not None:
            writer_fgr.close()

def seg_to_trimap(logit):
    val, idx = torch.sigmoid(logit).max(dim=2, keepdim=True) # ch
    # (bg, t, fg)
    tran_mask = idx == 1
    fg_mask = idx == 2
    return tran_mask*0.5 + fg_mask

def auto_downsample_ratio(h, w, target=512):
    """
    Automatically find a downsample ratio so that the largest side of the resolution be 512px.
    """
    # return min(target / min(h, w), 1)
    ratio = min(target / max(h, w), 1)
    print('auto size h, w = %f, %f' % (h*ratio, w*ratio))
    return ratio

    
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    # parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--input-source', type=str, required=True)
    parser.add_argument('--input-resize', type=int, default=None, nargs=2)
    parser.add_argument('--memory_img', type=str, required=True)
    parser.add_argument('--memory_mask', type=str, default='')
    parser.add_argument('--downsample-ratio', type=float)
    parser.add_argument('--output-composition', type=str)
    parser.add_argument('--output-alpha', type=str)
    parser.add_argument('--output-foreground', type=str)
    parser.add_argument('--output-type', type=str, default='video', choices=['video', 'png_sequence'])
    parser.add_argument('--output-video-mbps', type=int, default=1)
    parser.add_argument('--seq-chunk', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--disable-progress', action='store_true')
    parser.add_argument('--target_size', type=int, default=1024)
    args = parser.parse_args()
    
    device = 'cuda:%d' % args.gpu
    model_attr = inference_model_list[args.model]
    model = get_model_by_string(model_attr[1])().to(device=device)
    def check_and_load_model_dict(model, state_dict: dict):
        for k in set(state_dict.keys()) - set(model.state_dict().keys()):
            if 'refiner' in k:
                print('remove refiner', k)
                state_dict.pop(k)
        model.load_state_dict(state_dict)
    check_and_load_model_dict(model, torch.load(model_attr[3]))
    # print(model)
    # converter = Converter(args.variant, args.checkpoint, args.device)
    convert_video(
        model,
        input_source=args.input_source,
        input_resize=args.input_resize,
        memory_img = args.memory_img,
        memory_mask = args.memory_mask,
        downsample_ratio=args.downsample_ratio,
        output_type=args.output_type,
        output_composition=args.output_composition,
        output_alpha=args.output_alpha,
        output_foreground=args.output_foreground,
        output_video_mbps=args.output_video_mbps,
        seq_chunk=args.seq_chunk,
        num_workers=args.num_workers,
        progress=not args.disable_progress,
        device=device,
        target_size=args.target_size,
    )
    
    
