# FTP-VM

## Training
Just run `run.sh` or 

```sh
python train.py \
  -b 4 --lr 0.0001 -i 120000 \
  --id STCNFuseMatting_fullres_matnaive \
  --which_model STCNFuseMatting_fullres_matnaive \
  --num_worker $num_workers \
  --benchmark \
  --iter_switch_dataset 30000 \
  --seg_cd 20000 --seg_iter 10000 --seg_start 0 --seg_stop 100000 \
  --size 480
```

## Evaluation

Put `(experient name, model name, inference core, model weight path)` into `inference_model_list.py`, and add the `experient name` into `inference_normal.py`.

Then execute
```sh
python inference_normal.py \
  --gpu $gpu \
  --n_workers $n_workers \
  --batch_size $batch_size
```

## Testing

```sh
python inference_footages.py \
    --gpu $gpu_id \
    --root $input_root \
    --out_root $output_root
```
Folder format
- `$input_root`
  - `${vid_name_1}.mp4`
  - `${vid_name_1}_thumbnail.png`
  - `${vid_name_1}_trimap.png`
  - `${vid_name_2}.mp4`
  - `${vid_name_2}_thumbnail.png`
  - `${vid_name_2}_trimap.png`
  - ...
- `$output_root`
  - `${vid_name_1}_com.mp4`
  - `${vid_name_1}_fgr.mp4`
  - `${vid_name_1}_pha.mp4`
  - `${vid_name_2}_com.mp4`
  - `${vid_name_2}_fgr.mp4`
  - `${vid_name_2}_pha.mp4`

`${vid_name}.mp4` is the input video.

`${vid_name}_thumbnail.png` is the memory frame, and `${vid_name}_trimap.png` is the memory trimap.

It will output composition, foreground and alpha video.

## Interactive Demo

Check your webcam is enabled and run 
```sh
python webcam.py
```