CUDA_VISIBLE_DEVICES=0 python train.py \
-b 4 --lr 0.0001 -i 120000 \
--id STCNFuseMatting_fullres_matnaive \
--which_model STCNFuseMatting_fullres_matnaive \
--num_worker 12 \
--benchmark \
--iter_switch_dataset 30000 \
--seg_cd 20000 --seg_iter 10000 --seg_start 0 --seg_stop 100000 \
--size 480 \
--tvloss_type temp_seg_allclass_weight \
$1