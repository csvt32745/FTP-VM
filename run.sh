python train.py \
--id FTPVM \
--which_model FTPVM \
--num_worker 12 \
--benchmark \
--lr 0.0001 -i 120000 \
--iter_switch_dataset 30000 \
--use_background_video \
-b_seg 8 -b_img_mat 10 -b_vid_mat 4 \
-s_seg 8 -s_img_mat 4 -s_vid_mat 8 \
--seg_cd 20000 --seg_iter 10000 --seg_start 0 --seg_stop 100000 \
--size 480 \
--tvloss_type temp_seg_allclass_weight \
$1