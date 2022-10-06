
from inference_func import *
from STCNVM.model import *
from STCNVM.module import *
from STCNVM.inference_model import *


inference_model_list = [
    ('STCNFuseMatting', 'STCNFuseMatting', InferenceCoreRecurrentGFM, 
	'./saves/Jul28_13.01.55_STCNFuseMatting/Jul28_13.01.55_STCNFuseMatting_120000.pth'),
    ('STCNFuseMatting_gru_before_fuse', 'STCNFuseMatting_gru_before_fuse', InferenceCoreRecurrentGFM, 
	'./saves/Jul28_13.01.51_STCNFuseMatting_gru_before_fuse/Jul28_13.01.51_STCNFuseMatting_gru_before_fuse_120000.pth'),    
    ('STCNFuseMatting_fuse=naive', 'STCNFuseMatting_fuse=naive', InferenceCoreRecurrentGFM, 
	'./saves/Jul29_09.53.05_STCNFuseMatting_fuse=naive/Jul29_09.53.05_STCNFuseMatting_fuse=naive_120000.pth'),
	('STCNFuseMatting_480', 'STCNFuseMatting', InferenceCoreRecurrentGFM, 
	'./saves/Jul29_02.52.18_STCNFuseMatting_480/Jul29_02.52.18_STCNFuseMatting_480_120000.pth'),
    
	('STCNFuseMatting_fuse=bn', 'STCNFuseMatting_fuse=bn', InferenceCoreRecurrentGFM, 
	'./saves/Jul30_00.02.43_STCNFuseMatting_fuse=bn/Jul30_00.02.43_STCNFuseMatting_fuse=bn_120000.pth'),
	# ('STCNFuseMatting_fuse=fullres', 'STCNFuseMatting_fuse=fullres', InferenceCoreRecurrentGFM, 
	# './saves/Jul29_16.14.22_STCNFuseMatting_fuse=fullres/Jul29_16.14.22_STCNFuseMatting_fuse=fullres_120000.pth'),
	('STCNFuseMatting_randtri', 'STCNFuseMatting', InferenceCoreRecurrentGFM, 
	'./saves/Jul29_15.40.06_STCNFuseMatting_randtri/Jul29_15.40.06_STCNFuseMatting_randtri_120000.pth'),
    

    ('STCNFuseMatting_fuse=fullgate', 'STCNFuseMatting_fuse=fullgate', InferenceCoreRecurrentGFM, 
	'./saves/Aug03_10.42.16_STCNFuseMatting_fuse=fullgate/Aug03_10.42.16_STCNFuseMatting_fuse=fullgate_120000.pth'),
    # ('STCNFuseMatting_fuse=naive_480', 'STCNFuseMatting_fuse=naive', InferenceCoreRecurrentGFM, 
	# './saves/Aug03_00.30.22_STCNFuseMatting_fuse=naive_480/Aug03_00.30.22_STCNFuseMatting_fuse=naive_480_120000.pth'),
	('STCNFuseMatting_fuse=intrimap_only', 'STCNFuseMatting_fuse=intrimap_only', InferenceCoreRecurrentGFM, 
	'./saves/Aug02_16.46.55_STCNFuseMatting_fuse=intrimap_only/Aug02_16.46.55_STCNFuseMatting_fuse=intrimap_only_120000.pth'),
    ('STCNFuseMatting_fuse=small', 'STCNFuseMatting_fuse=small', InferenceCoreRecurrentGFM, 
	'./saves/Aug01_21.53.33_STCNFuseMatting_fuse=small/Aug01_21.53.33_STCNFuseMatting_fuse=small_120000.pth'),
    ('STCNFuseMatting_480_normalce', 'STCNFuseMatting', InferenceCoreRecurrentGFM, 
	'./saves/Aug01_21.34.56_STCNFuseMatting_480_normalce/Aug01_21.34.56_STCNFuseMatting_480_normalce_120000.pth'),
	('STCNFuseMatting_big', 'STCNFuseMatting_big', InferenceCoreRecurrentGFM, 
	'./saves/Aug01_21.34.51_STCNFuseMatting_big/Aug01_21.34.51_STCNFuseMatting_big_120000.pth'),
	('STCNFuseMatting_ytvos', 'STCNFuseMatting', InferenceCoreRecurrentGFM, 
	'./saves/Aug01_13.05.05_STCNFuseMatting_ytvos/Aug01_13.05.05_STCNFuseMatting_ytvos_120000.pth'),
	('STCNFuseMatting_fullres_mat', 'STCNFuseMatting_fullres_mat', InferenceCoreRecurrentGFM, 
	'./saves/Aug01_13.03.19_STCNFuseMatting_fullres_mat/Aug01_13.03.19_STCNFuseMatting_fullres_mat_120000.pth'),

    ('STCNFuseMatting_fuse=bn_480', 'STCNFuseMatting_fuse=bn', InferenceCoreRecurrentGFM, 
	'./saves/Aug04_19.01.22_STCNFuseMatting_fuse=bn_480/Aug04_19.01.22_STCNFuseMatting_fuse=bn_480_120000.pth'),
    
	('STCNFuseMatting_fuse=intrimap_only_fullres', 'STCNFuseMatting_fuse=intrimap_only_fullres', InferenceCoreRecurrentGFM, 
	'./saves/Aug04_02.38.27_STCNFuseMatting_fuse=intrimap_only_fullres/Aug04_02.38.27_STCNFuseMatting_fuse=intrimap_only_fullres_120000.pth'),
    ('STCNFuseMatting_fuse=fullres', 'STCNFuseMatting_fuse=fullres', InferenceCoreRecurrentGFM, 
	'./saves/Aug05_02.48.42_STCNFuseMatting_fuse=fullres/Aug05_02.48.42_STCNFuseMatting_fuse=fullres_120000.pth'),
    ('STCNFuseMatting_fuse=naive_480', 'STCNFuseMatting_fuse=naive', InferenceCoreRecurrentGFM, 
	'./saves/Aug06_01.31.09_STCNFuseMatting_fuse=naive_480/Aug06_01.31.09_STCNFuseMatting_fuse=naive_480_120000.pth'),
    ('STCNFuseMatting_fuse=bn_wo_consis_480', 'STCNFuseMatting_fuse=bn', InferenceCoreRecurrentGFM, 
	'./saves/Aug06_01.31.09_STCNFuseMatting_fuse=bn_wo_consis_480/Aug06_01.31.09_STCNFuseMatting_fuse=bn_wo_consis_480_120000.pth'),
    ('STCNFuseMatting_1xseg_4x2mat', 'STCNFuseMatting_1xseg_4x2mat', InferenceCoreRecurrentGFM, 
	'./saves/Aug06_16.19.43_STCNFuseMatting_1xseg_4x2mat/Aug06_16.19.43_STCNFuseMatting_1xseg_4x2mat_120000.pth'),
	# ('STCNFuseMatting_fuse=gn_480', 'STCNFuseMatting_fuse=gn', InferenceCoreRecurrentGFM, 
	# './saves/Aug06_16.19.33_STCNFuseMatting_fuse=gn_480/Aug06_16.19.33_STCNFuseMatting_fuse=gn_480_120000.pth'),
    
    ('STCNFuseMatting_fuse=bn_seg_consis_correctonly_480', 'STCNFuseMatting_fuse=bn', InferenceCoreRecurrentGFM, 
	'./saves/Aug07_18.16.20_STCNFuseMatting_fuse=bn_seg_consis_correctonly_480/Aug07_18.16.20_STCNFuseMatting_fuse=bn_seg_consis_correctonly_480_120000.pth'),

    ('STCNFuseMatting_fuse=gn_480', 'STCNFuseMatting_fuse=gn', InferenceCoreRecurrentGFM, 
	'./saves/Aug08_09.07.55_STCNFuseMatting_fuse=gn_480/Aug08_09.07.55_STCNFuseMatting_fuse=gn_480_120000.pth'),
    ('STCNFuseMatting_fullresseg', 'STCNFuseMatting_fullresseg', InferenceCoreRecurrentGFM, 
	'./saves/Aug08_09.07.55_STCNFuseMatting_fullresseg/Aug08_09.07.55_STCNFuseMatting_fullresseg_120000.pth'),
    ('STCNFuseMatting_fuse=bn2_480', 'STCNFuseMatting_fuse=bn2', InferenceCoreRecurrentGFM, 
	'./saves/Aug08_13.41.34_STCNFuseMatting_fuse=bn2_480/Aug08_13.41.34_STCNFuseMatting_fuse=bn2_480_120000.pth'),
    ('STCNFuseMatting_fullres_mat3_480', 'STCNFuseMatting_fullres_mat3', InferenceCoreRecurrentGFM, 
	'./saves/Aug08_13.41.27_STCNFuseMatting_fullres_mat3_480/Aug08_13.41.27_STCNFuseMatting_fullres_mat3_480_120000.pth'),

    ('STCNFuseMatting_fullres_480_none_temp_seg', 'STCNFuseMatting_fullres_gn', InferenceCoreRecurrentGFM, 
	'./saves/Aug09_13.26.00_STCNFuseMatting_fullres_480_none_temp_seg/Aug09_13.26.00_STCNFuseMatting_fullres_480_none_temp_seg_120000.pth'),
    ('STCNFuseMatting_fullres_480_temp_seg_allclass_weight_x1', 'STCNFuseMatting_fullres_gn', InferenceCoreRecurrentGFM, 
	'./saves/Aug10_08.09.55_STCNFuseMatting_fullres_480_temp_seg_allclass_weight_x1/Aug10_08.09.55_STCNFuseMatting_fullres_480_temp_seg_allclass_weight_x1_120000.pth'),

    ('STCNFuseMatting_fullres_matnaive', 'STCNFuseMatting_fullres_matnaive', InferenceCoreRecurrentGFM, 
	'./saves/Aug11_15.46.26_STCNFuseMatting_fullres_matnaive/Aug11_15.46.26_STCNFuseMatting_fullres_matnaive_120000.pth'),
	('STCNFuseMatting_fullres_matnaive_80k', 'STCNFuseMatting_fullres_matnaive', InferenceCoreRecurrentGFM, 
	'./saves/Aug11_00.38.33_STCNFuseMatting_fullres_matnaive/Aug11_00.38.33_STCNFuseMatting_fullres_matnaive_80000.pth'),

	
    ('STCNFuseMatting_fullres_matnaive_480_temp_seg', 'STCNFuseMatting_fullres_matnaive', InferenceCoreRecurrentGFM, 
	'./saves/Aug11_22.07.52_STCNFuseMatting_fullres_matnaive_480_temp_seg/Aug11_22.07.52_STCNFuseMatting_fullres_matnaive_480_temp_seg_120000.pth'),
    ('STCNFuseMatting_fullres_matnaive_seg2_480_temp_seg_allclass', 'STCNFuseMatting_fullres_matnaive_seg2', InferenceCoreRecurrentGFM, 
	'./saves/Aug12_01.18.24_STCNFuseMatting_fullres_matnaive_seg2_480_temp_seg_allclass/Aug12_01.18.24_STCNFuseMatting_fullres_matnaive_seg2_480_temp_seg_allclass_120000.pth'),
    
    ('STCNFuseMatting_fullres_gn_3chmask', 'STCNFuseMatting_fullres_gn_3chmask', InferenceCoreRecurrent3chTrimap, 
	'./saves/Aug12_14.39.08_STCNFuseMatting_fullres_gn_3chmask/Aug12_14.39.08_STCNFuseMatting_fullres_gn_3chmask_120000.pth'),
    
	# ==== Applied models ========================
    ('STCNFuseMatting_SingleDec', 'STCNFuseMatting_SingleDec', InferenceCoreRecurrent, 
	'./saves/Aug12_21.48.33_STCNFuseMatting_SingleDec/Aug12_21.48.33_STCNFuseMatting_SingleDec_120000.pth'),
    ('STCNFuseMatting_SingleDec_big', 'STCNFuseMatting_SingleDec_big', InferenceCoreRecurrent, 
	'./saves/Aug13_16.45.59_STCNFuseMatting_SingleDec_big/Aug13_16.45.59_STCNFuseMatting_SingleDec_big_120000.pth'),
	('STCNFuseMatting_SameDec_480', 'STCNFuseMatting_SameDec', InferenceCoreRecurrentGFM, 
	'./saves/Aug04_19.01.14_STCNFuseMatting_SameDec_480/Aug04_19.01.14_STCNFuseMatting_SameDec_480_120000.pth'),
	('RVM_vm108', 'RVM', InferenceCoreRecurrent, 
	'./saves/Aug14_21.58.39_RVM/Aug14_21.58.39_RVM_120000.pth'),
	
	
	('STCNFuseMatting_fullres_matnaive_seg2_480_temp_seg_allclass_weight', 'STCNFuseMatting_fullres_matnaive_seg2', InferenceCoreRecurrentGFM, 
	'./saves/Aug14_03.39.50_STCNFuseMatting_fullres_matnaive_seg2_480_temp_seg_allclass_weight/Aug14_03.39.50_STCNFuseMatting_fullres_matnaive_seg2_480_temp_seg_allclass_weight_120000.pth'),

	('STCNFuseMatting_fullres_matnaive_temp_seg_allclass_weight', 'STCNFuseMatting_fullres_matnaive', InferenceCoreRecurrentGFM, 
	'./saves/Aug14_16.02.10_STCNFuseMatting_fullres_matnaive_temp_seg_allclass_weight/Aug14_16.02.10_STCNFuseMatting_fullres_matnaive_temp_seg_allclass_weight_120000.pth'),
	('STCNFuseMatting_fullres_matnaive_l2attn', 'STCNFuseMatting_fullres_matnaive_l2attn', InferenceCoreRecurrentGFM, 
	'./saves/Aug16_12.23.41_STCNFuseMatting_fullres_matnaive_l2attn/Aug16_12.23.41_STCNFuseMatting_fullres_matnaive_l2attn_120000.pth'),
	('STCNFuseMatting_fullres_matnaive2_seg3', 'STCNFuseMatting_fullres_matnaive2_seg3', InferenceCoreRecurrentGFM, 
	'./saves/Aug16_12.23.43_STCNFuseMatting_fullres_matnaive2_seg3/Aug16_12.23.43_STCNFuseMatting_fullres_matnaive2_seg3_120000.pth'),
	('STCNFuseMatting_fullres_matnaive_naivefuse', 'STCNFuseMatting_fullres_matnaive_naivefuse', InferenceCoreRecurrentGFM, 
	'./saves/Aug17_12.49.34_STCNFuseMatting_fullres_matnaive_naivefuse/Aug17_12.49.34_STCNFuseMatting_fullres_matnaive_naivefuse_120000.pth'),
	('STCNFuseMatting_fullres_matnaive_l2gate', 'STCNFuseMatting_fullres_matnaive_l2gate', InferenceCoreRecurrentGFM, 
	'./saves/Aug17_12.49.34_STCNFuseMatting_fullres_matnaive_l2gate/Aug17_12.49.34_STCNFuseMatting_fullres_matnaive_l2gate_120000.pth'),
	('STCNFuseMatting_fullres_matnaive_backbonefuse', 'STCNFuseMatting_fullres_matnaive_backbonefuse', InferenceCoreRecurrentGFM, 
	'./saves/Aug18_18.38.37_STCNFuseMatting_fullres_matnaive_backbonefuse/Aug18_18.38.37_STCNFuseMatting_fullres_matnaive_backbonefuse_120000.pth'),
	('STCNFuseMatting_fullres_matnaive_multiobj', 'STCNFuseMatting_fullres_matnaive', InferenceCoreRecurrentGFM, 
	'./saves/Aug18_18.38.39_STCNFuseMatting_fullres_matnaive_multiobj/Aug18_18.38.39_STCNFuseMatting_fullres_matnaive_multiobj_120000.pth'),
	('STCNFuseMatting_fullres_matnaive_none_temp_seg', 'STCNFuseMatting_fullres_matnaive', InferenceCoreRecurrentGFM, 
	'./saves/Aug18_18.54.49_STCNFuseMatting_fullres_matnaive_none_temp_seg/Aug18_18.54.49_STCNFuseMatting_fullres_matnaive_none_temp_seg_120000.pth'),
	('STCNFuseMatting_fullres_matnaive_wodata_seg_d646', 'STCNFuseMatting_fullres_matnaive', InferenceCoreRecurrentGFM, 
	'./saves/Aug20_03.54.13_STCNFuseMatting_fullres_matnaive_wodata_seg_d646/Aug20_03.54.13_STCNFuseMatting_fullres_matnaive_wodata_seg_d646_120000.pth'),
	('STCNFuseMatting_fullres_matnaive_memalpha', 'STCNFuseMatting_fullres_matnaive_memalpha', InferenceCoreRecurrentMemAlpha, 
	'./saves/Aug20_03.54.00_STCNFuseMatting_fullres_matnaive_memalpha/Aug20_03.54.00_STCNFuseMatting_fullres_matnaive_memalpha_120000.pth'),
	('STCNFuseMatting_fullres_matnaive_ytvos', 'STCNFuseMatting_fullres_matnaive', InferenceCoreRecurrentGFM, 
	'./saves/Aug19_23.16.50_STCNFuseMatting_fullres_matnaive_ytvos/Aug19_23.16.50_STCNFuseMatting_fullres_matnaive_ytvos_120000.pth'),
	('STCNFuseMatting_fullres_matnaive_ppm1236', 'STCNFuseMatting_fullres_matnaive_ppm1236', InferenceCoreRecurrentGFM, 
	'./saves/Aug20_23.53.50_STCNFuseMatting_fullres_matnaive_ppm1236/Aug20_23.53.50_STCNFuseMatting_fullres_matnaive_ppm1236_120000.pth'),
	('2stage', '2stage', InferenceCoreRecurrentGFM, 
	'./saves/Aug21_04.08.42_2stage/Aug21_04.08.42_2stage_120000.pth'),
	('2stage_seg4x', '2stage_seg4x', InferenceCoreRecurrentGFM, 
	'./saves/Aug26_22.03.23_2stage_seg4x/Aug26_22.03.23_2stage_seg4x_120000.pth'),
	('STCNFuseMatting_fullres_matnaive_woPPM', 'STCNFuseMatting_fullres_matnaive_woPPM', InferenceCoreRecurrentGFM, 
	'./saves/Aug22_01.43.29_STCNFuseMatting_fullres_matnaive_woPPM/Aug22_01.43.29_STCNFuseMatting_fullres_matnaive_woPPM_120000.pth'),
	# ('STCNFuseMatting_fullres_matnaive_woCBAM', 'STCNFuseMatting_fullres_matnaive_woCBAM', InferenceCoreRecurrentGFM, 
	# './saves/Aug22_01.43.27_STCNFuseMatting_fullres_matnaive_woCBAM/Aug22_01.43.27_STCNFuseMatting_fullres_matnaive_woCBAM_120000.pth'),
	
	('STCNFuseMatting_fullres_matnaive2', 'STCNFuseMatting_fullres_matnaive2', InferenceCoreRecurrentGFM, 
	'./saves/Aug23_05.58.49_STCNFuseMatting_fullres_matnaive2/Aug23_05.58.49_STCNFuseMatting_fullres_matnaive2_120000.pth'),
	('STCNFuseMatting_fullres_matnaive_wogru', 'STCNFuseMatting_fullres_matnaive_wogru', InferenceCoreRecurrentGFM, 
	'./saves/Aug23_05.58.47_STCNFuseMatting_fullres_matnaive_wogru/Aug23_05.58.47_STCNFuseMatting_fullres_matnaive_wogru_120000.pth'),
	('STCNFuseMatting_fullres_matnaive_temp_seg_allclass', 'STCNFuseMatting_fullres_matnaive', InferenceCoreRecurrentGFM, 
	'./saves/Aug23_05.57.53_STCNFuseMatting_fullres_matnaive_temp_seg_allclass/Aug23_05.57.53_STCNFuseMatting_fullres_matnaive_temp_seg_allclass_120000.pth'),
	('STCNFuseMatting_fullres_matnaive_fullmatte', 'STCNFuseMatting_fullres_matnaive', InferenceCoreRecurrentGFM, 
	'./saves/Aug24_16.43.09_STCNFuseMatting_fullres_matnaive_fullmatte/Aug24_16.43.09_STCNFuseMatting_fullres_matnaive_fullmatte_120000.pth'),
	('STCNFuseMatting_fullres_matnaive4', 'STCNFuseMatting_fullres_matnaive4', InferenceCoreRecurrentGFM, 
	'./saves/Aug24_22.16.02_STCNFuseMatting_fullres_matnaive4/Aug24_22.16.02_STCNFuseMatting_fullres_matnaive4_120000.pth'),
	('STCNFuseMatting_fullres_matnaive3', 'STCNFuseMatting_fullres_matnaive3', InferenceCoreRecurrentGFM, 
	'./saves/Aug25_01.26.01_STCNFuseMatting_fullres_matnaive3/Aug25_01.26.01_STCNFuseMatting_fullres_matnaive3_120000.pth'),
	('STCNFuseMatting_fullres_matnaive_same_memque0.5', 'STCNFuseMatting_fullres_matnaive', InferenceCoreRecurrentGFM, 
	'./saves/Aug25_16.24.24_STCNFuseMatting_fullres_matnaive_same_memque0.5/Aug25_16.24.24_STCNFuseMatting_fullres_matnaive_same_memque0.5_120000.pth'),
	('STCNFuseMatting_fullres_matnaive_normalce', 'STCNFuseMatting_fullres_matnaive', InferenceCoreRecurrentGFM, 
	'./saves/Aug25_18.34.40_STCNFuseMatting_fullres_matnaive_normalce/Aug25_18.34.40_STCNFuseMatting_fullres_matnaive_normalce_120000.pth'),
	('STCNFuseMatting_fullres_matnaive_normalce_nonetemp', 'STCNFuseMatting_fullres_matnaive', InferenceCoreRecurrentGFM, 
	'./saves/Aug26_19.03.29_STCNFuseMatting_fullres_matnaive_normalce_nonetemp/Aug26_19.03.29_STCNFuseMatting_fullres_matnaive_normalce_nonetemp_120000.pth'),
	('STCNFuseMatting_fullres_matnaive_same_memque0.1', 'STCNFuseMatting_fullres_matnaive', InferenceCoreRecurrentGFM, 
	'./saves/Aug28_00.30.21_STCNFuseMatting_fullres_matnaive_same_memque0.1/Aug28_00.30.21_STCNFuseMatting_fullres_matnaive_same_memque0.1_120000.pth'),
	('STCNFuseMatting_fullres_matnaive_retry', 'STCNFuseMatting_fullres_matnaive', InferenceCoreRecurrentGFM, 
	'./saves/Aug28_17.58.17_STCNFuseMatting_fullres_matnaive_retry/Aug28_17.58.17_STCNFuseMatting_fullres_matnaive_retry_120000.pth'),
	('STCNFuseMatting_fullres_matnaive_tempweightl2', 'STCNFuseMatting_fullres_matnaive', InferenceCoreRecurrentGFM, 
	'./saves/Aug29_18.20.57_STCNFuseMatting_fullres_matnaive_tempweightl2/Aug29_18.20.57_STCNFuseMatting_fullres_matnaive_tempweightl2_120000.pth'),
	('STCNFuseMatting_fullres_matnaive_focal_weight', 'STCNFuseMatting_fullres_matnaive', InferenceCoreRecurrentGFM, 
	'./saves/Aug29_21.57.02_STCNFuseMatting_fullres_matnaive_focal_weight/Aug29_21.57.02_STCNFuseMatting_fullres_matnaive_focal_weight_120000.pth'),
	('STCNFuseMatting_fullres_matnaive_wo_seg', 'STCNFuseMatting_fullres_matnaive', InferenceCoreRecurrentGFM, 
	'./saves/Sep04_22.51.23_STCNFuseMatting_fullres_matnaive_wo_seg/Sep04_22.51.23_STCNFuseMatting_fullres_matnaive_wo_seg_120000.pth'),
	('STCNFuseMatting_fullres_matnaive_woCBAMPPM', 'STCNFuseMatting_fullres_matnaive_woCBAMPPM', InferenceCoreRecurrentGFM, 
	'./saves/Sep14_12.05.40_STCNFuseMatting_fullres_matnaive_woCBAMPPM/Sep14_12.05.40_STCNFuseMatting_fullres_matnaive_woCBAMPPM_120000.pth'),
	('STCNFuseMatting_fullres_matnaive_woCBAM', 'STCNFuseMatting_fullres_matnaive_woCBAM', InferenceCoreRecurrentGFM, 
	'./saves/Sep14_12.05.21_STCNFuseMatting_fullres_matnaive_woCBAM/Sep14_12.05.21_STCNFuseMatting_fullres_matnaive_woCBAM_120000.pth'),
]   



if __name__ == '__main__':
    print("="*50)
    print("Check if the model file exists of not...")
    for v in inference_model_list:
        if not os.path.isfile(v[-1]):
            print(v[0], v[-1])
    print("OK.")
    print("="*50)
    print("Check if the model name duplicates...")
    check_name = set()
    for i in inference_model_list:
        if i[0] in check_name:
            print(i[0], " is duplicated!")
        else:
            check_name.add(i[0])
    print("OK.")
    print("="*50)
    
inference_model_list = {i[0]: i for i in inference_model_list}