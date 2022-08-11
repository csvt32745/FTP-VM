
from inference_func import *
from STCNVM.model import *
from STCNVM.module import *
from STCNVM.inference_model import *


inference_model_list = [
    # ('GFM_FuseVM', 'GFM_FuseVM=GFMFuse3', InferenceCoreRecurrentGFM, 
    # '/home/csvt32745/matte/MaskPropagation/saves/Jul12_22.35.05_GFM_FuseVM/Jul12_22.35.05_GFM_FuseVM_120000.pth'),
    ('DualVM', lambda: DualMattingNetwork(is_output_fg=False), InferenceCoreRecurrent, 
    '/home/csvt32745/matte/MaskPropagation/saves/Jul01_23.59.45_DualVM/Jul01_23.59.45_DualVM_120000.pth'), 
        
    ('GFM_GatedFuseVM_convgru', lambda: GFM_GatedFuseVM(gru=ConvGRU), InferenceCoreDoubleRecurrentGFM,
    '/home/csvt32745/matte/MaskPropagation/saves/Jul04_10.47.39_GFM_GatedFuseVM_convgru/Jul04_10.47.39_GFM_GatedFuseVM_convgru_120000.pth'),
    # ('GFM_GatedFuseVM_convgru', GFM_GatedFuseVM_big, InferenceCoreDoubleRecurrentGFM,
    # '/home/csvt32745/matte/MaskPropagation/saves/Jul11_23.12.45_GFM_GatedFuseVM_big_convgru/Jul11_23.12.45_GFM_GatedFuseVM_big_convgru_120000.pth'),
    
    ('GFM_GatedFuseVM_AttnGRU', lambda: GFM_GatedFuseVM(gru=AttnGRU), InferenceCoreDoubleRecurrentGFM,
    '/home/csvt32745/matte/MaskPropagation/saves/Jul05_03.14.15_GFM_GatedFuseVM_attngru/Jul05_03.14.15_GFM_GatedFuseVM_attngru_120000.pth'),
    ('GFM_GatedFuseVM_AttnGRU2', lambda: GFM_GatedFuseVM(gru=AttnGRU2), InferenceCoreDoubleRecurrentGFM,
    '/home/csvt32745/matte/MaskPropagation/saves/Jul04_10.46.48_GFM_GatedFuseVM_attngru2/Jul04_10.46.48_GFM_GatedFuseVM_attngru2_120000.pth'),
    ('GFM_GatedFuseVM_FocalGRU', lambda: GFM_GatedFuseVM(gru=FocalGRU), InferenceCoreDoubleRecurrentGFM,
    '/home/csvt32745/matte/MaskPropagation/saves/Jul05_00.51.44_GFM_GatedFuseVM_focalgru/Jul05_00.51.44_GFM_GatedFuseVM_focalgru_120000.pth'),
    ('GFM_GatedFuseVM_FocalGRU_new', lambda: GFM_GatedFuseVM(gru=FocalGRU), InferenceCoreDoubleRecurrentGFM,
    '/home/csvt32745/matte/MaskPropagation/saves/Jul17_05.29.10_GFM_GatedFuseVM_focalgru_new/Jul17_05.29.10_GFM_GatedFuseVM_focalgru_new_120000.pth'),
    ('GFM_FuseVM2_AttnGRU_new', lambda: GFM_FuseVM(fuse='GFMFuse2', gru=AttnGRU), InferenceCoreDoubleRecurrentGFM, 
    '/home/csvt32745/matte/MaskPropagation/saves/Jun26_13.46.07_GFM_FuseVM_AttnGRU_wo_tvloss/Jun26_13.46.07_GFM_FuseVM_AttnGRU_wo_tvloss_120000.pth'),
    ('GatedVM', lambda: GatedVM(is_output_fg=False), InferenceCoreRecurrent, 
    '/home/csvt32745/matte/MaskPropagation/saves/Jun27_21.03.07_GatedVM/Jun27_21.03.07_GatedVM_120000.pth'),
    
    ('GFM_GatedFuseVM_4xfoucs_focalgru', lambda: GFM_GatedFuseVM_4xfoucs(gru=FocalGRU), InferenceCoreRecurrentGFM, 
    '/home/csvt32745/matte/MaskPropagation/saves/Jul07_13.23.24_GFM_GatedFuseVM_4xfoucs_focalgru/Jul07_13.23.24_GFM_GatedFuseVM_4xfoucs_focalgru_120000.pth'),
    ('GFM_GatedFuseVM_4xfoucs_fulltrimap', GFM_GatedFuseVM_4xfoucs, InferenceCoreRecurrentGFM, 
    '/home/csvt32745/matte/MaskPropagation/saves/Jul16_01.15.46_GFM_GatedFuseVM_4xfoucs/Jul16_01.15.46_GFM_GatedFuseVM_4xfoucs_120000.pth'),
    ('GFM_GatedFuseVM_4xfoucs_old', GFM_GatedFuseVM_4xfoucs, InferenceCoreRecurrentGFM, 
    '/home/csvt32745/matte/MaskPropagation/saves/Jul08_13.49.36_GFM_GatedFuseVM_4xfoucs_ch32/Jul08_13.49.36_GFM_GatedFuseVM_4xfoucs_ch32_120000.pth'),
    ('GFM_GatedFuseVM_bottleneck_dropout', 'GFM_GatedFuseVM_fuse=dropout', InferenceCoreRecurrentGFM, 
    '/home/csvt32745/matte/MaskPropagation/saves/Jul16_05.02.26_GFM_GatedFuseVM_bottleneck_dropout/Jul16_05.02.26_GFM_GatedFuseVM_bottleneck_dropout_120000.pth'),
    ('GFM_GatedFuseVM_fuse=splitstage', 'GFM_GatedFuseVM_fuse=splitstage', InferenceCoreRecurrentGFM, 
    '/home/csvt32745/matte/MaskPropagation/saves/Jul13_01.37.50_GFM_GatedFuseVM_fuse=splitstage/Jul13_01.37.50_GFM_GatedFuseVM_fuse=splitstage_120000.pth'),
    ('GFM_GatedFuseVM_fuse=samekv', 'GFM_GatedFuseVM_fuse=samekv', InferenceCoreRecurrentGFM, 
    '/home/csvt32745/matte/MaskPropagation/saves/Jul13_16.25.49_GFM_GatedFuseVM_fuse=samekv/Jul13_16.25.49_GFM_GatedFuseVM_fuse=samekv_120000.pth'),
    ('GFM_GatedFuseVM_fuse=sameqk', 'GFM_GatedFuseVM_fuse=sameqk', InferenceCoreRecurrentGFM, 
    '/home/csvt32745/matte/MaskPropagation/saves/Jul12_12.31.40_GFM_GatedFuseVM_fuse=sameqk/Jul12_12.31.40_GFM_GatedFuseVM_fuse=sameqk_120000.pth'),
    ('GFM_GatedFuseVM_fuse=head1', 'GFM_GatedFuseVM_fuse=head1', InferenceCoreRecurrentGFM, 
    '/home/csvt32745/matte/MaskPropagation/saves/Jul18_20.32.39_GFM_GatedFuseVM_fuse=head1/Jul18_20.32.39_GFM_GatedFuseVM_fuse=head1_120000.pth'),
    ('GFM_GatedFuseVM_fuse=splitfuse', 'GFM_GatedFuseVM_fuse=splitfuse', InferenceCoreRecurrentGFM, 
    '/home/csvt32745/matte/MaskPropagation/saves/Jul19_08.54.18_GFM_GatedFuseVM_fuse=splitfuse/Jul19_08.54.18_GFM_GatedFuseVM_fuse=splitfuse_120000.pth'),
    ('GFM_GatedFuseVM_fuse=sameqk_head1', 'GFM_GatedFuseVM_fuse=sameqk_head1', InferenceCoreRecurrentGFM, 
    '/home/csvt32745/matte/MaskPropagation/saves/Jul19_08.54.41_GFM_GatedFuseVM_fuse=sameqk_head1/Jul19_08.54.41_GFM_GatedFuseVM_fuse=sameqk_head1_120000.pth'),
    
    ('GFM_GatedFuseVM_4xfoucs', GFM_GatedFuseVM_4xfoucs, InferenceCoreRecurrentGFM, 
    '/home/csvt32745/matte/MaskPropagation/saves/Jul16_01.15.46_GFM_GatedFuseVM_4xfoucs/Jul16_01.15.46_GFM_GatedFuseVM_4xfoucs_120000.pth'),
    ('GFM_GatedFuseVM_4xfoucs_1', GFM_GatedFuseVM_4xfoucs_1, InferenceCoreRecurrentGFM, 
    '/home/csvt32745/matte/MaskPropagation/saves/Jul08_13.49.23_GFM_GatedFuseVM_4xfoucs_1/Jul08_13.49.23_GFM_GatedFuseVM_4xfoucs_1_120000.pth'),
    ('GFM_GatedFuseVM_4xfoucs_2', GFM_GatedFuseVM_4xfoucs_2, InferenceCoreRecurrentGFM, 
    '/home/csvt32745/matte/MaskPropagation/saves/Jul08_18.28.22_GFM_GatedFuseVM_4xfoucs_2/Jul08_18.28.22_GFM_GatedFuseVM_4xfoucs_2_120000.pth'),
    ('GFM_GatedFuseVM_to4xglance_4xfocus', GFM_GatedFuseVM_to4xglance_4xfocus, InferenceCoreRecurrentGFM, 
    '/home/csvt32745/matte/MaskPropagation/saves/Jul13_16.26.20_GFM_GatedFuseVM_to4xglance_4xfocus/Jul13_16.26.20_GFM_GatedFuseVM_to4xglance_4xfocus_120000.pth'),
    # ('GFM_GatedFuseVM_to4xglance_4xfocus_2', GFM_GatedFuseVM_to4xglance_4xfocus_2, InferenceCoreRecurrentGFM, 
    # '/home/csvt32745/matte/MaskPropagation/saves/Jul15_17.25.32_GFM_GatedFuseVM_to4xglance_4xfocus_2/Jul15_17.25.32_GFM_GatedFuseVM_to4xglance_4xfocus_2_120000.pth'),
    
    ('GFM_GatedFuseVM_3dtv_allclass', 'GFM_GatedFuseVM_attngru2', InferenceCoreRecurrentGFM, 
    '/home/csvt32745/matte/MaskPropagation/saves/Jul14_22.51.49_GFM_GatedFuseVM_3dtv_allclass/Jul14_22.51.49_GFM_GatedFuseVM_3dtv_allclass_120000.pth'),
    ('GFM_GatedFuseVM_3dtv_allclass_weight', 'GFM_GatedFuseVM_attngru2', InferenceCoreRecurrentGFM, 
    '/home/csvt32745/matte/MaskPropagation/saves/Jul14_22.53.12_GFM_GatedFuseVM_3dtv_allclass_weight/Jul14_22.53.12_GFM_GatedFuseVM_3dtv_allclass_weight_120000.pth'),
    ('GFM_GatedFuseVM_3dtv_trimap_mem==gt', 'GFM_GatedFuseVM_attngru2', InferenceCoreRecurrentGFM, 
    '/home/csvt32745/matte/MaskPropagation/saves/Jul14_23.52.51_GFM_GatedFuseVM_attngru2_trimap_mem==gt/Jul14_23.52.51_GFM_GatedFuseVM_attngru2_trimap_mem==gt_120000.pth'),
    ('GFM_GatedFuseVM_wo_3dtv==gt', 'GFM_GatedFuseVM_attngru2', InferenceCoreRecurrentGFM, 
    '/home/csvt32745/matte/MaskPropagation/saves/Jul16_02.01.31_GFM_GatedFuseVM_wo_3dtv/Jul16_02.01.31_GFM_GatedFuseVM_wo_3dtv_120000.pth'),
    ('GFM_GatedFuseVM_3dtv_vanilla', 'GFM_GatedFuseVM_attngru2', InferenceCoreRecurrentGFM, 
    '/home/csvt32745/matte/MaskPropagation/saves/Jul17_04.59.48_GFM_GatedFuseVM_3dtv_vanilla/Jul17_04.59.48_GFM_GatedFuseVM_3dtv_vanilla_120000.pth'),
    ('GFM_GatedFuseVM_3dtv_allclass_mean', 'GFM_GatedFuseVM_attngru2', InferenceCoreRecurrentGFM, 
    '/home/csvt32745/matte/MaskPropagation/saves/Jul16_22.08.03_GFM_GatedFuseVM_3dtv_allclass_mean/Jul16_22.08.03_GFM_GatedFuseVM_3dtv_allclass_mean_120000.pth'),
    ('GFM_GatedFuseVM_tempseg', 'GFM_GatedFuseVM_attngru2', InferenceCoreRecurrentGFM, 
    '/home/csvt32745/matte/MaskPropagation/saves/Jul17_05.00.25_GFM_GatedFuseVM_temp_seg/Jul17_05.00.25_GFM_GatedFuseVM_temp_seg_120000.pth'),
    ('GFM_GatedFuseVM_tempseg_allclass', 'GFM_GatedFuseVM_attngru2', InferenceCoreRecurrentGFM, 
    '/home/csvt32745/matte/MaskPropagation/saves/Jul18_02.38.18_GFM_GatedFuseVM_tempseg_allclass/Jul18_02.38.18_GFM_GatedFuseVM_tempseg_allclass_120000.pth'),
    ('GFM_GatedFuseVM_tempseg_allclass_weight', 'GFM_GatedFuseVM_attngru2', InferenceCoreRecurrentGFM, 
    '/home/csvt32745/matte/MaskPropagation/saves/Jul17_19.13.48_GFM_GatedFuseVM_tempseg_allclass_weight/Jul17_19.13.48_GFM_GatedFuseVM_tempseg_allclass_weight_120000.pth'),
    ('GFM_GatedFuseVM_tempseg_allclass_weight_fix', 'GFM_GatedFuseVM_attngru2', InferenceCoreRecurrentGFM, 
    '/home/csvt32745/matte/MaskPropagation/saves/Jul18_23.50.44_GFM_GatedFuseVM_tempseg_allclass_weight/Jul18_23.50.44_GFM_GatedFuseVM_tempseg_allclass_weight_120000.pth'),
    ('GFM_GatedFuseVM_3dtv_allclass_weight_fix', 'GFM_GatedFuseVM_attngru2', InferenceCoreRecurrentGFM, 
    '/home/csvt32745/matte/MaskPropagation/saves/Jul18_13.42.47_GFM_GatedFuseVM_3dseg_allclass_weight/Jul18_13.42.47_GFM_GatedFuseVM_3dseg_allclass_weight_120000.pth'),
    
    ('GFM_GatedFuseVM_3dtv_allclass_60000d646', 'GFM_GatedFuseVM_attngru2', InferenceCoreRecurrentGFM, 
    '/home/csvt32745/matte/MaskPropagation/saves/Jul16_02.38.09_GFM_GatedFuseVM_3dtv_allclass_60000d646/Jul16_02.38.09_GFM_GatedFuseVM_3dtv_allclass_60000d646_120000.pth'),
    
    ('GFM_GatedFuseVM_3dtv_allclass_3chmask', 'GFM_GatedFuseVM_fuse=3chmask', InferenceCoreRecurrent3chTrimap, 
    '/home/csvt32745/matte/MaskPropagation/saves/Jul17_22.31.23_GFM_GatedFuseVM_3dtv_allclass_3chmask/Jul17_22.31.23_GFM_GatedFuseVM_3dtv_allclass_3chmask_120000.pth'),
    ('GFM_GatedFuseVM_4xfoucs_randmemtrimap', 'GFM_GatedFuseVM_4xfoucs', InferenceCoreRecurrentGFM, 
    '/home/csvt32745/matte/MaskPropagation/saves/Jul17_12.43.10_GFM_GatedFuseVM_4xfoucs_randmemtrimap/Jul17_12.43.10_GFM_GatedFuseVM_4xfoucs_randmemtrimap_120000.pth'),

    


    ('GFM_GatedFuseVM_4xfoucs_focal_sameqk_head1', 'GFM_GatedFuseVM_4xfoucs_focal_sameqk_head1', InferenceCoreRecurrentGFM, 
    '/home/csvt32745/matte/MaskPropagation/saves/Jul20_02.07.53_GFM_GatedFuseVM_4xfoucs_focal_sameqk_head1/Jul20_02.07.53_GFM_GatedFuseVM_4xfoucs_focal_sameqk_head1_120000.pth'),
    ('GFM_GatedFuseVM_to4xglance_4xfocus_focal_sameqk_head1', 'GFM_GatedFuseVM_to4xglance_4xfocus_focal_sameqk_head1', InferenceCoreRecurrentGFM, 
    '/home/csvt32745/matte/MaskPropagation/saves/Jul19_15.48.22_GFM_GatedFuseVM_to4xglance_4xfocus_focal_sameqk_head1/Jul19_15.48.22_GFM_GatedFuseVM_to4xglance_4xfocus_focal_sameqk_head1_120000.pth'),
    
    ('GFM_GatedFuseVM_to4xglance_4xfocus_3', 'GFM_GatedFuseVM_to4xglance_4xfocus_3', InferenceCoreRecurrentGFM, '/home/csvt32745/matte/MaskPropagation/saves/Jul19_17.23.40_GFM_GatedFuseVM_to4xglance_4xfocus_3/Jul19_17.23.40_GFM_GatedFuseVM_to4xglance_4xfocus_3_120000.pth'),
    ('GFM_GatedFuseVM_to4xglance_4xfocus_3_naivefuse', 'GFM_GatedFuseVM_to4xglance_4xfocus_3_naivefuse', InferenceCoreRecurrentGFM, '/home/csvt32745/matte/MaskPropagation/saves/Jul21_01.11.16_GFM_GatedFuseVM_to4xglance_4xfocus_3_naivefuse/Jul21_01.11.16_GFM_GatedFuseVM_to4xglance_4xfocus_3_naivefuse_120000.pth'),
    ('GFM_GatedFuseVM_to4xglance_4xfocus_3_focal', 'GFM_GatedFuseVM_to4xglance_4xfocus_3_focal', InferenceCoreRecurrentGFM, '/home/csvt32745/matte/MaskPropagation/saves/Jul21_00.14.03_GFM_GatedFuseVM_to4xglance_4xfocus_3_focal/Jul21_00.14.03_GFM_GatedFuseVM_to4xglance_4xfocus_3_focal_120000.pth'),
    ('GFM_GatedFuseVM_to4xglance_4xfocus_3_focal_sameqk_head1', 'GFM_GatedFuseVM_to4xglance_4xfocus_3_focal_sameqk_head1', InferenceCoreRecurrentGFM, '/home/csvt32745/matte/MaskPropagation/saves/Jul20_18.15.55_GFM_GatedFuseVM_to4xglance_4xfocus_3_focal_sameqk_head1/Jul20_18.15.55_GFM_GatedFuseVM_to4xglance_4xfocus_3_focal_sameqk_head1_120000.pth'),
    # ('GFM_GatedFuseVM_to4xglance_4xfocus_3_fixtrimap', 'GFM_GatedFuseVM_to4xglance_4xfocus_3', InferenceCoreRecurrentGFM, '/home/csvt32745/matte/MaskPropagation/saves/Jul20_19.50.59_GFM_GatedFuseVM_to4xglance_4xfocus_3_fixtrimap/Jul20_19.50.59_GFM_GatedFuseVM_to4xglance_4xfocus_3_fixtrimap_120000.pth'),
    ('GFM_GatedFuseVM_to4xglance_4xfocus_4', 'GFM_GatedFuseVM_to4xglance_4xfocus_4', InferenceCoreRecurrentGFM, '/home/csvt32745/matte/MaskPropagation/saves/Jul20_23.17.16_GFM_GatedFuseVM_to4xglance_4xfocus_4/Jul20_23.17.16_GFM_GatedFuseVM_to4xglance_4xfocus_4_120000.pth'),
    
    ('GFM_GatedFuseVM_to4xglance_4xfocus_fixtrimap', 'GFM_GatedFuseVM_to4xglance_4xfocus', InferenceCoreRecurrentGFM, 
	'./saves/Jul23_02.17.16_GFM_GatedFuseVM_to4xglance_4xfocus_fixtrimap/Jul23_02.17.16_GFM_GatedFuseVM_to4xglance_4xfocus_fixtrimap_120000.pth'),
	('GFM_GatedFuseVM_to4xglance_4xfocus_3_bn', 'GFM_GatedFuseVM_to4xglance_4xfocus_3_fuse=bn', InferenceCoreRecurrentGFM, 
	'./saves/Jul23_02.17.09_GFM_GatedFuseVM_to4xglance_4xfocus_3_bn/Jul23_02.17.09_GFM_GatedFuseVM_to4xglance_4xfocus_3_bn_120000.pth'),
	('GFM_GatedFuseVM_to4xglance_4xfocus_3_fixtrimap', 'GFM_GatedFuseVM_to4xglance_4xfocus_3', InferenceCoreRecurrentGFM, 
	'./saves/Jul23_02.07.57_GFM_GatedFuseVM_to4xglance_4xfocus_3_fixtrimap/Jul23_02.07.57_GFM_GatedFuseVM_to4xglance_4xfocus_3_fixtrimap_120000.pth'),
	('GFM_GatedFuseVM_to4xglance_4xfocus_5', 'GFM_GatedFuseVM_to4xglance_4xfocus_5', InferenceCoreRecurrentGFM, 
	'./saves/Jul22_11.52.32_GFM_GatedFuseVM_to4xglance_4xfocus_5/Jul22_11.52.32_GFM_GatedFuseVM_to4xglance_4xfocus_5_120000.pth'),

    ('GFM_GatedFuseVM_to4xglance_4xfocus_6_sameqk_head1', 'GFM_GatedFuseVM_to4xglance_4xfocus_6_sameqk_head1', InferenceCoreRecurrentGFM, 
	'./saves/Jul25_01.46.29_GFM_GatedFuseVM_to4xglance_4xfocus_6_sameqk_head1/Jul25_01.46.29_GFM_GatedFuseVM_to4xglance_4xfocus_6_sameqk_head1_120000.pth'),
	('GFM_GatedFuseVM_to4xglance_4xfocus_5_sameqk_head1', 'GFM_GatedFuseVM_to4xglance_4xfocus_5_sameqk_head1', InferenceCoreRecurrentGFM, 
	'./saves/Jul25_01.46.21_GFM_GatedFuseVM_to4xglance_4xfocus_5_sameqk_head1/Jul25_01.46.21_GFM_GatedFuseVM_to4xglance_4xfocus_5_sameqk_head1_120000.pth'),
	('GFM_GatedFuseVM_to4xglance_4xfocus_3_sameqk_head1', 'GFM_GatedFuseVM_to4xglance_4xfocus_3_sameqk_head1', InferenceCoreRecurrentGFM, 
	'./saves/Jul24_12.58.16_GFM_GatedFuseVM_to4xglance_4xfocus_3_sameqk_head1/Jul24_12.58.16_GFM_GatedFuseVM_to4xglance_4xfocus_3_sameqk_head1_120000.pth'),

    ('GFM_GatedFuseVM_to4xglance_4xfocus_3_inputmaskonly', 'GFM_GatedFuseVM_to4xglance_4xfocus_3_fuse=inputmaskonly', InferenceCoreRecurrentGFM, './saves/Jul22_03.22.32_GFM_GatedFuseVM_to4xglance_4xfocus_3_inputmaskonly/Jul22_03.22.32_GFM_GatedFuseVM_to4xglance_4xfocus_3_inputmaskonly_120000.pth'),
	('GFM_GatedFuseVM_to4xglance_4xfocus_3_random_trimap', 'GFM_GatedFuseVM_to4xglance_4xfocus_3', InferenceCoreRecurrentGFM, './saves/Jul21_23.40.18_GFM_GatedFuseVM_to4xglance_4xfocus_3_random_trimap/Jul21_23.40.18_GFM_GatedFuseVM_to4xglance_4xfocus_3_random_trimap_120000.pth'),
	# ('GFM_GatedFuseVM_to4xglance_4xfocus_3_sameqk_head1', 'GFM_GatedFuseVM_to4xglance_4xfocus_3_sameqk_head1', InferenceCoreRecurrentGFM, './saves/Jul21_23.06.35_GFM_GatedFuseVM_to4xglance_4xfocus_3_sameqk_head1/Jul21_23.06.35_GFM_GatedFuseVM_to4xglance_4xfocus_3_sameqk_head1_120000.pth'),
    ('GFM_FuseVM', 'GFM_FuseVM=GFMFuse3', InferenceCoreRecurrentGFM, './saves/Jul21_15.40.15_GFM_FuseVM/Jul21_15.40.15_GFM_FuseVM_120000.pth'),
	('GFM_GatedFuseVM_to4xglance_4xfocus_3_convgru', 'GFM_GatedFuseVM_to4xglance_4xfocus_3_convgru', InferenceCoreRecurrentGFM, './saves/Jul21_03.27.02_GFM_GatedFuseVM_to4xglance_4xfocus_3_convgru/Jul21_03.27.02_GFM_GatedFuseVM_to4xglance_4xfocus_3_convgru_120000.pth'),
	('GFM_GatedFuseVM_to4xglance_4xfocus_2', 'GFM_GatedFuseVM_to4xglance_4xfocus_2', InferenceCoreRecurrentGFM, './saves/Jul21_02.46.36_GFM_GatedFuseVM_to4xglance_4xfocus_2/Jul21_02.46.36_GFM_GatedFuseVM_to4xglance_4xfocus_2_120000.pth'),
    ('GFM_GatedFuseVM_to4xglance_4xfocus_3_fusefeature', 'GFM_GatedFuseVM_to4xglance_4xfocus_3_fusefeature', InferenceCoreRecurrentGFM, 
	'./saves/Jul22_03.22.26_GFM_GatedFuseVM_to4xglance_4xfocus_3_fusefeature/Jul22_03.22.26_GFM_GatedFuseVM_to4xglance_4xfocus_3_fusefeature_120000.pth'),

    ('GFM_GatedFuseVM_to4xglance_4xfocus_7', 'GFM_GatedFuseVM_to4xglance_4xfocus_7', InferenceCoreRecurrentGFM, 
	'./saves/Jul25_15.57.37_GFM_GatedFuseVM_to4xglance_4xfocus_7/Jul25_15.57.37_GFM_GatedFuseVM_to4xglance_4xfocus_7_120000.pth'),
	('GFM_GatedFuseVM_to4xglance_4xfocus_3_fixtrimap_firstmat', 'GFM_GatedFuseVM_to4xglance_4xfocus_3', InferenceCoreRecurrentGFM, 
	'./saves/Jul25_15.50.12_GFM_GatedFuseVM_to4xglance_4xfocus_3_fixtrimap_firstmat/Jul25_15.50.12_GFM_GatedFuseVM_to4xglance_4xfocus_3_fixtrimap_firstmat_120000.pth'),
	# ('GFM_GatedFuseVM_to4xglance_4xfocus_3_sameqk_head1_convgru', 'GFM_GatedFuseVM_to4xglance_4xfocus_3_sameqk_head1_convgru', InferenceCoreRecurrentGFM, 
	# './saves/Jul25_02.55.06_GFM_GatedFuseVM_to4xglance_4xfocus_3_sameqk_head1_convgru/Jul25_02.55.06_GFM_GatedFuseVM_to4xglance_4xfocus_3_sameqk_head1_convgru_120000.pth'),

    ('GFM_GatedFuseVM_to4xglance_4xfocus_3_fixtrimap_ytvos', 'GFM_GatedFuseVM_to4xglance_4xfocus_3', InferenceCoreRecurrentGFM, 
	'./saves/Jul27_02.59.35_GFM_GatedFuseVM_to4xglance_4xfocus_3_fixtrimap_ytvos/Jul27_02.59.35_GFM_GatedFuseVM_to4xglance_4xfocus_3_fixtrimap_ytvos_120000.pth'),
	('GFM_GatedFuseVM_to4xglance_4xfocus_3_fuse_convgru=ff2', 'GFM_GatedFuseVM_to4xglance_4xfocus_3_fuse_convgru=ff2', InferenceCoreRecurrentGFM, 
	'./saves/Jul26_19.57.32_GFM_GatedFuseVM_to4xglance_4xfocus_3_fuse_convgru=ff2/Jul26_19.57.32_GFM_GatedFuseVM_to4xglance_4xfocus_3_fuse_convgru=ff2_120000.pth'),
	('GFM_GatedFuseVM_to4xglance_4xfocus_3_randtrimap', 'GFM_GatedFuseVM_to4xglance_4xfocus_3', InferenceCoreRecurrentGFM, 
	'./saves/Jul26_18.59.47_GFM_GatedFuseVM_to4xglance_4xfocus_3_randtrimap/Jul26_18.59.47_GFM_GatedFuseVM_to4xglance_4xfocus_3_randtrimap_120000.pth'),
    ('GFM_GatedFuseVM_to4xglance_4xfocus_3=grufuse', 'GFM_GatedFuseVM_to4xglance_4xfocus_3_fuse=grufuse', InferenceCoreRecurrentGFM, 
	'./saves/Jul26_02.21.00_GFM_GatedFuseVM_to4xglance_4xfocus_3=grufuse/Jul26_02.21.00_GFM_GatedFuseVM_to4xglance_4xfocus_3=grufuse_120000.pth'),
	('GFM_GatedFuseVM_to4xglance_4xfocus_3_fullresgate', 'GFM_GatedFuseVM_to4xglance_4xfocus_3_fullresgate', InferenceCoreRecurrentGFM, 
	'./saves/Jul26_02.08.00_GFM_GatedFuseVM_to4xglance_4xfocus_3_fullresgate/Jul26_02.08.00_GFM_GatedFuseVM_to4xglance_4xfocus_3_fullresgate_120000.pth'),
	('GFM_GatedFuseVM_to4xglance_4xfocus_3=naive_h1sqk', 'GFM_GatedFuseVM_to4xglance_4xfocus_3_fuse=naive_h1sqk', InferenceCoreRecurrentGFM, 
	'./saves/Jul26_01.58.59_GFM_GatedFuseVM_to4xglance_4xfocus_3=naive_h1sqk/Jul26_01.58.59_GFM_GatedFuseVM_to4xglance_4xfocus_3=naive_h1sqk_120000.pth'),

    ('GFM_GatedFuseVM_2dtv_tempcons_weightce_512', GFM_GatedFuseVM, InferenceCoreDoubleRecurrentGFM,
    '/home/csvt32745/matte/MaskPropagation/saves/Jul01_20.11.40_GFM_GatedFuseVM_tempcon_2dtv_weightce_512/Jul01_20.11.40_GFM_GatedFuseVM_tempcon_2dtv_weightce_512_120000.pth'),
    ('GFM_GatedFuseVM_3dtvloss_weightce_512', GFM_GatedFuseVM, InferenceCoreDoubleRecurrentGFM,
    '/home/csvt32745/matte/MaskPropagation/saves/Jul01_20.12.09_GFM_GatedFuseVM_3dtvloss_weightce_512/Jul01_20.12.09_GFM_GatedFuseVM_3dtvloss_weightce_512_120000.pth'),
    ('GFM_GatedFuseVM_normal_celoss_480', GFM_GatedFuseVM, InferenceCoreDoubleRecurrentGFM,
    '/home/csvt32745/matte/MaskPropagation/saves/Jul11_23.12.44_GFM_GatedFuseVM_normal_celoss_480/Jul11_23.12.44_GFM_GatedFuseVM_normal_celoss_480_120000.pth'),
    ('GFM_GatedFuseVM_3dtvloss_480', GFM_GatedFuseVM, InferenceCoreDoubleRecurrentGFM,
    '/home/csvt32745/matte/MaskPropagation/saves/Jul11_13.46.19_GFM_GatedFuseVM_3dtvloss_480/Jul11_13.46.19_GFM_GatedFuseVM_3dtvloss_480_120000.pth'),
	('GFM_GatedFuseVM_to4xglance_4xfocus_3_multiobj', 'GFM_GatedFuseVM_to4xglance_4xfocus_3', InferenceCoreRecurrentGFM, 
	'./saves/Jul27_16.22.17_GFM_GatedFuseVM_to4xglance_4xfocus_3_multiobj/Jul27_16.22.17_GFM_GatedFuseVM_to4xglance_4xfocus_3_multiobj_120000.pth'),
	('GFM_GatedFuseVM_to4xglance_4xfocus_3_sameqk_head1_convgru', 'GFM_GatedFuseVM_to4xglance_4xfocus_3', InferenceCoreRecurrentGFM, 
	'./saves/Jul27_16.14.51_GFM_GatedFuseVM_to4xglance_4xfocus_3_sameqk_head1_convgru/Jul27_16.14.51_GFM_GatedFuseVM_to4xglance_4xfocus_3_sameqk_head1_convgru_120000.pth'),
    

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
    ('STCNFuseMatting_SameDec_480', 'STCNFuseMatting_SameDec', InferenceCoreRecurrentGFM, 
	'./saves/Aug04_19.01.14_STCNFuseMatting_SameDec_480/Aug04_19.01.14_STCNFuseMatting_SameDec_480_120000.pth'),
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

    ('STCNFuseMatting_fullres_matnaive_480_temp_seg', 'STCNFuseMatting_fullres_matnaive', InferenceCoreRecurrentGFM, 
	'./saves/Aug11_22.07.52_STCNFuseMatting_fullres_matnaive_480_temp_seg/Aug11_22.07.52_STCNFuseMatting_fullres_matnaive_480_temp_seg_120000.pth'),
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