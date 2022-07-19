
from inference_func import *
from STCNVM.model import *
from STCNVM.module import *
from STCNVM.inference_model import *


inference_model_list = [
    ('GFM_FuseVM', 'GFM_FuseVM=GFMFuse3', InferenceCoreRecurrentGFM, 
    '/home/csvt32745/matte/MaskPropagation/saves/Jul12_22.35.05_GFM_FuseVM/Jul12_22.35.05_GFM_FuseVM_120000.pth'),
    ('DualVM', lambda: DualMattingNetwork(is_output_fg=False), InferenceCoreRecurrent, 
    '/home/csvt32745/matte/MaskPropagation/saves/Jul01_23.59.45_DualVM/Jul01_23.59.45_DualVM_120000.pth'), 
        
    ('GFM_GatedFuseVM_convgru', lambda: GFM_GatedFuseVM(gru=ConvGRU), InferenceCoreDoubleRecurrentGFM,
    '/home/csvt32745/matte/MaskPropagation/saves/Jul04_10.47.39_GFM_GatedFuseVM_convgru/Jul04_10.47.39_GFM_GatedFuseVM_convgru_120000.pth'),
    ('GFM_GatedFuseVM_convgru', GFM_GatedFuseVM_big, InferenceCoreDoubleRecurrentGFM,
    '/home/csvt32745/matte/MaskPropagation/saves/Jul11_23.12.45_GFM_GatedFuseVM_big_convgru/Jul11_23.12.45_GFM_GatedFuseVM_big_convgru_120000.pth'),
    
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
    ('GFM_GatedFuseVM_to4xglance_4xfocus_2', GFM_GatedFuseVM_to4xglance_4xfocus_2, InferenceCoreRecurrentGFM, 
    '/home/csvt32745/matte/MaskPropagation/saves/Jul15_17.25.32_GFM_GatedFuseVM_to4xglance_4xfocus_2/Jul15_17.25.32_GFM_GatedFuseVM_to4xglance_4xfocus_2_120000.pth'),
    
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
    
    

    ('GFM_GatedFuseVM_2dtv_tempcons_weightce_512', GFM_GatedFuseVM, InferenceCoreDoubleRecurrentGFM,
    '/home/csvt32745/matte/MaskPropagation/saves/Jul01_20.11.40_GFM_GatedFuseVM_tempcon_2dtv_weightce_512/Jul01_20.11.40_GFM_GatedFuseVM_tempcon_2dtv_weightce_512_120000.pth'),
    ('GFM_GatedFuseVM_3dtvloss_weightce_512', GFM_GatedFuseVM, InferenceCoreDoubleRecurrentGFM,
    '/home/csvt32745/matte/MaskPropagation/saves/Jul01_20.12.09_GFM_GatedFuseVM_3dtvloss_weightce_512/Jul01_20.12.09_GFM_GatedFuseVM_3dtvloss_weightce_512_120000.pth'),
]

inference_model_list = {i[0]: i for i in inference_model_list}