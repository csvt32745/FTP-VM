# from FTPVM.rvm.model import RVM
from FTPVM.model import *

def get_model_by_string(input_string):
    # which_model=which_module
    if len(model_names := input_string.split('=')) > 1:
        which_module = model_names[1]
    which_model = model_names[0]
    return {
        'STCNFuseMatting': STCNFuseMatting,
        'test': lambda :STCNFuseMatting(backbone_arch='efficientnet_b3a'),
        'STCNFuseMatting_big': STCNFuseMatting_big,
        'STCNFuseMatting_fuse': lambda: STCNFuseMatting(trimap_fusion=which_module),
        'STCNFuseMatting_fullres': STCNFuseMatting_fullres_mat,
        'STCNFuseMatting_fullres_gn': lambda: STCNFuseMatting_fullres_mat(trimap_fusion='gn'),
        'STCNFuseMatting_fullres_mat2': lambda: STCNFuseMatting_fullres_mat(mat_decoder='4x_2'),
        'STCNFuseMatting_fullres_mat3': lambda: STCNFuseMatting_fullres_mat(mat_decoder='4x_3'),
        'STCNFuseMatting_fullres_matnaive': lambda: STCNFuseMatting_fullres_mat(mat_decoder='naive', trimap_fusion='gn'),
        'STCNFuseMatting_fullres_matnaive_memalpha': lambda: STCNFuseMatting_fullres_mat(mat_decoder='naive', trimap_fusion='gn', ch_mask=2),
        'STCNFuseMatting_fullres_matnaive_fullgate': lambda: STCNFuseMatting_fullres_mat(mat_decoder='naive', trimap_fusion='fullgate'),
        'STCNFuseMatting_fullres_matnaive_backbonefuse': lambda: STCNFuseMatting_fullres_mat(mat_decoder='naive', trimap_fusion='backbone'),
        'STCNFuseMatting_fullres_matnaive_backbonefuse_f16values': lambda: STCNFuseMatting_fullres_mat(mat_decoder='naive', trimap_fusion='backbone', bottleneck_fusion='f16') ,
        'STCNFuseMatting_fullres_matnaive_bn': lambda: STCNFuseMatting_fullres_mat(mat_decoder='naive', trimap_fusion='bn'),
        'STCNFuseMatting_fullres_matnaive_seg2': lambda: STCNFuseMatting_fullres_mat(mat_decoder='naive', trimap_fusion='gn', seg_decoder='4x_2'),
        'STCNFuseMatting_fullres_gn_3chmask': lambda: STCNFuseMatting_fullres_mat(trimap_fusion='gn', ch_mask=3),
        'STCNFuseMatting_SameDec': STCNFuseMatting_SameDec,
        # 'STCNFuseMatting_1xseg': lambda: STCNFuseMatting(seg_decoder='1x', trimap_fusion='bn2'),
        # 'STCNFuseMatting_1xseg_4x2mat': lambda: STCNFuseMatting(seg_decoder='1x', mat_decoder='4x_2', trimap_fusion='gn'),
        'STCNFuseMatting_SingleDec': lambda: STCNFuseMatting_SingleDec(trimap_fusion='gn'),
        'STCNFuseMatting_SingleDec_big': lambda: STCNFuseMatting_SingleDec(trimap_fusion='gn', ch_decode=[128, 96, 64, 32]),
        'STCNFuseMatting_fullres_matnaive_f16value': lambda: STCNFuseMatting_fullres_mat(mat_decoder='naive', trimap_fusion='gn', bottleneck_fusion='f16'),
        'STCNFuseMatting_fullres_matnaive_l2attn': lambda: STCNFuseMatting_fullres_mat(mat_decoder='naive', trimap_fusion='gn', bottleneck_fusion='l2'),
        'STCNFuseMatting_fullres_matnaive_l2gate': lambda: STCNFuseMatting_fullres_mat(mat_decoder='naive', trimap_fusion='gn', bottleneck_fusion='gate'),
        'STCNFuseMatting_fullres_matnaive2_seg3': lambda: STCNFuseMatting_fullres_mat(seg_decoder='4x_3', mat_decoder='naive2', trimap_fusion='gn'),
        'STCNFuseMatting_fullres_matnaive2': lambda: STCNFuseMatting_fullres_mat(mat_decoder='naive2', trimap_fusion='gn'),
        'STCNFuseMatting_fullres_matnaive3': lambda: STCNFuseMatting_fullres_mat(mat_decoder='naive3', trimap_fusion='gn'),
        'STCNFuseMatting_fullres_matnaive4': lambda: STCNFuseMatting_fullres_mat(mat_decoder='naive4', trimap_fusion='gn'),
        'STCNFuseMatting_fullres_matnaive_naivefuse': lambda: STCNFuseMatting_fullres_mat(mat_decoder='naive', trimap_fusion='naive'),
        'STCNFuseMatting_fullres_matnaive_seg3': lambda: STCNFuseMatting_fullres_mat(mat_decoder='naive', trimap_fusion='gn', seg_decoder='4x_3'),
        'STCNFuseMatting_fullres_mat_big': lambda: STCNFuseMatting_fullres_mat_big(mat_decoder='naive', trimap_fusion='gn'),
        
        'STCNFuseMatting_fullres_matnaive_ppm1236': lambda: STCNFuseMatting_fullres_mat(mat_decoder='naive', trimap_fusion='gn', bottleneck_fusion='1236'),
        'STCNFuseMatting_fullres_matnaive_woPPM': lambda: STCNFuseMatting_fullres_mat(mat_decoder='naive', trimap_fusion='gn', bottleneck_fusion='woppm'),
        'STCNFuseMatting_fullres_matnaive_woCBAM': lambda: STCNFuseMatting_fullres_mat(mat_decoder='naive', trimap_fusion='gn', bottleneck_fusion='wocbam'),
        'STCNFuseMatting_fullres_matnaive_woCBAMPPM': lambda: STCNFuseMatting_fullres_mat(mat_decoder='naive', trimap_fusion='gn', bottleneck_fusion='wocbamppm'),
        'STCNFuseMatting_fullres_matnaive_mat-gru': lambda: STCNFuseMatting_fullres_mat(mat_decoder='naive-gru', trimap_fusion='gn'),
        'STCNFuseMatting_fullres_matnaive_seg-gru': lambda: STCNFuseMatting_fullres_mat(mat_decoder='naive', seg_decoder='4x-gru', trimap_fusion='gn'),
        'STCNFuseMatting_fullres_matnaive_wogru': lambda: STCNFuseMatting_fullres_mat(mat_decoder='naive-gru', seg_decoder='4x-gru', trimap_fusion='gn'),
        '2stage':  SeperateNetwork,
        '2stage_seg4x':  lambda: SeperateNetwork(seg='4x'),

        # 'RVM': RVM,

    }[which_model]

