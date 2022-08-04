from STCNVM.model import *

def get_model_by_string(input_string):
    # which_model=which_module
    if len(model_names := input_string.split('=')) > 1:
        which_module = model_names[1]
    which_model = model_names[0]
    return {
        'DualVM': lambda: DualMattingNetwork(is_output_fg=False),
        'GatedVM': lambda: GatedVM(is_output_fg=False),
        'DualVM_fm': lambda: DualMattingNetwork(gru=FocalGRU, is_output_fg=False),
        'DualVM_attngru_effb3a': lambda: DualMattingNetwork(backbone_arch='efficientnet_b3a', gru=AttnGRU, is_output_fg=False),
        'DualVM_attngru': lambda: DualMattingNetwork(gru=AttnGRU, is_output_fg=False),
        'GFM_FuseVM': lambda: GFM_FuseVM(fuse=which_module, gru=AttnGRU2),
        'GFM_FuseVM_FocalGRU': lambda: GFM_FuseVM(fuse=which_module, gru=FocalGRU),
        'GFM_FuseVM_AttnGRU': lambda: GFM_FuseVM(fuse=which_module, gru=AttnGRU),
        'STCN_GFM_VM': STCN_GFM_VM,
        'GatedSTCN_GFM_VM': GatedSTCN_GFM_VM,
        'GatedSTCN_GFM_VM2': lambda: GatedSTCN_GFM_VM(encoder=2),
        'GFM_GatedFuseVM': GFM_GatedFuseVM,
        'GFM_GatedFuseVM_convgru': lambda: GFM_GatedFuseVM(gru=ConvGRU),
        'GFM_GatedFuseVM_bigconvgru': lambda: GFM_GatedFuseVM(gru=ConvGRU_big),
        # 'test': lambda: GFM_GatedFuseVM(gru=ConvGRU, fuse='test'),
        'GFM_GatedFuseVM_attngru2': lambda: GFM_GatedFuseVM(gru=AttnGRU2),
        'GFM_GatedFuseVM_attngru': lambda: GFM_GatedFuseVM(gru=AttnGRU),
        'GFM_GatedFuseVM_focalgru': lambda: GFM_GatedFuseVM(gru=FocalGRU),
        'GFM_GatedFuseVM_fuse': lambda: GFM_GatedFuseVM(gru=AttnGRU2, fuse=which_module),
        'GFM_GatedFuseVM_4xfoucs': GFM_GatedFuseVM_4xfoucs,
        'GFM_GatedFuseVM_4xfoucs_dropout': lambda: GFM_GatedFuseVM_4xfoucs(gru=lambda *x: AttnGRU(*x, dropout=0.1)),
        'GFM_GatedFuseVM_4xfoucs_focalgru': lambda: GFM_GatedFuseVM_4xfoucs(gru=FocalGRU),
        'GFM_GatedFuseVM_4xfoucs_convgru': lambda: GFM_GatedFuseVM_4xfoucs(gru=ConvGRU),
        'GFM_GatedFuseVM_4xfoucs_1': GFM_GatedFuseVM_4xfoucs_1,
        'GFM_GatedFuseVM_4xfoucs_2': GFM_GatedFuseVM_4xfoucs_2,
        'GFM_GatedFuseVM_big': GFM_GatedFuseVM_big,
        'GFM_GatedFuseVM_to4xglance_4xfocus': GFM_GatedFuseVM_to4xglance_4xfocus,
        'GFM_GatedFuseVM_to4xglance_4xfocus_2': GFM_GatedFuseVM_to4xglance_4xfocus_2,
        'GFM_GatedFuseVM_to4xglance_4xfocus_3': lambda: GFM_GatedFuseVM_to4xglance_4xfocus_3(fuse='sameqk_head1', gru=ConvGRU),
        # 'GFM_GatedFuseVM_to4xglance_4xfocus_3': GFM_GatedFuseVM_to4xglance_4xfocus_3,
        'GFM_GatedFuseVM_to4xglance_4xfocus_4': GFM_GatedFuseVM_to4xglance_4xfocus_4,
        'GFM_GatedFuseVM_to4xglance_4xfocus_5': GFM_GatedFuseVM_to4xglance_4xfocus_5,
        'GFM_GatedFuseVM_to4xglance_4xfocus_6': GFM_GatedFuseVM_to4xglance_4xfocus_6,
        'GFM_GatedFuseVM_to4xglance_4xfocus_focal_sameqk_head1': lambda: GFM_GatedFuseVM_to4xglance_4xfocus(gru=FocalGRU, fuse='sameqk_head1'),
        'GFM_GatedFuseVM_4xfoucs_focal_sameqk_head1': lambda: GFM_GatedFuseVM_4xfoucs(gru=FocalGRU, fuse='sameqk_head1'),
        'GFM_GatedFuseVM_to4xglance_4xfocus_3_focal_sameqk_head1': lambda: GFM_GatedFuseVM_to4xglance_4xfocus_3(gru=FocalGRU, fuse='sameqk_head1'),
        'GFM_GatedFuseVM_to4xglance_4xfocus_3_focal': lambda: GFM_GatedFuseVM_to4xglance_4xfocus_3(gru=FocalGRU),
        'GFM_GatedFuseVM_to4xglance_4xfocus_3_convgru': lambda: GFM_GatedFuseVM_to4xglance_4xfocus_3(gru=ConvGRU),
        'GFM_GatedFuseVM_to4xglance_4xfocus_3_naivefuse': lambda: GFM_GatedFuseVM_to4xglance_4xfocus_3(fuse='naivefuse'),
        
        'GFM_GatedFuseVM_to4xglance_4xfocus_3_sameqk_head1': lambda: GFM_GatedFuseVM_to4xglance_4xfocus_3(fuse='sameqk_head1'),
        'GFM_GatedFuseVM_to4xglance_4xfocus_3_sameqk_head1_convgru': lambda: GFM_GatedFuseVM_to4xglance_4xfocus_3(fuse='sameqk_head1', gru=ConvGRU),

        'GFM_GatedFuseVM_to4xglance_4xfocus_3_fuse': lambda: GFM_GatedFuseVM_to4xglance_4xfocus_3(fuse=which_module),
        'GFM_GatedFuseVM_to4xglance_4xfocus_3_fuse_convgru': lambda: GFM_GatedFuseVM_to4xglance_4xfocus_3(fuse=which_module, gru=ConvGRU),
        'GFM_GatedFuseVM_to4xglance_4xfocus_3_fusefeature': GFM_GatedFuseVM_to4xglance_4xfocus_3_fusefeature(fuse='sameqk_head1'),
        'GFM_GatedFuseVM_to4xglance_4xfocus_5_sameqk_head1': lambda: GFM_GatedFuseVM_to4xglance_4xfocus_6(fuse='sameqk_head1'),
        'GFM_GatedFuseVM_to4xglance_4xfocus_6_sameqk_head1': lambda: GFM_GatedFuseVM_to4xglance_4xfocus_5(fuse='sameqk_head1'),
        'GFM_GatedFuseVM_to4xglance_4xfocus_7': lambda: GFM_GatedFuseVM_to4xglance_4xfocus_7(fuse='sameqk_head1'),
        'GFM_GatedFuseVM_to4xglance_4xfocus_3_fullresgate': GFM_GatedFuseVM_to4xglance_4xfocus_3_fullresgate,
        'STCNFuseMatting': STCNFuseMatting,
        'STCNFuseMatting_big': STCNFuseMatting_big,
        'STCNFuseMatting_gru_before_fuse': STCNFuseMatting_gru_before_fuse,
        'STCNFuseMatting_fuse': lambda: STCNFuseMatting(trimap_fusion=which_module),
        'STCNFuseMatting_fullres_mat': STCNFuseMatting_fullres_mat,
        'STCNFuseMatting_SameDec': STCNFuseMatting_SameDec,


    }[which_model]