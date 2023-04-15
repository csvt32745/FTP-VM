from FTPVM.model import *

def get_model_by_string(input_string):
    # which_model=which_module
    if len(model_names := input_string.split('=')) > 1:
        which_module = model_names[1]
    which_model = model_names[0]
    
    return {
        'FTPVM': FastTrimapPropagationVideoMatting,
    }[which_model]

