
from inference_func import *
from FTPVM.model import *
from FTPVM.module import *
from FTPVM.inference_model import *


inference_model_list = [
    ('STCNFuseMatting_fullres_matnaive', 'STCNFuseMatting_fullres_matnaive', InferenceCoreRecurrentGFM, 
	'./saves/Aug11_15.46.26_STCNFuseMatting_fullres_matnaive/Aug11_15.46.26_STCNFuseMatting_fullres_matnaive_120000.pth'),
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