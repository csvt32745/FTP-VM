import argparse
import torch
from tqdm import tqdm
from time import time
from STCNVM.model import STCNFuseMatting
from model.which_model import get_model_by_string

torch.backends.cudnn.benchmark = True

class InferenceSpeedTest:
    def __init__(self):
        self.parse_args()
        self.init_model()
        self.loop()
        
    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_name', type=str, required=True)
        parser.add_argument('--resolution', type=int, nargs=2, default=[512, 512])
        # parser.add_argument('--downsample-ratio', type=float, required=True)
        parser.add_argument('--precision', type=str, default='float32')
        parser.add_argument('--disable-refiner', action='store_true')
        parser.add_argument('--gpu', type=int, default=0)
        parser.add_argument('--pad', type=int, default=16)
        self.args = parser.parse_args()
        
    def init_model(self):
        print(self.args)
        self.device = f'cuda'
        # self.device = f'cuda:{self.args.gpu}'
        self.precision = {'float32': torch.float32, 'float16': torch.float16}[self.args.precision]
        # self.model = get_model_by_string(self.args.model_name)()
        self.model = STCNFuseMatting()
        self.model = self.model.to(device=self.device, dtype=self.precision).eval()
        # self.model = torch.jit.script(self.model)
        # self.model = torch.jit.freeze(self.model)
    
    def loop(self):
        # w, h = (512, 512)
        w, h = self.args.resolution
        if (mod := w%self.args.pad) > 0:
            w += self.args.pad-mod
        if (mod := h%self.args.pad) > 0:
            h += self.args.pad-mod
        print('H, W = ', h, w)
        qimg = torch.randn((1, 1, 3, h, w), device=self.device, dtype=self.precision)
        mimg = torch.randn((1, 1, 3, h, w), device=self.device, dtype=self.precision)
        mask = torch.randn((1, 1, 1, h, w), device=self.device, dtype=self.precision)
        N = 1000
        with torch.no_grad():
            if 'default_rec' in dir(self.model):
                rec = self.model.default_rec
            else:
                rec = [None]*4
                # rec = [rec]*2
                # rec = [rec, [None]]
            print(rec)
            # mk, mv = self.model.encode_imgs_to_value(mimg, mask)
            rec = self.model(qimg, mimg, mask, *rec)[-2]

            t = time()
            for _ in tqdm(range(N)):
                # rec = self.model.forward_with_memory(qimg, mk, mv, *rec)[-2]
                rec = self.model(qimg, mimg, mask, *rec)[-2]
                torch.cuda.synchronize()
            t = time()-t
        print("FPS: ", N / t)

if __name__ == '__main__':
    InferenceSpeedTest()