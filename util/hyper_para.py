from argparse import ArgumentParser
import json

def none_or_default(x, default):
    return x if x is not None else default

class HyperParameters():
    def parse(self, unknown_arg_ok=False):
        parser = ArgumentParser()
        parser.add_argument('--which_model', help='Use which model', default='NULL')
        parser.add_argument('--tvloss_type', help='Loss type of total variation', default='3d_seg', type=str)
        parser.add_argument('--celoss_type', help='Loss type of cross entropy: [focal, normal]', default='focal', type=str)
        parser.add_argument('--split_trimap', help='split 1ch-trimap into 3ch masks', action='store_true')

        # Enable torch.backends.cudnn.benchmark -- Faster in some cases, test in your own environment
        parser.add_argument('--benchmark', action='store_true')
        parser.add_argument('--num_worker', help='num_workers of dataloader', default=16, type=int)
        parser.add_argument('--nb_frame_only', help='neighbor frames only', action='store_true')
        parser.add_argument('--get_bgr_pha', help='get pha of bgr by lastframe affine', action='store_true')
        parser.add_argument('--random_memtrimap', help='Given memory trimap != GT trimap', action='store_true')
        parser.add_argument('--size', help='dataset img size', default=256, type=int)

        # Generic learning parameters
        parser.add_argument('--long_seq', help='Adjust seq length and batch size', action='store_true', default=False)

        parser.add_argument('--iter_switch_dataset', help='switch to vid dataset at which epoch', type=int, default=5)
        parser.add_argument('-b', '--batch_size', default=None, type=int)
        parser.add_argument('-i', '--iterations', default=None, type=int)
        
        parser.add_argument('--seg_start', help='Start iter of segmentation training', default=0, type=int)
        parser.add_argument('--seg_cd', help='Cooldown of segmentation training', default=20000, type=int)
        parser.add_argument('--seg_iter', help='Iter of segmentation training', default=10000, type=int)
        parser.add_argument('--seg_stop', help='At which iter to stop segmentation training', default=80000, type=int)

        parser.add_argument('--lr', help='Initial learning rate', default=1e-4, type=float)

        # Loading
        parser.add_argument('--load_network', help='Path to pretrained network weight only')
        parser.add_argument('--load_model', help='Path to the model file, including network, optimizer and such')

        # Logging information
        parser.add_argument('--id', help='Experiment UNIQUE id, use NULL to disable logging to tensorboard', default='NULL')
        parser.add_argument('--debug', help='Debug mode which logs information more often', action='store_true')

        if unknown_arg_ok:
            args, _ = parser.parse_known_args()
            self.args = vars(args)
        else:
            self.args = vars(parser.parse_args())

    def __getitem__(self, key):
        return self.args[key]

    def __setitem__(self, key, value):
        self.args[key] = value

    def __str__(self):
        return str(self.args)

    def save(self, savepath):
        with open(savepath, 'w') as f:
            json.dump(self.args, f, indent=4) 