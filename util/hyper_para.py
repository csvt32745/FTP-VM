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
        parser.add_argument('--full_matte', help='predict full matte in the matting decoder', action='store_true')
        parser.add_argument('--lambda_segtv', help='lambda of segmentation consistency(total variation loss)', default=10, type=float)
        parser.add_argument('--start_segtv', help='start iteration of segmentation consistency(total variation loss)', default=-1, type=int)
        parser.add_argument('--same_mem_que', help='memory frame also in query frames ', default=1, type=float)
        parser.add_argument('--lambda_tc', help='lambda of segmentation consistency(total variation loss)', default=5, type=float)

        parser.add_argument('--split_trimap', help='split 1ch-trimap into 3ch masks', action='store_true')
        parser.add_argument('--memory_alpha', help='input memory alpha along with memory trimap', action='store_true')
        parser.add_argument('--memory_out_alpha_start', help='input memory alpha along with memory trimap', default=30000, type=int)
        parser.add_argument('--compose_multiobj', help='fuse other fgs into bgs in the batch', action='store_true')
        parser.add_argument('--ytvos', help='use YTVOS dataset intead of VIS for segmentation training', action='store_true')

        # Enable torch.backends.cudnn.benchmark -- Faster in some cases, test in your own environment
        parser.add_argument('--benchmark', action='store_true')
        parser.add_argument('--num_worker', help='num_workers of dataloader', default=16, type=int)

        # Dataset setting
        parser.add_argument('--use_background_dataset', help='Composite data with background video as well', action='store_true')
        parser.add_argument('--nb_frame_only', help='neighbor frames only', action='store_true')
        parser.add_argument('--get_bgr_pha', help='get pha of bgr by lastframe affine', action='store_true')
        parser.add_argument('--random_memtrimap', help='Given memory trimap != GT trimap', action='store_true')
        parser.add_argument('--size', help='dataset img size', default=256, type=int)

        # Generic learning setting
        parser.add_argument('-i', '--iterations', default=None, type=int)
        
        parser.add_argument('-b_seg', '--batch_size_seg', help='batch size of video segmentation', default=8, type=int)
        parser.add_argument('-b_img_mat', '--batch_size_image_matte', help='batch size of image matting dataset', default=10, type=int)
        parser.add_argument('-b_vid_mat', '--batch_size_video_matte', help='batch size of video matting dataset',default=4, type=int)
        
        parser.add_argument('-s_seg', '--seq_len_seg', help='sequence length of video segmentation', default=8, type=int)
        parser.add_argument('-s_img_mat', '--seq_len_image_matte', help='sequence length of image matting dataset', default=10, type=int)
        parser.add_argument('-s_vid_mat', '--seq_len_video_matte', help='sequence length of video matting dataset',default=4, type=int)
        
        parser.add_argument('--iter_switch_dataset', help='switch to vid dataset from img dataset at which epoch', type=int, default=5)
        parser.add_argument('--seg_start', help='Start iter of segmentation training', default=0, type=int)
        parser.add_argument('--seg_cd', help='Cooldown of segmentation training', default=20000, type=int)
        parser.add_argument('--seg_iter', help='Iter of segmentation training', default=10000, type=int)
        parser.add_argument('--seg_stop', help='At which iter to stop segmentation training', default=80000, type=int)

        parser.add_argument('--lr', help='Initial learning rate', default=1e-4, type=float)

        # Loading
        parser.add_argument('--load_network', help='Path to pretrained network weight only')
        parser.add_argument('--load_model', help='Path to the model file, including network, optimizer and such')
        parser.add_argument('--resume', help='resume the training by searching the latest checkpoint', action='store_true')

        # Logging information
        parser.add_argument('--report_interval', help='Interval of report metrics', default=100, type=int)
        parser.add_argument('--save_im_interval', help='Interval of save images', default=800, type=int)
        parser.add_argument('--save_model_interval', help='Interval of save model', default=20000, type=int)
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