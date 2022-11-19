import numpy as np
import cv2
import torch

torch.set_grad_enabled(False)

from torchvision import transforms
from model.which_model import get_model_by_string
from inference_model_list import inference_model_list
from FTPVM.memory_bank import MemoryBank

class TrimapScribbler:
    def __init__(self, callback, display_ratio=1.):
        self.display_ratio = display_ratio
        self.MODE_FG = 255
        self.MODE_BG = 0
        self.MODE_TRAN = 255//2

        self.srb_mode = self.MODE_FG
        self.srb_default_size = 50
        self.srb_size = self.srb_default_size
        
        # Used for drawing
        self.pressed = False
        self.last_ex = self.last_ey = None
        self.need_update = False
        self.need_draw = False
        self.name = 'Trimap Scribbler'
        self.callback = callback

    def start(self, img):
        self.img = img
        self.img_size = self.img.shape[:2]

        # scribbles trimap
        self.last_mask = None
        self.mask = np.zeros(self.img_size, dtype=np.uint8)
        

        # Used for drawing
        self.pressed = False
        self.last_ex = self.last_ey = None
        self.need_update = False
        cv2.namedWindow(self.name)
        # cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.name, self.callback)
        return self.run()

    def run(self):
        is_updated = True
        while 1:
            if self.control():
                break

            if self.need_update or is_updated:
                # if display_comp == DisplayMode.Comparance:
                #     display = comp_image(image, np_mask, manager.p_srb, manager.n_srb, manager.t_srb)
                # elif display_comp == DisplayMode.Mask:
                #     # display = np_mask
                #     display = np.tile(np_mask, 3)
                # else:# display_comp == DisplayMode.FG:
                #     alpha = (np_mask/255.)
                #     display = (alpha*image + (1-alpha)*new_bg).astype(np.uint8)
                #     # display = np.stack([distance_transform_edt(1-srb) for srb in [manager.p_srb, manager.n_srb, manager.t_srb]], axis=-1)/manager.norm
                #     # display = distance_transform_edt(1-manager.n_srb)/manager.norm
                display = np.clip((self.img * 0.5) + (self.mask[..., None]*0.5), 0, 255).astype(np.uint8)
                self.need_draw = is_updated = False

            final_display = display.copy()
            cv2.circle(final_display, (self.last_ex, self.last_ey), radius=self.srb_size-3, color=[self.srb_mode]*3, thickness=-1)
            # if show_mask_aside:
            #     # print(final_display.shape, show_mask.shape, show_comp.shape)
            # print(final_display.shape, final_display.dtype)
            # print(self.mask.max())
            final_display = np.concatenate([final_display, np.repeat(self.mask[..., None], 3, axis=-1)], axis=0)
            final_display = cv2.resize(final_display, None, fx = self.display_ratio, fy = self.display_ratio)
            cv2.imshow(self.name, final_display)
        cv2.destroyWindow(self.name)
        return self.mask

    def mouse_down(self, ex, ey):
        ex = int(ex / self.display_ratio)
        ey = int(ey / self.display_ratio)
        if ex >= self.img_size[1] or ey >= self.img_size[0]:
            return
        self.last_ex = ex
        self.last_ey = ey
        self.pressed = True
        cv2.circle(self.mask, (ex, ey), radius=self.srb_size, color=(self.srb_mode), thickness=-1)
        self.need_update = True

    def mouse_move(self, ex, ey):
        ex = int(ex / self.display_ratio)
        ey = int(ey / self.display_ratio)
        if ex >= self.img_size[1] or ey >= self.img_size[0]:
            return
        if not self.pressed:
            self.last_ex = ex
            self.last_ey = ey
            return
        cv2.line(self.mask, (self.last_ex, self.last_ey), (ex, ey), (self.srb_mode), thickness=self.srb_size)
        self.last_ex = ex
        self.last_ey = ey
        self.need_update = True

    def mouse_up(self):
        self.pressed = False
        self.need_update = True

    def control(self):
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            return True
        if k == ord('d'):
            pass
            # display_comp = (display_comp+1) % DisplayMode.__len__()
            # self.need_update = True
        elif k == ord('r'):
            self.clean_up()
            self.need_update = True
        elif k == ord('e'):
            self.resize_srb(5)
        elif k == ord('q'):
            self.resize_srb(-5)
        elif k == ord('w'):
            self.resize_srb(0)
        elif k == ord('a'):
            self.next_scribble_mode()
        return False

    def resize_srb(self, delta):
        if delta == 0:
           self.srb_size = self.srb_default_size
        elif self.srb_size+delta > 0:
            self.srb_size += delta
        print(f'Scribble size = [ {self.srb_size} ]')

    def clean_up(self):
        self.mask.fill(self.MODE_BG)
        self.need_update = True

    def next_scribble_mode(self):
        if self.srb_mode == self.MODE_FG:
            self.srb_mode = self.MODE_TRAN
        elif self.srb_mode == self.MODE_TRAN:
            self.srb_mode = self.MODE_BG
        elif self.srb_mode == self.MODE_BG:
            self.srb_mode = self.MODE_FG

class WebcamMatting:
    def __init__(self, model: torch.nn.Module, trimap_scribbler: TrimapScribbler):
        self.transform = transforms.ToTensor()
        self.trimap_scribbler = trimap_scribbler
        self.memory = None
        self.model = model
        self.rec = self.model.default_rec
        self.downsample_ratio = 1
        self.name = 'WebcamMatting'
        # memory = model.encode_imgs_to_value(m_img, m_mask, downsample_ratio=downsample_ratio)
        # cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        self.bg_color = torch.tensor([120, 255, 155], device='cuda').div(255).view(1, 1, 3, 1, 1)
        self.memory_bank = MemoryBank()
        
    def run(self, mirror=False):
        cam = cv2.VideoCapture(0)
        while True:
            available, self.img = cam.read()
            # print(available, self.img)
            if not available:
                continue

            if mirror: 
                self.img = cv2.flip(self.img, 1)
            self.shape = self.img.shape[:2]
            self.qimg = self.transform(self.img).cuda().unsqueeze(0).unsqueeze(0)

            if self.control():
                break
                
            # else:
            # if self.memory is None:
            #     print(self.qimg.shape)
            #     self.encode_value(self.qimg, torch.ones(1, 1, 1, *self.shape).cuda()*0.5)
            memory = self.memory_bank.get_memory()
            if memory is not None:
            # if self.memory is not None:
                trimap, matte, pha, self.rec, _ = self.model.forward_with_memory(self.qimg, *memory, *self.rec, downsample_ratio=self.downsample_ratio)
                # trimap, matte, pha, self.rec, _ = self.model.forward_with_memory(self.qimg, *self.memory, *self.rec, downsample_ratio=self.downsample_ratio)
                out = (self.qimg*pha+self.bg_color*(1-pha)).squeeze().permute(1, 2, 0).cpu().numpy()
                out = np.concatenate([out, self.mcomp], axis=1)
            else:
                out = self.img
            scale = 800/out.shape[0]
            out = cv2.resize(out, None, fx=scale, fy=scale)
            cv2.imshow(self.name, out)
        cv2.destroyAllWindows()
    
    def control(self):
        key = cv2.waitKey(1) & 0xFF
        # print(key)
        if key == 27 or key == ord('q') :
            # ESC or q
            return True

        if key == ord('t'):
            # print('draw tri')
            self.draw_trimap()
            # cv2.imshow('draw trimap', self.img)
        elif key == ord('c'):
            print("Clean recurrent memory")
            self.rec = self.model.default_rec
        elif key == ord('v'):
            print("Clean trimap memory")
            self.memory_bank = MemoryBank()
        return False

    def encode_value(self, img, mask):
        # self.memory = self.model.encode_imgs_to_value(img, mask, downsample_ratio=self.downsample_ratio)
        self.memory_bank.add_memory(*self.model.encode_imgs_to_value(img, mask, downsample_ratio=self.downsample_ratio))
        
    def draw_trimap(self):
        mask = self.trimap_scribbler.start(self.img) # (h, w)
        self.mmask = mask
        self.ming = self.img
        self.mcomp = ((self.img * 0.5) + (mask[..., None]*0.5))/255
        mask = (torch.from_numpy(mask)/255.).cuda().unsqueeze(0).unsqueeze(0).unsqueeze(0)
        img = self.transform(self.img).cuda().unsqueeze(0).unsqueeze(0)
        
        self.encode_value(img, mask)
        
        





def mouse_callback(event, x, y, flags, *args):
    if event == cv2.EVENT_LBUTTONDOWN:
        trimap_srb.mouse_down(x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        trimap_srb.mouse_up()
    elif event == cv2.EVENT_MBUTTONDOWN:
        trimap_srb.next_scribble_mode()
    # if event == cv2.EVENT_MOUSEWHEEL:
    #     # print(x, y, flags)
    #     if flags > 0:
    #         trimap_srb.resize_srb(5)
    #     else:
    #         trimap_srb.resize_srb(-5)
    # Draw
    if event == cv2.EVENT_MOUSEMOVE:
        trimap_srb.mouse_move(x, y)


if __name__ == '__main__':
    model_name = 'STCNFuseMatting_fullres_matnaive'
    model_attr = inference_model_list[model_name]
    model = get_model_by_string(model_attr[1])().cuda()
    def check_and_load_model_dict(model, state_dict: dict):
        for k in set(state_dict.keys()) - set(model.state_dict().keys()):
            if 'refiner' in k:
                print('remove refiner', k)
                state_dict.pop(k)
        model.load_state_dict(state_dict)
    check_and_load_model_dict(model, torch.load(model_attr[3]))
    
    trimap_srb = TrimapScribbler(callback=mouse_callback)
    webcam = WebcamMatting(model, trimap_srb)
    
    

    webcam.run()