import os
import sys
from torchvision.utils import save_image
from torchvision.transforms import ToPILImage


def save_tensor(path, img_tens):
    save_image(img_tens, path)


def get_curr_path():
    return os.getcwd()


def check_ex(path, create=False):
    if not os.path.exists(path):
        print(f'{path}\ndoes not exist.')
        if create:
            os.mkdir(path)
            print(f'{path} is created.')
        else:
            print('Program will be terminated!')
            sys.exit()

def tens2pils(tens_list):
    tens2pil = ToPILImage()

    for idx in range(len(tens_list)):
        tens_list[idx] = tens2pil(tens_list[idx][:4])

    return tens_list

def save_gif(tensor_list, path):
    pil_list = tens2pils(tensor_list)
    pil_list[0].save(path, save_all=True, append_images=pil_list[1:])



