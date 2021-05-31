import os
import sys
from torchvision.transforms import ToPILImage
import torch


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


def tensor2image(tensor_list):
    tens2pil = ToPILImage(mode='RGB')

    for idx in range(len(tensor_list)):
        tensor_list[idx] = tens2pil(torch.clamp(tensor_list[idx][:3], 0, 1))

    return tensor_list


def save_gif(tensor_list, path):
    pil_list = tensor2image(tensor_list)
    pil_list[0].save(path, save_all=True, append_images=pil_list[1:], loop = 0)



