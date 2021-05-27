import os
import sys
from torchvision.utils import save_image

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