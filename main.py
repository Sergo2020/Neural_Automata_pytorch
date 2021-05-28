from test import *
from train import *
from utils import *


if __name__ == "__main__":
    img_path = Path(get_curr_path() + r"\images\pika.png")
    dest_path = Path(get_curr_path() + r"\results")
    check_path = Path(get_curr_path() + r"\check_points")

    method_load_path = Path(check_path, )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hyper = {'Image path': img_path,
             'Batch Size': 8, 'Epochs': 60, 'Device': device,
             'Channels': 16, 'Image size': 40, 'Target padding': 4,
             'Random Seed': 0, 'Perceive': True, 'Update pool': True,
             'Pool size': 1024,
             'Learning rate': 1e-3, 'Learning gamma': 0.9999, 'Fire rate': 0.5,
             'Hidden dim.': 128, 'Min. Steps': 64, 'Max. Steps': 96}

    train_model(dest_path, check_path, hyper)
    test_model(method_load_path, 'test_60', dest_path,
               new_img_path=img_path,
               steps=hyper['Max. Steps'])
