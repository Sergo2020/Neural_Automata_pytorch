from pathlib import Path

import torch
from tqdm import tqdm

import data
import trainer
from utils import *


def train_model(dest_dir, save_dir, hyper, load_path=None):

    check_ex(dest_dir, True)
    check_ex(save_dir, True)

    if load_path is None:

        pool_set = data.Single_image(hyper)
        torch.save(hyper, save_dir / 'train_hypers.pt')
        method = trainer.Trainer(hyper)
        init_e = 0
    else:
        check_ex(load_path)
        method = torch.load(load_path[0])
        pool_set = torch.load(load_path[1])
        init_e = len(method.train_obj_loss)

    epochs = list(range(1 + init_e, hyper['Epochs'] + 1 + init_e))
    pbar = tqdm(total=len(epochs), desc='Train progress')

    factor = hyper['Epochs'] // 10

    for ep in epochs:
        test = method.train_model(pool_set)

        pbar.update()
        pbar.postfix = f'Loss {method.train_loss[-1]:.4f}'

        if (ep % factor) == 0:
            save_tensor(dest_dir / f'test_{ep}.png', test[:, :4])
            method.save_method(save_dir, f'test_{ep}')
            method.plot_loss()

def test_model(method_path, prefix, res_dir, new_img_path = None, steps=32):

    check_ex(method_path)
    check_ex(res_dir, True)

    hyper = torch.load(method_path / 'train_hypers.pt')

    if new_img_path:
        check_ex(new_img_path)
        hyper = new_img_path

    seeds_pool = data.Single_image(hyper)
    method = trainer.Trainer(hyper)
    method.load_method(method_path, prefix)

    res = method.simulate_cells(seeds_pool.seed, steps)
    save_gif(res, dest_path / f'{prefix}_gif.gif')


if __name__ == "__main__":
    img_path = Path(get_curr_path() + r"\images\pika.png")
    dest_path = Path(get_curr_path() + r"\results")
    check_path = Path(get_curr_path() + r"\check_points")

    method_load_path = Path(check_path, )
    data_load_path = Path(check_path / r"pika_data.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hyper = {'Image path': img_path,
             'Batch Size': 8, 'Epochs': 60, 'Device': device,
             'Channels': 16, 'Image size': 40, 'Target padding': 4,
             'Random Seed': 0, 'Perceive': True, 'Update pool': True,
             'Pool size': 1024,
             'Learning rate': 1e-3, 'Learning gamma': 0.9999, 'Fire rate': 0.5,
             'Hidden dim.': 128, 'Min. Steps': 64, 'Max. Steps': 96}

    #train_model(dest_path, check_path, hyper)
    test_model(method_load_path, 'test_60', dest_path, steps=hyper['Max. Steps'])