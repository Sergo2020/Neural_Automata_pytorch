import pickle
from pathlib import Path

import argparse

import torch
from torchvision.utils import save_image
from tqdm import tqdm

import data
import trainer
from utils import check_ex


def train_model(img_path, dest_dir, save_dir, hyper, load_prefix=None):
    check_ex(dest_dir, create=True)
    check_ex(save_dir, create=True)
    method = trainer.Trainer(hyper)

    if load_prefix is None:
        pool_set = data.Single_image(img_path, hyper)
        pickle.dump(hyper, open((save_dir / 'train_hypers.pt'), 'wb'))

        init_e = 0
    else:
        method.load_method(save_dir, load_prefix)
        check_ex(load_prefix)
        init_e = len(method.train_obj_loss)

    epochs = list(range(1 + init_e, hyper['Epochs'] + 1 + init_e))
    pbar = tqdm(total=len(epochs), desc='Train')

    factor = hyper['Epochs'] // 10

    for ep in epochs:
        test = method.train_model(pool_set)

        pbar.update()
        pbar.postfix = f'Loss {method.train_loss[-1]:.4f}'

        if (ep % factor) == 0:
            save_image(test[:, :4], dest_dir / f'test_{ep}.png')
            method.save_method(save_dir, f'test_{ep}')
            method.plot_loss()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--im_path", default=Path('images') / 'pika.png', required=False)
    parser.add_argument("-d", "--dest_path", default=Path('results'), required=False)
    parser.add_argument("-c", "--check_path", default=Path('check_points'), required=False)
    parser.add_argument("-b", "--batch_size", type=int, default=8, required=False)
    parser.add_argument("-e", "--epochs", type=int, default=60, required=False)
    parser.add_argument("-ch", "--channels", type=int, default=16, required=False)
    parser.add_argument("-is", "--image_size", type=int, default=40, required=False)
    parser.add_argument("-p", "--padd", type=int, default=4, required=False)
    parser.add_argument("-mi", "--min_steps", type=int, default=64, required=False)
    parser.add_argument("-ma", "--max_steps", type=int, default=96, required=False)
    parser.add_argument("-ps", "--pool_size", type=int, default=1024, required=False)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3, required=False)
    parser.add_argument("-hd", "--hidden_dim", type=int, default=128, required=False)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hyper = {'Batch Size': args.batch_size, 'Epochs': args.epochs, 'Device': device,
             'Channels': args.channels, 'Image size': args.image_size, 'Target padding': args.padd,
             'Random Seed': 0, 'Perceive': True, 'Update pool': True,
             'Pool size': args.pool_size,
             'Learning rate': args.learning_rate, 'Learning gamma': 0.9999, 'Fire rate': 0.5,
             'Hidden dim.': args.hidden_dim, 'Min. Steps': args.min_steps, 'Max. Steps': args.max_steps}

    train_model(Path(args.im_path), Path(args.dest_path), Path(args.check_path), hyper)

