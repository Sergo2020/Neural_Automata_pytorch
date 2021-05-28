from pathlib import Path

import argparse
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
            save_image(dest_dir / f'test_{ep}.png', test[:, :4])
            method.save_method(save_dir, f'test_{ep}')
            method.plot_loss()

if __name__ == "__main__":

    proj_path = get_curr_path()
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--im_path", default=proj_path + r"\images\pika.png", required=False)
    parser.add_argument("-d", "--dest_path", default=proj_path + r"\results", required=False)
    parser.add_argument("-c", "--check_path", default=proj_path + r"\check_points", required=False)
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

    hyper = {'Image path': Path(args.im_path),
             'Batch Size': args.batch_size, 'Epochs': args.epochs, 'Device': device,
             'Channels': args.channels, 'Image size': args.image_size, 'Target padding': args.padd,
             'Random Seed': 0, 'Perceive': True, 'Update pool': True,
             'Pool size': args.pool_size,
             'Learning rate': args.learning_rate, 'Learning gamma': 0.9999, 'Fire rate': 0.5,
             'Hidden dim.': args.hidden_dim, 'Min. Steps': args.min_steps, 'Max. Steps': args.max_steps}

    train_model(Path(args.dest_path), Path(args.check_path), hyper)

