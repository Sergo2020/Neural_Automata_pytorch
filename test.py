from pathlib import Path
import argparse

import data
import trainer
from utils import *


def test_model(method_path, prefix, res_dir, new_img_path=None, steps=32):
    check_ex(method_path)
    check_ex(res_dir, True)

    hyper = torch.load(method_path / 'train_hypers.pt')

    if new_img_path:
        check_ex(new_img_path)
        hyper['Image path'] = new_img_path

    seeds_pool = data.Single_image(hyper)

    method = trainer.Trainer(hyper)
    method.load_method(method_path, prefix)

    print(f'Growing {prefix}!')
    res = method.simulate_cells(seeds_pool.seed, steps)
    save_gif(res, res_dir / f'{prefix}_gif.gif')


if __name__ == "__main__":
    proj_path = get_curr_path()
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--im_path", default=proj_path + r"\images\pika.png", required=False)
    parser.add_argument("-d", "--dest_path", default=proj_path + r"\results", required=False)
    parser.add_argument("-c", "--check_path", default=proj_path + r"\check_points", required=False)
    parser.add_argument("-m", "--max_steps", type=int, default=96, required=False)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_model(Path(args.check_path), 'test_60', Path(args.dest_path),
               new_img_path=Path(args.im_path),
               steps=args.max_steps)
