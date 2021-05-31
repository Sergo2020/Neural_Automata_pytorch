import pickle
from pathlib import Path
import argparse
import torch
import data
import trainer
from utils import check_ex, save_gif


def test_model(method_path: Path, prefix, res_dir: Path, new_img_path: (Path, None), steps=32):
    check_ex(method_path)
    check_ex(res_dir, create=True)

    hyper = pickle.load(open(str(method_path / 'train_hypers.pt'), 'rb'))

    if new_img_path:
        check_ex(new_img_path)
        hyper['Image path'] = new_img_path

    seeds_pool = data.Single_image(hyper)

    method = trainer.Trainer(hyper)
    method.load_method(method_path, prefix)

    print(f'Growing {prefix}')
    res = method.simulate_cells(seeds_pool.seed, steps)
    save_path = str(res_dir / f'{prefix}_gif.gif')
    save_gif(res, save_path)
    print(f'Results save to {save_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--im_path", default=Path.cwd() / 'images' / 'pika.png', required=False)
    parser.add_argument("-d", "--dest_path", default=Path.cwd() / 'results', required=False)
    parser.add_argument("-c", "--check_path", default=Path.cwd() / 'check_points', required=False)
    parser.add_argument("-m", "--max_steps", type=int, default=96, required=False)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_model(Path(args.check_path), 'test_60', Path(args.dest_path),
               new_img_path=Path(args.im_path),
               steps=args.max_steps)
