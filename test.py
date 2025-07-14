import argparse
import os

import numpy as np
import torch
import yaml
from torchvision.io import read_image, write_png
from torchvision.transforms.functional import convert_image_dtype

import datasets
from models import DenoisingDiffusion
from utils.metrics import metric


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("-b", "--batch-size", type=int, default=512)
    parser.add_argument("-p", "--patch_size", type=int, nargs="?")
    parser.add_argument("-g", "--grid_r", type=int, default=16)
    parser.add_argument("-t", "--sampling_timesteps", type=int, default=25)
    parser.add_argument("-d", "--dir", default="data/test/results_debug_v2", type=str)
    parser.add_argument("--seed", default=0, type=int, metavar="N")
    parser.add_argument("--correct", action="store_true")
    parser.add_argument("--gray", action="store_true")
    parser.add_argument("--no-patch", action="store_true")
    return parser.parse_args()


def parse_args_and_config():
    args = parse_args()

    def dict2namespace(config):
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace

    with open(args.config) as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    return args, new_config


def main():
    args, config = parse_args_and_config()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.device = device
    if args.patch_size is not None:
        config.data.image_size = args.patch_size

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    Dataset = getattr(datasets, config.data.dataset)
    dataset = Dataset(config)
    test_loader = dataset.test_loader()

    model = DenoisingDiffusion(args, config)
    if args.resume != "":
        model.load(args.resume, ema=False)
    model.model.eval()

    save_root = args.dir
    os.makedirs(save_root, exist_ok=True)

    grid_r = None if args.no_patch else args.grid_r

    scores_dict = {}
    for (lows, highs), labels in test_loader:
        lows, highs = lows.to(device), highs.to(device)
        for low, high, label in zip(lows, highs, labels):
            name = label.item()
            path = os.path.join(save_root, f"{name}.png")
            if not os.path.exists(path):
                image_size = high.shape[-2:]
                restored = model.restore_single(low, grid_r, image_size, batch_size=args.batch_size)
                restored = convert_image_dtype(restored, dtype=torch.uint8)
                restored = restored.cpu()
                write_png(restored, path)
            else:
                restored = read_image(path)
            restored, high = restored.cpu(), high.cpu()
            score_dict = metric(
                restored, high,
                correct=args.correct,
                ssim_y_channels=args.gray,
            )
            for k, v in score_dict.items():
                scores_dict.setdefault(k, []).append(v)
            print(f"{name}: " + ", ".join(
                "{k}:{v:.4}".format(k=k, v=v)
                for k, v in score_dict.items()
            ))

    scores_dict = {k: sum(v) / len(v) for k, v in scores_dict.items()}
    print(", ".join(
        f"{k}:{v:.4}" for k, v in scores_dict.items()
    ))


if __name__ == "__main__":
    main()
