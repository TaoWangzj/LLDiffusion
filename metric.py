import argparse

from utils.metrics import evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+")
    parser.add_argument("--gt", type=str, default="data/sets/LOL/eval15/high")
    parser.add_argument("--correct", action="store_true")
    parser.add_argument("--gray", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dirs = args.dirs
    gt_path = args.gt
    correct = args.correct
    gray = args.gray
    verbose = args.verbose

    for dir in dirs:
        res = evaluate(
            dir, 
            gt_path,
            correct=correct,
            ssim_y_channels=gray,
            verbose=verbose,
        )
        print(f"{dir}: " + ", ".join(
            f"{k}:{v:.4}" for k, v in res.items()
        ))


if __name__ == "__main__":
    main()
