import argparse
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
sys.path.append(current_dir.parent.as_posix())

from colorize import Trainer


def make_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default="../assets/data",
        type=str,
        help="plese set data root",
    )
    parser.add_argument(
        "--size",
        default=32,
        type=int,
        help="plese set image size",
    )
    parser.add_argument(
        "--epoch",
        default=100,
        type=int,
        help="plese set train epoch",
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="plese set batch size",
    )
    parser.add_argument(
        "--log_dir",
        default="../logs",
        type=str,
        help="plese set logdir",
    )

    return parser.parse_args()


def main():
    args = make_parse()

    trainer = Trainer(args)
    trainer.fit()


if __name__ == "__main__":
    main()
