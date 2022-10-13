import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

current_dir = Path(__file__).resolve().parent
sys.path.append(current_dir.parent.as_posix())

from colorize.model import Colorizer
from colorize.transforms import get_transforms


def make_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
        default="../assets/ckpt/best_ckpt.pth",
        type=str,
        help="plese set ckpt path",
    )
    parser.add_argument(
        "--image",
        default="../assets/images/gray.png",
        type=str,
        help="plese set gray image path",
    )

    return parser.parse_args()


def main():
    args = make_parse()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    transforms = get_transforms()["test"]

    model = Colorizer()
    model.load_state_dict(
        torch.load(args.ckpt, map_location=torch.device(device))
    )

    gray = Image.open(args.image)
    images = transforms(gray).unsqueeze(dim=0)
    color = model(images).to('cpu').detach().squeeze().permute(1, 2, 0)

    plt.imshow(color)
    plt.show()


if __name__ == "__main__":
    main()
