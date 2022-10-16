import sys
from pathlib import Path

import pytest
from PIL import Image

current_dir = Path(__file__).resolve().parent
sys.path.append(current_dir.parent.as_posix())

from colorize.transforms import get_transforms


@pytest.mark.parametrize("mode", ["train", "validation", "test"])
def test_transforms(mode: str):
    size = (32, 32)
    image = Image.new(mode="RGB", size=size, color=(128, 128, 128))

    transforms = get_transforms(size=size)[mode]

    if mode == "test":
        gray = transforms(image.convert(mode="L"))
        assert gray.shape[1:] == size
    else:
        color, gray = transforms(image)
        assert color.shape[1:] == gray.shape[1:] == size
