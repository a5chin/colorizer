import sys
from pathlib import Path

import pytest
import torch

current_dir = Path(__file__).resolve().parent
sys.path.append(current_dir.parent.as_posix())

from colorize.model import Colorizer


@pytest.mark.parametrize("batch_size", [4])
def test_model(batch_size: int):
    size = (28, 28)
    images = torch.rand(size=(batch_size, 1, *size))

    model = Colorizer()
    out = model(images)

    assert out.shape == (batch_size, 3, *size)
