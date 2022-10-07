from typing import Dict, Tuple, Union

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F


class Compose(transforms.Compose):
    def __init__(self, transforms_dict: Dict[str, torch.nn.Module]):
        self.transforms = transforms_dict

    def __call__(
        self, *img: Union[Image.Image, Tuple[Image.Image, ...]]
    ) -> Union[Image.Image, Tuple[Image.Image, ...]]:
        for t in self.transforms:
            if isinstance(img, Image.Image):
                img = t(img)
            elif isinstance(img, tuple):
                img = t(*img)
            else:
                raise TypeError("Image should be a Image.Image or tuple")

        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"

        return format_string


class ColorAndGray(torch.nn.Module):
    def __call__(self, x: Image.Image) -> Tuple[Image.Image, Image.Image]:
        color = x
        gray = x.convert("L")

        return color, gray


class ToTensor(torch.nn.Module):
    def __call__(
        self, *imgs: Tuple[Image.Image, ...]
    ) -> Tuple[torch.Tensor, ...]:
        return tuple(F.to_tensor(img) for img in imgs)


def get_transforms(size: int = 32) -> Dict[str, Compose]:
    return {
        "train": Compose(
            [
                transforms.RandomResizedCrop(
                    size=size, scale=(0.08, 1), ratio=(3 / 4, 4 / 3)
                ),
                ColorAndGray(),
                ToTensor(),
            ]
        ),
        "validation": Compose(
            [
                ColorAndGray(),
                ToTensor(),
            ]
        ),
        "test": Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    }
