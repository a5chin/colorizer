import argparse
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100


def get_loader(
    args: argparse.Namespace, transforms: torch.nn.Module
) -> Tuple[DataLoader, DataLoader]:
    traindataset = CIFAR100(
        root=args.root,
        train=True,
        transform=transforms["train"],
        download=True,
    )
    traindataloader = DataLoader(
        dataset=traindataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    valdataset = CIFAR100(
        root=args.root,
        train=False,
        transform=transforms["validation"],
        download=True,
    )
    valdataloader = DataLoader(
        dataset=valdataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
    )

    return traindataloader, valdataloader
