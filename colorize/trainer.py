import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .dataset import get_loaders
from .model import Colorizer
from .transforms import get_transforms
from .utils import AverageMeter, Logger


class Trainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = Path(args.root).expanduser()
        self.logger = Logger()
        self.transforms = get_transforms(size=args.size)
        self.traindataloader, self.valdataloader = get_loaders(
            args=args, transforms=self.transforms
        )
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.model = Colorizer()
        self.best_loss = float("inf")
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.log_dir = Path(args.log_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir.as_posix())

    def fit(self) -> None:
        self.model.apply(Colorizer.init_weights)
        self.model.to(self.device)

        for epoch in range(self.args.epoch):
            self.model.train()

            losses = AverageMeter("train_loss")

            with tqdm(self.traindataloader) as pbar:
                pbar.set_description(f"[Epoch {epoch + 1}/{self.args.epoch}]")

                for images, _ in pbar:
                    color = images[0].to(self.device)
                    gray = images[1].to(self.device)

                    out = self.model(gray)

                    loss = self.criterion(out, color)
                    losses.update(loss.item())

                    loss.backward()
                    self.optimizer.step()

                    pbar.set_postfix(OrderedDict(Loss=losses.value))

            self.writer.add_scalar("train/loss", losses.avg, epoch + 1)

            torch.save(self.model.state_dict(), self.log_dir / "last_ckpt.pth")

            self.evaluate(self.model, epoch + 1)

    @torch.inference_mode()
    def evaluate(self, model: nn.Module, epoch: Optional[int] = None) -> None:
        model.eval()

        losses = AverageMeter("valid_loss")

        for images, _ in self.valdataloader:
            color = images[0].to(self.device)
            gray = images[1].to(self.device)

            preds = model(gray)
            loss = self.criterion(preds, color)

            losses.update(loss.item())

        self.logger.log(f"Loss: {losses.avg}")

        if epoch is not None:
            self.writer.add_scalar("val/loss", losses.avg, epoch + 1)
            if losses.avg <= self.best_loss:
                self.best_loss = losses.avg
                torch.save(model.state_dict(), self.log_dir / "best_ckpt.pth")
