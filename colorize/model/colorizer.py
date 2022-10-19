import torch
from torch import nn

from ..losses import CrayLoss


class Colorizer(nn.Module):
    BASE_CHANNEl = 64

    def __init__(self) -> None:
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.cray_loss = CrayLoss()

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=Colorizer.BASE_CHANNEl,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(Colorizer.BASE_CHANNEl),
            nn.LeakyReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=Colorizer.BASE_CHANNEl,
                out_channels=Colorizer.BASE_CHANNEl * 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(Colorizer.BASE_CHANNEl * 2),
            nn.LeakyReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=Colorizer.BASE_CHANNEl * 2,
                out_channels=Colorizer.BASE_CHANNEl * 3,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(Colorizer.BASE_CHANNEl * 3),
            nn.LeakyReLU(),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(
                in_channels=Colorizer.BASE_CHANNEl * 3,
                out_channels=Colorizer.BASE_CHANNEl * 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(Colorizer.BASE_CHANNEl * 2),
            nn.LeakyReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(
                in_channels=Colorizer.BASE_CHANNEl * 2,
                out_channels=Colorizer.BASE_CHANNEl,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(Colorizer.BASE_CHANNEl),
            nn.LeakyReLU(),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(
                in_channels=Colorizer.BASE_CHANNEl,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(3),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3) + x2
        x5 = self.layer5(x4) + x1
        x6 = self.layer6(x5)

        return x6

    def loss(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = self.mse_loss(input, target)
        cray = self.cray_loss(input, target)

        return mse + cray

    @staticmethod
    def init_weights(m: nn.Module) -> None:
        """Initialize model weights. Defaults to True.
        Args:
            m (nn.Module): The model.
        """
        classname = m.__class__.__name__
        if classname.find("Linear") != -1 or classname.find("Bilinear") != -1:
            nn.init.kaiming_uniform_(
                a=2, mode="fan_in", nonlinearity="leaky_relu", tensor=m.weight
            )
            if m.bias is not None:
                nn.init.zeros_(tensor=m.bias)

        elif classname.find("Conv") != -1:
            nn.init.kaiming_uniform_(
                a=2, mode="fan_in", nonlinearity="leaky_relu", tensor=m.weight
            )
            if m.bias is not None:
                nn.init.zeros_(tensor=m.bias)

        elif (
            classname.find("BatchNorm") != -1
            or classname.find("GroupNorm") != -1
            or classname.find("LayerNorm") != -1
        ):
            nn.init.uniform_(a=0, b=1, tensor=m.weight)
            nn.init.zeros_(tensor=m.bias)

        elif classname.find("Cell") != -1:
            nn.init.xavier_uniform_(gain=1, tensor=m.weiht_hh)
            nn.init.xavier_uniform_(gain=1, tensor=m.weiht_ih)
            nn.init.ones_(tensor=m.bias_hh)
            nn.init.ones_(tensor=m.bias_ih)

        elif (
            classname.find("RNN") != -1
            or classname.find("LSTM") != -1
            or classname.find("GRU") != -1
        ):
            for w in m.all_weights:
                nn.init.xavier_uniform_(gain=1, tensor=w[2].data)
                nn.init.xavier_uniform_(gain=1, tensor=w[3].data)
                nn.init.ones_(tensor=w[0].data)
                nn.init.ones_(tensor=w[1].data)

        elif classname.find("Embedding") != -1:
            nn.init.kaiming_uniform_(
                a=2, mode="fan_in", nonlinearity="leaky_relu", tensor=m.weight
            )
