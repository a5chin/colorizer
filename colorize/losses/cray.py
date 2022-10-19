from torch import Tensor, nn
from torch.nn import functional as F


class CrayLoss(nn.Module):
    R, G, B = 0, 1, 2
    R2GRAY, G2GRAY, B2GRAY = 0.299, 0.587, 0.114

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input_g = (
            input[:, CrayLoss.R, ...] * CrayLoss.R2GRAY
            + input[:, CrayLoss.G, ...] * CrayLoss.G2GRAY
            + input[:, CrayLoss.B, ...] * CrayLoss.B2GRAY
        )
        target_g = (
            target[:, CrayLoss.R, ...] * CrayLoss.R2GRAY
            + target[:, CrayLoss.G, ...] * CrayLoss.G2GRAY
            + target[:, CrayLoss.B, ...] * CrayLoss.B2GRAY
        )

        return F.mse_loss(input_g, target_g)
