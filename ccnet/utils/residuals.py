from torch import nn
from .smu import SMU

class ResidualBlock(nn.Module):
    def __init__(self, width, smooth=True, normalized=False):
        super().__init__()

        layers = [
            nn.Linear(width, width),
            nn.GELU() if smooth else nn.ReLU(),
            nn.Linear(width, width),
        ]
        if normalized:
            layers.insert(1, nn.BatchNorm1d(width))
            layers.append(nn.BatchNorm1d(width))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)
