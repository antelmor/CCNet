from torch import nn
from .smu import SMU

class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
            SMU(),
            nn.Linear(width, width),
            nn.BatchNorm1d(width)
        )

    def forward(self, x):
        return x + self.block(x)
