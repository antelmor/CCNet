from torch import nn
from .smu import SMU

class ResidualBlock(nn.Module):
    def __init__(self, width, smooth=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(width, width),
            SMU() if smooth else nn.ReLU(),
            nn.Linear(width, width),
        )

    def forward(self, x):
        return x + self.block(x)
