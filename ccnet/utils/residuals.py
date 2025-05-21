from torch import nn
from .smu import SMU

class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(width, width),
            SMU(),
            nn.Linear(width, width),
        )

    def forward(self, x):
        return x + self.block(x)
