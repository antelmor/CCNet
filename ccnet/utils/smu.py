import torch
import torch.nn as nn

class SMU(nn.Module):

    def __init__(self, beta=1.0):
        super().__init__()

        self.log_beta = nn.Parameter(
            torch.tensor(float(beta)).log()
        )

    def forward(self, x):
        beta = torch.exp(self.log_beta)
        return x * (1.0 + torch.erf(beta * x))
