from .ansatz import Ansatz
from .quantum_states import get_HF_state
from .functions import Heaviside
from .residuals import ResidualBlock
from .smu import SMU

def heaviside(x, k=10.0):
    return Heaviside.apply(x, k)
