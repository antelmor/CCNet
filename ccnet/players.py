import torch
from torch import nn
from math import comb

from .vqe import VQE
from .utils import get_HF_state, Ansatz, heaviside
from .operator import HermitianOp
from .utils import ResidualBlock

class Player(nn.Module):

    def __init__(self, num_states=2, width=64, depth=4, smooth=True, normalized=False):
        super(Player, self).__init__()
        self.num_states = num_states

        num_first_pairs = comb(num_states, 2)
        num_second_pairs = comb(num_first_pairs, 2)
        self.size = 2*num_second_pairs + 3*num_first_pairs + num_states

        layers = [
            ResidualBlock(width, smooth=smooth, normalized=normalized) 
            for _ in range(depth-1)
        ]
        self.base_fc = nn.Sequential(
            nn.Linear(self.size, width),
            nn.GELU() if smooth else nn.ReLU(),
            *layers
        )

        self.head = nn.Linear(width, self.size)

    def forward(self, x):
        
        x = self.base_fc(x)
        coefficients = self.head(x)

        return coefficients

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):

        player = cls(**kwargs)
        player.load_state_dict(state_dict)

        return player

class Proposer(Player):

    def __init__(self, smooth=False, normalized=True, **kwargs):
        super(Proposer, self).__init__(smooth=smooth, normalized=normalized, **kwargs)

        width = kwargs.pop('width', 64)
        self.head = nn.Sequential(
            nn.Linear(width, self.size),
            nn.Tanh()
        )

    def forward(self, x):
        x = super(Proposer, self).forward(x)

        return 4*x

    def propose_hamiltonian(self, batch_size=1):

        noise = torch.rand(batch_size, self.size, dtype=torch.float64)
        outputs = self.forward(noise)

        hamiltonian = HermitianOp(self.num_states, batch_shape=(batch_size,))
        hamiltonian.update_from_flat_coefficients(outputs)

        return hamiltonian

class Solver(Player):

    def __init__(self, 
            pool_size=5, 
            k_param=10.0,
            num_states=2,
            width=64,
            depth=2,
            smooth=False,
            normalized=False
        ):
        super(Solver, self).__init__(
            num_states=num_states,
            width=width,
            depth=depth,
            smooth=smooth,
            normalized=normalized
        ) 
        self.pool_size = pool_size
        self.k = k_param

        num_first_pairs = comb(num_states, 2)
        num_second_pairs = comb(num_first_pairs, 2)
        self.num_sectors = num_states + 1
        self.out_size = num_second_pairs + 2*num_first_pairs + num_states
        self.coeff_size = pool_size * self.out_size
        self.init_indices = (1 << torch.arange(self.num_sectors)) - 1

        self.head = nn.Linear(width, self.coeff_size + self.num_sectors)

    def discretize(self, x):
        x = x - x.max(dim=-1, keepdim=True)[0]

        return heaviside(x, k=self.k)

    def forward(self, x):
        
        x = self.base_fc(x)
        x = self.head(x)

        coefficients = x[..., :self.coeff_size].reshape(-1, self.pool_size, self.out_size)
        init_state_weights = x[..., self.coeff_size:]

        coefficients = self.discretize(coefficients)
        init_state_weights = self.discretize(init_state_weights)

        return coefficients, init_state_weights

    def update_ansatz(self, inputs, ansatz):
        
        coeffs, weights = self.forward(inputs)
        idiag = ansatz._diagonal_index
        ansatz._coefficients = coeffs.to(torch.complex128)
        ansatz._coefficients[..., :idiag] *= 1j
        ansatz._tensor = ansatz.to_tensor()

        new_state0 = torch.zeros_like(ansatz._state0)
        new_state0[..., self.init_indices] = torch.complex(weights, torch.zeros_like(weights))
        ansatz._state0 = new_state0

    def generate_ansatz(self, inputs):

        ansatz = Ansatz(
                self.num_states, 
                num_parameters=self.pool_size, 
                batch_shape=inputs.shape[:-1]
        )
        init_state = torch.zeros(
                *inputs.shape[:-1], 
                1 << self.num_states, 
                dtype=torch.complex128,
                device=next(self.parameters()).device
        )
        ansatz.init_state = init_state
        
        self.update_ansatz(inputs, ansatz)

        return ansatz

    def assemble_vqe(self, hamiltonian, **options):

        if hamiltonian.num_spin_orbitals != self.num_states:
            raise ValueError(
                f"The solver can only deal with hamiltonians with {self.num_states} states."
            )

        ir = hamiltonian._diagonal_index
        coefficients = hamiltonian.coefficients
        inputs = torch.concatenate([coefficients.real, coefficients.imag[:, ir:]], dim=1)
        
        ansatz = self.generate_ansatz(inputs)

        return VQE(hamiltonian, ansatz, **options)
