import torch
from torch import nn
from math import comb

from .vqe import VQE
from .utils import get_HF_state, Ansatz, heaviside
from .operator import HermitianOp
from .utils import ResidualBlock

class Player(nn.Module):

    def __init__(self, num_states=2, width=64, depth=4):
        super(Player, self).__init__()
        self.num_states = num_states

        num_first_pairs = comb(num_states, 2)
        num_second_pairs = comb(num_first_pairs, 2)
        self.size = 2*num_second_pairs + 3*num_first_pairs + num_states

        layers = [ResidualBlock(width) for _ in range(depth-1)]
        self.base_fc = nn.Sequential(
            nn.Linear(self.size, width),
            nn.BatchNorm1d(width),
            nn.ReLU(),
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

    def __init__(self, **kwargs):
        super(Proposer, self).__init__(**kwargs)

        width = kwargs.pop('width', 64)
        self.head = nn.Sequential(
            nn.Linear(width, self.size),
            nn.Tanh()
        )

    def forward(self, x):
        x = super(Proposer, self).forward(x)

        return x

    def propose_hamiltonian(self, batch_size=1):

        noise = torch.rand(batch_size, self.size, dtype=torch.float64)
        outputs = self.forward(noise)

        hamiltonian = HermitianOp(self.num_states, batch_shape=(batch_size,))
        hamiltonian.update_from_flat_coefficients(outputs)

        return hamiltonian

class Solver(Player):

    def __init__(self, pool_size=5, **kwargs):
        super(Solver, self).__init__(**kwargs)
        
        self.pool_size = pool_size
        width = kwargs.pop('width', 64)
        self.head = nn.Sequential(
            nn.Linear(width, pool_size*self.size),
            nn.Tanh()
        )

    def forward(self, x):
        
        x = self.base_fc(x)
        x = self.head(x).reshape(-1, self.pool_size, self.size)

        return x

    def update_ansatz(self, inputs, ansatz):
        
        coeffs = self.forward(inputs)
        ansatz.update_from_flat_coefficients(coeffs)
        ansatz._tensor = ansatz.to_tensor()

    def generate_ansatz(self, inputs):

        ansatz = Ansatz(
                self.num_states, num_parameters=self.pool_size, batch_shape=inputs.shape[:-1]
        )
        ansatz.init_state = get_HF_state(self.num_states, num_electrons=self.num_states // 2)
        self.update_ansatz(inputs, ansatz)

        return ansatz

    def solve(self, hamiltonian):

        if hamiltonian.num_spin_orbitals != self.num_states:
            raise ValueError(
                f"The solver can only deal with hamiltonians with {self.num_states} states."
            )

        ir = hamiltonian._diagonal_index
        coefficients = hamiltonian.coefficients
        inputs = torch.concatenate([coefficients.real, coefficients.imag[:, ir:]], dim=1)
        
        H = hamiltonian.to_tensor()
        ansatz = self.generate_ansatz(inputs)

        propagator = ansatz.get_propagator()
        uccsd_state = (propagator * ansatz._state0[..., None, :]).sum(dim=-1)
        uccsd_energy = (
            uccsd_state[..., None].conj() * H * uccsd_state[..., None, :]
        ).sum(dim=(-1, -2)).real

        return uccsd_energy, uccsd_state
