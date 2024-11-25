import torch
from torch import nn
from math import comb

from .vqe import VQE
from .utils import get_HF_state, Ansatz
from .operator import HermitianOp

class Player(nn.Module):

    def __init__(self, num_states, width=64, depth=4):
        super(Player, self).__init__()
        self.num_states = num_states

        num_first_pairs = comb(num_states, 2)
        num_second_pairs = comb(num_first_pairs, 2)
        self.size = 2*num_second_pairs + 3*num_first_pairs + num_states

        layers = [nn.Linear(width, width), nn.ReLU()]*(depth-1)
        self.base_fc = nn.Sequential(
            nn.Linear(self.size, width),
            nn.ReLU(),
            *layers
        )

        self.head = nn.Linear(width, self.size)

    def forward(self, x):
        
        x = self.base_fc(x)
        coefficients = self.head(x)

        return coefficients

class Proposer(Player):

    def __init__(self, num_states, **kwargs):
        super(Proposer, self).__init__(num_states, **kwargs)

    def forward(self, x):
        x = super(Proposer, self).forward(x)

        return x / x.abs().amax(dim=-1)[:, None]

    def propose_hamiltonian(self, batch_size=1):

        noise = torch.rand(batch_size, self.size, dtype=torch.float64)
        outputs = self.forward(noise)

        hamiltonian = HermitianOp(self.num_states, batch_shape=(batch_size,))
        hamiltonian.update_from_flat_coefficients(outputs)

        return hamiltonian

class Solver(Player):

    _sigmoid_factor = 1e+6

    def __init__(self, num_states, pool_size=5, width=64, depth=4):
        super(Solver, self).__init__(num_states, width=width, depth=depth)
        self.pool_size = pool_size

        self.out_size = self.head.out_features
        self.head = nn.Linear(width, pool_size*self.out_size)

        self.state_fc = nn.Sequential(
            nn.Linear(self.size, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU()
        )
        self.head_state = nn.Linear(width, 1 << num_states)

    def pseudo_heaviside(self, x):

        probs = nn.functional.softmax(x, dim=-1)
        probs = probs - probs.max(dim=-1)[0][..., None]
        result = 2*torch.sigmoid(self._sigmoid_factor * probs)

        return result

    def forward(self, x):
        
        x1 = self.base_fc(x)
        x2 = self.state_fc(x)

        x1 = self.head(x1).reshape(-1, self.pool_size, self.out_size)
        coefficients = self.pseudo_heaviside(x1)

        x2 = self.head_state(x2)
        init_state = self.pseudo_heaviside(x2)

        return init_state, coefficients

    def update_ansatz(self, inputs, ansatz):

        idiag = ansatz._diagonal_index
        init_state, coeffs = self.forward(inputs)

        ansatz._state0 = init_state.to(torch.complex128)
        ansatz._coefficients = coeffs.to(torch.complex128)
        ansatz._coefficients[..., :idiag] *= 1j
        ansatz._coefficients[..., idiag:] *= torch.exp(2*torch.pi*1j* coeffs[..., idiag:])
        ansatz._tensor = ansatz.to_tensor()

    def generate_ansatz(self, inputs):

        ansatz = Ansatz(
                self.num_states, num_parameters=self.pool_size, batch_shape=inputs.shape[:-1]
        )
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
