import torch
from torch import nn
from math import comb

from .vqe import VQE
from .utils import get_HF_state, Ansatz
from .operator import HermitianOp

class Player(nn.Module):

    def __init__(self, num_states, hidden_size=128):
        super(Player, self).__init__()
        self.num_states = num_states
        self._hidden_size = hidden_size

        num_first_pairs = comb(num_states, 2)
        num_second_pairs = comb(num_first_pairs, 2)
        self.size = 2*num_second_pairs + 3*num_first_pairs + num_states

        self.fc1 = nn.Linear(self.size, self._hidden_size)
        self.fc2 = nn.Linear(self._hidden_size, self._hidden_size)
        self.fc3 = nn.Linear(self._hidden_size, self.size)

    def forward(self, x):
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        coefficients = self.fc3(x)

        return coefficients

class Proposer(Player):

    def __init__(self, num_states):
        super(Proposer, self).__init__(num_states)

    def forward(self, x):
        x = super(Proposer, self).forward(x)

        return x / x.abs().amax(dim=-1)[:, None]

    def propose_hamiltonian(self, batch_size=1):

        noise = torch.rand(batch_size, self.size, dtype=torch.float64)
        outputs = self.forward(noise)

        hamiltonian = HermitianOp(self.num_states, batch_size=batch_size)
        hamiltonian.update_from_flat_coefficients(outputs)

        return hamiltonian

class Solver(Player):

    _sigmoid_factor = 1e+6

    def __init__(self, num_states, pool_size=5, hidden_size=128):
        super(Solver, self).__init__(num_states, hidden_size=hidden_size)
        self.pool_size = pool_size

        num_first_pairs = comb(num_states, 2)
        num_second_pairs = comb(num_first_pairs, 2)
        self.out_size = num_second_pairs + 2*num_first_pairs + num_states

        self.fc3 = nn.Linear(self._hidden_size, pool_size*self.out_size)

    def forward(self, x):
        
        x = super(Solver, self).forward(x).reshape(-1, self.out_size)
        probs = nn.functional.softmax(x, dim=1)
        probs = probs - probs.max(dim=1)[0][:, None]
        coefficients = 2*torch.sigmoid(self._sigmoid_factor * probs)

        return coefficients

    def create_ansatz(self, inputs, ansatz=None):

        outputs = self.forward(inputs)
        if ansatz is None:
            ansatz = Ansatz(self.num_states, num_parameters=self.pool_size)
        
        ansatz._coefficients = outputs.to(torch.complex128)
        ansatz._coefficients[:, :ansatz._diagonal_index] *= 1j
        ansatz._tensor = ansatz.to_tensor()

        return ansatz

    def assemble_vqe(self, hamiltonian, **options):

        if hamiltonian.num_spin_orbitals != self.num_states:
            raise ValueError(
                f"The solver can only deal with hamiltonians with {self.num_states} states."
            )

        ir = hamiltonian._diagonal_index
        coefficients = hamiltonian.coefficients
        inputs = torch.concatenate([coefficients.real, coefficients.imag[:, ir:]], dim=1)
        
        ansatz = self.create_ansatz(inputs)

        self.vqe = VQE(hamiltonian, ansatz, **options)
