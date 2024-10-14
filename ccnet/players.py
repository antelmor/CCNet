import torch
from torch import nn
from math import comb

from .utils import get_HF_state
from .operator import HermitianOp, AntiHermitianOp

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

    def __init__(self, num_states, pool_size=5, hidden_size=128):
        super(Solver, self).__init__(num_states, hidden_size=hidden_size)
        self.pool_size = pool_size

        self.fc3 = nn.Linear(self._hidden_size, pool_size*self.size)

    def forward(self, x):
        x = super(Solver, self).forward(x)

        return x.reshape(-1, self.pool_size, self.size)

    def run_vqe(self, hamiltonian, num_electrons=1):

        if hamiltonian.num_spin_orbitals != self.num_states:
            raise ValueError(
                f"The solver can only deal with hamiltonians with {self.num_states} states."
            )

        ir = hamiltonian._diagonal_index
        coefficients = hamiltonian.coefficients

        ansatz = AntiHermitianOp(self.num_states, batch_size=self.num_sets*coefficients.shape[0])

        inputs = torch.concatenate([coefficients.real, coefficients.imag[:, ir:]], dim=1)
        outputs = self.forward(inputs).reshape(-1, self.size)

        size = 1 << self.num_states
        ansatz.update_from_flat_coefficients(outputs)
        A = ansatz.to_tensor().reshape(-1, self.num_sets, size, size)
        exp = torch.matrix_exp(A)

        U = exp[:, 0]
        for i in range(1, self.num_sets):
            U @= exp[:, i]

        hf_state = get_HF_state(self.num_states, num_electrons)
        ground_state = U @ hf_state
        H = hamiltonian.to_tensor()
        energy = torch.einsum('ni,nij,nj->n', ground_state.conj(), H, ground_state)

        return energy, ground_state, U
