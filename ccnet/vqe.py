import torch
from .utils import Ansatz, get_HF_state
from .operator import HermitianOp

class VQE:

    def __init__(
            self, 
            hamiltonian: HermitianOp,
            ansatz: Ansatz,
            num_electrons: int = 1,
            optimizer_type: str = 'Adam'
        ):

        if hamiltonian.num_spin_orbitals != ansatz.num_spin_orbitals:
            raise ValueError("The hamiltonian and ansatz must have the same number of states.")
        
        self.hamiltonian = hamiltonian
        self.ansatz = ansatz
        self.num_electrons = num_electrons
        self.num_qubits = hamiltonian.num_spin_orbitals

        shape = ansatz.coefficients.shape[:-1]
        self.angles = torch.rand(shape, dtype=torch.float64, requires_grad=True)

        if ansatz.init_state is None:
            hf_state = get_HF_state(self.num_qubits, num_electrons=num_electrons)
            shape = ansatz.coefficients.shape[:-2]
            ansatz.init_state = hf_state.repeat(shape, 1)

        if optimizer_type == 'Adam':
            self.optimizer = torch.optim.Adam([self.angles])
        elif optimizer_type == 'SGD':
            self.optimizer = torch.optim.SGD([self.angles])
        elif optimizer_type == 'LBFGS':
            self.optimizer = torch.optim.LBFGS([self.angles])
        else:
            raise ValueError(
                f"Unrecognized optimizer mode '{optimizer_type}'. Valid options are: Adam, SGD, or LBFGS"
            )

    def modify_optimizer(self, **options):

        for group in self.optimizer.param_groups:
            group |= options

    def compute_energy(self):

        propagator = self.ansatz.get_propagator(self.angles)
        ground_state = torch.einsum('...ij,...j->...i', propagator, self.ansatz.init_state)
        self.energy = torch.einsum(
            '...i,...ij,...j->...', 
            ground_state.conj(), self.H, ground_state
        ).real

    def closure(self):

        self.optimizer.zero_grad()
        self.compute_energy()
        self.loss = self.energy.sum()
        self.loss.backward(retain_graph=True)

        return self.loss

    def run(self, max_iterations=50, etol=1e-5, verbosity=False, **kwargs):
        
        self.modify_optimizer(**kwargs)
        self.H = self.hamiltonian.to_tensor()
        self.energy = torch.zeros(self.H.shape[0], dtype=torch.float64)

        for _ in range(max_iterations):
            previous = self.energy
            self.optimizer.step(self.closure)
            ediff = torch.abs(previous - self.energy).max()
            if verbosity:
                print(f'Energy: {self.loss.item()}, Ediff: {ediff.item()}')
            if ediff < etol:
                break

        self.propagator = self.ansatz.get_propagator(self.angles)
        self.ground_state = torch.einsum(
            '...ij,...j->...i', self.propagator, self.ansatz.init_state
        )
