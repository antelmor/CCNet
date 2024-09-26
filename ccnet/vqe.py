import torch
import numpy as np
from typing import Sequence, Union
from hamiltonian import HamiltonianOp
from qiskit.quantum_info import SparsePauliOp

class VQE:

    def __init__(
            self, 
            hamiltonian: Union[SparsePauliOp, HamiltonianOp, np.ndarray] = None, 
            ansatz: Union[Sequence[Union[SparsePauliOp, HamiltonianOp]], np.ndarray] = None, 
            num_electrons: int = 1
        ):

        self._matrix = None

        self.num_electrons = num_electrons
        self.num_qubits = None
        self.hamiltonian = hamiltonian
        self.ansatz = ansatz

        self.energy = 0.0
        self.ground_state = None
        self.angles = None
        self.propagator = None

    @property
    def hamiltonian(self):
        return self._hamiltonian

    @hamiltonian.setter
    def hamiltonian(self, value):

        if isinstance(value, (HamiltonianOp, SparsePauliOp) ):
            if not value.is_hermitian():
                raise ValueError("The hamiltonian must be Hermitian.")
            self._hamiltonian = value
            self.matrix = value.to_matrix()

            try:
                self.num_qubits = value.num_spin_orbitals
            except AttributeError:
                self.num_qubits = value.num_qubits

        elif isinstance(value, (np.ndarray, torch.Tensor) ):
            self._hamiltonian = None
            self.matrix = value

        elif value is not None:
            raise TypeError(
                "Hamiltonian allowed types: 'HamiltonianOp', 'SparsePauliOp', 'numpy.ndarray', 'torch.Tensor'."
            )

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        
        if value is None:
            self._matrix = value
            self.num_qubits = value
        else:
            
            try:
                tensor = torch.tensor(value, dtype=torch.complex128)
            except (ValueError, TypeError):
                raise TypeError("The matrix must be a sequence of numerical data.")

            if len(tensor.shape) != 2:
                raise ValueError(
                    "The hamiltonian matrix must be square of shape (2**n, 2**n) for an integer n."
                )
            elif tensor.shape[0] != tensor.shape[1] or np.log2(tensor.shape[0]) % 1 != 0:
                raise ValueError(
                    "The hamiltonian matrix must be square of shape (2**n, 2**n) for an integer n."
                )
            elif not torch.allclose(tensor, tensor.T.conj()):
                raise ValueError(
                    "The hamiltonian matrix must be Hermitian."
                )
                
            self._matrix = tensor
            self.num_qubits = tensor.shape[-1].bit_length() - 1

    @property
    def ansatz(self):
        return self._ansatz

    @ansatz.setter
    def ansatz(self, value):

        if value is None:
            pass
        elif self.matrix is None and self.hamiltonian is None:
            raise AttributeError(
                "The 'hamiltonian' or 'matrix' attribute must be set before 'ansatz'."
            )
        elif isinstance(value, (np.ndarray, torch.Tensor) ):
            dim = 1 << self.num_qubits
            if len(value.shape) != 3 or value.shape[1:] != (dim, dim):
                raise ValueError(
                    "The anstaz array must be of shape (m, nqubits, nqubits)."
                )
        elif isinstance(value, Sequence):
            if not all( isinstance(element, (HamiltonianOp, SparsePauliOp)) for element in value ):
                raise TypeError(
                    "The ansatz must be a list of Hamiltonian or Qubit operators."
                )
        else:
            raise TypeError(
                "The ansatz must be a numpy.ndarray, a torch.Tensor, or a list of operators."
            )

        if isinstance(value, Sequence):
            self._ansatz = list(value) 
            array = np.stack([operator.to_matrix() for operator in self._ansatz])
            self._ansatz_tensor = torch.tensor(array, dtype=torch.complex128)
        elif isinstance(value, np.ndarray):
            self._ansatz = value
            self._ansatz_tensor = torch.tensor(value, dtype=torch.complex128)
        else:
            self._ansatz = value
            self._ansatz_tensor = value

    def run(self, **kwargs):
        
        H = self._matrix
        ansatz = self._ansatz_tensor
        #torch.autograd.set_detect_anomaly(True)

        hf_state = torch.zeros(2**self.num_qubits, dtype=torch.complex128)
        hf_bin = '0'*( (self.num_qubits-self.num_electrons) // 2) + '1'*(self.num_electrons // 2)
        hf_state[int(2*hf_bin, 2)] = 1.0
        
        def energy(theta):
            propagator = torch.linalg.multi_dot([*torch.matrix_exp(theta[:, None, None] * ansatz)])
            ground_state = propagator @ hf_state
            return (ground_state.conj() @ H @ ground_state).real

        def closure():
            optimizer.zero_grad()
            loss = energy(theta)
            loss.backward(retain_graph=True)
            return loss

        num_iterations = kwargs.pop('num_iterations', 100)
        theta = torch.rand(len(ansatz), requires_grad=True)
        optimizer = torch.optim.LBFGS([theta], **kwargs)

        for i in range(num_iterations):
            optimizer.step(closure)

        self.angles = theta
        self.propagator = torch.linalg.multi_dot( [*torch.matrix_exp(theta[:, None, None] * ansatz)])
        self.ground_state = self.propagator @ hf_state
        self.energy = energy(theta)
