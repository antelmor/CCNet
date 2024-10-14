import torch
from ..hamiltonian import AntiHermitianOp

class Ansatz(AntiHermitianOp):

    def __init__(self, num_spin_orbitals, num_parameters=5, device=None):
        super().__init__(num_spin_orbitals, batch_size=num_parameters, device=device)
       
        self.num_parameters = num_parameters
        self._unitary = torch.matrix_exp( self.to_tensor() )

    @property
    def unitary(self):
        return self._unitary

    def get_propagator(self, angles):

        U = angles[:, :, None, None] * self._unitary

        propagator = U[:, 0]
        for mat in torch.unbind(U[:, 1:], dim=1):
            propagator @= mat

        return propagator
