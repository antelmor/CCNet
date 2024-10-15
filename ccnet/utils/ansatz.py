import torch
from ..operator import AntiHermitianOp

class Ansatz(AntiHermitianOp):

    def __init__(self, num_spin_orbitals, num_parameters=5, device=None):
        super().__init__(num_spin_orbitals, batch_size=num_parameters, device=device)
       
        self.num_parameters = num_parameters

    @AntiHermitianOp.coefficients.setter
    def coefficients(self, values):

        AntiHermitianOp.coefficients.fset(self, values)
        self._tensor = self.to_tensor()

    def get_propagator(self, angles):

        U = torch.matrix_exp(angles[:, :, None, None] * self._tensor)

        propagator = U[:, 0]
        for mat in torch.unbind(U[:, 1:], dim=1):
            propagator @= mat

        return propagator
