import torch
from ..operator import AntiHermitianOp

class Ansatz(AntiHermitianOp):

    def __init__(self, num_spin_orbitals, batch_shape=(), num_parameters=5, device=None):

        shape = (*batch_shape, num_parameters)
        super().__init__(num_spin_orbitals, batch_shape=shape, device=device)
        self.num_parameters = num_parameters

    @AntiHermitianOp.coefficients.setter
    def coefficients(self, values):

        AntiHermitianOp.coefficients.fset(self, values)
        self._tensor = self.to_tensor()

    def get_propagator(self, angles):

        U = torch.matrix_exp(angles[..., None, None] * self._tensor)

        propagator = U[..., 0, :, :]
        for mat in torch.unbind(U[..., 1:, :, :], dim=-3):
            propagator @= mat

        return propagator
