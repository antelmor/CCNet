import torch
from .quantum_states import get_HF_state 
from ..operator import AntiHermitianOp

class Ansatz(AntiHermitianOp):

    def __init__(self, num_spin_orbitals, batch_shape=(), num_parameters=5, device=None):

        shape = (*batch_shape, num_parameters)
        super().__init__(num_spin_orbitals, batch_shape=shape, device=device)
        self.num_parameters = num_parameters
        
        n_sectors = num_spin_orbitals + 1
        init_state = torch.zeros(n_sectors, 1 << num_spin_orbitals, dtype=torch.complex128)
        init_indices = (1 << torch.arange(n_sectors)) - 1
        init_state[torch.arange(n_sectors), init_indices] = 1.0
        self.init_state = init_state

    @AntiHermitianOp.coefficients.setter
    def coefficients(self, values):

        AntiHermitianOp.coefficients.fset(self, values)
        self._tensor = self.to_tensor()

    @property
    def init_state(self):
        return self._state0

    @init_state.setter
    def init_state(self, value):

        try:
            init_state = torch.as_tensor(value)
        except (TypeError, ValueError, RuntimeError) as err:
            raise type(err)(f"'init_state' must be a tensor like object with numerical data.")

        size = 1 << self.num_spin_orbitals
        if init_state.shape[-1] != size:
            raise ValueError(f"'init_state' size is inconsistent. Last dim must be '{size}'")

        self._state0 = init_state.to(self.device)

    def get_propagator(self, angles=None):

        if angles is None:
            U = torch.linalg.matrix_exp(self._tensor)
        else:
            U = torch.linalg.matrix_exp(angles.clone()[..., None, None] * self._tensor)

        propagator = U[..., 0, :, :]
        for mat in torch.unbind(U[..., 1:, :, :], dim=-3):
            propagator = propagator @ mat

        return propagator
