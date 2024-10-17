import torch
import numpy as np
from itertools import combinations, combinations_with_replacement

def hamming_mod2_weight(binary, size):

    temp = binary.clone()
    result = 0
    for i in range(size):
        result ^= temp % 2
        temp >>= 1

    return result

def get_binaries(*label):

    n = len(label) // 2
    m, p, z = 0, 0, 0
    for i in range(n):
        a = 1 << label[i]
        b = 1 << label[i+n]
        m |= a
        p |= b
        z ^= (a-1) ^ (b-1)
    z &= ~(m | p)

    return m, p, z

class FermionOp:

    def __init__(self, labels, num_spin_orbitals=1, batch_shape=(), device=None):

        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = device

        try:
            the_labels = [ tuple(int(i) for i in label) for label in labels ]
        except (TypeError, ValueError) as err:
            raise type(err)("input must be an iterable of tuples containing integers")

        if any( len(label) % 2 != 0 for label in the_labels ):
            raise ValueError("Each label must have even length.")
        elif any( min(label) < 0 for label in the_labels ):
            raise ValueError("The integers of each label must be positive.")

        if num_spin_orbitals == -1:
            num_spin_orbitals = max( max(the_labels) )

        if not isinstance(num_spin_orbitals, int):
            raise TypeError("'num_spin_orbitals' must be a positive integer.")
        elif num_spin_orbitals < 1:
            raise ValueError("'num_spin_orbitals' must be a positive integer.")

        try:
            batch_shape = tuple(int(n) for n in batch_shape)
        except (TypeError, ValueError):
            raise type(err)("'batch_shape' must be an iterable of integers")
        else:
            if any(n < 1 for n in batch_shape):
                raise ValueError("The integers of 'batch_shape' must be positive.")

        indices = list(range(num_spin_orbitals)) 

        size = len(the_labels)
        shape = (*batch_shape, size)
        self.num_spin_orbitals = num_spin_orbitals

        self._size = size
        self._labels = the_labels
        self._coefficients = torch.ones( shape, dtype=torch.complex128, device=device )
        self._set_like_array()
        self._set_factors()

    def _set_like_array(self):

        like_array = [get_binaries(*label) for label in self._labels]        
        self._like_array = torch.as_tensor(like_array, device=self.device).T

    def _set_factors(self):

        factors = [
            -1 if len(label) > 2 or label[1] < label[0] else 1 for label in self._labels
        ]
        self._factors = torch.as_tensor(factors, dtype=torch.float64, device=self.device)

    @property
    def size(self):
        return self._size

    @property
    def labels(self):
        return self._labels

    @property
    def m_like(self):
        return self._like_array[0]

    @property
    def p_like(self):
        return self._like_array[1]

    @property
    def z_like(self):
        return self._like_array[2]

    @property
    def factors(self):
        return self._factors

    @property
    def coefficients(self):
        return self._coefficients

    @coefficients.setter
    def coefficients(self, values):
        self._coefficients = self._get_valid_coefficients(values)

    def _get_valid_coefficients(self, values):

        try:
            coefficients = torch.as_tensor(values)
        except (TypeError, ValueError, RuntimeError) as err:
            raise type(err)("The coefficients must be a tensor like object with numerical data.")

        num_coefficients = self._coefficients.shape[-1]
        if coefficients.shape[-1] != num_coefficients:
            raise ValueError(
                f"The coefficients must be of shape (...,{num_coefficients})"
            )

        return coefficients

    @property
    def terms(self):
        return {label: coeff for label, coeff in zip(self._labels, self._coefficients)}

    def to_tensor(self):

        size = 1 << self.num_spin_orbitals
        batch_shape = self.coefficients.shape[:-1]
        i = torch.arange(size, device=self.device)
        tensor = torch.zeros(*batch_shape, size, size, dtype=torch.complex128, device=self.device)
        
        m_like, p_like = self.m_like, self.p_like
        x = m_like ^ p_like
        a = m_like & p_like

        bin_tensor = i[:, None] & (p_like[None, :] ^ a[None, :])
        bin_tensor |= ~i[:, None] & (m_like[None, :] ^ a[None, :])
        bin_tensor |= ~i[:, None] & a[None, :]

        idx, jdx = torch.where(bin_tensor == 0)
        kdx = idx ^ x[jdx]
        factors = self.factors[jdx]
        weights = hamming_mod2_weight(idx & self.z_like[jdx], size)
        factors[weights != 0] *= -1

        N = len(batch_shape) + 1
        broadcast = [ [slice(None) if i == j else None for j in range(N)] for i in range(N)]
        indices = tuple(torch.arange(n)[*broadcast[i]] for i, n in enumerate(batch_shape))
        indices += (idx[*broadcast[-1]], kdx[*broadcast[-1]])
        
        tensor = tensor.index_put(
            indices, factors * self.coefficients[..., jdx], accumulate=True
        )

        return tensor

class HermitianOp(FermionOp):

    def __init__(self, num_spin_orbitals, batch_shape=(), device=None):

        if not isinstance(num_spin_orbitals, int):
            raise TypeError("Position arguments must be positive integers.")
        elif num_spin_orbitals < 1:
            raise ValueError("Position arguments must be positive integers.")

        indices = list(range(num_spin_orbitals))

        real_labels = [
            (i, j, i, j) if i != j else (i, j)
            for i, j in combinations_with_replacement(indices, 2)
        ]
        self._diagonal_index = len(real_labels)

        complex_labels = [
            (i, j, k, l) if (i, j) != (k, l) else (i, j)
            for (i, j), (k, l) in combinations_with_replacement(combinations(indices, 2), 2)
        ]
        labels = real_labels + complex_labels

        super().__init__(
            labels, num_spin_orbitals=num_spin_orbitals, batch_shape=batch_shape, device=device
        )

        self._factors[:self._diagonal_index] /= 2

    @property
    def coefficients(self):
        return self._coefficients

    @coefficients.setter
    def coefficients(self, values):

        the_coefficients = self._get_valid_coefficients(values)

        ir = self._diagonal_index
        if the_coefficients[..., :ir].imag.any():
            raise ValueError(
                f"The first {ir} coefficients must be real for the operator to be Hermitian."
            )

        self._coefficients = the_coefficients

    def update_from_flat_coefficients(self, values):

        coefficients = torch.zeros(
            self._coefficients.shape, dtype=torch.complex128, device=self.device
        )
        
        si = self.size
        di = self._diagonal_index
        coefficients[..., :di] = values[..., :di]
        coefficients[..., di:] = values[..., di:si] + 1j*values[..., si:]
    
        self._coefficients = coefficients

    def to_tensor(self):

        tensor = super().to_tensor()
        tensor += tensor.swapaxes(-1, -2).conj()

        return tensor

class AntiHermitianOp(HermitianOp):

    def __init__(self, num_spin_orbitals, batch_shape=(), device=None): 

        super().__init__(num_spin_orbitals, batch_shape=batch_shape, device=device)

    @property
    def coefficients(self):
        return self._coefficients

    @coefficients.setter
    def coefficients(self, values):

        the_coefficients = self._get_valid_coefficients(values)

        ii = self._diagonal_index
        if the_coefficients[..., :ii].imag.any():
            raise ValueError(
                f"The first {ii} coefficients must be real for the operator to be AntiHermitian."
            )

        the_coefficients[..., :ii] *= 1j
        self._coefficients = the_coefficients

    def update_from_flat_coefficients(self, values):

        di = self._diagonal_index
        super(AntiHermitianOp, self).update_from_flat_coefficients(values)
        self._coefficients[..., :di] *= 1j

    def to_tensor(self):

        tensor = super(HermitianOp, self).to_tensor()
        tensor -= tensor.swapaxes(-1, -2).conj()

        return tensor
