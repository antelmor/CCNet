import torch
import numpy as np

pauli = {
    'I': np.array([
        [1, 0],
        [0, 1]], dtype=complex),
    'X': np.array([
        [0, 1],
        [1, 0]], dtype=complex),
    'Y': 1j*np.array([
        [0, -1],
        [1, 0]], dtype=complex),
    'Z': np.array([
        [1, 0],
        [0, -1]], dtype=complex)
}

def convert(term, size):

    string = size*['I',]
    for i, char in term:
        string[-i-1] = char

    result = np.array(1)
    for char in string:
        result = np.kron(pauli[char], result)

    return result

def openfermion2matrix(operator, num_qubits):

    size = 1 << num_qubits
    matrix = np.zeros((size, size), dtype=complex)
    for term, coefficient in operator.terms.items():
        matrix += coefficient*convert(term, num_qubits)

    return matrix

def get_HF_state(num_qubits, num_electrons=1):

    hf_state = torch.zeros(1 << num_qubits, dtype=torch.complex128)
    hf_bin = '0'*( (num_qubits-num_electrons) // 2) + '1'*(num_electrons // 2)
    hf_state[int(2*hf_bin, 2)] = 1.0

    return hf_state
