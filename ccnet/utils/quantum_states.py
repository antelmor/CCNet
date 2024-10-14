import torch

def get_HF_state(num_qubits, num_electrons=1):

    hf_state = torch.zeros(1 << num_qubits, dtype=torch.complex128)
    hf_bin = '0'*( (num_qubits-num_electrons) // 2) + '1'*(num_electrons // 2)
    hf_state[int(2*hf_bin, 2)] = 1.0

    return hf_state
