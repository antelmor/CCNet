import torch

from .utils import get_HF_state
from .players import Solver, Proposer
from .hamiltonian import HermitianOp, AntiHermitianOp

class BasicTraining:

    def __init__(self, 
            solver=None, 
            num_states=None, 
            num_electrons=1, 
            num_sets=1,
            hidden_size=128
        ):

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        if solver is None and num_states is None:
            raise ValueError("You must provide the solver or the number of states.")
        elif solver is None:
            solver = Solver(num_states, num_sets=num_sets, hidden_size=hidden_size).double()

        self.solver = solver.to(self.device)
        self.num_states = solver.num_states
        self.num_sets = solver.num_sets
        self.size = 1 << self.num_states
        self.hamiltonian = HermitianOp(self.num_states, device=self.device)
        self.ansatz = AntiHermitianOp(self.num_states)
        self.hf_state  = get_HF_state(self.num_states, num_electrons=num_electrons).to(self.device)

    def calculate_exact_energy(self):

        self.H = self.hamiltonian.to_tensor()
        eigvals, eigvecs = torch.linalg.eigh(self.H)
        self.exact_energy = eigvals[:, 0]
        self.exact_gstate = eigvecs[:, :, 0]

    def criterion_step(self, retain_graph=False):

        A = self.ansatz.to_tensor().reshape(-1, self.num_sets, self.size, self.size)
        exp = torch.matrix_exp(A)
        U = exp[:, 0]
        for i in range(1, exp.shape[1]):
            U @= exp[:, i]

        ground_state = U @ self.hf_state
        energy = torch.einsum('ni,nij,nj->n', ground_state.conj(), self.H, ground_state).real

        loss = ( (energy - self.exact_energy) / (self.exact_energy + energy) )**2
        loss = torch.log(loss + 1)
        self.loss = loss.mean()
        self.loss.backward(retain_graph=retain_graph)

    def criterion_step0(self, retain_graph=False):

        A = self.ansatz.to_tensor().reshape(-1, self.num_sets, self.size, self.size)
        exp = torch.matrix_exp(A)
        U = exp[:, 0]
        for i in range(1, exp.shape[1]):
            U @= exp[:, i]

        ground_state = U @ self.hf_state
        difference = ground_state - self.exact_gstate
        loss = torch.linalg.norm(difference, dim=1) / 2
        self.loss = loss.mean()
        self.loss.backward(retain_graph=retain_graph)

    def generate(self):
        pass
        
    def run(self, 
            training_steps=100,
            num_epochs=4,
            batch_size=5,
            retain_graph=False,
            verbosity=torch.inf
            ):

        self.solver.train()
        optimizer = torch.optim.Adam(self.solver.parameters(), lr=1e-3)

        self.hamiltonian.coefficients = torch.zeros(
            batch_size, self.hamiltonian.size, dtype=torch.complex128, device=self.device
        )
        self.ansatz.coefficients = torch.zeros(
            batch_size*self.num_sets, self.ansatz.size, dtype=torch.complex128, device=self.device
        )

        for step in range(training_steps):
            inputs = self.generate()
            self.hamiltonian.update_from_flat_coefficients(inputs)
            self.calculate_exact_energy()

            for epoch in range(num_epochs):
                self.solver.train()
                optimizer.zero_grad()
                outputs = self.solver(inputs).reshape(-1, self.solver.size)
                self.ansatz.update_from_flat_coefficients(outputs)
                self.criterion_step(retain_graph=retain_graph)
                optimizer.step()

            if step % verbosity == 0:
                print(f'Step {step}, Loss = {self.loss.item()}')

class Random(BasicTraining):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate(self):

        return 2*torch.rand(self.inputs_shape, dtype=torch.float64, device=self.device) - 1

    def run(self, batch_size=5, **kwargs):

        self.inputs_shape = (batch_size, 2*self.hamiltonian.size - self.hamiltonian._diagonal_index)
        super().run(batch_size=batch_size, **kwargs)

class Step(BasicTraining):

    def __init__(self, step_size=10, **kwargs):

        super().__init__(**kwargs)
        self.iteration = 0
        self.step_size = step_size

    def generate(self):

        i = self.iteration // self.step_size + 1
        shape = (self.inputs_shape[0], i)
        coefficients = torch.zeros(self.inputs_shape, dtype=torch.float64, device=self.device)
        coefficients[:, :i] = 2*torch.rand(shape, dtype=torch.float64, device=self.device) - 1
        self.iteration += 1

        return coefficients

    def run(self, batch_size=5, **kwargs):

        size = 2*self.hamiltonian.size - self.hamiltonian._diagonal_index
        self.inputs_shape = (batch_size, size)
        kwargs['training_steps'] = size*self.step_size - 1
        super().run(batch_size=batch_size, **kwargs)

class Game(BasicTraining):

    def __init__(self,
            proposer=None,
            solver=None,
            num_states=None,
            num_electrons=1,
            num_sets=1,
            hidden_size=128
            ):

        if solver is None and num_states is None:
            if proposer is None:
                raise ValueError(
                    "You must provide the proposer, the solver or the number of states."
                )
            else:
                num_states = proposer.num_states

        super().__init__(
            solver=solver,
            num_states=num_states,
            num_electrons=num_electrons,
            num_sets=num_sets,
            hidden_size=hidden_size
        )

        if proposer is None:
            proposer = Proposer(self.num_states).double().to(self.device) 

        if proposer.num_states != self.solver.num_states:
            raise ValueError(
                "The number of states in the proposer and solver are not consistent."
            )

        self.proposer = proposer

    def generate(self):

        try:
            for param in self.proposer.parameters():
                param.grad *= -1
        except TypeError:
            pass
        else:
            self.prop_optimizer.step()
            self.prop_optimizer.zero_grad()

        self.noise = torch.rand(self.inputs_shape, dtype=torch.float64, device=self.device)
        data = self.proposer(self.noise)

        return data

    def run(self, batch_size=5, **kwargs):

        self.proposer.train()
        self.inputs_shape = (batch_size, self.proposer.size)
        self.prop_optimizer = torch.optim.Adam(self.proposer.parameters(), lr=1e-3)

        kwargs.pop('retain_graph', None)
        super().run(batch_size=batch_size, retain_graph=True, **kwargs)
