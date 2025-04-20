import torch
from pytorch_optimizer import Lookahead

from .utils import get_HF_state, Ansatz
from .players import Solver, Proposer
from .operator import HermitianOp

class BasicTraining:

    def __init__(self, 
            solver=None, 
            num_states=None, 
            pool_size=5,
            width=64,
            depth=4
        ):

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        if solver is None and num_states is None:
            raise ValueError("You must provide the solver or the number of states.")
        elif solver is None:
            solver = Solver(num_states=num_states, pool_size=pool_size, width=width, depth=depth).double()

        self.solver = solver.to(self.device)
        self.num_states = solver.num_states
        self.pool_size = solver.pool_size
        self.size = 1 << self.num_states
        self.hamiltonian = HermitianOp(self.num_states, device=self.device)

    def calculate_exact(self):

        self.H = self.hamiltonian.to_tensor()
        eigvals = torch.linalg.eigvalsh(self.H)
        self.exact_energy = eigvals[..., 0]

    def calculate_uccsd(self):

        propagator = self.ansatz.get_propagator()
        gstate = (propagator * self.ansatz._state0[..., None, :]).sum(dim=-1)
        self.uccsd_energy = (
            gstate[..., None].conj() * self.H * gstate[..., None, :]
        ).sum(dim=(-1, -2)).real
        self.uccsd_state = gstate

    def criterion_step(self, retain_graph=False):

        energy_loss = self.huberloss(self.exact_energy, self.uccsd_energy)

        mel = energy_loss.mean()
        self.loss = mel
        self.loss.backward(retain_graph=retain_graph)

    def generate(self):
        pass
        
    def run(self, 
            training_steps=100,
            num_epochs=4,
            batch_size=5,
            retain_graph=False,
            verbosity=torch.inf,
            delta=1.0,
            lr=1e-3,
            weight_decay=1e-4
        ):

        self.solver.train()
        base_optimizer = torch.optim.Adam(self.solver.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(base_optimizer, T_0=100)

        self.hamiltonian.coefficients = torch.zeros(
            batch_size, self.hamiltonian.size, dtype=torch.complex128, device=self.device
        ) 
        self.ansatz = self.solver.generate_ansatz(
            2*torch.rand(batch_size, self.solver.size, dtype=torch.float64, device=self.device)
        )
        self.huberloss = torch.nn.HuberLoss(delta=delta)

        self.energy_diff = []
        for step in range(1, training_steps+1):
            self.inputs = self.generate()
            self.hamiltonian.update_from_flat_coefficients(self.inputs)
            self.calculate_exact()

            for epoch in range(num_epochs):
                optimizer.zero_grad()
                self.solver.update_ansatz(self.inputs, self.ansatz)
                self.calculate_uccsd()
                self.criterion_step(retain_graph=retain_graph)
                optimizer.step()
                scheduler.step()
            self.energy_diff.append([self.exact_energy.mean(), self.uccsd_energy.mean()])

            if step % verbosity == 0:
                print(f'Step {step}, Loss = {self.loss.item()}')
        self.energy_diff = torch.as_tensor(self.energy_diff)

class Random(BasicTraining):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate(self):
        inputs = 2*torch.rand(self.inputs_shape, dtype=torch.float64, device=self.device) - 1
        return inputs.requires_grad_()

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
            pool_size=5,
            width=64,
            depth=4
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
            pool_size=pool_size,
            width=width,
            depth=depth
        )

        if proposer is None:
            proposer = Proposer(num_states=self.num_states).double().to(self.device)

        if proposer.num_states != self.solver.num_states:
            raise ValueError(
                "The number of states in the proposer and solver are not consistent."
            )

        self.proposer = proposer

    def generate(self):

        self.prop_optimizer.step()
        self.prop_optimizer.zero_grad()

        self.noise = torch.rand(self.inputs_shape, dtype=torch.float64, device=self.device)
        data = self.proposer(self.noise)

        return data

    def run(self, batch_size=5, **kwargs):

        self.proposer.train()
        self.inputs_shape = (batch_size, self.proposer.size)
        self.prop_optimizer = torch.optim.Adam(self.proposer.parameters(), lr=1e-3, maximize=True)

        kwargs.pop('retain_graph', None)
        super().run(batch_size=batch_size, retain_graph=True, **kwargs)
