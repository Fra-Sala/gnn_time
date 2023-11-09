import torch
from torch import nn
from torchdiffeq import odeint_adjoint as odeint

class RecurrentODESolver(nn.Module):
    def __init__(self, func, t0, t1, h):
        super().__init__()
        self.func = func
        self.t0 = t0
        self.t1 = t1
        self.h = h

    def forward(self, x0):
        t = torch.linspace(self.t0, self.t1, int((self.t1 - self.t0) / self.h) + 1)
        x = odeint(self.func, x0, t, method='dopri5')
        return x[-1]

class ODEFunc(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.Tanh(),
            nn.Linear(64, dim)
        )

    def forward(self, t, x):
        return self.net(x)

# Example usage
ode_func = ODEFunc(2)
solver = RecurrentODESolver(ode_func, 0, 1, 0.1)
x0 = torch.tensor([1.0, 0.0])
x1 = solver(x0)
print(x1)

