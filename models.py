"""
This file contains the model architecture for the small PINN.
"""
import torch
import torch.nn as nn
import math


class Siren(nn.Module):
    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


def siren_init(layer, w0=30.0, is_first=False):
    with torch.no_grad():
        in_dim = layer.weight.size(-1)
        if is_first:
            bound = 1.0 / in_dim
        else:
            bound = math.sqrt(6.0 / in_dim) / w0
        layer.weight.uniform_(-bound, bound)
        layer.bias.zero_()


class Model1(nn.Module):
    """
    Inputs: x in [0,1], r0, Z. Output: V(x; r0, Z).
    Uses direct input with Siren MLP.
    """

    def __init__(self, params):
        super().__init__()
        # Get hyperparameters from config
        hidden = params.hidden
        layers = params.nlayers
        w0 = params.w0

        # Direct input: x, r0, Z (3 dimensions)
        in_dim = 3
        self.first = nn.Linear(in_dim, hidden)
        self.hiddens = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(layers-2)])
        self.out = nn.Linear(hidden, 1)
        self.act = Siren(w0=w0)
        # Siren init
        siren_init(self.first, w0=w0, is_first=True)
        for h in self.hiddens:
            siren_init(h, w0=w0, is_first=False)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, xrz):
        """
        Args:
            xrz: shape [N,3] -> columns: x, r0, Z

        Returns:
            V: shape [N,1]
        """
        h = self.act(self.first(xrz))
        for layer in self.hiddens:
            h = self.act(layer(h))
        V = self.out(h)
        return V


class Model2(nn.Module):
    """
    Inputs: x in [0,1], r0, Z. Output: V(x; r0, Z).
    Uses direct input with Siren MLP.
    Hard constraint: projection at the end to enforce V'(1) = 0. 
    The x·g factor keeps V finite and well-behaved at the origin in a spherical cell.
    """

    def __init__(self, params):
        super().__init__()
        # Get hyperparameters from config
        hidden = params.hidden
        layers = params.nlayers
        w0 = params.w0

        # Direct input: x, r0, Z (3 dimensions)
        in_dim = 3
        self.first = nn.Linear(in_dim, hidden)
        self.hiddens = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(layers-2)])
        self.out = nn.Linear(hidden, 1)
        self.act = Siren(w0=w0)
        # Siren init
        siren_init(self.first, w0=w0, is_first=True)
        for h in self.hiddens:
            siren_init(h, w0=w0, is_first=False)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        # scaling factor for the boundary condition that can be learned
        self.bc_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, xrz):
        """
        Args:
            xrz: shape [N,3] -> columns: x, r0, Z

        Returns:
            V: shape [N,1]
        """
        x = xrz[..., 0:1]
        h = self.act(self.first(xrz))
        for layer in self.hiddens:
            h = self.act(layer(h))
        g = self.out(h)
        V = x * g + self.bc_scale * (1.0 - x)**2
        return V
