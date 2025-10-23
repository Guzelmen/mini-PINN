"""
This file contains the model architecture for the small PINN.
"""
import torch
import torch.nn as nn
import math


class FourierFeatures(nn.Module):
    """
    Fourier features for the input x.

    Args:
        num_frequencies: number of frequencies
        max_freq: maximum frequency
        include_input: whether to include the input x
    """

    def __init__(self, num_frequencies=12, max_freq=12.0, include_input=True):
        super().__init__()
        self.include_input = include_input
        # linear frequency grid 1..max_freq, or use geometric: f_k = f_min * (f_max/f_min)**((k-1)/(K-1))
        self.register_buffer('freqs', torch.linspace(
            1.0, max_freq, num_frequencies))

    def forward(self, x):  # x shape [N,1] in [0,1]
        # 2π f x
        # broadcast: [N,1]*[F] -> [N,F]
        angles = 2.0 * math.pi * x * self.freqs
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        if self.include_input:
            return torch.cat([x, sin, cos], dim=-1)
        else:
            return torch.cat([sin, cos], dim=-1)


class Sine(nn.Module):
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
    Uses Fourier features on x and SIREN MLP.
    """

    def __init__(self, hidden=128, layers=6, fourier_freqs=12, fourier_max=12.0, w0=30.0):
        super().__init__()
        self.ff = FourierFeatures(
            num_frequencies=fourier_freqs, max_freq=fourier_max, include_input=True)
        # input dims: r0,Z (2) + encoded x dims: 1 + 2*F
        in_dim = 2 + (1 + 2*fourier_freqs)
        self.first = nn.Linear(in_dim, hidden)
        self.hiddens = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(layers-2)])
        self.out = nn.Linear(hidden, 1)
        self.act = Sine(w0=w0)
        # SIREN init
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
        x = xrz[..., 0:1]
        r0 = xrz[..., 1:2]
        Z = xrz[..., 2:3]
        x_enc = self.ff(x)
        inp = torch.cat([r0, Z, x_enc], dim=-1)
        h = self.act(self.first(inp))
        for layer in self.hiddens:
            h = self.act(layer(h))
        V = self.out(h)
        return V


class Model2(nn.Module):
    """
    Inputs: x in [0,1], r0, Z. Output: V(x; r0, Z).
    Uses Fourier features on x and SIREN MLP.
    Hard constraint: projection at the end to enforce V'(1) = 0. 
    The x·g factor keeps V finite and well-behaved at the origin in a spherical cell.
    """

    def __init__(self, hidden=128, layers=6, fourier_freqs=12, fourier_max=12.0, w0=30.0):
        super().__init__()
        self.ff = FourierFeatures(
            num_frequencies=fourier_freqs, max_freq=fourier_max, include_input=True)
        # input dims: r0,Z (2) + encoded x dims: 1 + 2*F
        in_dim = 2 + (1 + 2*fourier_freqs)
        self.first = nn.Linear(in_dim, hidden)
        self.hiddens = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(layers-2)])
        self.out = nn.Linear(hidden, 1)
        self.act = Sine(w0=w0)
        # SIREN init
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
        r0 = xrz[..., 1:2]
        Z = xrz[..., 2:3]
        x_enc = self.ff(x)
        inp = torch.cat([r0, Z, x_enc], dim=-1)
        h = self.act(self.first(inp))
        for layer in self.hiddens:
            h = self.act(layer(h))
        g = self.out(h)
        V = x * g + self.bc_scale * (1.0 - x)**2
        return V
