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


class Model_soft_phase1(nn.Module):
    """
    Inputs: x in [0,1]. Output: Psi(x).
    Uses direct input with Siren MLP.
    """

    def __init__(self, params):
        super().__init__()
        # Get hyperparameters from config
        hidden = params.hidden
        layers = params.nlayers
        w0 = params.w0

        # Direct input: x
        in_dim = 1
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

    def forward(self, x):
        """
        Args:
            x: shape N

        Returns:
            Psi: shape N
        """
        h = self.act(self.first(x))
        for layer in self.hiddens:
            h = self.act(layer(h))
        Psi = self.out(h)
        return Psi


class Model_hard_phase1(nn.Module):
    """
    Inputs: x in [0,1]. Output: Psi(x).
    Uses direct input with Siren MLP.
    Hard constraint: transform to ensure 2 constraints:
     - Psi(0) = 1
     - Psi(1) = Psi'(1)
    Tranform: Psi(x) = x^2 + 1 + xN(x) + x(1 - x)N'(x)
    """

    def __init__(self, params):
        super().__init__()
        # Get hyperparameters from config
        hidden = params.hidden
        layers = params.nlayers
        w0 = params.w0
        activation = params.activation

        # Direct input: x
        in_dim = 1
        self.first = nn.Linear(in_dim, hidden)
        self.hiddens = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(layers-2)])
        self.out = nn.Linear(hidden, 1)

        # Initialize activation function
        if activation == "SiLU":
            act = nn.SiLU()
        elif activation == "Identity":
            act = nn.Identity()
        elif activation == "ReLU":
            act = nn.ReLU()
        elif activation == "SIREN":
            act = Siren(w0=w0)
            # Siren init
            siren_init(self.first, w0=w0, is_first=True)
            for h in self.hiddens:
                siren_init(h, w0=w0, is_first=False)
        else:
            raise ValueError(
                f"Unknown activation: {activation}. Choose from: SiLU, Identity, ReLU, SIREN")

        self.act = act
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        """
        Args:
            x: shape N

        Returns:
            Psi: shape N
        """
        # Don't detach! x already has requires_grad=True from trainer
        # We need the graph connection for higher-order derivatives
        h = self.act(self.first(x))
        for layer in self.hiddens:
            h = self.act(layer(h))
        N = self.out(h)
        N_prime_outputs = torch.autograd.grad(
            N, x,
            grad_outputs=torch.ones_like(N),
            create_graph=True,  # needed if you want higher derivatives later
            retain_graph=True,  # only needed if you reuse graph, safe to include for now
            only_inputs=True,   # Only compute gradients w.r.t. x
            allow_unused=False  # Should not be unused - raise error if disconnected
        )

        if len(N_prime_outputs) == 0 or N_prime_outputs[0] is None:
            raise RuntimeError(
                "N_prime computation failed in Model2. "
                "Check that x is properly connected in the computational graph."
            )
        N_prime = N_prime_outputs[0]

        Psi = x**2 + 1 + x*N + x*(1-x)*N_prime

        return Psi


class Model_hard_phase2(nn.Module):
    """
    Inputs: x in [0,1]. Output: Psi(x).
    Uses direct input with diff activation types.
    Hard constraint: transform to ensure 2 constraints:
     - Psi(0) = 1
     - Psi(1) = Psi'(1)
    Tranform: Psi(x) = x^2 + 1 + xN(x) + x(1 - x)N'(x)
    """

    def __init__(self, params):
        super().__init__()
        # Get hyperparameters from config
        hidden = params.hidden
        layers = params.nlayers
        w0 = params.w0
        activation = params.activation

        # Direct inputs: x, r0
        in_dim = 2
        self.first = nn.Linear(in_dim, hidden)
        self.hiddens = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(layers-2)])
        self.out = nn.Linear(hidden, 1)

        # Initialize activation function
        if activation == "SiLU":
            act = nn.SiLU()
        elif activation == "Identity":
            act = nn.Identity()
        elif activation == "ReLU":
            act = nn.ReLU()
        elif activation == "SIREN":
            act = Siren(w0=w0)
            # Siren init
            siren_init(self.first, w0=w0, is_first=True)
            for h in self.hiddens:
                siren_init(h, w0=w0, is_first=False)
        else:
            raise ValueError(
                f"Unknown activation: {activation}. Choose from: SiLU, Identity, ReLU, SIREN")

        self.act = act
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        norm_mode = params.norm_mode
        if norm_mode == "standardize":
            self.register_buffer(
                "a_mean", torch.tensor(params.standard_mean))
            self.register_buffer(
                "a_std",  torch.tensor(params.standard_std))
        elif norm_mode == "minmax":
            self.register_buffer(
                "a_min",  torch.tensor(params.minmax_min))
            self.register_buffer(
                "a_max",  torch.tensor(params.minmax_max))
        else:
            raise ValueError("mode must be 'standardize' or 'minmax'")

        self.norm_mode = norm_mode

    def scale_alpha(self, alpha):
        a = torch.log1p(alpha)
        if self.norm_mode == "standardize":
            normalised = (a - self.a_mean) / (self.a_std + 1e-8)
        elif self.norm_mode == "minmax":
            a_mm = (a - self.a_min) / (self.a_max - self.a_min + 1e-12)
            normalised = 2.0 * a_mm - 1.0
        else:
            normalised = 0
            print("What the helly")

        return normalised

    def forward(self, inputs):
        """
        Args:
            inputs: shape [N, 2]

        Returns:
            Psi: shape N
        """

        x = inputs[:, 0:1]
        alpha = inputs[:, 1:2]
        norm_alpha = self.scale_alpha(alpha)
        norm_inp = torch.cat([x, norm_alpha], dim=-1)

        h = self.act(self.first(norm_inp))
        for layer in self.hiddens:
            h = self.act(layer(h))
        N = self.out(h)

        w = 0.5 * x**2 + x * (1 - x)**2 * N

        Psi = torch.exp(w)

        return Psi
