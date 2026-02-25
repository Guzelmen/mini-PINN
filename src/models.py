"""
This file contains the model architecture for the small PINN.
"""
import torch
import torch.nn as nn
import math
from .utils import first_deriv_auto, sec_deriv_auto, phi_w_transform, phi_0_transform, phi_bc_transform_phase4
from .fd_integrals import B_M, C0_M  # Physical constants for phase 4


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
        in_dim = params.inp_dim
        self.first = nn.Linear(in_dim, hidden)
        self.hiddens = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(layers-2)])
        self.out = nn.Linear(hidden, 1)

        # Initialize activation function
        if activation == "SiLU":
            act = nn.SiLU()
        elif activation == "Tanh":
            act = nn.Tanh()
        elif activation == "Mish":
            act = nn.Mish()
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
    Inputs: x in [0,1], alpha between ~1 and 100-1000. Output: Psi(x).
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
        self.k_val = params.k_val
        self.add_phi0 = params.add_phi0

        in_dim = params.inp_dim
        self.first = nn.Linear(in_dim, hidden)
        self.hiddens = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(layers-2)])
        self.out = nn.Linear(hidden, 1)

        # Initialize activation function
        if activation == "SiLU":
            act = nn.SiLU()
        elif activation == "Tanh":
            act = nn.Tanh()
        elif activation == "Mish":
            act = nn.Mish()
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
            std_mean = getattr(params, "standard_mean", None)
            std_std = getattr(params, "standard_std", None)
            # If not provided, will be estimated online in scale_alpha during training
            init_mean = torch.tensor(0.0) if std_mean is None else torch.tensor(float(std_mean))
            init_std  = torch.tensor(1.0) if std_std  is None else torch.tensor(float(std_std))
            self.register_buffer("a_mean", init_mean)
            self.register_buffer("a_std",  init_std)
            # Online running stats (Welford) updated in training mode
            self.register_buffer("running_mean", torch.tensor(0.0))
            self.register_buffer("running_var", torch.tensor(1.0))
            self.register_buffer("running_count", torch.tensor(0.0))
            # Flag to indicate if stats have been fitted at least once
            self.register_buffer("norm_fitted", torch.tensor(0, dtype=torch.int64))
        elif norm_mode == "minmax":
            self.register_buffer(
                "a_min",  torch.tensor(params.minmax_min))
            self.register_buffer(
                "a_max",  torch.tensor(params.minmax_max))
        else:
            raise ValueError("mode must be 'standardize' or 'minmax'")

        self.norm_mode = norm_mode
        self.debug_mode = getattr(params, "debug_mode", False)
        self.params = params

    def scale_alpha(self, alpha):
        a = torch.log1p(alpha)

        if self.norm_mode == "standardize":
            normalised = (a - self.a_mean) / (self.a_std + 1e-8)
        elif self.norm_mode == "minmax":
            a_mm = (a - self.a_min) / (self.a_max - self.a_min + 1e-12)
            normalised = 2.0 * a_mm - 1.0
        else:
            normalised = 0
            print("Not correct norm mode")

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
        if self.debug_mode is True:
            # Debug: check raw vs normalized alpha statistics
            # (mean/std here are over the current batch)
            a_mean = alpha.mean().item()
            a_std = alpha.std(unbiased=False).item()
            na_mean = norm_alpha.mean().item()
            na_std = norm_alpha.std(unbiased=False).item()
            print(
                f"[During fwd] alpha stats: mean={a_mean:.6g}, std={a_std:.6g} | "
                f"norm_alpha stats: mean={na_mean:.6g}, std={na_std:.6g}"
            )
        norm_inp = torch.cat([x, norm_alpha], dim=-1)

        h = self.act(self.first(norm_inp))
        for layer in self.hiddens:
            h = self.act(layer(h))
        N = self.out(h)

        N_prime = first_deriv_auto(N, norm_inp, var1=0)

        #debugging
        if self.debug_mode is True:
            print(f"[During fwd] network output before transform (N) -> Min: {N.min().item()}, Max: {N.max().item()}")
            print(f"[During fwd] N_prime (dN/dnorm_inp, then take wrt x) -> Min: {N_prime.min().item()}, Max: {N_prime.max().item()}")
        
        Phi = phi_w_transform(x=x, N=N, N_prime=N_prime, params=self.params)
        if self.debug_mode is True:
            print(f"[During fwd] phi (exp(w)) -> Min: {Phi.min().item()}, Max: {Phi.max().item()}")
        if (Phi < 0).any() and self.debug_mode is True:
            print("[During fwd] Watch out: phi (exp(w)) is negative.")

        if self.add_phi0:
            Phi_0 = phi_0_transform(x=x, alpha=alpha, k=self.k_val)
            #print(f"[During fwd] phi_0 -> Min: {Phi_0.min().item()}, Max: {Phi_0.max().item()}")
            if (Phi_0 < 0).any() and self.debug_mode is True:
                print("[During fwd] Watch out: phi_0 is negative.")
            Phi = Phi + Phi_0

        return Phi


class Model_hard_phase3(nn.Module):
    """
    Inputs: x in [0,1], sqrt(x), alpha between ~1 and 100-100. Output: Psi(x).
    Uses direct input with diff activation types.
    Hard constraint: transform to ensure 2 constraints:
     - Psi(0) = 1
     - Psi(1) = Psi'(1)
    Tranform: w(x) = x^2 + 1 + xN(x) + x(1 - x)N'(x)
    Psi(x) = exp(w(x)), enforce positivity
    """

    def __init__(self, params):
        super().__init__()
        # Get hyperparameters from config
        hidden = params.hidden
        layers = params.nlayers
        w0 = params.w0
        activation = params.activation
        self.k_val = params.k_val
        self.add_phi0 = params.add_phi0

        in_dim = params.inp_dim
        self.first = nn.Linear(in_dim, hidden)
        self.hiddens = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(layers-2)])
        self.out = nn.Linear(hidden, 1)

        # Initialize activation function
        if activation == "SiLU":
            act = nn.SiLU()
        elif activation == "Tanh":
            act = nn.Tanh()
        elif activation == "Mish":
            act = nn.Mish()
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
            std_mean = getattr(params, "standard_mean", None)
            std_std = getattr(params, "standard_std", None)
            # If not provided, will be estimated online in scale_alpha during training
            init_mean = torch.tensor(0.0) if std_mean is None else torch.tensor(float(std_mean))
            init_std  = torch.tensor(1.0) if std_std  is None else torch.tensor(float(std_std))
            self.register_buffer("a_mean", init_mean)
            self.register_buffer("a_std",  init_std)
            # Online running stats (Welford) updated in training mode
            self.register_buffer("running_mean", torch.tensor(0.0))
            self.register_buffer("running_var", torch.tensor(1.0))
            self.register_buffer("running_count", torch.tensor(0.0))
            # Flag to indicate if stats have been fitted at least once
            self.register_buffer("norm_fitted", torch.tensor(0, dtype=torch.int64))
        elif norm_mode == "minmax":
            self.register_buffer(
                "a_min",  torch.tensor(params.minmax_min))
            self.register_buffer(
                "a_max",  torch.tensor(params.minmax_max))
        else:
            raise ValueError("mode must be 'standardize' or 'minmax'")

        self.norm_mode = norm_mode
        self.debug_mode = getattr(params, "debug_mode", False)
        self.params = params

    def scale_alpha(self, alpha):
        a = torch.log1p(alpha)

        if self.norm_mode == "standardize":
            normalised = (a - self.a_mean) / (self.a_std + 1e-8)
        elif self.norm_mode == "minmax":
            a_mm = (a - self.a_min) / (self.a_max - self.a_min + 1e-12)
            normalised = 2.0 * a_mm - 1.0
        else:
            normalised = 0
            print("Not correct norm mode")

        return normalised

    def forward(self, inputs):
        """
        Args:
            inputs: shape [N, 2]

        Then calculates sqrt(x) and gives 3 inputs to network

        Returns:
            Psi: shape N
        """

        x = inputs[:, 0:1]
        alpha = inputs[:, 1:2]
        sqrt_x = torch.sqrt(x)
        norm_alpha = self.scale_alpha(alpha)
        if self.debug_mode is True:
            # Debug: check raw vs normalized alpha statistics
            # (mean/std here are over the current batch)
            a_mean = alpha.mean().item()
            a_std = alpha.std(unbiased=False).item()
            na_mean = norm_alpha.mean().item()
            na_std = norm_alpha.std(unbiased=False).item()
            print(
                f"[During fwd] alpha stats: mean={a_mean:.6g}, std={a_std:.6g} | "
                f"norm_alpha stats: mean={na_mean:.6g}, std={na_std:.6g}"
            )
        enriched_inp = torch.cat([x, norm_alpha, sqrt_x], dim=-1)

        h = self.act(self.first(enriched_inp))
        for layer in self.hiddens:
            h = self.act(layer(h))
        N = self.out(h)

        N_prime = first_deriv_auto(N, inputs, var1=0)

        #debugging
        if self.debug_mode is True:
            print(f"[During fwd] network output before transform (N) -> Min: {N.min().item()}, Max: {N.max().item()}")
            print(f"[During fwd] N_prime (dN/dnorm_inp, then take wrt x) -> Min: {N_prime.min().item()}, Max: {N_prime.max().item()}")

        Phi = phi_w_transform(x=x, N=N, N_prime=N_prime, params=self.params)
        if self.debug_mode is True:
            print(f"[During fwd] phi (exp(w)) -> Min: {Phi.min().item()}, Max: {Phi.max().item()}")
        if (Phi < 0).any() and self.debug_mode is True:
            print("[During fwd] Watch out: phi (exp(w)) is negative.")

        if self.add_phi0:
            Phi_0 = phi_0_transform(x=x, alpha=alpha, k=self.k_val)
            #print(f"[During fwd] phi_0 -> Min: {Phi_0.min().item()}, Max: {Phi_0.max().item()}")
            if (Phi_0 < 0).any() and self.debug_mode is True:
                print("[During fwd] Watch out: phi_0 is negative.")
            Phi = Phi + Phi_0

        return Phi


class Model_hard_phase4(nn.Module):
    """
    Phase 4: Temperature-dependent Thomas-Fermi model.

    Inputs: x in [0,1], alpha (~1 to 100), T_kV (0.1 to 100 keV). Output: Phi(x).

    Hard constraint: transform to ensure 2 constraints:
     - Phi(0) = 1
     - Phi(1) = Phi'(1)
    Transform: Phi(x) = x^2 + 1 + xN(x) + x(1 - x)N'(x)

    Network receives (controlled by params.inp_dim):
     - inp_dim=3: [x, norm_alpha, norm_T]
     - inp_dim=4: [x, norm_alpha, norm_T, sqrt(x)]
    """

    def __init__(self, params):
        super().__init__()
        # Get hyperparameters from config
        hidden = params.hidden
        layers = params.nlayers
        w0 = params.w0
        activation = params.activation
        self.k_val = params.k_val
        self.add_phi0 = params.add_phi0

        in_dim = params.inp_dim  # 3: [x, norm_alpha, norm_T] or 4: [x, norm_alpha, norm_T, sqrt_x]
        self.inp_dim = in_dim  # Store for forward()
        self.first = nn.Linear(in_dim, hidden)
        self.hiddens = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(layers-2)])
        self.out = nn.Linear(hidden, 1)

        # Initialize activation function
        if activation == "SiLU":
            act = nn.SiLU()
        elif activation == "Tanh":
            act = nn.Tanh()
        elif activation == "Mish":
            act = nn.Mish()
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
                f"Unknown activation: {activation}. Choose from: SiLU, Identity, ReLU, SIREN, Tanh, Mish")

        self.act = act
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        # Alpha normalization (same as phase 2/3)
        norm_mode = params.norm_mode
        if norm_mode == "standardize":
            std_mean = getattr(params, "standard_mean", None)
            std_std = getattr(params, "standard_std", None)
            init_mean = torch.tensor(0.0) if std_mean is None else torch.tensor(float(std_mean))
            init_std = torch.tensor(1.0) if std_std is None else torch.tensor(float(std_std))
            self.register_buffer("a_mean", init_mean)
            self.register_buffer("a_std", init_std)
        elif norm_mode == "minmax":
            self.register_buffer("a_min", torch.tensor(params.minmax_min))
            self.register_buffer("a_max", torch.tensor(params.minmax_max))
        else:
            raise ValueError("norm_mode must be 'standardize' or 'minmax'")

        # Temperature normalization (new for phase 4)
        if norm_mode == "standardize":
            # Try T_mean/T_std first (from main.py), fallback to T_standard_mean/T_standard_std
            T_mean_val = getattr(params, "T_mean", None)
            T_std_val = getattr(params, "T_std", None)
            init_T_mean = torch.tensor(0.0) if T_mean_val is None else torch.tensor(float(T_mean_val))
            init_T_std = torch.tensor(1.0) if T_std_val is None else torch.tensor(float(T_std_val))
            self.register_buffer("T_mean", init_T_mean)
            self.register_buffer("T_std", init_T_std)
        elif norm_mode == "minmax":
            self.register_buffer("T_min", torch.tensor(getattr(params, "T_minmax_min", -1.0)))
            self.register_buffer("T_max", torch.tensor(getattr(params, "T_minmax_max", 1.0)))

        
        self.use_log_x = getattr(params, "use_log_x", False):
        if self.use_log_x:
            if norm_mode == "standardize":
                x_mean_val = getattr(params, "x_log_mean", 0.0)
                x_std_val = getattr(params, "x_log_std", 1.0)
                self.register_buffer("x_log_mean", torch.tensor(float(x_mean_val)))
                self.register_buffer("x_log_std", torch.tensor(float(x_std_val)))
            elif norm_mode = "minmax":
                self.register_buffer("x_min", torch.tensor(getattr(params, "x_minmax_min", -1.0)))
                self.register_buffer("x_max", torch.tensor(getattr(params, "x_minmax_max", 1.0)))

        self.norm_mode = norm_mode
        self.debug_mode = getattr(params, "debug_mode", False)
        self.params = params

    def scale_alpha(self, alpha):
        """Normalize alpha using log1p + standardize/minmax."""
        a = torch.log1p(alpha)

        if self.norm_mode == "standardize":
            normalised = (a - self.a_mean) / (self.a_std + 1e-8)
        elif self.norm_mode == "minmax":
            a_mm = (a - self.a_min) / (self.a_max - self.a_min + 1e-12)
            normalised = 2.0 * a_mm - 1.0
        else:
            normalised = a
            print("Warning: incorrect norm_mode")

        return normalised

    def scale_T(self, T_kV):
        """Normalize temperature using log1p + standardize/minmax."""
        t = torch.log1p(T_kV)

        if self.norm_mode == "standardize":
            normalised = (t - self.T_mean) / (self.T_std + 1e-8)
        elif self.norm_mode == "minmax":
            t_mm = (t - self.T_min) / (self.T_max - self.T_min + 1e-12)
            normalised = 2.0 * t_mm - 1.0
        else:
            normalised = t
            print("Warning: incorrect norm_mode")

        return normalised

    def scale_x(self, x):
        log_x = torch.log(x + 1e-8)
        if self.norm_mode == "standardize":
            normalised = (log_x - self.x_log_mean) / (self.x_log_std + 1e-8)
        elif self.norm_mode == "minmax":
            x_mm = (log_x - self.x_min) / (self.x_max - self.x_min + 1e-12)
            normalised = 2.0 * x_mm - 1.0
        else:
            normalised = log_x
            print("Warning: incorrect norm_mode")
        
        return normalised

    def forward(self, inputs):
        """
        Args:
            inputs: shape [N, 3] with columns [x, alpha, T_kV]

        Computes sqrt(x), normalizes alpha and T and optionally x, then passes 3 or 4 inputs to network.

        Returns:
            Psi: shape [N, 1]
        """
        x = inputs[:, 0:1]
        alpha = inputs[:, 1:2]
        T_kV = inputs[:, 2:3]

        norm_alpha = self.scale_alpha(alpha)
        norm_T = self.scale_T(T_kV)
        if self.use_log_x:
            norm_x = self.scale_x(x)
        else:
            norm_x = x

        if self.debug_mode:
            # Debug: check raw vs normalized statistics
            print(f"[Phase4 fwd] x: min={x.min().item():.6g}, max={x.max().item():.6g}")
            print(f"[Phase4 fwd] norm_x: min={norm_x.min().item():.6g}, max={norm_x.max().item():.6g}")
            print(f"[Phase4 fwd] alpha: mean={alpha.mean().item():.6g}, std={alpha.std().item():.6g}")
            print(f"[Phase4 fwd] norm_alpha: mean={norm_alpha.mean().item():.6g}, std={norm_alpha.std().item():.6g}")
            print(f"[Phase4 fwd] T_kV: mean={T_kV.mean().item():.6g}, std={T_kV.std().item():.6g}")
            print(f"[Phase4 fwd] norm_T: mean={norm_T.mean().item():.6g}, std={norm_T.std().item():.6g}")

        # Build network input based on inp_dim
        # inp_dim=3: [x, norm_alpha, norm_T]
        # inp_dim=4: [x, norm_alpha, norm_T, sqrt_x]
        if self.inp_dim == 4:
            sqrt_x = torch.sqrt(norm_x)
            network_inp = torch.cat([norm_x, norm_alpha, norm_T, sqrt_x], dim=-1)
        else:
            network_inp = torch.cat([norm_x, norm_alpha, norm_T], dim=-1)

        h = self.act(self.first(network_inp))
        for layer in self.hiddens:
            h = self.act(layer(h))
        N = self.out(h)

        # Compute N' w.r.t. x (need grad through original inputs for PDE loss)
        N_prime = first_deriv_auto(N, inputs, var1=0)

        if self.debug_mode:
            print(f"[Phase4 fwd] N: min={N.min().item():.6g}, max={N.max().item():.6g}")
            print(f"[Phase4 fwd] N_prime: min={N_prime.min().item():.6g}, max={N_prime.max().item():.6g}")

        Phi = phi_bc_transform_phase4(x=x, N=N, N_prime=N_prime, params=self.params)

        if self.debug_mode:
            print(f"[Phase4 fwd] Phi: min={Phi.min().item():.6g}, max={Phi.max().item():.6g}")

        # Optional phi_0 term (physics-informed initialization)
        if self.add_phi0:
            Phi_0 = phi_0_transform(x=x, alpha=alpha, k=self.k_val)
            if self.debug_mode:
                print(f"[Phase4 fwd] Phi_0: min={Phi_0.min().item():.6g}, max={Phi_0.max().item():.6g}")
            if (Phi_0 < 0).any() and self.debug_mode:
                print("[Phase4 fwd] Warning: Phi_0 is negative")
            Phi = Phi + Phi_0

        return Phi
