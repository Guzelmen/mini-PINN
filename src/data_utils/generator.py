import math
import torch
from pathlib import Path
import os
from ..utils import PROJECT_ROOT
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["OMP_NUM_THREADS"] = "1"


def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_x_edge_focused(n_points=200, seed=42):
    """
    Build an edge-focused 1D grid on [0,1] with exactly n_points values.
    Strategy: mix uniform/global with bands [0,0.2] and [0.8,1], dedupe, 
    pad, and enforce endpoints.
    """
    g = torch.Generator().manual_seed(seed)
    n_total = n_points
    n_u = n_total // 2
    n_l = n_total // 4
    n_r = n_total - n_u - n_l
    xu = torch.rand(n_u, generator=g)
    xl = torch.rand(n_l, generator=g) * 0.2
    xr = 0.8 + torch.rand(n_r, generator=g) * 0.2
    x = torch.cat([xu, xl, xr], dim=0).clamp(0.0, 1.0)
    x = torch.unique(torch.sort(x).values)
    while x.numel() < n_total:
        add = torch.rand(n_total - x.numel(), generator=g)
        x = torch.unique(torch.sort(torch.cat([x, add], dim=0)).values)
    if x.numel() > n_total:
        x = x[:n_total]
    if (x[-1] < 1.0):
        x[-1] = 1.0
    if (x[0] > 0.0):
        x[0] = 0.0
    x, _ = torch.sort(x)
    return x.view(-1, 1)  # [n_points,1]


def make_r0_mm_20():
    # 1e-4 to 1e-2 with log scale → 20 points
    return torch.logspace(-4, -2, 20).view(-1, 1)


def make_Z_10():
    return torch.tensor([1., 2., 4., 8., 16., 24., 32., 48., 64., 80.]).view(-1, 1)


def cartesian(x, r0, Z):
    Nx, Nr, Nz = x.shape[0], r0.shape[0], Z.shape[0]
    xT = x.repeat(Nr * Nz, 1)
    r0T = r0.repeat_interleave(Nx, dim=0).repeat(Nz, 1)
    ZT = Z.repeat_interleave(Nx * Nr, dim=0)
    return torch.cat([xT, r0T, ZT], dim=1)


def generate_x_r0_64k(out_path=PROJECT_ROOT / "data/64k_x_log_r0.pt", seed=42, n_points=64000):
    """
    Generate dataset with n_points (default 64k) points and 2 columns [x, r0]:
      - x: evenly spaced in (0.0001, 0.9999), shape [n_points, 1]
      - r0: log-uniform in [1e-9, 1e-5], shape [n_points, 1]
    Saves dict with keys: 'X', 'x_minmax', 'r0_range', 'seed'.
    """
    set_seed(seed)
    # x grid (exclusive endpoints per spec)
    x = torch.linspace(0.0001, 0.9999, n_points).view(-1, 1)
    # r0 ~ log-uniform between 1e-9 and 1e-5
    g = torch.Generator().manual_seed(seed)
    exp_min, exp_max = -9.0, -5.0
    u = torch.rand(n_points, 1, generator=g)  # uniform [0,1]
    exps = exp_min + (exp_max - exp_min) * u
    r0 = 10.0 ** exps
    X = torch.cat([x, r0], dim=1)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"X": X, "x_minmax": (0.0001, 0.9999), "r0_range": (
            1e-9, 1e-5), "seed": seed},
        out_path,
    )
    print(f"Saved 2-column dataset with {X.shape[0]} points to {out_path}")
    print(f"  x in (0.0001, 0.9999), r0 log-uniform in [1e-9, 1e-5]")


def old_main(out_path=PROJECT_ROOT / "data/40kgridpoints_logr0_randZ_noendpoints.pt",
             seed=42, n_points=40000):
    """
    Generate dataset with n_points (default 40k) points.
    - x: evenly spaced grid in [0, 1] (inclusive) with n_points values
    - r0: randomly sampled from logspace(1e-4, 1e-2, 20) for each x
    - Z: randomly sampled from [1, 2, 4, 8, 16, 24, 32, 48, 64, 80] for each x
    """
    set_seed(seed)

    # Create evenly spaced x grid from 0 to 1 (inclusive)
    x = torch.linspace(0.0001, 0.9999, n_points).view(-1, 1)  # [n_points, 1]

    # Define r0 and Z ranges (same as before)
    r0_values = make_r0_mm_20()  # [20, 1] - logspace from 1e-4 to 1e-2
    Z_values = make_Z_10()  # [10, 1] - [1, 2, 4, 8, 16, 24, 32, 48, 64, 80]

    # Randomly sample r0 and Z for each x point
    g = torch.Generator().manual_seed(seed)
    r0_indices = torch.randint(0, r0_values.shape[0], (n_points,), generator=g)
    Z_indices = torch.randint(0, Z_values.shape[0], (n_points,), generator=g)

    r0 = r0_values[r0_indices, :]  # [n_points, 1] - select rows
    Z = Z_values[Z_indices, :]     # [n_points, 1] - select rows

    # Combine into [x, r0, Z] format
    X = torch.cat([x, r0, Z], dim=1)  # [n_points, 3]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"X": X, "x": x, "r0": r0_values,
               "Z": Z_values, "seed": seed}, out_path)
    print(f"Saved dataset with {X.shape[0]} points to {out_path}")
    print(f"  x: evenly spaced grid in (0, 1) with {n_points} points")
    print(
        f"  r0: randomly sampled from {r0_values.shape[0]} values in [1e-4, 1e-2]")
    print(f"  Z: randomly sampled from {Z_values.shape[0]} values")


def generate_phase2_final(out_path=PROJECT_ROOT / "data/phase_2_small_alpha.pt",
                          seed=42, n_points=64000):
    torch.manual_seed(seed)
    x = torch.linspace(0.0, 1.0, steps=n_points).unsqueeze(1)
    r0 = torch.logspace(math.log10(5e-11), math.log10(5e-8),
                        steps=n_points).unsqueeze(1)
    a0 = 5.291772105e-11  # in meters
    b = 0.25 * (4.5 * math.pi**2)**(1/3) * a0
    alpha = r0 / b
    X = torch.cat([x, alpha], dim=1)  # shape [N, 2] = [x, alpha]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(X, out_path)
    print(f"Saved tensor of shape {tuple(X.shape)} to {out_path}")


def generate_phase2_combo(out_path=PROJECT_ROOT / "data/phase_2_LOG500x_RED128alpha.pt",
                          seed=42, n_x=500, n_alpha=128):
    torch.manual_seed(seed)
    # Use log10 for torch.logspace (it expects base-10 exponents)
    x = torch.logspace(math.log10(1e-6), math.log10(1.0), steps=n_x).unsqueeze(1)
    # x = torch.linspace(0.0, 1.0, steps=n_x).unsqueeze(1)
    r0 = torch.logspace(math.log10(5e-11), math.log10(5e-9),
                        steps=n_alpha).unsqueeze(1)
    a0 = 5.291772105e-11  # in meters
    b = 0.25 * (4.5 * math.pi**2)**(1/3) * a0
    alpha = r0 / b

    # Build Cartesian product of x (size n_x) and alpha (size n_alpha)
    # Result: X has shape [n_x * n_alpha, 2] = [N, 2] with columns [x, alpha]
    Nx, Na = x.shape[0], alpha.shape[0]
    xT = x.repeat(Na, 1)                     # [Nx * Na, 1]
    alphaT = alpha.repeat_interleave(Nx, 0)  # [Nx * Na, 1]
    X = torch.cat([xT, alphaT], dim=1)       # [Nx * Na, 2]
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(X, out_path)
    print(f"Saved tensor of shape {tuple(X.shape)} to {out_path}")


def generate_logx_logr0(out_path=PROJECT_ROOT / "data/phase_2_log_x_log_alpha_64k.pt",
                          seed=42, n_points=64000):
    torch.manual_seed(seed)
    x = torch.logspace(math.log10(1e-12), math.log10(1.0), 
                        steps=n_points).unsqueeze(1)
    r0 = torch.logspace(math.log10(1e-9), math.log10(1e-5),
                        steps=n_points).unsqueeze(1)
    a0 = 5.291772105e-11  # in meters
    b = 0.25 * (4.5 * math.pi**2)**(1/3) * a0
    alpha = r0 / b
    X = torch.cat([x, alpha], dim=1)  # shape [N, 2] = [x, alpha]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(X, out_path)
    print(f"Saved tensor of shape {tuple(X.shape)} to {out_path}")

if __name__ == "__main__":
    generate_phase2_combo()
