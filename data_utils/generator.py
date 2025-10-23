
import torch
from pathlib import Path
import os
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
    Strategy: mix uniform/global with bands [0,0.2] and [0.8,1], dedupe, pad, and enforce endpoints.
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


def main(out_path="./data/40kpoints_logr0.pt", seed=42, n_x=200):
    set_seed(seed)
    x = make_x_edge_focused(n_points=n_x, seed=seed)
    r0 = make_r0_mm_20()
    Z = make_Z_10()
    X = cartesian(x, r0, Z)  # [n_x*20*10, 3]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"X": X, "x": x, "r0": r0, "Z": Z, "seed": seed}, out_path)
    print(f"Saved dataset with {X.shape[0]} rows (n_x={n_x}) to {out_path}")


if __name__ == "__main__":
    main()
