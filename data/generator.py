import math
import torch
from torch.utils.data import Dataset

# Physical constant e can be absorbed in scaling; keep as 1.0 by default.


def compute_C(Z, r0, e=1.0):
    # C = 3 Z e / (4 pi r0^3)
    return 3.0 * Z * e / (4.0 * math.pi * (r0 ** 3))


def sample_x(N, endpoint_fraction=0.4, beta_a=0.3, beta_b=0.3, device="cpu"):
    k = int(N * endpoint_fraction)
    n_u = N - k
    x_u = torch.rand(n_u, 1, device=device)  # uniform [0,1]
    # Beta concentrated near 0 and 1 via mix of Beta(a,b) and 1 - Beta(a,b)
    xb = torch.distributions.Beta(beta_a, beta_b).sample((k, 1)).to(device)
    mask = torch.rand_like(xb) < 0.5
    xb = torch.where(mask > 0, xb, 1.0 - xb)
    return torch.cat([x_u, xb], dim=0)


def sample_params(Np, r0_range, Z_set, log_r0=True, device="cpu"):
    # r0_range = (r0_min, r0_max)
    r0_min, r0_max = r0_range
    if log_r0:
        u = torch.rand(Np, 1, device=device)
        lr0 = torch.log(torch.tensor(r0_min, device=device)) * \
            (1-u) + torch.log(torch.tensor(r0_max, device=device)) * u
        r0 = torch.exp(lr0)
    else:
        r0 = torch.rand(Np, 1, device=device) * (r0_max - r0_min) + r0_min
    Z_choices = torch.tensor(Z_set, device=device, dtype=r0.dtype)
    Z_idx = torch.randint(low=0, high=len(Z_set), size=(Np,), device=device)
    Z = Z_choices[Z_idx].view(-1, 1)
    return r0, Z


def tile_params_over_x(x, r0, Z):
    # Given x of shape [Nx,1] and params [Np,1], tile to [Nx*Np, 1]
    Nx = x.shape[0]
    Np = r0.shape[0]
    xT = x.repeat(Np, 1)
    r0T = r0.repeat_interleave(Nx, dim=0)
    ZT = Z.repeat_interleave(Nx, dim=0)
    return xT, r0T, ZT


def make_collocation_batch(
    Nx=4096,
    Np=8,
    r0_range=(1e-2, 1e+2),
    Z_set=(1., 2., 6., 10., 13., 26.),
    endpoint_fraction=0.4,
    device="cpu",
    e=1.0,
):
    x = sample_x(Nx, endpoint_fraction=endpoint_fraction, device=device)
    r0, Z = sample_params(Np, r0_range=r0_range, Z_set=Z_set, device=device)
    xT, r0T, ZT = tile_params_over_x(x, r0, Z)
    C = compute_C(ZT, r0T, e=e)
    # usually all False here; boundary handled separately
    is_boundary = (xT == 1.0)
    return {
        "x": xT, "r0": r0T, "Z": ZT, "C": C, "is_boundary": is_boundary
    }


def make_boundary_batch(
    Nb_per_param=128,
    Np=8,
    r0_range=(1e-2, 1e+2),
    Z_set=(1., 2., 6., 10., 13., 26.),
    device="cpu",
    e=1.0,
):
    x = torch.ones(Nb_per_param, 1, device=device)
    r0, Z = sample_params(Np, r0_range=r0_range, Z_set=Z_set, device=device)
    xT, r0T, ZT = tile_params_over_x(x, r0, Z)
    C = compute_C(ZT, r0T, e=e)
    is_boundary = torch.ones_like(xT, dtype=torch.bool, device=device)
    return {
        "x": xT, "r0": r0T, "Z": ZT, "C": C, "is_boundary": is_boundary
    }


class CollocationDataset(Dataset):
    """
    Optional pre-generated dataset: pass tensors with matching first dimension.
    """

    def __init__(self, x, r0, Z, C, is_boundary):
        self.x = x
        self.r0 = r0
        self.Z = Z
        self.C = C
        self.is_boundary = is_boundary

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return {
            "x": self.x[idx:idx+1],
            "r0": self.r0[idx:idx+1],
            "Z": self.Z[idx:idx+1],
            "C": self.C[idx:idx+1],
            "is_boundary": self.is_boundary[idx:idx+1],
        }
