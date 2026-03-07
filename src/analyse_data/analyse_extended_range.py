import torch
import numpy as np
from pathlib import Path

COARSE_PATH = Path(__file__).resolve().parents[2] / "data/phase_4_solver_extended_coarse.pt"
FINE_PATH   = Path(__file__).resolve().parents[2] / "data/phase_4_solver_extended_fine.pt"

def get_unique_param_vals(inputs):
    """Get unique alpha and T values (pair-level, not point-level)."""
    alpha_vals = torch.unique(inputs[:, 1])
    T_vals     = torch.unique(inputs[:, 2])
    return alpha_vals, T_vals

def log_percentile_edges(vals, lo_pct, hi_pct):
    """Return lower and upper cutoff in original space for middle pct% in log space."""
    log_vals = torch.log(vals)
    lo = float(torch.quantile(log_vals, lo_pct / 100))
    hi = float(torch.quantile(log_vals, 1 - hi_pct / 100))
    return np.exp(lo), np.exp(hi)

def analyse_edges(coarse_path, fine_path):
    print("Loading coarse...")
    coarse = torch.load(coarse_path)
    coarse_inputs = coarse['inputs']

    print("Loading fine...")
    fine = torch.load(fine_path)
    fine_inputs = fine['inputs']

    alpha_c, T_c = get_unique_param_vals(coarse_inputs)
    print(f"\nCoarse unique alpha values: {len(alpha_c)}, T values: {len(T_c)}")
    print(f"  alpha range: [{alpha_c.min():.4f}, {alpha_c.max():.4f}]")
    print(f"  T range:     [{T_c.min():.6f}, {T_c.max():.4f}] keV")

    alpha_f, T_f = get_unique_param_vals(fine_inputs)
    print(f"\nFine unique alpha values: {len(alpha_f)}, T values: {len(T_f)}")
    print(f"  alpha range: [{alpha_f.min():.4f}, {alpha_f.max():.4f}]")
    print(f"  T range:     [{T_f.min():.6f}, {T_f.max():.4f}] keV")

    for _ in range(1):
        a_lo, a_hi = log_percentile_edges(alpha_c, 10, 5)
        T_lo, T_hi = log_percentile_edges(T_c, 5, 5)
        print(f"\n--- Middle 85% edges (log-space percentiles on coarse unique vals) ---")
        print(f"  alpha: [{a_lo:.4f}, {a_hi:.4f}]")
        print(f"  T:     [{T_lo:.6f}, {T_hi:.4f}] keV")

        # Count fine points outside these edges (OOD)
        fine_alpha = fine_inputs[:, 1]
        fine_T     = fine_inputs[:, 2]

        ood_alpha = (fine_alpha < a_lo) | (fine_alpha > a_hi)
        ood_T     = (fine_T     < T_lo) | (fine_T     > T_hi)
        ood_either = ood_alpha | ood_T
        ood_both   = ood_alpha & ood_T

        n_fine = len(fine_inputs)
        print(f"  Fine points outside alpha edges:    {ood_alpha.sum().item():>8d} / {n_fine} ({100*ood_alpha.float().mean():.2f}%)")
        print(f"  Fine points outside T edges:        {ood_T.sum().item():>8d} / {n_fine} ({100*ood_T.float().mean():.2f}%)")
        print(f"  Fine points outside either:         {ood_either.sum().item():>8d} / {n_fine} ({100*ood_either.float().mean():.2f}%)")
        print(f"  Fine points outside both (corners): {ood_both.sum().item():>8d} / {n_fine} ({100*ood_both.float().mean():.2f}%)")

if __name__ == "__main__":
    analyse_edges(COARSE_PATH, FINE_PATH)