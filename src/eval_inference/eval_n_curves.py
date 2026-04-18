"""
Evaluation script: 3x3 grid of n(r) curves comparing PINN predictions vs BVP solver targets.

For each (alpha, T) combination, phi(x) is converted to electron number density n(r):
  1. gamma = 0.0899 / T^(3/4),  lam = alpha * b * T^(1/4) / C0
  2. xi(x) = gamma * phi(x) / (lam * x)
  3. n(x)  = C1 * (T_keV * KEV_TO_J)^(3/2) * F_{1/2}(xi(x))
  4. r = x * alpha * b / a0   [Bohr radii]

Both axes log-scaled. n(r) is always positive; report and clip any negatives.

Usage:
    python -m src.eval_inference.eval_n_curves \\
        <wandb_run_path> \\
        --run_name <local_run_name> \\
        --alphas 1.0 7.0 15.0 \\
        --temps 0.03 0.25 3.0 \\
        [--epoch <N>] \\
        [--device auto] \\
        [--output path/to/figure.png] \\
        [--batch_size 4096]
"""
import argparse
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from ..fd_integrals import (
    compute_gamma, compute_lambda, fermi_dirac_half,
    A0_M, B_M,
)
from ..utils import PROJECT_ROOT
from .color_map import find_latest_state_path
from .eval_residual_phase2 import fetch_config_from_wandb, build_model
from .eval_interp_range import find_state_path_for_epoch

# Physical constants not in fd_integrals.py
M_E    = 9.1093837015e-31        # electron mass [kg]
HBAR   = 1.054571817e-34         # reduced Planck constant [J·s]
KEV_TO_J = 1.602176634e-16       # 1 keV in Joules
# Density of states prefactor: C1 = (1/2pi^2) * (2*m_e/hbar^2)^(3/2)  [m^-3 J^-3/2]
C1 = (1.0 / (2.0 * math.pi**2)) * (2.0 * M_E / HBAR**2) ** 1.5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot 3x3 grid of n(r): PINN prediction vs BVP solver target."
    )
    parser.add_argument("wandb_run_path", help="Full wandb run path: entity/project/run_id")
    parser.add_argument("--run_name", required=True,
                        help="Local run name (subfolder in saving_weights/).")
    parser.add_argument("--alphas", type=float, nargs=3, required=True, metavar="ALPHA",
                        help="Three alpha values for the rows of the grid.")
    parser.add_argument("--temps", type=float, nargs=3, required=True, metavar="T",
                        help="Three T values (keV) for the columns of the grid.")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Checkpoint epoch to load. Defaults to latest.")
    parser.add_argument("--device", default="auto",
                        help='Device: "auto", "cpu", "cuda:0", etc.')
    parser.add_argument("--output", default=None,
                        help="Output path. Default: true_vs_target_n_plots/<run_name>/...")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--add_uniform", action="store_true", default=False,
                        help="Add a horizontal uniform-density curve: n = 3/(4*pi*R0^3).")
    return parser.parse_args()


def snap_pair_to_data(req_alpha: float, req_T: float,
                      pair_alphas: torch.Tensor, pair_Ts: torch.Tensor):
    """Find the (alpha, T) pair present in the data closest to (req_alpha, req_T).
    Uses normalised Euclidean distance. Returns (sa, sT, alpha_tol, T_tol).
    """
    d_alpha = (pair_alphas - req_alpha) / (req_alpha + 1e-12)
    d_T     = (pair_Ts     - req_T)     / (req_T     + 1e-12)
    idx     = torch.argmin(torch.sqrt(d_alpha**2 + d_T**2))
    sa      = pair_alphas[idx].item()
    sT      = pair_Ts[idx].item()
    if abs(sa - req_alpha) / (req_alpha + 1e-12) > 0.05:
        print(f"  Warning: alpha {req_alpha:.4g} -> {sa:.4g}")
    if abs(sT - req_T) / (req_T + 1e-12) > 0.05:
        print(f"  Warning: T {req_T:.4g} -> {sT:.4g}")
    alpha_tol = max(abs(sa) * 0.01, 1e-9)
    T_tol     = max(abs(sT) * 0.01, 1e-9)
    return sa, sT, alpha_tol, T_tol


def run_model_batched(model, x_vals: torch.Tensor, alpha_val: float, T_val: float,
                      device, batch_size: int) -> np.ndarray:
    """Run model on x_vals in batches, return phi as numpy array."""
    results = []
    n = x_vals.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x_b     = x_vals[start:end].unsqueeze(1).to(device).requires_grad_(True)
        alpha_b = torch.full((end - start, 1), alpha_val, dtype=torch.float32, device=device)
        T_b     = torch.full((end - start, 1), T_val,     dtype=torch.float32, device=device)
        inp = torch.cat([x_b, alpha_b, T_b], dim=1)
        with torch.enable_grad():
            phi_b = model(inp)
        results.append(phi_b.detach().cpu().squeeze(1))
    return torch.cat(results, dim=0).numpy()


def phi_to_n(phi: np.ndarray, x: np.ndarray, alpha: float, T: float) -> np.ndarray:
    """
    Convert phi(x) to electron number density n(r) [m^-3].

    xi(x) = gamma * phi / (lam * x)
    n(x)  = C1 * (T_keV * KEV_TO_J)^(3/2) * F_{1/2}(xi)
    """
    gamma = compute_gamma(T)
    lam   = compute_lambda(alpha, T)
    xi    = gamma * phi / (lam * x)          # shape [N]
    # fermi_dirac_half expects a torch tensor with shape [N, 1]
    xi_t  = torch.tensor(xi, dtype=torch.float32).unsqueeze(1)
    F12   = fermi_dirac_half(xi_t).squeeze(1).numpy()
    return C1 * (T * KEV_TO_J) ** 1.5 * F12


def x_to_bohr(x: np.ndarray, alpha: float) -> np.ndarray:
    """Convert x in [0,1] to r in Bohr radii. r = x * alpha * b / a0."""
    return x * alpha * B_M / A0_M


def main():
    args = parse_args()

    if args.device.lower() == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    print(f"Fetching config from wandb: {args.wandb_run_path}")
    config = fetch_config_from_wandb(args.wandb_run_path)
    params = SimpleNamespace(**config)
    params.device = device

    for field in ("x_min_threshold", "random_seed", "batch_size"):
        if hasattr(params, field):
            try:
                setattr(params, field, int(float(getattr(params, field))))
            except (TypeError, ValueError):
                pass
    if not hasattr(params, "debug_mode"):
        params.debug_mode = False

    if args.epoch is not None:
        state_path = find_state_path_for_epoch(args.run_name, args.epoch)
    else:
        state_path = find_latest_state_path(args.run_name)
    print(f"Loading checkpoint: {state_path}")
    state_dict = torch.load(state_path, map_location=device)

    model = build_model(params, state_dict, device)
    model.eval()
    torch.set_grad_enabled(True)

    fine_path = PROJECT_ROOT / params.fine_data_path
    print(f"Loading fine data from {fine_path} ...")
    fine_raw     = torch.load(fine_path, map_location="cpu")
    fine_inputs  = fine_raw["inputs"].float()
    fine_targets = fine_raw["targets"].float()
    fine_phi     = fine_targets[:, 0]

    if getattr(params, "filter_x_min", False):
        thresh = float(getattr(params, "x_min_threshold", 1e-6))
        mask_x = fine_inputs[:, 0] >= thresh
        fine_inputs = fine_inputs[mask_x]
        fine_phi    = fine_phi[mask_x]

    # Build set of (alpha, T) pairs that actually have solver data
    seen = {}
    pair_a_list, pair_T_list = [], []
    for a, t in zip(fine_inputs[:, 1].tolist(), fine_inputs[:, 2].tolist()):
        key = (round(a, 6), round(t, 6))
        if key not in seen:
            seen[key] = True
            pair_a_list.append(a)
            pair_T_list.append(t)
    pair_alphas_t = torch.tensor(pair_a_list, dtype=torch.float32)
    pair_Ts_t     = torch.tensor(pair_T_list, dtype=torch.float32)
    print(f"Unique (alpha, T) pairs with solver data: {len(pair_a_list)}")

    # First snap each T independently (shared across all alpha rows)
    uniq_T = torch.unique(pair_Ts_t)
    def snap_T(req_T):
        idx = torch.argmin(torch.abs(uniq_T - req_T))
        sT = uniq_T[idx].item()
        if abs(sT - req_T) / (req_T + 1e-12) > 0.05:
            print(f"  Warning: T {req_T:.4g} -> {sT:.4g}")
        return sT, max(abs(sT) * 0.01, 1e-9)

    snapped_Ts   = []
    T_tolerances = []
    for req_T in args.temps:
        sT, Ttol = snap_T(req_T)
        snapped_Ts.append(sT)
        T_tolerances.append(Ttol)

    # For each requested alpha, find the closest alpha that has data for ALL 3 snapped Ts.
    # Collect all unique alpha values, then for each candidate check all 3 Ts exist.
    uniq_alpha = torch.unique(pair_alphas_t)
    pair_set = set(zip([round(a, 6) for a in pair_a_list],
                       [round(t, 6) for t in pair_T_list]))

    def has_all_Ts(candidate_alpha, snapped_Ts, T_tolerances):
        for sT, Ttol in zip(snapped_Ts, T_tolerances):
            # Find closest T in data for this alpha
            alpha_mask = torch.abs(pair_alphas_t - candidate_alpha) < candidate_alpha * 0.01 + 1e-9
            if not alpha_mask.any():
                return False
            Ts_for_alpha = pair_Ts_t[alpha_mask]
            closest_T = Ts_for_alpha[torch.argmin(torch.abs(Ts_for_alpha - sT))].item()
            if abs(closest_T - sT) > Ttol:
                return False
        return True

    print("\nSnapping alpha values (homogeneous — same alpha used for all T columns):")
    snapped_alphas   = []
    alpha_tolerances = []
    for req_a in args.alphas:
        # Sort unique alphas by distance to requested, pick first that has all 3 Ts
        dists = torch.abs(uniq_alpha - req_a)
        order = torch.argsort(dists)
        chosen = None
        for idx in order:
            candidate = uniq_alpha[idx].item()
            if has_all_Ts(candidate, snapped_Ts, T_tolerances):
                chosen = candidate
                break
        if chosen is None:
            print(f"  Warning: no alpha near {req_a:.4g} has data for all 3 T values, using closest")
            chosen = uniq_alpha[torch.argmin(torch.abs(uniq_alpha - req_a))].item()
        if abs(chosen - req_a) / (req_a + 1e-12) > 0.05:
            print(f"  Warning: alpha {req_a:.4g} -> {chosen:.4g}")
        else:
            print(f"  alpha {req_a:.4g} -> {chosen:.4g}")
        snapped_alphas.append(chosen)
        alpha_tolerances.append(max(abs(chosen) * 0.01, 1e-9))

    # Build grid
    grid = {}
    for row_idx, (sa, atol) in enumerate(zip(snapped_alphas, alpha_tolerances)):
        for col_idx, (sT, Ttol) in enumerate(zip(snapped_Ts, T_tolerances)):
            grid[(row_idx, col_idx)] = (sa, sT, atol, Ttol)

    label_alphas = snapped_alphas
    label_Ts     = snapped_Ts
    print(f"\nRow alpha values used: {label_alphas}")
    print(f"Col T values used (keV): {label_Ts}")

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    legend_handles = None

    for row_idx in range(3):
        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            alpha, T, alpha_tol, T_tol = grid[(row_idx, col_idx)]

            mask = (
                (torch.abs(fine_inputs[:, 1] - alpha) < alpha_tol) &
                (torch.abs(fine_inputs[:, 2] - T)     < T_tol)
            )
            cell_x   = fine_inputs[mask, 0]
            cell_phi = fine_phi[mask]
            print(f"  [alpha={alpha:.4g}, T={T:.4g}] rows found: {cell_x.shape[0]}")

            if cell_x.shape[0] == 0:
                ax.text(0.5, 0.5, "no solver data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=11, color="red")
            else:
                sort_idx    = torch.argsort(cell_x)
                cell_x_np   = cell_x[sort_idx].numpy()
                cell_phi_np = cell_phi[sort_idx].numpy()

                r_bohr    = x_to_bohr(cell_x_np, alpha)
                n_solver  = phi_to_n(cell_phi_np, cell_x_np, alpha, T)

                pinn_phi_np = run_model_batched(
                    model, cell_x[sort_idx], alpha, T, device, args.batch_size
                )
                n_pinn = phi_to_n(pinn_phi_np, cell_x_np, alpha, T)

                # n(r) should always be positive; clip and report if not
                for label, arr in [("solver", n_solver), ("PINN", n_pinn)]:
                    neg_mask = arr < 0
                    n_neg = neg_mask.sum()
                    if n_neg > 0:
                        neg_vals = arr[neg_mask]
                        print(f"  [alpha={alpha:.4g}, T={T:.4g}] {label}: "
                              f"{n_neg} negative n values | "
                              f"most negative={neg_vals.min():.4g} m^-3 | "
                              f"std={neg_vals.std():.4g} m^-3")
                n_solver = np.clip(n_solver, 1e-30, None)
                n_pinn   = np.clip(n_pinn,   1e-30, None)
                # Replace any inf from overflow with NaN so they're skipped in plot
                n_solver = np.where(np.isfinite(n_solver), n_solver, np.nan)
                n_pinn   = np.where(np.isfinite(n_pinn),   n_pinn,   np.nan)

                h_solver, = ax.plot(r_bohr, n_solver, color="C0", linewidth=1.5,
                                    label="BVP solver")
                h_pinn,   = ax.plot(r_bohr, n_pinn,   color="C1", linewidth=1.5,
                                    linestyle="--", label="PINN")
                ax.fill_between(r_bohr, n_solver, n_pinn,
                                alpha=0.3, color="C1")

                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=3))
                ax.xaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10))
                ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=3))
                ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10))

                if args.add_uniform:
                    # Lock axes to the data range first, then draw uniform line.
                    # This prevents n_uniform (potentially far off) from rescaling the axes.
                    ax.autoscale_view()
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    R0_m = alpha * B_M
                    n_uniform = 3.0 / (4.0 * math.pi * R0_m**3)
                    h_uniform, = ax.plot(r_bohr, np.full_like(r_bohr, n_uniform),
                                         color="darkgreen", linewidth=1.5,
                                         linestyle="-.", label="Uniform density")
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)

                if legend_handles is None:
                    legend_handles = [h_solver, h_pinn]
                    if args.add_uniform:
                        legend_handles.append(h_uniform)

            # x-axis: tick labels on every plot (r varies per row with alpha)
            # axis label text only on bottom row
            if row_idx == 2:
                ax.set_xlabel(r"$r/a_0$", fontsize=24)
            ax.tick_params(axis="x", labelsize=20)
            ax.tick_params(axis="y", labelsize=22)

            if row_idx == 0:
                ax.set_title(f"$T = {label_Ts[col_idx]:.2f}$ keV", fontsize=24)

            if col_idx == 0:
                r0_bohr = label_alphas[row_idx] * B_M / A0_M
                ax.text(-0.35, 0.5, rf"$R_0 = {r0_bohr:.2f}\,a_0$",
                        transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=24)

    if legend_handles is not None:
        legend_labels = ["BVP solver", "PINN"]
        if args.add_uniform:
            legend_labels.append("Uniform density")
        fig.legend(legend_handles, legend_labels,
                   loc="upper center", ncol=len(legend_labels), fontsize=24,
                   bbox_to_anchor=(0.5, 1.02), frameon=True)

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if args.output is not None:
        output_path = Path(args.output)
    else:
        alpha_str = "_".join(f"{a:.2f}".replace(".", "p") for a in label_alphas)
        T_str     = "_".join(f"{T:.2f}".replace(".", "p") for T in label_Ts)
        out_dir   = PROJECT_ROOT / "true_vs_target_n_plots" / args.run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        uniform_tag = "_with_uniform" if args.add_uniform else ""
        output_path = out_dir / f"alpha{alpha_str}-T{T_str}{uniform_tag}.pdf"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
