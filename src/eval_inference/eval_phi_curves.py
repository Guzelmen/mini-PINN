"""
Evaluation script: 3x3 grid of phi(x) curves comparing PINN predictions vs BVP solver targets.

Each subplot shows phi(x) for one (alpha, T) combination. Requested alpha and T values are
snapped to the nearest grid point that has solver data.

Usage:
    python -m src.eval_inference.eval_phi_curves \\
        <wandb_run_path> \\
        --run_name <local_run_name> \\
        --alphas 1.0 5.0 20.0 \\
        --temps 0.01 0.1 1.0 \\
        [--epoch <N>] \\
        [--device auto] \\
        [--output path/to/figure.png] \\
        [--batch_size 4096]
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..utils import PROJECT_ROOT
from ..fd_integrals import B_M, A0_M
from .color_map import find_latest_state_path
from .eval_residual_phase2 import fetch_config_from_wandb, build_model
from .eval_interp_range import find_state_path_for_epoch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot 3x3 grid of phi(x): PINN prediction vs BVP solver target."
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
                        help="Output path. Default: true_vs_target_phi_plots/<run_name>/...")
    parser.add_argument("--batch_size", type=int, default=4096)
    return parser.parse_args()


def snap_pair_to_data(req_alpha: float, req_T: float,
                      pair_alphas: torch.Tensor, pair_Ts: torch.Tensor):
    """Find the (alpha, T) pair present in the data closest to (req_alpha, req_T).
    Uses normalised Euclidean distance so neither axis dominates.
    Returns (snapped_alpha, snapped_T, alpha_tol, T_tol).
    """
    d_alpha = (pair_alphas - req_alpha) / (req_alpha + 1e-12)
    d_T     = (pair_Ts     - req_T)     / (req_T     + 1e-12)
    dist    = torch.sqrt(d_alpha**2 + d_T**2)
    idx     = torch.argmin(dist)
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
        x_b = x_vals[start:end].unsqueeze(1).to(device).requires_grad_(True)
        alpha_b = torch.full((end - start, 1), alpha_val, dtype=torch.float32, device=device)
        T_b     = torch.full((end - start, 1), T_val,     dtype=torch.float32, device=device)
        inp = torch.cat([x_b, alpha_b, T_b], dim=1)
        with torch.enable_grad():
            phi_b = model(inp)
        results.append(phi_b.detach().cpu().squeeze(1))
    return torch.cat(results, dim=0).numpy()


def main():
    args = parse_args()

    # Device
    if args.device.lower() == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load wandb config and build model
    print(f"Fetching config from wandb: {args.wandb_run_path}")
    config = fetch_config_from_wandb(args.wandb_run_path)
    params = SimpleNamespace(**config)
    params.device = device

    # Cast fields that may come back as strings from wandb
    for field in ("x_min_threshold", "random_seed", "batch_size"):
        if hasattr(params, field):
            try:
                setattr(params, field, int(float(getattr(params, field))))
            except (TypeError, ValueError):
                pass
    if not hasattr(params, "debug_mode"):
        params.debug_mode = False

    # Load checkpoint
    if args.epoch is not None:
        state_path = find_state_path_for_epoch(args.run_name, args.epoch)
    else:
        state_path = find_latest_state_path(args.run_name)
    print(f"Loading checkpoint: {state_path}")
    state_dict = torch.load(state_path, map_location=device)

    model = build_model(params, state_dict, device)
    model.eval()
    torch.set_grad_enabled(True)

    # Load fine data
    fine_path = PROJECT_ROOT / params.fine_data_path
    print(f"Loading fine data from {fine_path} ...")
    fine_raw     = torch.load(fine_path, map_location="cpu")
    fine_inputs  = fine_raw["inputs"].float()   # [N, 3]: x, alpha, T_kV
    fine_targets = fine_raw["targets"].float()  # [N, 2]: psi, dpsi_dx
    fine_phi     = fine_targets[:, 0]

    # Apply x_min filter consistent with training
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

    snapped_Ts, T_tolerances = [], []
    for req_T in args.temps:
        sT, Ttol = snap_T(req_T)
        snapped_Ts.append(sT)
        T_tolerances.append(Ttol)

    # For each requested alpha, find closest alpha with data for ALL 3 snapped Ts
    uniq_alpha = torch.unique(pair_alphas_t)

    def has_all_Ts(candidate_alpha, snapped_Ts, T_tolerances):
        atol = candidate_alpha * 0.01 + 1e-9
        alpha_mask = torch.abs(pair_alphas_t - candidate_alpha) < atol
        if not alpha_mask.any():
            return False
        Ts_for_alpha = pair_Ts_t[alpha_mask]
        for sT, Ttol in zip(snapped_Ts, T_tolerances):
            closest_T = Ts_for_alpha[torch.argmin(torch.abs(Ts_for_alpha - sT))].item()
            if abs(closest_T - sT) > Ttol:
                return False
        return True

    print("\nSnapping alpha values (homogeneous — same alpha for all T columns):")
    snapped_alphas, alpha_tolerances = [], []
    for req_a in args.alphas:
        order = torch.argsort(torch.abs(uniq_alpha - req_a))
        chosen = None
        for idx in order:
            candidate = uniq_alpha[idx].item()
            if has_all_Ts(candidate, snapped_Ts, T_tolerances):
                chosen = candidate
                break
        if chosen is None:
            chosen = uniq_alpha[torch.argmin(torch.abs(uniq_alpha - req_a))].item()
            print(f"  Warning: no alpha near {req_a:.4g} with all Ts, using {chosen:.4g}")
        else:
            print(f"  alpha {req_a:.4g} -> {chosen:.4g}")
        snapped_alphas.append(chosen)
        alpha_tolerances.append(max(abs(chosen) * 0.01, 1e-9))

    grid = {}
    for row_idx, (sa, atol) in enumerate(zip(snapped_alphas, alpha_tolerances)):
        for col_idx, (sT, Ttol) in enumerate(zip(snapped_Ts, T_tolerances)):
            grid[(row_idx, col_idx)] = (sa, sT, atol, Ttol)

    label_alphas = snapped_alphas
    label_Ts     = snapped_Ts
    print(f"\nRow alpha values used: {label_alphas}")
    print(f"Col T values used (keV): {label_Ts}")

    # Build figure
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

            if cell_x.shape[0] == 0:
                ax.text(0.5, 0.5, "no solver data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=11, color="red")
                ax.set_xlim(0, 1)
            else:
                sort_idx = torch.argsort(cell_x)
                cell_x_np   = cell_x[sort_idx].numpy()
                cell_phi_np = cell_phi[sort_idx].numpy()

                pinn_phi = run_model_batched(
                    model, cell_x[sort_idx], alpha, T, device, args.batch_size
                )

                h_solver, = ax.plot(cell_x_np, cell_phi_np, color="C0", linewidth=1.5,
                                    label="BVP solver")
                h_pinn,   = ax.plot(cell_x_np, pinn_phi,    color="C1", linewidth=1.5,
                                    linestyle="--", label="PINN")

                if legend_handles is None:
                    legend_handles = [h_solver, h_pinn]

            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            if row_idx < 2:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("x", fontsize=18)
            ax.tick_params(axis="x", labelsize=16)

            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            ax.tick_params(axis="y", labelsize=16)

            if col_idx == 0:
                ax.set_ylabel(r"$\varphi(x)$", fontsize=18)

            if row_idx == 0:
                ax.set_title(f"$T = {label_Ts[col_idx]:.2f}$ keV", fontsize=18)

            if col_idx == 0:
                r0_bohr = label_alphas[row_idx] * B_M / A0_M
                ax.text(-0.35, 0.5, rf"$r_0 = {r0_bohr:.2f}\,a_0$",
                        transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=18)

    if legend_handles is not None:
        fig.legend(legend_handles, ["BVP solver", "PINN"],
                   loc="upper center", ncol=2, fontsize=18,
                   bbox_to_anchor=(0.5, 1.0), frameon=True)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Output path
    if args.output is not None:
        output_path = Path(args.output)
    else:
        alpha_str = ",".join(f"{a:.4g}" for a in label_alphas)
        T_str     = ",".join(f"{T:.4g}" for T in label_Ts)
        out_dir   = PROJECT_ROOT / "true_vs_target_phi_plots" / args.run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{alpha_str}-{T_str}.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
