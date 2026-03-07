"""
Evaluation script: interpolation-range (x, T) colormaps for a trained Phase 4 model.

For each of N selected alpha values (from fine dataset, not seen in coarse training),
produces two colormaps showing performance over the (x, T) plane:
  1. Relative L1 PDE residual: |residual| / |RHS|
  2. Relative L1 phi error vs stored targets: |phi_pred - phi_target| / (|phi_target| + 1e-8)

Alpha values are selected from the fine dataset within the training alpha range [alpha_lo, alpha_hi]
but excluding alpha values present in the coarse training set.

T values for each alpha are restricted to [T_lo, T_hi] (interpolation range)
and must not appear in the coarse training set (unseen T only).

Usage:
    python -m src.eval_interp_range <wandb_run_id> \\
        --run_name <run_name> \\
        [--device auto] \\
        [--output_dir extended_range_eval/interpolation] \\
        [--n_alpha 10] \\
        [--log_x] \\
        [--batch_size 4096]
"""
import argparse
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from ..utils import PROJECT_ROOT, sec_deriv_auto
from ..fd_integrals import fermi_dirac_half, compute_lambda, compute_gamma
from .color_map import find_latest_state_path
from .eval_residual_phase2 import fetch_config_from_wandb, build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 4 interpolation-range (x, T) colormaps for unseen alpha values."
    )
    parser.add_argument(
        "run_id",
        help="Wandb run ID.",
    )
    parser.add_argument(
        "--run_name", required=True,
        help="Local run name (subfolder in saving_weights/).",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--output_dir", type=str, default="extended_range_eval/interpolation",
        help="Output directory (relative to PROJECT_ROOT).",
    )
    parser.add_argument(
        "--n_alpha", type=int, default=10,
        help="Number of alpha values to evaluate.",
    )
    parser.add_argument(
        "--log_x", action="store_true",
        help="Log-scale the x-axis of colormaps.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4096,
    )
    parser.add_argument(
        "--epoch", type=int, default=None,
        help="Epoch of checkpoint to load (e.g. --epoch 1000). Defaults to latest.",
    )
    return parser.parse_args()


def _log_edge(unique_vals, lo_pct, hi_pct):
    """Compute log-space quantile edges, matching extended_loader.py logic."""
    log_vals = torch.log(unique_vals)
    lo = float(torch.quantile(log_vals, lo_pct / 100.0))
    hi = float(torch.quantile(log_vals, 1.0 - hi_pct / 100.0))
    return math.exp(lo), math.exp(hi)


def find_state_path_for_epoch(run_name: str, epoch: int) -> Path:
    """Return the checkpoint path for a specific epoch."""
    weights_dir = PROJECT_ROOT / "saving_weights" / run_name
    p = weights_dir / f"weights_epoch_{epoch}"
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    return p


def compute_rel_pde_residual(model, inputs, batch_size):
    """
    Compute pointwise relative L1 PDE residual: |residual| / (|RHS| + 1e-8)

    PDE: d²psi/dx² - (lambda^3 / gamma) * x * I_{1/2}(gamma*psi / (lambda*x)) = 0

    Args:
        model: trained Phase 4 model
        inputs: [N, 3] tensor [x, alpha, T_kV]
        batch_size: chunk size for processing

    Returns:
        [N] numpy array of pointwise relative L1 residuals
    """
    device = next(model.parameters()).device
    n = inputs.shape[0]
    results = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        inp = inputs[start:end].to(device).detach().requires_grad_(True)

        outputs = model(inp)

        x = inp[:, 0:1]
        alpha = inp[:, 1:2]
        T_kV = inp[:, 2:3]

        d2phi_dx2 = sec_deriv_auto(outputs, inp, var1=0, var2=0)

        phi = outputs if (outputs.ndim == 2 and outputs.shape[1] == 1) else outputs.unsqueeze(1)

        lam = compute_lambda(alpha, T_kV)
        gamma = compute_gamma(T_kV, Z=1)

        eta = gamma * phi / (lam * x + 1e-12)
        fd_half = fermi_dirac_half(eta)

        coeff = (lam ** 3) / gamma
        rhs = coeff * x * fd_half
        residual = d2phi_dx2 - rhs
        rel_res = torch.abs(residual) / (torch.abs(rhs) + 1e-8)

        results.append(rel_res.detach().cpu().squeeze())

    return torch.cat(results, dim=0).numpy()


def compute_rel_phi_error(model, inputs, targets, batch_size):
    """
    Compute pointwise relative L1 phi error vs stored targets.

    Args:
        model: trained Phase 4 model
        inputs: [N, 3] tensor [x, alpha, T_kV]
        targets: [N, 1] tensor of reference phi values
        batch_size: chunk size

    Returns:
        [N] numpy array of pointwise relative L1 errors
    """
    device = next(model.parameters()).device
    n = inputs.shape[0]
    results = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        inp = inputs[start:end].to(device).requires_grad_(True)
        tgt = targets[start:end].to(device)

        outputs = model(inp)
        phi = outputs if (outputs.ndim == 2 and outputs.shape[1] == 1) else outputs.unsqueeze(1)

        # targets may have shape [N, 2] (double-target format); use first column only
        tgt_phi = tgt[:, 0:1]
        rel_err = torch.abs(phi - tgt_phi) / (torch.abs(tgt_phi) + 1e-8)
        results.append(rel_err.detach().cpu().reshape(-1))

    return torch.cat(results, dim=0).numpy()


def plot_colormap(x_np, T_np, values_np, alpha_val, metric_name, metric_label,
                  out_dir, log_x):
    """Scatter colormap over (x, T) plane."""
    fig, ax = plt.subplots(figsize=(7, 5))

    vmin = max(float(values_np.min()), 1e-30)
    vmax = float(values_np.max())
    if vmin >= vmax:
        vmax = vmin * 10

    sc = ax.scatter(
        x_np, T_np,
        c=values_np,
        cmap="viridis",
        norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
        s=20, marker="s", edgecolors="none",
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(metric_label)

    if log_x:
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$T$ (keV)")
    ax.set_title(rf"$\alpha = {alpha_val:.4g}$  —  {metric_name}")

    fig.tight_layout()

    x_tag = "_logx" if log_x else "_linx"
    fname = f"alpha{alpha_val:.6g}_{metric_name}_logT{x_tag}.png"
    save_path = out_dir / fname
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def main():
    args = parse_args()

    # Device
    if args.device.lower() == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load config and model
    config = fetch_config_from_wandb(args.run_id)
    params = SimpleNamespace(**config)
    params.device = device
    if not hasattr(params, "debug_mode"):
        params.debug_mode = False
    if hasattr(params, "x_min_threshold"):
        params.x_min_threshold = float(params.x_min_threshold)
    if hasattr(params, "random_seed"):
        params.random_seed = int(params.random_seed)

    phase = int(params.phase)
    if phase != 4:
        raise ValueError(f"This script is for Phase 4 only, got phase={phase}")

    if args.epoch is not None:
        state_path = find_state_path_for_epoch(args.run_name, args.epoch)
    else:
        state_path = find_latest_state_path(args.run_name)
    print(f"Loading checkpoint: {state_path}")
    state_dict = torch.load(state_path, map_location=device)
    model = build_model(params, state_dict, device)

    torch.set_grad_enabled(True)

    # Load both datasets
    coarse_path = str(PROJECT_ROOT / params.data_path)
    fine_path   = str(PROJECT_ROOT / params.fine_data_path)

    print(f"Loading coarse data from {coarse_path}...")
    coarse_raw = torch.load(coarse_path)
    print(f"Loading fine data from {fine_path}...")
    fine_raw = torch.load(fine_path)

    coarse_inputs  = coarse_raw['inputs'].float()
    coarse_targets = coarse_raw['targets'].float()
    fine_inputs    = fine_raw['inputs'].float()
    fine_targets   = fine_raw['targets'].float()

    # x filtering
    if getattr(params, 'filter_x_min', False):
        thresh = getattr(params, 'x_min_threshold', 1e-6)
        mask_c = coarse_inputs[:, 0] >= thresh
        coarse_inputs  = coarse_inputs[mask_c]
        coarse_targets = coarse_targets[mask_c]
        mask_f = fine_inputs[:, 0] >= thresh
        fine_inputs  = fine_inputs[mask_f]
        fine_targets = fine_targets[mask_f]
        print(f"Filtered x < {thresh} from coarse and fine datasets")

    # Compute training boundaries (same logic as extended_loader.py)
    alpha_lo_pct = getattr(params, 'alpha_lo_pct', 10)
    alpha_hi_pct = getattr(params, 'alpha_hi_pct', 5)
    T_lo_pct     = getattr(params, 'T_lo_pct', 5)
    T_hi_pct     = getattr(params, 'T_hi_pct', 5)

    coarse_alpha_unique = torch.unique(coarse_inputs[:, 1])
    coarse_T_unique     = torch.unique(coarse_inputs[:, 2])

    alpha_lo, alpha_hi = _log_edge(coarse_alpha_unique, alpha_lo_pct, alpha_hi_pct)
    T_lo,     T_hi     = _log_edge(coarse_T_unique,     T_lo_pct,     T_hi_pct)

    print(f"Training edges: alpha=[{alpha_lo:.4f}, {alpha_hi:.4f}], T=[{T_lo:.6f}, {T_hi:.4f}] keV")

    # Coarse alpha and T values as sets (for exclusion)
    coarse_alpha_set = set(coarse_alpha_unique.tolist())
    coarse_T_set     = set(coarse_T_unique.tolist())

    # Select alpha values: fine, inside training alpha range, not in coarse
    fine_alpha_unique = torch.unique(fine_inputs[:, 1])
    candidate_alphas = [
        float(a) for a in fine_alpha_unique.tolist()
        if alpha_lo <= float(a) <= alpha_hi and float(a) not in coarse_alpha_set
    ]
    candidate_alphas.sort()

    if len(candidate_alphas) == 0:
        print("ERROR: No unseen alpha values found in the fine dataset within training range.")
        sys.exit(1)

    n_alpha = min(args.n_alpha, len(candidate_alphas))
    if n_alpha < args.n_alpha:
        print(f"Warning: only {len(candidate_alphas)} candidate alpha values available, using all {n_alpha}.")

    # Pick evenly by index
    indices = np.round(np.linspace(0, len(candidate_alphas) - 1, n_alpha)).astype(int)
    selected_alphas = [candidate_alphas[i] for i in indices]
    print(f"\nSelected {n_alpha} alpha values: {[f'{a:.4g}' for a in selected_alphas]}")

    # Output directory
    subdir = args.run_name if args.epoch is None else f"{args.run_name}_EPOCH_{args.epoch}"
    base_dir = PROJECT_ROOT / args.output_dir / subdir
    pde_dir    = base_dir / "pde"
    target_dir = base_dir / "target"
    pde_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {base_dir}\n")

    # Process each alpha
    for alpha_val in selected_alphas:
        print(f"\n=== alpha = {alpha_val:.4g} ===")

        # Filter fine data to this alpha
        alpha_mask = torch.abs(fine_inputs[:, 1] - alpha_val) < 1e-9
        alpha_inputs  = fine_inputs[alpha_mask]
        alpha_targets = fine_targets[alpha_mask]

        if alpha_inputs.shape[0] == 0:
            print(f"  No fine points found for alpha={alpha_val:.4g}, skipping.")
            continue

        # Filter to T within training range and not in coarse T set
        T_col = alpha_inputs[:, 2]
        T_mask = (T_col >= T_lo) & (T_col <= T_hi)
        # Exclude coarse T values
        not_coarse_T = torch.tensor(
            [float(t) not in coarse_T_set for t in T_col.tolist()],
            dtype=torch.bool,
        )
        final_mask = T_mask & not_coarse_T

        eval_inputs  = alpha_inputs[final_mask]
        eval_targets = alpha_targets[final_mask]

        n_pts = eval_inputs.shape[0]
        print(f"  {n_pts} fine points (T in training range, T not in coarse)")

        if n_pts == 0:
            print(f"  No valid points for alpha={alpha_val:.4g}, skipping.")
            continue

        n_T_vals = len(torch.unique(eval_inputs[:, 2]))
        print(f"  Unique T values: {n_T_vals}")

        # --- PDE residual ---
        print("  Computing PDE residual...")
        pde_res = compute_rel_pde_residual(model, eval_inputs, args.batch_size)

        # --- Phi error ---
        print("  Computing phi error...")
        phi_err = compute_rel_phi_error(model, eval_inputs, eval_targets, args.batch_size)

        x_np = eval_inputs[:, 0].numpy()
        T_np = eval_inputs[:, 2].numpy()

        print(f"  PDE residual: mean={pde_res.mean():.3e}, median={np.median(pde_res):.3e}, max={pde_res.max():.3e}")
        print(f"  Phi error:    mean={phi_err.mean():.3e}, median={np.median(phi_err):.3e}, max={phi_err.max():.3e}")

        # --- Plots ---
        plot_colormap(
            x_np, T_np, pde_res, alpha_val,
            metric_name="pde_residual",
            metric_label=r"$|\mathrm{residual}| / |\mathrm{RHS}|$",
            out_dir=pde_dir,
            log_x=args.log_x,
        )
        plot_colormap(
            x_np, T_np, phi_err, alpha_val,
            metric_name="phi_error",
            metric_label=r"$|\phi_\mathrm{pred} - \phi_\mathrm{target}| / |\phi_\mathrm{target}|$",
            out_dir=target_dir,
            log_x=args.log_x,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
