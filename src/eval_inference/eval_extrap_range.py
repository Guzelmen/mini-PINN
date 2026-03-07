"""
Evaluation script: extrapolation-range (ood_single) colormaps for a trained Phase 4 model.

For OOD-alpha values (alpha outside training range, T inside):
  Produces (x, T) colormaps showing PDE residual and phi error.

For OOD-T values (T outside training range, alpha inside):
  Produces (x, alpha) colormaps showing PDE residual and phi error.

Selects n_vals values below AND above each boundary (default 3 each = 6 total per axis).
Values are chosen from the fine dataset, excluding any that appear in the coarse training set.

Usage:
    python -m src.eval_inference.eval_extrap_range <wandb_run_id> \\
        --run_name <run_name> \\
        [--device auto] \\
        [--output_dir extended_range_eval/ood_single] \\
        [--n_vals 3] \\
        [--log_x] \\
        [--batch_size 4096] \\
        [--epoch <int>]
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
        description="Phase 4 extrapolation-range (ood_single) colormaps."
    )
    parser.add_argument("run_id", help="Wandb run ID.")
    parser.add_argument("--run_name", required=True,
                        help="Local run name (subfolder in saving_weights/).")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--output_dir", type=str, default="extended_range_eval/ood_single",
        help="Output directory (relative to PROJECT_ROOT).",
    )
    parser.add_argument(
        "--n_vals", type=int, default=3,
        help="Number of OOD values to select below AND above each boundary (6 total per axis).",
    )
    parser.add_argument("--log_x", action="store_true",
                        help="Log-scale the x-axis of colormaps.")
    parser.add_argument("--batch_size", type=int, default=4096)
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


def _closest_fine_val(fine_vals_sorted, target, exclude_set):
    """Return the fine value closest to target that is not in exclude_set, or None."""
    candidates = [v for v in fine_vals_sorted if v not in exclude_set]
    if not candidates:
        return None
    return min(candidates, key=lambda v: abs(v - target))


def compute_rel_pde_residual(model, inputs, batch_size):
    """
    Compute pointwise relative L1 PDE residual: |residual| / (|RHS| + 1e-8)

    PDE: d²psi/dx² - (lambda^3 / gamma) * x * I_{1/2}(gamma*psi / (lambda*x)) = 0

    Args:
        model: trained Phase 4 model
        inputs: [N, 3] tensor [x, alpha, T_kV]
        batch_size: chunk size

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

        results.append(rel_res.detach().cpu().reshape(-1))

    return torch.cat(results, dim=0).numpy()


def compute_rel_phi_error(model, inputs, targets, batch_size):
    """
    Compute pointwise relative L1 phi error vs stored targets.

    Args:
        model: trained Phase 4 model
        inputs: [N, 3] tensor [x, alpha, T_kV]
        targets: [N, 1 or 2] tensor of reference values (first column is phi)
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

        tgt_phi = tgt[:, 0:1]
        rel_err = torch.abs(phi - tgt_phi) / (torch.abs(tgt_phi) + 1e-8)
        results.append(rel_err.detach().cpu().reshape(-1))

    return torch.cat(results, dim=0).numpy()


def plot_colormap_xT(x_np, T_np, values_np, label_val, label_prefix,
                     metric_name, metric_label, out_dir, log_x):
    """Scatter colormap over (x, T) plane — for OOD alpha values."""
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
    ax.set_title(rf"$\alpha = {label_val:.4g}$ (OOD)  —  {metric_name}")

    fig.tight_layout()

    x_tag = "_logx" if log_x else "_linx"
    fname = f"{label_prefix}{label_val:.6g}_{metric_name}_logT{x_tag}.png"
    save_path = out_dir / fname
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_colormap_xAlpha(x_np, alpha_np, values_np, label_val, label_prefix,
                         metric_name, metric_label, out_dir, log_x):
    """Scatter colormap over (x, alpha) plane — for OOD T values."""
    fig, ax = plt.subplots(figsize=(7, 5))

    vmin = max(float(values_np.min()), 1e-30)
    vmax = float(values_np.max())
    if vmin >= vmax:
        vmax = vmin * 10

    sc = ax.scatter(
        x_np, alpha_np,
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
    ax.set_ylabel(r"$\alpha$")
    ax.set_title(rf"$T = {label_val:.4g}$ keV (OOD)  —  {metric_name}")

    fig.tight_layout()

    x_tag = "_logx" if log_x else "_linx"
    fname = f"{label_prefix}{label_val:.6g}_{metric_name}_logalpha{x_tag}.png"
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
    fine_inputs    = fine_raw['inputs'].float()
    fine_targets   = fine_raw['targets'].float()

    # x filtering
    if getattr(params, 'filter_x_min', False):
        thresh = getattr(params, 'x_min_threshold', 1e-6)
        mask_c = coarse_inputs[:, 0] >= thresh
        coarse_inputs = coarse_inputs[mask_c]
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

    coarse_alpha_set = set(coarse_alpha_unique.tolist())
    coarse_T_set     = set(coarse_T_unique.tolist())

    # Output directory
    subdir = args.run_name if args.epoch is None else f"{args.run_name}_EPOCH_{args.epoch}"
    base_dir = PROJECT_ROOT / args.output_dir / subdir

    pde_alpha_dir    = base_dir / "pde"    / "alpha_ood"
    pde_T_dir        = base_dir / "pde"    / "T_ood"
    target_alpha_dir = base_dir / "target" / "alpha_ood"
    target_T_dir     = base_dir / "target" / "T_ood"

    for d in [pde_alpha_dir, pde_T_dir, target_alpha_dir, target_T_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {base_dir}\n")

    # -----------------------------------------------------------------------
    # Part 1: OOD alpha values — (x, T) colormaps
    # -----------------------------------------------------------------------
    fine_alpha_unique = sorted([float(a) for a in torch.unique(fine_inputs[:, 1]).tolist()])

    alpha_below = sorted([a for a in fine_alpha_unique if a < alpha_lo and a not in coarse_alpha_set])
    alpha_above = sorted([a for a in fine_alpha_unique if a > alpha_hi and a not in coarse_alpha_set])

    selected_alpha_below = alpha_below[-args.n_vals:] if len(alpha_below) >= args.n_vals else alpha_below
    selected_alpha_above = alpha_above[:args.n_vals]  if len(alpha_above) >= args.n_vals else alpha_above

    # Append fine dataset min/max (absolute boundaries from generation).
    # Do NOT exclude coarse values here — we specifically want the extreme edge points.
    already_selected_alpha = set(selected_alpha_below + selected_alpha_above)
    alpha_min_val = _closest_fine_val(fine_alpha_unique, fine_alpha_unique[0], already_selected_alpha)
    alpha_max_val = _closest_fine_val(fine_alpha_unique, fine_alpha_unique[-1], already_selected_alpha)
    if alpha_min_val is not None and alpha_min_val < alpha_lo:
        selected_alpha_below = sorted(selected_alpha_below + [alpha_min_val])
    if alpha_max_val is not None and alpha_max_val > alpha_hi:
        selected_alpha_above = sorted(selected_alpha_above + [alpha_max_val])

    selected_alphas = selected_alpha_below + selected_alpha_above

    print(f"OOD alpha values: {len(selected_alpha_below)} below, {len(selected_alpha_above)} above")
    print(f"  Below: {[f'{a:.4g}' for a in selected_alpha_below]}")
    print(f"  Above: {[f'{a:.4g}' for a in selected_alpha_above]}\n")

    for alpha_val in selected_alphas:
        print(f"=== alpha = {alpha_val:.4g} (OOD) ===")

        alpha_mask = torch.abs(fine_inputs[:, 1] - alpha_val) < 1e-9
        eval_inputs  = fine_inputs[alpha_mask]
        eval_targets = fine_targets[alpha_mask]

        n_pts = eval_inputs.shape[0]
        print(f"  {n_pts} fine points, {len(torch.unique(eval_inputs[:, 2]))} unique T values")

        if n_pts == 0:
            print(f"  No points found, skipping.")
            continue

        x_np = eval_inputs[:, 0].numpy()
        T_np = eval_inputs[:, 2].numpy()

        print("  Computing PDE residual...")
        pde_res = compute_rel_pde_residual(model, eval_inputs, args.batch_size)
        print("  Computing phi error...")
        phi_err = compute_rel_phi_error(model, eval_inputs, eval_targets, args.batch_size)

        print(f"  PDE residual: mean={pde_res.mean():.3e}, median={np.median(pde_res):.3e}, max={pde_res.max():.3e}")
        print(f"  Phi error:    mean={phi_err.mean():.3e}, median={np.median(phi_err):.3e}, max={phi_err.max():.3e}")

        plot_colormap_xT(x_np, T_np, pde_res, alpha_val, "alpha",
                         metric_name="pde_residual",
                         metric_label=r"$|\mathrm{residual}| / |\mathrm{RHS}|$",
                         out_dir=pde_alpha_dir, log_x=args.log_x)
        plot_colormap_xT(x_np, T_np, phi_err, alpha_val, "alpha",
                         metric_name="phi_error",
                         metric_label=r"$|\phi_\mathrm{pred} - \phi_\mathrm{target}| / |\phi_\mathrm{target}|$",
                         out_dir=target_alpha_dir, log_x=args.log_x)

    # -----------------------------------------------------------------------
    # Part 2: OOD T values — (x, alpha) colormaps
    # -----------------------------------------------------------------------
    fine_T_unique = sorted([float(t) for t in torch.unique(fine_inputs[:, 2]).tolist()])

    T_below = sorted([t for t in fine_T_unique if t < T_lo and t not in coarse_T_set])
    T_above = sorted([t for t in fine_T_unique if t > T_hi and t not in coarse_T_set])

    selected_T_below = T_below[-args.n_vals:] if len(T_below) >= args.n_vals else T_below
    selected_T_above = T_above[:args.n_vals]  if len(T_above) >= args.n_vals else T_above

    # Append fine dataset min/max (absolute boundaries from generation).
    # Do NOT exclude coarse values here — we specifically want the extreme edge points.
    already_selected_T = set(selected_T_below + selected_T_above)
    T_min_val = _closest_fine_val(fine_T_unique, fine_T_unique[0], already_selected_T)
    T_max_val = _closest_fine_val(fine_T_unique, fine_T_unique[-1], already_selected_T)
    if T_min_val is not None and T_min_val < T_lo:
        selected_T_below = sorted(selected_T_below + [T_min_val])
    if T_max_val is not None and T_max_val > T_hi:
        selected_T_above = sorted(selected_T_above + [T_max_val])

    selected_Ts = selected_T_below + selected_T_above

    print(f"\nOOD T values: {len(selected_T_below)} below, {len(selected_T_above)} above")
    print(f"  Below: {[f'{t:.4g}' for t in selected_T_below]}")
    print(f"  Above: {[f'{t:.4g}' for t in selected_T_above]}\n")

    for T_val in selected_Ts:
        print(f"=== T = {T_val:.4g} keV (OOD) ===")

        T_mask = torch.abs(fine_inputs[:, 2] - T_val) < 1e-9
        eval_inputs  = fine_inputs[T_mask]
        eval_targets = fine_targets[T_mask]

        n_pts = eval_inputs.shape[0]
        print(f"  {n_pts} fine points, {len(torch.unique(eval_inputs[:, 1]))} unique alpha values")

        if n_pts == 0:
            print(f"  No points found, skipping.")
            continue

        x_np     = eval_inputs[:, 0].numpy()
        alpha_np = eval_inputs[:, 1].numpy()

        print("  Computing PDE residual...")
        pde_res = compute_rel_pde_residual(model, eval_inputs, args.batch_size)
        print("  Computing phi error...")
        phi_err = compute_rel_phi_error(model, eval_inputs, eval_targets, args.batch_size)

        print(f"  PDE residual: mean={pde_res.mean():.3e}, median={np.median(pde_res):.3e}, max={pde_res.max():.3e}")
        print(f"  Phi error:    mean={phi_err.mean():.3e}, median={np.median(phi_err):.3e}, max={phi_err.max():.3e}")

        plot_colormap_xAlpha(x_np, alpha_np, pde_res, T_val, "T",
                             metric_name="pde_residual",
                             metric_label=r"$|\mathrm{residual}| / |\mathrm{RHS}|$",
                             out_dir=pde_T_dir, log_x=args.log_x)
        plot_colormap_xAlpha(x_np, alpha_np, phi_err, T_val, "T",
                             metric_name="phi_error",
                             metric_label=r"$|\phi_\mathrm{pred} - \phi_\mathrm{target}| / |\phi_\mathrm{target}|$",
                             out_dir=target_T_dir, log_x=args.log_x)

    print("\nDone.")


if __name__ == "__main__":
    main()
