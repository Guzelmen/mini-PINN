"""
Standalone evaluation script: compute detailed PDE residual diagnostics
and generate (x, alpha) colormaps at fixed T slices for a trained Phase 4 model.

Produces one colormap per T value showing |PDE residual| over (x, alpha),
using actual validation points (not a synthetic grid).

Usage:
    python -m src.eval_residual_phase4 <wandb_run_id> \
        --run_name <run_name> \
        [--device auto] [--output_dir "plot_predictions/eval_phase4"] \
        [--T_values 1 3 5 7 10]
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.tri as tri

from .utils import PROJECT_ROOT, sec_deriv_auto
from .fd_integrals import fermi_dirac_half, compute_lambda, compute_gamma
from .color_map import find_latest_state_path
from .eval_residual_phase2 import fetch_config_from_wandb, build_model
from .data_utils.loader import load_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 4 PDE residual colormaps at fixed T slices (using val points)."
    )
    parser.add_argument(
        "run_id",
        help="Wandb run ID (appended to guzelmen_msci_project/mini_pinn/).",
    )
    parser.add_argument(
        "--run_name", required=True,
        help="Local run name (subfolder in saving_weights/).",
    )
    parser.add_argument("--device", type=str, default="auto", help='Device: "auto", "cpu", "cuda:0", etc.')
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (relative to PROJECT_ROOT). Defaults to eval_phase4/<run_name>/.",
    )
    parser.add_argument(
        "--T_values", nargs="+", type=float, default=[1, 3, 5, 7, 10],
        help="Temperature values (keV) for each colormap slice.",
    )
    return parser.parse_args()


def compute_pointwise_residual_phase4(model, inputs, batch_size=4096):
    """
    Compute pointwise |PDE residual| for Phase 4 on given inputs.

    PDE: d²ψ/dx² - (λ³/γ) · x · I_{1/2}(γψ/(λx)) = 0

    Args:
        model: trained Phase 4 model
        inputs: [N, 3] tensor with columns [x, alpha, T_kV], requires_grad=True
        batch_size: process in chunks to avoid OOM

    Returns:
        abs_residual: [N] tensor of |residual| per point
    """
    device = next(model.parameters()).device
    n = inputs.shape[0]
    abs_residuals = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        inp = inputs[start:end].to(device)
        inp.requires_grad_(True)

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
        rel_residual = residual / (torch.abs(rhs) + 1e-8)

        abs_residuals.append(torch.abs(rel_residual).detach().cpu().squeeze())

    return torch.cat(abs_residuals, dim=0)


def print_diagnostics_for_slice(abs_residual_np, x_np, alpha_np, T_val):
    """Print residual statistics for a single T slice."""
    mean_res = float(np.mean(abs_residual_np))
    max_res = float(np.max(abs_residual_np))
    max_idx = int(np.argmax(abs_residual_np))

    pcts = [50, 90, 95, 99]
    pct_vals = np.percentile(abs_residual_np, pcts)

    print(f"\n--- T = {T_val} keV  ({len(abs_residual_np)} val points) ---")
    print(f"  Mean |residual|:  {mean_res:.6e}")
    print(f"  Max  |residual|:  {max_res:.6e}  at x={x_np[max_idx]:.6e}, α={alpha_np[max_idx]:.6e}")
    for p, v in zip(pcts, pct_vals):
        print(f"  {p:>2d}th percentile:  {v:.6e}")


def plot_residual_colormap_at_T(x_np, alpha_np, abs_residual_np, T_val, out_dir, run_name):
    fig, ax = plt.subplots(figsize=(7, 5))
    
    sc = ax.scatter(
        x_np, alpha_np, 
        c=abs_residual_np, 
        cmap="viridis",
        norm=mcolors.LogNorm(vmin=max(abs_residual_np.min(), 1e-30), vmax=abs_residual_np.max()),
        s=30, edgecolors="none"
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"$|\mathrm{residual}| / |\mathrm{RHS}|$")
    
    # ax.set_xscale("log")  # uncomment for log x-axis
    ax.set_yscale("log")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\alpha$")
    ax.set_title(f"|Relative PDE residual| at T = {T_val:.2f} keV")

    fig.tight_layout()
    save_path = out_dir / f"pde_rel_residual_alpha_x_cmap_T{T_val}keV_linx.png"  # change to _logx if using log x-axis
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def main():
    args = parse_args()

    # Device
    if args.device.lower() == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Fetch config from wandb
    wandb_run_path = f"{args.run_id}"
    config = fetch_config_from_wandb(wandb_run_path)
    from types import SimpleNamespace
    params = SimpleNamespace(**config)
    params.device = device
    if not hasattr(params, "debug_mode"):
        params.debug_mode = False

    phase = int(params.phase)
    if phase != 4:
        raise ValueError(f"This script is for Phase 4 only, got phase={phase}")

    # Wandb config values may come back as strings — cast numeric fields
    if hasattr(params, "x_min_threshold"):
        params.x_min_threshold = float(params.x_min_threshold)
    if hasattr(params, "random_seed"):
        params.random_seed = int(params.random_seed)

    # Load weights
    state_path = find_latest_state_path(args.run_name)
    print(f"Loading checkpoint: {state_path}")
    state_dict = torch.load(state_path, map_location=device)

    # Build model
    model = build_model(params, state_dict, device)

    # Enable gradients for autograd derivatives
    torch.set_grad_enabled(True)

    # Load validation data (same split as training)
    data_dict = load_data(params)
    val_inputs = data_dict["val"].clone()  # [N_val, 3]
    print(f"Validation set: {val_inputs.shape[0]} points")

    # Compute pointwise residuals on entire val set
    print("Computing pointwise PDE residuals on validation set...")
    abs_residual = compute_pointwise_residual_phase4(model, val_inputs)
    abs_residual_np = abs_residual.numpy()

    val_inputs_np = val_inputs.numpy()
    x_all = val_inputs_np[:, 0]
    alpha_all = val_inputs_np[:, 1]
    T_all = val_inputs_np[:, 2]

    # Find unique T values in val set
    unique_T = np.unique(T_all)
    print(f"Unique T values in val set: {unique_T}")

    # Output directory
    if args.output_dir is not None:
        out_dir = PROJECT_ROOT / args.output_dir
    else:
        out_dir = PROJECT_ROOT / "eval_phase4" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process each target T
    print(f"\n=== Phase 4 PDE Residual Diagnostics: {args.run_name} ===")
    for T_target in args.T_values:
        # Find nearest unique T
        nearest_idx = int(np.argmin(np.abs(unique_T - T_target)))
        nearest_T = unique_T[nearest_idx]
        if abs(nearest_T - T_target) > 0.5:
            print(f"\nWarning: target T={T_target} keV, nearest in data is T={nearest_T:.4f} keV (gap > 0.5 keV)")

        mask = T_all == nearest_T
        x_slice = x_all[mask]
        alpha_slice = alpha_all[mask]
        res_slice = abs_residual_np[mask]

        if len(x_slice) == 0:
            print(f"\nNo val points found near T={T_target} keV, skipping.")
            continue

        print_diagnostics_for_slice(res_slice, x_slice, alpha_slice, nearest_T)
        plot_residual_colormap_at_T(x_slice, alpha_slice, res_slice, nearest_T, out_dir, args.run_name)

    print("\nDone.")


if __name__ == "__main__":
    main()
