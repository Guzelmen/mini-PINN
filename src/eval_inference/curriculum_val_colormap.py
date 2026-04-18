"""
Standalone eval script: relative L1 phi error colormap at x=1 over the (alpha, T) plane
for a curriculum-trained Phase 4 model.

The alpha/T range to evaluate is read directly from the wandb config fields:
    alpha_min_train, alpha_max_train, T_min_train, T_max_train

Only fine-data points with exactly x=1.0 are used (one per (alpha, T) pair).

Usage:
    python -m src.eval_inference.curriculum_val_colormap <wandb_run_id> \
        --run_name <run_name> \
        [--device auto] \
        [--epoch <N>]
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
import matplotlib.colors as mcolors

from ..utils import PROJECT_ROOT
from .color_map import find_latest_state_path
from .eval_interp_range import compute_rel_phi_error, find_state_path_for_epoch
from .eval_residual_phase2 import fetch_config_from_wandb, build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Curriculum val: phi error colormap at x=1 over (alpha, T) plane."
    )
    parser.add_argument("run_id", help="Wandb run ID.")
    parser.add_argument("--run_name", required=True,
                        help="Local run name (subfolder in saving_weights/).")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Checkpoint epoch to load. Defaults to latest.")
    parser.add_argument("--alpha_min", type=float, default=None,
                        help="Override alpha_min_train from config.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Device
    if args.device.lower() == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load config from wandb
    config = fetch_config_from_wandb(args.run_id)
    params = SimpleNamespace(**config)
    params.device = device
    if not hasattr(params, "debug_mode"):
        params.debug_mode = False

    # Cast numeric fields that may come back as strings from wandb
    for field in ("x_min_threshold", "random_seed", "batch_size"):
        if hasattr(params, field):
            try:
                setattr(params, field, int(float(getattr(params, field))))
            except (TypeError, ValueError):
                pass

    # Load checkpoint
    if args.epoch is not None:
        state_path = find_state_path_for_epoch(args.run_name, args.epoch)
    else:
        state_path = find_latest_state_path(args.run_name)
    print(f"Loading checkpoint: {state_path}")
    state_dict = torch.load(state_path, map_location=device)

    # Build model (also reads norm stats from state dict into params)
    model = build_model(params, state_dict, device)
    model.eval()
    torch.set_grad_enabled(True)  # hard_phase4 forward requires autograd

    # Print norm stats loaded from checkpoint (used by the model's forward pass)
    sd_keys = set(state_dict.keys())
    print("\n--- Norm stats loaded from checkpoint ---")
    if "a_mean" in sd_keys and "a_std" in sd_keys:
        print(f"  alpha (log): mean={params.standard_mean:.6g}, std={params.standard_std:.6g}  (norm_mode={params.norm_mode})")
    elif "a_min" in sd_keys and "a_max" in sd_keys:
        print(f"  alpha: min={state_dict['a_min']}, max={state_dict['a_max']}  (norm_mode=minmax)")
    else:
        print("  alpha: no norm stats found in checkpoint")
    if "T_mean" in sd_keys and "T_std" in sd_keys:
        print(f"  T (log):     mean={params.T_mean:.6g}, std={params.T_std:.6g}")
    else:
        print("  T: no norm stats found in checkpoint")
    if "x_log_mean" in sd_keys and "x_log_std" in sd_keys:
        print(f"  x (log):     mean={params.x_log_mean:.6g}, std={params.x_log_std:.6g}")
    else:
        print("  x (log): no norm stats found in checkpoint")
    print("-----------------------------------------\n")

    # Read training range from config
    alpha_lo = args.alpha_min if args.alpha_min is not None else float(params.alpha_min_train)
    alpha_hi = float(params.alpha_max_train)
    T_lo     = float(params.T_min_train)
    T_hi     = float(params.T_max_train)
    print(f"Training range: alpha=[{alpha_lo:.4g}, {alpha_hi:.4g}], T=[{T_lo:.4g}, {T_hi:.4g}] keV")

    # Load fine data
    fine_path = PROJECT_ROOT / params.fine_data_path
    print(f"Loading fine data from {fine_path}...")
    fine_raw     = torch.load(fine_path, map_location="cpu")
    fine_inputs  = fine_raw["inputs"].float()
    fine_targets = fine_raw["targets"].float()

    # Apply x_min filter consistent with training
    if getattr(params, "filter_x_min", False):
        thresh = float(getattr(params, "x_min_threshold", 1e-6))
        mask_x = fine_inputs[:, 0] >= thresh
        fine_inputs  = fine_inputs[mask_x]
        fine_targets = fine_targets[mask_x]

    # Filter to training alpha/T range
    range_mask = (
        (fine_inputs[:, 1] >= alpha_lo) & (fine_inputs[:, 1] <= alpha_hi) &
        (fine_inputs[:, 2] >= T_lo)     & (fine_inputs[:, 2] <= T_hi)
    )
    fine_inputs  = fine_inputs[range_mask]
    fine_targets = fine_targets[range_mask]
    print(f"Points inside training range: {fine_inputs.shape[0]}")

    # Select exactly x=1.0 points
    x1_mask  = fine_inputs[:, 0] == 1.0
    at_inputs  = fine_inputs[x1_mask]
    at_targets = fine_targets[x1_mask]
    print(f"Points at x=1.0: {at_inputs.shape[0]}")

    if at_inputs.shape[0] == 0:
        print("ERROR: no x=1.0 points found in the filtered data. Exiting.")
        sys.exit(1)

    # Compute relative L1 phi error
    batch_size = int(getattr(params, "batch_size", 4096))
    phi_vals = compute_rel_phi_error(model, at_inputs, at_targets, batch_size)

    # Plot
    alpha_np = at_inputs[:, 1].numpy()
    T_np     = at_inputs[:, 2].numpy()

    fig, ax = plt.subplots(figsize=(7, 5))

    vmin = max(float(phi_vals.min()), 1e-30)
    vmax = float(phi_vals.max())
    if vmin >= vmax:
        vmax = vmin * 10

    sc = ax.scatter(
        alpha_np, T_np,
        c=phi_vals,
        cmap="viridis",
        norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
        s=400, marker="s", edgecolors="none",
    )
    fig.colorbar(sc, ax=ax)

    ax.set_yscale("log")
    ax.set_xlabel(r"$\alpha$", fontsize=16)
    ax.set_ylabel(r"$T$ (keV)", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)

    fig.tight_layout()

    out_dir = PROJECT_ROOT / "curriculum_val_plots" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "alpha_T_phi_error.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
