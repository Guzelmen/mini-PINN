"""
Plot phi(x) prediction vs BVP target curves for selected (alpha, T) values.

For each alpha in ALPHA_VALUES, produces one PNG with two subplots side-by-side:
  Left:  BVP target    phi(x) for each T in T_VALUES
  Right: Model prediction phi(x) for the same T values

T and alpha values are snapped to the nearest available in the fine dataset.
Mean relative L1 error (%) is printed per (alpha, T) pair.

Usage:
    python -m src.eval_inference.stage_1_prediction_plots <wandb_run_id> \
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

from ..utils import PROJECT_ROOT
from .color_map import find_latest_state_path
from .eval_interp_range import find_state_path_for_epoch
from .eval_residual_phase2 import fetch_config_from_wandb, build_model

# ── Configurable ─────────────────────────────────────────────────────────────
T_VALUES     = [0.01]   # keV — snapped to nearest in data
ALPHA_VALUES = [1.0, 5.0, 10.0]
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="phi(x) prediction vs BVP target line plots."
    )
    parser.add_argument("run_id", help="Wandb run ID.")
    parser.add_argument("--run_name", required=True,
                        help="Local run name (subfolder in saving_weights/).")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Checkpoint epoch to load. Defaults to latest.")
    return parser.parse_args()


def _infer_phi(model, x_sorted, nearest_alpha, nearest_T, batch_size, device):
    """Run model inference and return phi predictions as a numpy array."""
    n = x_sorted.shape[0]
    ones = torch.ones(n, 1)
    inp = torch.cat([x_sorted.unsqueeze(1), ones * float(nearest_alpha), ones * float(nearest_T)], dim=1)

    results = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = inp[start:end].to(device).requires_grad_(True)
        with torch.enable_grad():
            out = model(chunk)
        phi = out.detach().cpu().squeeze(-1)
        results.append(phi)
    return torch.cat(results).numpy()


def main():
    args = parse_args()

    # Device
    if args.device.lower() == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Wandb config
    config = fetch_config_from_wandb(args.run_id)
    params = SimpleNamespace(**config)
    params.device = device
    if not hasattr(params, "debug_mode"):
        params.debug_mode = False
    for field in ("x_min_threshold", "random_seed", "batch_size"):
        if hasattr(params, field):
            try:
                setattr(params, field, int(float(getattr(params, field))))
            except (TypeError, ValueError):
                pass

    # Load checkpoint + model
    if args.epoch is not None:
        state_path = find_state_path_for_epoch(args.run_name, args.epoch)
    else:
        state_path = find_latest_state_path(args.run_name)
    print(f"Loading checkpoint: {state_path}")
    state_dict = torch.load(state_path, map_location=device)
    model = build_model(params, state_dict, device)
    model.eval()
    torch.set_grad_enabled(True)  # hard_phase4 forward requires autograd

    batch_size = int(getattr(params, "batch_size", 4096))

    # Load fine data
    fine_path = PROJECT_ROOT / params.fine_data_path
    print(f"Loading fine data from {fine_path}...")
    fine_raw     = torch.load(fine_path, map_location="cpu")
    fine_inputs  = fine_raw["inputs"].float()
    fine_targets = fine_raw["targets"].float()

    if getattr(params, "filter_x_min", False):
        thresh = float(getattr(params, "x_min_threshold", 1e-6))
        mask_x = fine_inputs[:, 0] >= thresh
        fine_inputs  = fine_inputs[mask_x]
        fine_targets = fine_targets[mask_x]

    out_dir = PROJECT_ROOT / "stage_1_predic_plots" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # ── Loop over alpha values ────────────────────────────────────────────────
    unique_alphas = fine_inputs[:, 1].unique()

    for alpha_target in ALPHA_VALUES:
        nearest_alpha = unique_alphas[torch.argmin(torch.abs(unique_alphas - alpha_target))]
        print(f"\nalpha requested: {alpha_target}, using: {float(nearest_alpha):.4g}")

        alpha_mask = fine_inputs[:, 1] == nearest_alpha
        unique_Ts  = fine_inputs[alpha_mask, 2].unique()

        fig, (ax_bvp, ax_pred) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

        for i, T_target in enumerate(T_VALUES):
            color = colors[i % len(colors)]

            nearest_T = unique_Ts[torch.argmin(torch.abs(unique_Ts - T_target))]
            print(f"  T requested: {T_target}, using: {float(nearest_T):.4g} keV")

            # Extract slice, sorted by x
            mask     = alpha_mask & (fine_inputs[:, 2] == nearest_T)
            x_vals   = fine_inputs[mask, 0]
            phi_tgt  = fine_targets[mask, 0]
            sort_idx = torch.argsort(x_vals)
            x_sorted     = x_vals[sort_idx]
            phi_tgt_sorted = phi_tgt[sort_idx].numpy()

            # Model inference
            phi_pred = _infer_phi(model, x_sorted, nearest_alpha, nearest_T, batch_size, device)

            # Error
            rel_l1 = float(
                (np.abs(phi_pred - phi_tgt_sorted) / (np.abs(phi_tgt_sorted) + 1e-8)).mean()
            ) * 100
            print(f"    Mean rel L1 error: {rel_l1:.3f}%")

            label = f"T = {float(nearest_T):.3g} keV"
            x_np  = x_sorted.numpy()
            ax_bvp.plot(x_np, phi_tgt_sorted, color=color, label=label)
            ax_pred.plot(x_np, phi_pred,       color=color, label=label)

        ax_bvp.set_xlabel(r"$x$", fontsize=16)
        ax_bvp.set_ylabel(r"$\phi$", fontsize=16)
        ax_bvp.tick_params(axis="both", labelsize=14)
        ax_bvp.legend(fontsize=14)

        ax_pred.set_xlabel(r"$x$", fontsize=16)
        ax_pred.tick_params(axis="both", labelsize=14)
        ax_pred.legend(fontsize=14)

        fig.tight_layout()
        save_path = out_dir / f"alpha_{float(nearest_alpha):.4g}.png"
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {save_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
