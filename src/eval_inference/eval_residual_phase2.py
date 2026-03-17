"""
Standalone evaluation script: compute detailed PDE residual diagnostics
and generate colormaps for a trained model checkpoint.

Usage:
    python -m src.eval_residual_phase2 \
        --wandb_run_path "guzelmen_msci_project/mini_pinn/<run_id>" \
        --run_name <run_name> \
        [--n_grid 200] [--device auto] [--log_x] [--log_alpha] \
        [--min_alpha 1] [--max_alpha 100] [--output_dir "plot_predictions/eval"]
"""
import sys
import argparse
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .. import models
from ..utils import PROJECT_ROOT, sec_deriv_auto
from .color_map import find_latest_state_path, build_grid


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate PDE residuals for a trained model checkpoint."
    )
    parser.add_argument(
        "--wandb_run_path", required=True,
        help="Full wandb run path: entity/project/run_id",
    )
    parser.add_argument(
        "--run_name", required=True,
        help="Local run name (subfolder in saving_weights/).",
    )
    parser.add_argument("--n_grid", type=int, default=200, help="Grid resolution per axis.")
    parser.add_argument("--device", type=str, default="auto", help='Device: "auto", "cpu", "cuda:0", etc.')
    parser.add_argument("--log_x", action="store_true", help="Log-space the x axis.")
    parser.add_argument("--log_alpha", action="store_true", help="Log-space the alpha axis.")
    parser.add_argument("--min_alpha", type=float, default=1.0, help="Minimum alpha for grid.")
    parser.add_argument("--max_alpha", type=float, default=100.0, help="Maximum alpha for grid.")
    parser.add_argument(
        "--output_dir", type=str, default="plot_predictions/eval",
        help="Output directory (relative to PROJECT_ROOT).",
    )
    return parser.parse_args()


def fetch_config_from_wandb(wandb_run_path: str) -> dict:
    """Fetch the run config dict from the wandb API."""
    try:
        import wandb
        api = wandb.Api()
        wandb_run = api.run(wandb_run_path)
        return dict(wandb_run.config)
    except Exception as e:
        print(f"ERROR: Could not fetch config from wandb ({wandb_run_path}): {e}")
        sys.exit(1)


def build_model(params, state_dict, device):
    """
    Construct the model from params + state_dict, aligning normalization
    buffers with the checkpoint, then load weights.
    """
    # Align norm_mode with checkpoint buffers
    sd_keys = set(state_dict.keys())
    if "a_mean" in sd_keys and "a_std" in sd_keys:
        params.norm_mode = "standardize"
        try:
            val = state_dict["a_mean"]
            params.standard_mean = float(val.item() if hasattr(val, "item") else val)
            val = state_dict["a_std"]
            params.standard_std = float(val.item() if hasattr(val, "item") else val)
        except Exception:
            pass
    elif "a_min" in sd_keys and "a_max" in sd_keys:
        params.norm_mode = "minmax"

    # Phase 4: temperature normalization buffers
    if int(params.phase) == 4:
        if "T_mean" in sd_keys and "T_std" in sd_keys:
            try:
                val = state_dict["T_mean"]
                params.T_mean = float(val.item() if hasattr(val, "item") else val)
                val = state_dict["T_std"]
                params.T_std = float(val.item() if hasattr(val, "item") else val)
            except Exception:
                pass

        # Phase 4: log-x normalization buffers
        if "x_log_mean" in sd_keys and "x_log_std" in sd_keys:
            try:
                val = state_dict["x_log_mean"]
                params.x_log_mean = float(val.item() if hasattr(val, "item") else val)
                val = state_dict["x_log_std"]
                params.x_log_std = float(val.item() if hasattr(val, "item") else val)
            except Exception:
                pass

    # Instantiate model class
    mode = str(params.mode).strip()
    phase = int(params.phase)
    model_class_name = f"Model_{mode}_phase{phase}"
    if not hasattr(models, model_class_name):
        raise ValueError(
            f"Model class '{model_class_name}' not found in src/models.py "
            f"(mode={mode}, phase={phase})."
        )
    ModelClass = getattr(models, model_class_name)
    model = ModelClass(params)

    # Load state dict
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys in state dict: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys in state dict: {unexpected}")

    model.to(device)
    model.eval()

    # Print loaded model summary
    print(f"\n=== Model loaded: {model_class_name} ===")
    print(f"  mode={mode}, phase={phase}")
    print(f"  hidden={getattr(params, 'hidden', '?')}, nlayers={getattr(params, 'nlayers', '?')}")
    print(f"  activation={getattr(params, 'activation', '?')}, w0={getattr(params, 'w0', '?')}")
    print(f"  inp_dim={getattr(params, 'inp_dim', '?')}, norm_mode={getattr(params, 'norm_mode', '?')}")
    print(f"  add_phi0={getattr(params, 'add_phi0', '?')}, k_val={getattr(params, 'k_val', '?')}")
    if getattr(params, 'norm_mode', None) == "standardize":
        print(f"  standard_mean={getattr(params, 'standard_mean', '?')}, standard_std={getattr(params, 'standard_std', '?')}")
    elif getattr(params, 'norm_mode', None) == "minmax":
        print(f"  minmax_min={getattr(params, 'minmax_min', '?')}, minmax_max={getattr(params, 'minmax_max', '?')}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  total parameters: {n_params}")
    print()

    return model


def compute_residuals(model, inputs, params):
    """
    Forward pass + PDE residual computation.
    Returns (outputs, residual, rel_residual) tensors.
    """
    outputs = model(inputs)

    phase = int(params.phase)
    if phase in (2, 3):
        d2phi_dx2 = sec_deriv_auto(outputs, inputs, var1=0, var2=0)
        x = inputs[:, 0:1]
        alpha = inputs[:, 1:2]

        phi = outputs if (outputs.ndim == 2 and outputs.shape[1] == 1) else outputs.unsqueeze(1)

        # Debug: check for NaN/Inf in intermediate quantities
        print(f"[debug] phi   — min: {phi.min().item():.6e}, max: {phi.max().item():.6e}, "
              f"nan: {torch.isnan(phi).sum().item()}, inf: {torch.isinf(phi).sum().item()}")
        print(f"[debug] d2phi — min: {d2phi_dx2.min().item():.6e}, max: {d2phi_dx2.max().item():.6e}, "
              f"nan: {torch.isnan(d2phi_dx2).sum().item()}, inf: {torch.isinf(d2phi_dx2).sum().item()}")
        print(f"[debug] x     — min: {x.min().item():.6e}, max: {x.max().item():.6e}")
        print(f"[debug] alpha — min: {alpha.min().item():.6e}, max: {alpha.max().item():.6e}")

        lhs = (d2phi_dx2 * x ** 0.5) / (alpha ** 1.5)
        rhs = phi ** 1.5

        print(f"[debug] lhs   — nan: {torch.isnan(lhs).sum().item()}, inf: {torch.isinf(lhs).sum().item()}")
        print(f"[debug] rhs   — nan: {torch.isnan(rhs).sum().item()}, inf: {torch.isinf(rhs).sum().item()}")

        residual = lhs - rhs
        rel_residual = residual / (torch.abs(rhs) + 1e-8)
    elif phase == 4:
        raise NotImplementedError("Phase 4 eval not yet implemented")
    else:
        raise ValueError(f"Unsupported phase: {phase}")

    return outputs, residual, rel_residual


def print_diagnostics(rel_residual, inputs, args):
    """Print detailed PDE residual diagnostics to stdout."""
    rel_res_np = rel_residual.detach().cpu().numpy().reshape(-1)
    inp_np = inputs.detach().cpu().numpy()

    rel_res_sq = rel_res_np ** 2
    abs_rel = np.abs(rel_res_np)

    mean_rel_l2 = float(np.mean(rel_res_sq))
    rms_rel = float(np.sqrt(mean_rel_l2))
    max_rel_sq = float(np.max(rel_res_sq))
    max_abs_rel = float(np.max(abs_rel))
    max_idx = int(np.argmax(abs_rel))
    max_x = float(inp_np[max_idx, 0])
    max_alpha = float(inp_np[max_idx, 1])

    pcts = [50, 90, 95, 99]
    pct_vals = np.percentile(abs_rel, pcts)

    print(f"\n=== PDE Residual Diagnostics: {args.run_name} ===")
    print(f"Wandb run: {args.wandb_run_path}")
    print(f"Phase: {args.phase}, Mode: {args.mode}")
    print(f"Grid: {args.n_grid}x{args.n_grid}, alpha range: [{args.min_alpha}, {args.max_alpha}]")
    print()
    print(f"Mean rel L2 residual²:  {mean_rel_l2:.6e}")
    print(f"RMS rel residual:       {rms_rel:.6e}  ({rms_rel * 100:.2f}%)")
    print(f"Max  rel residual²:     {max_rel_sq:.6e}")
    print(f"Max  |rel residual|:    {max_abs_rel:.6e}  ({max_abs_rel * 100:.2f}%)")
    print(f"Max error location:     x = {max_x:.6e}, alpha = {max_alpha:.6e}")
    print()
    print("Percentile breakdown (|rel_residual|):")
    for p, v in zip(pcts, pct_vals):
        print(f"  {p:>2d}th: {v:.4e}  ({v * 100:.2f}%)")
    print()


def plot_colormap(X_grid, A_grid, rel_residual, args, out_dir):
    """Save a colormap of relative residual² over (x, alpha)."""
    n = args.n_grid
    X_np = X_grid.detach().cpu().numpy()
    A_np = A_grid.detach().cpu().numpy()
    rel_res_sq = (rel_residual.detach().cpu().numpy().reshape(n, n)) ** 2

    # Replace NaN/Inf and clamp tiny values for LogNorm
    rel_res_sq = np.nan_to_num(rel_res_sq, nan=1e-30, posinf=1e10, neginf=1e-30)
    floor = 1e-30
    rel_res_sq = np.clip(rel_res_sq, floor, None)
    vmin, vmax = float(rel_res_sq.min()), float(rel_res_sq.max())
    if vmin >= vmax:
        vmax = vmin * 10  # ensure vmin < vmax for LogNorm

    fig, ax = plt.subplots(figsize=(7, 5))
    pcm = ax.pcolormesh(
        X_np, A_np, rel_res_sq,
        shading="auto",
        cmap="viridis",
        norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
    )
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(r"$(\mathrm{residual} / |\mathrm{RHS}|)^2$")

    if args.log_x:
        ax.set_xscale("log")
    if args.log_alpha:
        ax.set_yscale("log")

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\alpha$")
    ax.set_title(f"Phase {args.phase} relative residual² — {args.run_name}")
    fig.tight_layout()

    save_path = out_dir / f"{args.run_name}_rel_residual_sq_colormap.png"
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved colormap: {save_path}")


def plot_1d_lines(X_grid, A_grid, rel_residual, args, out_dir):
    """Plot |rel_residual| vs x for 5 alpha slices."""
    n = args.n_grid
    rel_res_np = np.abs(rel_residual.detach().cpu().numpy().reshape(n, n))
    X_np = X_grid.detach().cpu().numpy()
    A_np = A_grid.detach().cpu().numpy()

    # Alpha values along the first column (rows of A_grid)
    alpha_vals = A_np[:, 0]

    # Pick 5 indices evenly log-spaced
    log_alphas = np.log10(alpha_vals)
    target_logs = np.linspace(log_alphas.min(), log_alphas.max(), 5)
    row_idxs = [int(np.argmin(np.abs(log_alphas - t))) for t in target_logs]

    fig, ax = plt.subplots(figsize=(7, 5))
    for idx in row_idxs:
        a_val = alpha_vals[idx]
        x_line = X_np[idx, :]
        res_line = rel_res_np[idx, :]
        ax.plot(x_line, res_line, label=rf"$\alpha={a_val:.2f}$", linewidth=1.0)

    ax.set_yscale("log")
    if args.log_x:
        ax.set_xscale("log")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$|\mathrm{rel\ residual}|$")
    ax.set_title(f"Phase {args.phase} |rel residual| vs x — {args.run_name}")
    ax.legend(fontsize=8)
    fig.tight_layout()

    save_path = out_dir / f"{args.run_name}_rel_residual_vs_x.png"
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved 1D line plot: {save_path}")


def plot_phi_vs_x(model, device, args, out_dir):
    """Plot phi(x) vs x for a few fixed alpha values, with physical axis labels."""
    import math
    a0 = 5.291772105e-11  # Bohr radius in meters
    b = 0.25 * (4.5 * math.pi**2)**(1/3) * a0  # TF length scale

    alpha_values = [1.0, 3.0, 5.0]
    n_x = 500
    x_vals = torch.linspace(1e-6, 1.0, n_x, device=device)

    fig, ax = plt.subplots(figsize=(7, 5))

    for alpha_val in alpha_values:
        r0 = alpha_val * b
        r0_a0 = r0 / a0  # r0 in units of a0
        alpha_col = torch.full((n_x, 1), alpha_val, device=device)
        inputs = torch.cat([x_vals.unsqueeze(1), alpha_col], dim=1)
        inputs.requires_grad_(True)
        outputs = model(inputs)
        phi_np = outputs.detach().cpu().numpy().reshape(-1)
        x_np = x_vals.cpu().numpy()
        ax.plot(x_np, phi_np,
                label=rf"$r_0 = {r0_a0:.2f}\,a_0$",
                linewidth=1.5)

    ax.set_xlabel(r"$r / r_0$", fontsize=18)
    ax.set_ylabel(r"$\phi(r/r_0)$", fontsize=18)
    ax.legend(fontsize=15)
    fig.tight_layout()

    save_path = out_dir / f"{args.run_name}_phi_vs_x.png"
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved phi vs x plot: {save_path}")


def main():
    args = parse_args()

    # --- Device ---
    if args.device.lower() == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # --- Fetch config from wandb ---
    config = fetch_config_from_wandb(args.wandb_run_path)
    params = SimpleNamespace(**config)
    params.device = device
    # Ensure debug_mode exists (models read it)
    if not hasattr(params, "debug_mode"):
        params.debug_mode = False

    # Stash phase/mode on args for diagnostics printing
    args.phase = int(params.phase)
    args.mode = str(params.mode).strip()

    # --- Load weights ---
    state_path = find_latest_state_path(args.run_name)
    print(f"Loading checkpoint: {state_path}")
    state_dict = torch.load(state_path, map_location=device)

    # --- Build model ---
    model = build_model(params, state_dict, device)

    # --- Enable gradients for autograd derivatives ---
    torch.set_grad_enabled(True)

    # --- Build grid ---
    # Force log_x=True so x starts at 1e-6 instead of 0.
    # The PDE has a singularity at x=0 (d2phi/dx2 diverges), and training
    # data never includes x=0 exactly, so we match that here.
    if not args.log_x:
        print("[info] Overriding --log_x to True (x=0 causes PDE singularity)")
        args.log_x = True
    X_grid, A_grid, inputs = build_grid(
        n=args.n_grid,
        device=device,
        log_x=args.log_x,
        log_alpha=args.log_alpha,
        min_alpha=args.min_alpha,
        max_alpha=args.max_alpha,
    )

    # --- Forward + residuals ---
    outputs, residual, rel_residual = compute_residuals(model, inputs, params)

    # --- Diagnostics ---
    print_diagnostics(rel_residual, inputs, args)

    # --- Plots ---
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_colormap(X_grid, A_grid, rel_residual, args, out_dir)
    plot_1d_lines(X_grid, A_grid, rel_residual, args, out_dir)
    plot_phi_vs_x(model, device, args, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
