"""
Generate and save a colormap of the phase-2 residual (per point) over a grid in (x, alpha).

You should edit the constants below before running:
  - RUN_NAME:     name of the training run (subfolder in saving_weights/)
  - CONFIG_NAME:  YAML config stem in src/yamls/ (without .yaml)
  - N:            number of grid points in each direction (x and alpha)
  - LOG_X:        if True, x grid is log-spaced between a small >0 min and 1
  - LOG_ALPHA:    if True, alpha grid is log-spaced between MIN_ALPHA and MAX_ALPHA
  - MIN_ALPHA:    minimum alpha (for the grid)
  - MAX_ALPHA:    maximum alpha (for the grid)
  - DEVICE:       "auto" to pick GPU if available, else CPU; or explicit string like "cuda:0"
"""

from pathlib import Path
from typing import Tuple, Optional

import math

import numpy as np
import torch
import matplotlib.pyplot as plt

from .utils import PROJECT_ROOT, sec_deriv_auto
from .infer_model import load_model_from_config_and_state


# ====== USER-EDITABLE CONSTANTS ======
RUN_NAME: str = "phi0_logx_redalpha_m05_k10_tanh"
# CHECK ACTIVATION OF THE RUN
CONFIG_NAME: str = "phase_2_phi0"
N: int = 200  # grid resolution in each direction

# Grid controls
a0 = 5.291772105e-11  # in meters
b = 0.25 * (4.5 * math.pi**2)**(1/3) * a0
LOG_X: bool = True
LOG_ALPHA: bool = True
MIN_ALPHA: float = 1
MAX_ALPHA: float = 100

# Device selection: "auto" chooses CUDA if available, else CPU
DEVICE: str = "auto"


def find_latest_state_path(run_name: str) -> Path:
    """
    Find the latest saved weights file for a given run name,
    assuming files are saved as `weights_epoch_<int>` under:
        PROJECT_ROOT/saving_weights/<run_name>/
    """
    weights_dir = PROJECT_ROOT / "saving_weights" / run_name
    if not weights_dir.exists():
        raise FileNotFoundError(f"Weights directory not found: {weights_dir}")

    latest_epoch: Optional[int] = None
    latest_path: Optional[Path] = None

    for p in weights_dir.iterdir():
        name = p.name
        if not name.startswith("weights_epoch_"):
            continue
        try:
            epoch_str = name.split("weights_epoch_")[1]
            epoch = int(epoch_str)
        except (IndexError, ValueError):
            continue

        if (latest_epoch is None) or (epoch > latest_epoch):
            latest_epoch = epoch
            latest_path = p

    if latest_path is None:
        raise FileNotFoundError(
            f"No checkpoint files of form 'weights_epoch_<int>' found in {weights_dir}"
        )

    return latest_path


def build_grid(
    n: int,
    device: torch.device,
    log_x: bool = None,
    log_alpha: bool = None,
    min_alpha: float = None,
    max_alpha: float = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a grid of (x, alpha) values.

    x      in [0.0, 1.0]  (log-spaced >0..1 if log_x is True)
    alpha  in [min_alpha, max_alpha] (log-spaced if log_alpha is True)

    Returns:
        X_grid:   [N, N] tensor of x values
        A_grid:   [N, N] tensor of alpha values
        inputs:   [N*N, 2] tensor of (x, alpha) pairs (requires_grad=True)
    """
    # Use module-level defaults if not provided
    if log_x is None:
        log_x = LOG_X
    if log_alpha is None:
        log_alpha = LOG_ALPHA
    if min_alpha is None:
        min_alpha = MIN_ALPHA
    if max_alpha is None:
        max_alpha = MAX_ALPHA
    
    # x grid
    if log_x:
        # Avoid log(0); start from a small positive value
        x_min = 1e-6
        if x_min <= 0.0:
            raise ValueError("x_min for log-spaced x must be > 0.")
        x_vals = torch.logspace(
            start=torch.log10(torch.tensor(x_min, device=device)),
            end=torch.log10(torch.tensor(1.0, device=device)),
            steps=n,
            device=device,
        )
    else:
        x_vals = torch.linspace(0.0, 1.0, steps=n, device=device)

    # alpha grid
    if min_alpha <= 0.0 and log_alpha:
        raise ValueError("min_alpha must be > 0 when log_alpha is True.")
    if log_alpha:
        alpha_vals = torch.logspace(
            start=torch.log10(torch.tensor(min_alpha, device=device)),
            end=torch.log10(torch.tensor(max_alpha, device=device)),
            steps=n,
            device=device,
        )
    else:
        alpha_vals = torch.linspace(min_alpha, max_alpha, steps=n, device=device)

    # Meshgrid with alpha as rows (y-axis) and x as columns (x-axis)
    A_grid, X_grid = torch.meshgrid(alpha_vals, x_vals, indexing="ij")

    # Flatten into [N*N, 2] for model input
    x_flat = X_grid.reshape(-1, 1)
    a_flat = A_grid.reshape(-1, 1)

    inputs = torch.cat([x_flat, a_flat], dim=-1)

    inputs.requires_grad_(True)

    return X_grid, A_grid, inputs


def compute_pointwise_residual_sq(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """
    Compute the squared residual at each point for the phase-2 PDE:

        psi''(x) * x^(1/2) / C - psi(x)^(3/2) = 0,
        where C = alpha^(3/2).

    We return (residual ** 2) for each point, without averaging over the batch.
    """
    # inputs: [batch, 2] -> x is col 0, alpha is col 1
    x = inputs[:, 0:1]
    alpha = inputs[:, 1:2]

    # Second derivative w.r.t. x
    d2phi_dx2 = sec_deriv_auto(outputs, inputs, var1=0, var2=0)

    # C = alpha^(3/2)
    c = alpha ** (3.0 / 2.0)

    # Ensure phi has shape [batch, 1]
    if outputs.ndim == 2 and outputs.shape[1] == 1:
        phi = outputs
    elif outputs.ndim == 1:
        phi = outputs.unsqueeze(1)
    else:
        phi = outputs

    # Residual: (d2phi_dx2 * x**0.5) / c - phi**1.5
    residual = (d2phi_dx2 * x ** 0.5) / c - phi ** 1.5

    # Pointwise squared residual (no averaging)
    residual_sq = residual ** 2

    return residual_sq


def plot_and_save_colormap(
    X_grid: torch.Tensor,
    A_grid: torch.Tensor,
    residual_sq: torch.Tensor,
    run_name: str,
    output_dir: str = "color_maps",
    n: int = None,
    log_x: bool = None,
    log_alpha: bool = None,
) -> None:
    """
    Plot a colormap of the residual squared over (x, alpha) and save to:
        PROJECT_ROOT / output_dir / f"{run_name}.png"
    """
    # Use module-level defaults if not provided
    if log_x is None:
        log_x = LOG_X
    if log_alpha is None:
        log_alpha = LOG_ALPHA
    if n is None:
        n = N
    # Move to CPU and reshape residuals to [N, N]
    X_np = X_grid.detach().cpu().numpy()
    A_np = A_grid.detach().cpu().numpy()

    # infer N from grid shape
    n_alpha, n_x = A_np.shape
    residual_np = residual_sq.detach().cpu().numpy().reshape(n_alpha, n_x)

    fig, ax = plt.subplots(figsize=(6, 5))

    cmap = plt.get_cmap("viridis")
    # Use pcolormesh with shading for smooth visualization
    pcm = ax.pcolormesh(X_np, A_np, residual_np, shading="auto", cmap=cmap)
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(r"Residual$^2$")

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\alpha$")

    # If we used log spacing for either axis, show that explicitly in the plot
    if log_x:
        ax.set_xscale("log")
    if log_alpha:
        ax.set_yscale("log")
    ax.set_title(f"Phase 2 residual$^2$ map: {run_name}")

    fig.tight_layout()

    out_dir = PROJECT_ROOT / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = ""
    if log_alpha:
        tag += "_logalpha"
    if log_x:
        tag += "_logx"
    out_path = out_dir / f"{run_name}_colormap_{n}x{n}{tag}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def simple_plot_latest_epoch_only(
    outputs: torch.Tensor,
    inputs: torch.Tensor,
    run_name: str = None,
    output_dir: str = "color_maps",
    log_x: bool = None,
    log_alpha: bool = None,
) -> None:
    """
    Quick helper to plot psi(x) vs x for the current grid, mainly for debugging.
    Uses the model outputs and the first column of inputs (x).
    """
    # Use module-level defaults if not provided
    if run_name is None:
        run_name = RUN_NAME
    if log_x is None:
        log_x = LOG_X
    if log_alpha is None:
        log_alpha = LOG_ALPHA
    # Extract x (first column) and move everything to numpy for plotting
    x_t = inputs[:, 0:1].detach().cpu()
    y_t = outputs.detach().cpu()

    x_np = x_t.numpy().reshape(-1)
    y_np = y_t.numpy().reshape(-1)

    if x_np.shape[0] != y_np.shape[0]:
        print(
            f"Warning: shape mismatch in simple_plot_latest_epoch_only: "
            f"x has {x_np.shape}, y has {y_np.shape}"
        )
        return

    sort_idx = np.argsort(x_np)
    x_sorted = x_np[sort_idx]
    y_sorted = y_np[sort_idx]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_sorted, y_sorted, "+", color="red", ms=0.5, label=r"$\psi(x)$")

    ax.set_xlabel("x")
    ax.set_ylabel(r"$\psi(x)$")
    ax.set_title(f"{run_name} – prediction check")
    ax.legend(loc="best", framealpha=0.9)
    
    # Apply log scale if x was log-spaced
    if log_x:
        ax.set_xscale("log")

    fig.tight_layout()
    tag = ""
    if log_alpha:
        tag += "_logalpha"
    if log_x:
        tag += "_logx"
    save_path = PROJECT_ROOT / output_dir / f"{run_name}_pred_vs_x{tag}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def main():
    if RUN_NAME == "YOUR_RUN_NAME_HERE" or CONFIG_NAME == "YOUR_CONFIG_NAME_HERE":
        raise ValueError(
            "Please set RUN_NAME and CONFIG_NAME at the top of color_map.py before running."
        )

    # Choose device
    if DEVICE.lower() == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(DEVICE)

    # Find latest checkpoint for this run
    state_path = find_latest_state_path(RUN_NAME)

    # Load model and params from config + state dict
    model, params = load_model_from_config_and_state(
        config_name=CONFIG_NAME,
        state_path=str(state_path),
        device=str(device),
    )

    # We need gradients for second derivatives, so re-enable autograd globally.
    torch.set_grad_enabled(True)

    # Build (x, alpha) grid and input tensor
    X_grid, A_grid, inputs = build_grid(
        N, 
        device=device,
        log_x=LOG_X,
        log_alpha=LOG_ALPHA,
        min_alpha=MIN_ALPHA,
        max_alpha=MAX_ALPHA
    )

    # Forward pass
    outputs = model(inputs)

    simple_plot_latest_epoch_only(
        outputs, 
        inputs,
        log_x=LOG_X,
        log_alpha=LOG_ALPHA,
        output_dir=f"plot_predictions/newfiles/{RUN_NAME}"
    )

    # Compute pointwise squared residuals
    residual_sq = compute_pointwise_residual_sq(outputs, inputs)

    # Plot and save colormap
    plot_and_save_colormap(
        X_grid, 
        A_grid, 
        residual_sq, 
        RUN_NAME,
        log_x=LOG_X,
        log_alpha=LOG_ALPHA,
        output_dir=f"plot_predictions/newfiles/{RUN_NAME}"
    )


if __name__ == "__main__":
    main()


