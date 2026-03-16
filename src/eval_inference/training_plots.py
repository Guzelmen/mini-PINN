"""
Diagnostic colormap plots generated periodically during Phase 4 training.

Enabled via config params:
    save_plots_during_training: True
    save_training_plots_every: 500   # epoch interval

Output structure:
    plots_during_training/<run_name>/epoch_<ep>/
        boundary_x_plots/
            alpha_T_pde_residual.png
            alpha_T_phi_error.png
        x_T_colormaps/
            PDE/     alpha<val>_pde_residual_logT_logx.png  (×10)
            target/  alpha<val>_phi_error_logT_logx.png     (×10)
        x_alpha_colormaps/
            PDE/     T<val>keV_pde_residual_logx.png  (×10)
            target/  T<val>keV_phi_error_logx.png     (×10)
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from ..utils import PROJECT_ROOT
from .eval_interp_range import (
    _log_edge,
    compute_rel_pde_residual,
    compute_rel_phi_error,
    plot_colormap,
)


@dataclass
class PlotData:
    # Training range edges
    alpha_lo: float
    alpha_hi: float
    T_lo: float
    T_hi: float

    # Full val pool (fine, inside training range)
    val_inputs: torch.Tensor   # [N, 3]  float32
    val_targets: torch.Tensor  # [N, >=1] float32

    # alpha-T boundary plot: inputs/targets at x closest to 1.0 per (alpha, T) pair
    at_inputs: torch.Tensor    # [M, 3]  — M <= 50*50
    at_targets: torch.Tensor   # [M, >=1]

    # 10 T values for x-alpha colormaps (logspaced within val)
    T_vals_for_xalpha: list    # list of float, length <= 10

    # 10 alpha values for x-T colormaps (linspaced within val)
    alpha_vals_for_xT: list    # list of float, length <= 10


def prepare_plot_data(params) -> Optional[PlotData]:
    """
    Load raw data and pre-select all points needed for periodic diagnostic plots.
    Called once before the training loop starts.

    Returns None if data cannot be prepared (missing fine_data_path, etc.).
    """
    if not getattr(params, 'use_extended_loader', False):
        print("[training_plots] use_extended_loader=False — skipping plot data prep.")
        return None

    coarse_path = str(PROJECT_ROOT / params.data_path)
    fine_path   = str(PROJECT_ROOT / params.fine_data_path)

    print(f"[training_plots] Loading coarse data from {coarse_path}...")
    coarse_raw = torch.load(coarse_path, map_location="cpu")
    print(f"[training_plots] Loading fine data from {fine_path}...")
    fine_raw   = torch.load(fine_path,   map_location="cpu")

    coarse_inputs  = coarse_raw['inputs'].float()
    fine_inputs    = fine_raw['inputs'].float()
    fine_targets   = fine_raw['targets'].float()

    # Apply x_min filter consistent with training
    if getattr(params, 'filter_x_min', False):
        thresh = float(getattr(params, 'x_min_threshold', 1e-6))
        coarse_inputs = coarse_inputs[coarse_inputs[:, 0] >= thresh]
        mask_f        = fine_inputs[:, 0] >= thresh
        fine_inputs   = fine_inputs[mask_f]
        fine_targets  = fine_targets[mask_f]

    # Compute training range edges from coarse unique values
    alpha_lo_pct = getattr(params, 'alpha_lo_pct', 10)
    alpha_hi_pct = getattr(params, 'alpha_hi_pct', 5)
    T_lo_pct     = getattr(params, 'T_lo_pct', 5)
    T_hi_pct     = getattr(params, 'T_hi_pct', 5)

    coarse_alpha_unique = torch.unique(coarse_inputs[:, 1])
    coarse_T_unique     = torch.unique(coarse_inputs[:, 2])

    alpha_lo, alpha_hi = _log_edge(coarse_alpha_unique, alpha_lo_pct, alpha_hi_pct)
    T_lo,     T_hi     = _log_edge(coarse_T_unique,     T_lo_pct,     T_hi_pct)

    print(f"[training_plots] Training edges: alpha=[{alpha_lo:.4g}, {alpha_hi:.4g}], "
          f"T=[{T_lo:.4g}, {T_hi:.4g}] keV")

    # Val pool: fine points inside training range
    f_alpha_inside = (fine_inputs[:, 1] >= alpha_lo) & (fine_inputs[:, 1] <= alpha_hi)
    f_T_inside     = (fine_inputs[:, 2] >= T_lo)     & (fine_inputs[:, 2] <= T_hi)
    interp_mask    = f_alpha_inside & f_T_inside

    val_inputs  = fine_inputs[interp_mask]
    val_targets = fine_targets[interp_mask]

    if val_inputs.shape[0] == 0:
        print("[training_plots] WARNING: no fine val points found inside training range.")
        return None

    print(f"[training_plots] Val pool: {val_inputs.shape[0]} points")

    # --- Build alpha-T grid (50×50) at x≈1 ---
    val_alpha_unique = torch.unique(val_inputs[:, 1])
    val_T_unique     = torch.unique(val_inputs[:, 2])

    n_grid = 50
    # 50 alpha values: linspaced by index
    a_idx   = np.round(np.linspace(0, len(val_alpha_unique) - 1, n_grid)).astype(int)
    grid_alphas = [float(val_alpha_unique[i]) for i in a_idx]
    # 50 T values: logspaced by index in log-space (unique vals are already sorted)
    t_idx   = np.round(np.linspace(0, len(val_T_unique) - 1, n_grid)).astype(int)
    grid_Ts = [float(val_T_unique[i]) for i in t_idx]

    at_inp_list  = []
    at_tgt_list  = []

    val_x_col     = val_inputs[:, 0]
    val_alpha_col = val_inputs[:, 1]
    val_T_col     = val_inputs[:, 2]

    for a_val in grid_alphas:
        for t_val in grid_Ts:
            a_mask = torch.abs(val_alpha_col - a_val) < 1e-9
            t_mask = torch.abs(val_T_col     - t_val) < 1e-9
            pair_mask = a_mask & t_mask
            if pair_mask.sum() == 0:
                continue
            pair_x = val_x_col[pair_mask]
            # Pick the point with x closest to 1.0
            best_idx = torch.argmin(torch.abs(pair_x - 1.0))
            # Map back to global val indices
            global_indices = torch.where(pair_mask)[0]
            chosen = global_indices[best_idx]
            at_inp_list.append(val_inputs[chosen])
            at_tgt_list.append(val_targets[chosen])

    if len(at_inp_list) == 0:
        print("[training_plots] WARNING: no alpha-T grid points found.")
        at_inputs  = torch.zeros(0, val_inputs.shape[1])
        at_targets = torch.zeros(0, val_targets.shape[1])
    else:
        at_inputs  = torch.stack(at_inp_list,  dim=0)
        at_targets = torch.stack(at_tgt_list,  dim=0)

    print(f"[training_plots] alpha-T grid: {at_inputs.shape[0]} / {n_grid*n_grid} pairs populated")

    # --- 10 T values for x-alpha colormaps (logspaced by index) ---
    n_slice = 10
    t_slice_idx = np.round(np.linspace(0, len(val_T_unique) - 1, n_slice)).astype(int)
    T_vals_for_xalpha = [float(val_T_unique[i]) for i in t_slice_idx]

    # --- 10 alpha values for x-T colormaps (linspaced by index) ---
    a_slice_idx = np.round(np.linspace(0, len(val_alpha_unique) - 1, n_slice)).astype(int)
    alpha_vals_for_xT = [float(val_alpha_unique[i]) for i in a_slice_idx]

    return PlotData(
        alpha_lo=alpha_lo, alpha_hi=alpha_hi,
        T_lo=T_lo, T_hi=T_hi,
        val_inputs=val_inputs,
        val_targets=val_targets,
        at_inputs=at_inputs,
        at_targets=at_targets,
        T_vals_for_xalpha=T_vals_for_xalpha,
        alpha_vals_for_xT=alpha_vals_for_xT,
    )


def _plot_alpha_T_colormap(alpha_np, T_np, values_np, metric_name, metric_label, out_dir):
    """Scatter colormap over (alpha, T) plane at x≈1."""
    fig, ax = plt.subplots(figsize=(7, 5))

    vmin = max(float(values_np.min()), 1e-30)
    vmax = float(values_np.max())
    if vmin >= vmax:
        vmax = vmin * 10

    sc = ax.scatter(
        alpha_np, T_np,
        c=values_np,
        cmap="viridis",
        norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
        s=30, marker="s", edgecolors="none",
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(metric_label)

    ax.set_yscale("log")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$T$ (keV)")
    ax.set_title(rf"$x \approx 1$ — {metric_name}")

    fig.tight_layout()
    fname = f"alpha_T_{metric_name}.png"
    save_path = out_dir / fname
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def _plot_x_alpha_colormap(x_np, alpha_np, values_np, T_val, metric_name, metric_label, out_dir):
    """Scatter colormap over (x, alpha) plane for a fixed T value."""
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

    ax.set_xscale("log")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\alpha$")
    ax.set_title(rf"$T = {T_val:.4g}$ keV — {metric_name}")

    fig.tight_layout()
    fname = f"T{T_val:.6g}keV_{metric_name}_logx.png"
    save_path = out_dir / fname
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def generate_epoch_plots(model, plot_data: PlotData, ep: int, params, batch_size: int = 4096):
    """
    Generate all diagnostic colormaps for a given epoch.
    model should be in eval() mode before calling; caller restores train() after.
    """
    run_name = getattr(params, 'run_name', 'unknown_run')
    epoch_dir = PROJECT_ROOT / "plots_during_training" / run_name / f"epoch_{ep}"

    bx_dir       = epoch_dir / "boundary_x_plots"
    xT_pde_dir   = epoch_dir / "x_T_colormaps" / "PDE"
    xT_tgt_dir   = epoch_dir / "x_T_colormaps" / "target"
    xa_pde_dir   = epoch_dir / "x_alpha_colormaps" / "PDE"
    xa_tgt_dir   = epoch_dir / "x_alpha_colormaps" / "target"

    for d in (bx_dir, xT_pde_dir, xT_tgt_dir, xa_pde_dir, xa_tgt_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n[training_plots] Generating plots for epoch {ep} -> {epoch_dir}")

    # ------------------------------------------------------------------ #
    # 1. alpha-T colormap (x≈1)                                           #
    # ------------------------------------------------------------------ #
    if plot_data.at_inputs.shape[0] > 0:
        print("  [1/3] alpha-T boundary plots...")
        torch.set_grad_enabled(True)
        pde_at  = compute_rel_pde_residual(model, plot_data.at_inputs, batch_size)
        phi_at  = compute_rel_phi_error(model, plot_data.at_inputs, plot_data.at_targets, batch_size)
        torch.set_grad_enabled(False)

        alpha_np = plot_data.at_inputs[:, 1].numpy()
        T_np     = plot_data.at_inputs[:, 2].numpy()

        _plot_alpha_T_colormap(
            alpha_np, T_np, pde_at,
            metric_name="pde_residual",
            metric_label=r"$|\mathrm{residual}| / |\mathrm{RHS}|$",
            out_dir=bx_dir,
        )
        _plot_alpha_T_colormap(
            alpha_np, T_np, phi_at,
            metric_name="phi_error",
            metric_label=r"$|\phi_\mathrm{pred} - \phi_\mathrm{target}| / |\phi_\mathrm{target}|$",
            out_dir=bx_dir,
        )
    else:
        print("  [1/3] Skipping alpha-T plots (no grid points).")

    # ------------------------------------------------------------------ #
    # 2. x-T colormaps for 10 alpha values                                #
    # ------------------------------------------------------------------ #
    print(f"  [2/3] x-T colormaps for {len(plot_data.alpha_vals_for_xT)} alpha values...")
    val_alpha_col = plot_data.val_inputs[:, 1]
    val_T_col     = plot_data.val_inputs[:, 2]

    for alpha_val in plot_data.alpha_vals_for_xT:
        mask = torch.abs(val_alpha_col - alpha_val) < 1e-9
        if mask.sum() == 0:
            continue
        inp  = plot_data.val_inputs[mask]
        tgt  = plot_data.val_targets[mask]

        torch.set_grad_enabled(True)
        pde_vals = compute_rel_pde_residual(model, inp, batch_size)
        torch.set_grad_enabled(False)
        phi_vals = compute_rel_phi_error(model, inp, tgt, batch_size)

        x_np = inp[:, 0].numpy()
        T_np = inp[:, 2].numpy()

        plot_colormap(
            x_np, T_np, pde_vals, alpha_val,
            metric_name="pde_residual",
            metric_label=r"$|\mathrm{residual}| / |\mathrm{RHS}|$",
            out_dir=xT_pde_dir,
            log_x=True,
        )
        plot_colormap(
            x_np, T_np, phi_vals, alpha_val,
            metric_name="phi_error",
            metric_label=r"$|\phi_\mathrm{pred} - \phi_\mathrm{target}| / |\phi_\mathrm{target}|$",
            out_dir=xT_tgt_dir,
            log_x=True,
        )

    # ------------------------------------------------------------------ #
    # 3. x-alpha colormaps for 10 T values                                #
    # ------------------------------------------------------------------ #
    print(f"  [3/3] x-alpha colormaps for {len(plot_data.T_vals_for_xalpha)} T values...")

    for T_val in plot_data.T_vals_for_xalpha:
        mask = torch.abs(val_T_col - T_val) < 1e-9
        if mask.sum() == 0:
            continue
        inp  = plot_data.val_inputs[mask]
        tgt  = plot_data.val_targets[mask]

        torch.set_grad_enabled(True)
        pde_vals = compute_rel_pde_residual(model, inp, batch_size)
        torch.set_grad_enabled(False)
        phi_vals = compute_rel_phi_error(model, inp, tgt, batch_size)

        x_np     = inp[:, 0].numpy()
        alpha_np = inp[:, 1].numpy()

        _plot_x_alpha_colormap(
            x_np, alpha_np, pde_vals, T_val,
            metric_name="pde_residual",
            metric_label=r"$|\mathrm{residual}| / |\mathrm{RHS}|$",
            out_dir=xa_pde_dir,
        )
        _plot_x_alpha_colormap(
            x_np, alpha_np, phi_vals, T_val,
            metric_name="phi_error",
            metric_label=r"$|\phi_\mathrm{pred} - \phi_\mathrm{target}| / |\phi_\mathrm{target}|$",
            out_dir=xa_tgt_dir,
        )

    print(f"[training_plots] Done for epoch {ep}.")
