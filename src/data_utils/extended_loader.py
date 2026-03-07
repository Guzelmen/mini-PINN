import math
import torch
from ..utils import PROJECT_ROOT


def load_extended_data(params):
    """
    Load two-file (coarse + fine) dataset and split by parameter-space regions.

    Returns a dict with keys:
        train, train_targets,
        interp_val, interp_val_targets,
        ood_single, ood_single_targets,
        ood_corner, ood_corner_targets,
        norm_stats, edges
    """
    if int(params.phase) != 4:
        raise ValueError(
            f"[extended_loader] Only phase=4 is supported, got phase={params.phase}"
        )

    # ------------------------------------------------------------------
    # Load both files
    # ------------------------------------------------------------------
    coarse_path = str(PROJECT_ROOT / params.data_path)
    fine_path   = str(PROJECT_ROOT / params.fine_data_path)

    print(f"[extended_loader] Loading coarse data from {coarse_path}...")
    coarse_raw = torch.load(coarse_path)
    print(f"[extended_loader] Loading fine data from {fine_path}...")
    fine_raw = torch.load(fine_path)

    coarse_inputs  = coarse_raw['inputs']
    coarse_targets = coarse_raw['targets']
    fine_inputs    = fine_raw['inputs']
    fine_targets   = fine_raw['targets']

    print(f"[extended_loader] Coarse: inputs={coarse_inputs.shape}, targets={coarse_targets.shape}")
    print(f"[extended_loader] Fine:   inputs={fine_inputs.shape},   targets={fine_targets.shape}")

    # ------------------------------------------------------------------
    # x filtering (mirrors loader.py)
    # ------------------------------------------------------------------
    if getattr(params, 'filter_x_min', False):
        thresh = getattr(params, 'x_min_threshold', 1e-6)

        mask_c = coarse_inputs[:, 0] >= thresh
        n_before_c = coarse_inputs.shape[0]
        coarse_inputs  = coarse_inputs[mask_c]
        coarse_targets = coarse_targets[mask_c]
        print(f"[extended_loader] coarse: filtered x < {thresh}: "
              f"{n_before_c} -> {coarse_inputs.shape[0]} ({n_before_c - coarse_inputs.shape[0]} removed)")

        mask_f = fine_inputs[:, 0] >= thresh
        n_before_f = fine_inputs.shape[0]
        fine_inputs  = fine_inputs[mask_f]
        fine_targets = fine_targets[mask_f]
        print(f"[extended_loader] fine:   filtered x < {thresh}: "
              f"{n_before_f} -> {fine_inputs.shape[0]} ({n_before_f - fine_inputs.shape[0]} removed)")

    # ------------------------------------------------------------------
    # Compute boundary edges from coarse unique (alpha, T) values
    # ------------------------------------------------------------------
    alpha_vals = torch.unique(coarse_inputs[:, 1])
    T_vals     = torch.unique(coarse_inputs[:, 2])

    def _log_edge(unique_vals, lo_pct, hi_pct):
        log_vals = torch.log(unique_vals)
        lo = float(torch.quantile(log_vals, lo_pct / 100.0))
        hi = float(torch.quantile(log_vals, 1.0 - hi_pct / 100.0))
        return math.exp(lo), math.exp(hi)

    alpha_lo_pct = getattr(params, 'alpha_lo_pct', 10)
    alpha_hi_pct = getattr(params, 'alpha_hi_pct', 5)
    T_lo_pct     = getattr(params, 'T_lo_pct', 5)
    T_hi_pct     = getattr(params, 'T_hi_pct', 5)

    alpha_lo, alpha_hi = _log_edge(alpha_vals, alpha_lo_pct, alpha_hi_pct)
    T_lo,     T_hi     = _log_edge(T_vals,     T_lo_pct,     T_hi_pct)

    # ------------------------------------------------------------------
    # Build splits
    # ------------------------------------------------------------------
    # Coarse: train = inside on both axes
    c_alpha_inside = (coarse_inputs[:, 1] >= alpha_lo) & (coarse_inputs[:, 1] <= alpha_hi)
    c_T_inside     = (coarse_inputs[:, 2] >= T_lo)     & (coarse_inputs[:, 2] <= T_hi)
    train_mask     = c_alpha_inside & c_T_inside
    train_inputs   = coarse_inputs[train_mask]
    train_targets  = coarse_targets[train_mask]

    # Fine: interp_val / ood_single / ood_corner
    f_alpha_inside = (fine_inputs[:, 1] >= alpha_lo) & (fine_inputs[:, 1] <= alpha_hi)
    f_T_inside     = (fine_inputs[:, 2] >= T_lo)     & (fine_inputs[:, 2] <= T_hi)

    interp_mask = f_alpha_inside & f_T_inside
    single_mask = (f_alpha_inside & ~f_T_inside) | (~f_alpha_inside & f_T_inside)
    corner_mask = ~f_alpha_inside & ~f_T_inside

    # interp_val: subsample with fixed seed
    interp_all_inp = fine_inputs[interp_mask]
    interp_all_tgt = fine_targets[interp_mask]
    n_before       = interp_all_inp.shape[0]
    n_samples      = min(n_before, getattr(params, 'interp_val_n_samples', 50000))
    g = torch.Generator()
    g.manual_seed(params.random_seed)
    perm               = torch.randperm(n_before, generator=g)[:n_samples]
    interp_val_inputs  = interp_all_inp[perm]
    interp_val_targets = interp_all_tgt[perm]

    ood_single_inputs  = fine_inputs[single_mask]
    ood_single_targets = fine_targets[single_mask]
    ood_corner_inputs  = fine_inputs[corner_mask]
    ood_corner_targets = fine_targets[corner_mask]

    # ------------------------------------------------------------------
    # Normalisation stats from training set only
    # ------------------------------------------------------------------
    train_log_alpha = torch.log1p(train_inputs[:, 1])
    train_log_T     = torch.log1p(train_inputs[:, 2])
    train_log_x     = torch.log(train_inputs[:, 0] + 1e-8)

    norm_stats = {
        'a_mean':     float(train_log_alpha.mean()),
        'a_std':      float(train_log_alpha.std(unbiased=False)),
        'T_mean':     float(train_log_T.mean()),
        'T_std':      float(train_log_T.std(unbiased=False)),
        'x_log_mean': float(train_log_x.mean()),
        'x_log_std':  float(train_log_x.std(unbiased=False)),
    }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"[extended_loader] Edges: alpha=[{alpha_lo:.4f}, {alpha_hi:.4f}], "
          f"T=[{T_lo:.6f}, {T_hi:.4f}] keV")
    print(f"[extended_loader] train:       {train_inputs.shape[0]} points")
    print(f"[extended_loader] interp_val:  {interp_val_inputs.shape[0]} points "
          f"(subsampled from {n_before})")
    print(f"[extended_loader] ood_single:  {ood_single_inputs.shape[0]} points")
    print(f"[extended_loader] ood_corner:  {ood_corner_inputs.shape[0]} points")
    print(f"[extended_loader] Norm stats: "
          f"a_mean={norm_stats['a_mean']:.4f}, a_std={norm_stats['a_std']:.4f}, "
          f"T_mean={norm_stats['T_mean']:.4f}, T_std={norm_stats['T_std']:.4f}")

    return {
        'train':               train_inputs,
        'train_targets':       train_targets,
        'interp_val':          interp_val_inputs,
        'interp_val_targets':  interp_val_targets,
        'ood_single':          ood_single_inputs,
        'ood_single_targets':  ood_single_targets,
        'ood_corner':          ood_corner_inputs,
        'ood_corner_targets':  ood_corner_targets,
        'norm_stats':          norm_stats,
        'edges': {
            'alpha_lo': alpha_lo,
            'alpha_hi': alpha_hi,
            'T_lo':     T_lo,
            'T_hi':     T_hi,
        },
    }
