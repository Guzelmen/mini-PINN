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
    # Compute boundary edges from absolute min/max config params
    # (null → use actual coarse data min/max)
    # ------------------------------------------------------------------
    alpha_lo = getattr(params, 'alpha_min_train', None)
    alpha_hi = getattr(params, 'alpha_max_train', None)
    T_lo     = getattr(params, 'T_min_train',     None)
    T_hi     = getattr(params, 'T_max_train',     None)

    coarse_alpha = coarse_inputs[:, 1]
    coarse_T     = coarse_inputs[:, 2]

    if alpha_lo is None: alpha_lo = float(coarse_alpha.min())
    if alpha_hi is None: alpha_hi = float(coarse_alpha.max())
    if T_lo     is None: T_lo     = float(coarse_T.min())
    if T_hi     is None: T_hi     = float(coarse_T.max())

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
    # Normalisation stats from FULL fine dataset
    # (constant regardless of curriculum cutoffs applied above)
    # Note: log(T) not log1p(T) — T spans 4 orders of magnitude (0.0001-10)
    # ------------------------------------------------------------------
    fine_log_alpha = torch.log1p(fine_inputs[:, 1])
    fine_log_T     = torch.log(fine_inputs[:, 2] + 1e-12)

    use_log_lam_x = getattr(params, 'use_log_lam_x', False)
    if use_log_lam_x:
        from ..fd_integrals import B_M, C0_M
        fine_lam   = fine_inputs[:, 1:2] * B_M * (fine_inputs[:, 2:3] ** 0.25) / C0_M
        fine_log_x = torch.log(fine_lam * fine_inputs[:, 0:1] + 1e-12).squeeze(1)
    else:
        fine_log_x = torch.log(fine_inputs[:, 0] + 1e-8)

    norm_stats = {
        'a_mean':  float(fine_log_alpha.mean()),
        'a_std':   float(fine_log_alpha.std(unbiased=False)),
        'T_mean':  float(fine_log_T.mean()),
        'T_std':   float(fine_log_T.std(unbiased=False)),
    }
    if use_log_lam_x:
        norm_stats['x_log_lam_mean'] = float(fine_log_x.mean())
        norm_stats['x_log_lam_std']  = float(fine_log_x.std(unbiased=False))
    else:
        norm_stats['x_log_mean'] = float(fine_log_x.mean())
        norm_stats['x_log_std']  = float(fine_log_x.std(unbiased=False))

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
    if use_log_lam_x:
        print(f"[extended_loader] Norm stats (from full fine dataset, log T): "
              f"a_mean={norm_stats['a_mean']:.4f}, a_std={norm_stats['a_std']:.4f}, "
              f"T_mean={norm_stats['T_mean']:.4f}, T_std={norm_stats['T_std']:.4f}, "
              f"x_log_lam_mean={norm_stats['x_log_lam_mean']:.4f}, x_log_lam_std={norm_stats['x_log_lam_std']:.4f}")
    else:
        print(f"[extended_loader] Norm stats (from full fine dataset, log T): "
              f"a_mean={norm_stats['a_mean']:.4f}, a_std={norm_stats['a_std']:.4f}, "
              f"T_mean={norm_stats['T_mean']:.4f}, T_std={norm_stats['T_std']:.4f}, "
              f"x_log_mean={norm_stats['x_log_mean']:.4f}, x_log_std={norm_stats['x_log_std']:.4f}")

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
