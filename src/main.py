"""
Will run training and inference from here.
"""
import random
from . import models
from .color_map import (
    find_latest_state_path,
    build_grid,
    compute_pointwise_residual_sq,
    plot_and_save_colormap,
    simple_plot_latest_epoch_only,
)
from .trainer import trainer
from .data_utils.loader import load_data, get_data_loaders
from . import eval_predictions as ev
import torch
import numpy as np
import argparse
import wandb
from .YParams import YParams
from .utils import PROJECT_ROOT
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    yaml = args.config

    params = YParams(
        PROJECT_ROOT / "src/yamls" / f"{yaml}.yaml",
        "base_config",
        print_params=True,
    )

    wandb.login()
    wandb.init(
        name=params.run_name,
        group=params.wandb_group,
        project=params.wandb_project,
        entity=params.wandb_entity,
    )

    # Log all hyperparameters from config
    # Convert params to dict to ensure all parameters are captured
    config_dict = {}
    for key in dir(params):
        # Skip private attributes and methods
        if not key.startswith("_") and not callable(getattr(params, key, None)):
            try:
                value = getattr(params, key)
                # Only include serializable values (skip complex objects)
                if isinstance(value, (int, float, str, bool, list, dict, type(None))):
                    config_dict[key] = value
            except (AttributeError, TypeError):
                pass

    wandb.config.update(config_dict)

    print("Logged hyperparameters to wandb config.")

    torch.manual_seed(params.random_seed)
    random.seed(params.random_seed)
    np.random.seed(params.random_seed)

    data = load_data(params)
    

    # Compute normalization stats on log1p(alpha) using TRAIN ONLY (avoid val/test leakage),
    # then log diagnostics for val/test distributions. These stats are stored on `params`
    # for the model to use during forward passes.
    try:
        if int(params.phase) != 1 and str(params.mode).strip() == "hard":
            # Fit mean/std on train only
            X_train = data["train"]
            X_val = data["val"]
            X_test = data["test"]

            train_log_alpha = torch.log1p(X_train[:, 1:2])
            a_mean = float(train_log_alpha.mean().item())
            a_std = float(train_log_alpha.std(unbiased=False).item())

            params.norm_mode = "standardize"
            params.standard_mean = a_mean
            params.standard_std = a_std
            print(
                f"[norm] Train standardize log1p(alpha): mean={a_mean:.6g}, std={a_std:.6g}"
            )

            # Log diagnostics for val/test (raw and standardized with train stats)
            diag = {
                "norm/train_log1p_alpha_mean": float(train_log_alpha.mean().item()),
                "norm/train_log1p_alpha_std": float(train_log_alpha.std(unbiased=False).item()),
                "norm/fit_log1p_alpha_mean": a_mean,
                "norm/fit_log1p_alpha_std": a_std,
            }

            if X_val is not None:
                val_log_alpha = torch.log1p(X_val[:, 1:2])
                val_stdzd = (val_log_alpha - a_mean) / (a_std + 1e-12)
                diag.update({
                    "norm/val_log1p_alpha_mean": float(val_log_alpha.mean().item()),
                    "norm/val_log1p_alpha_std": float(val_log_alpha.std(unbiased=False).item()),
                    "norm/val_log1p_alpha_stdzd_mean": float(val_stdzd.mean().item()),
                    "norm/val_log1p_alpha_stdzd_std": float(val_stdzd.std(unbiased=False).item()),
                })
                print(
                    f"[norm] Val log1p(alpha): mean={diag['norm/val_log1p_alpha_mean']:.6g}, "
                    f"std={diag['norm/val_log1p_alpha_std']:.6g} | "
                    f"stdzd(mean,std)=({diag['norm/val_log1p_alpha_stdzd_mean']:.6g}, "
                    f"{diag['norm/val_log1p_alpha_stdzd_std']:.6g})"
                )

            if X_test is not None:
                test_log_alpha = torch.log1p(X_test[:, 1:2])
                test_stdzd = (test_log_alpha - a_mean) / (a_std + 1e-12)
                diag.update({
                    "norm/test_log1p_alpha_mean": float(test_log_alpha.mean().item()),
                    "norm/test_log1p_alpha_std": float(test_log_alpha.std(unbiased=False).item()),
                    "norm/test_log1p_alpha_stdzd_mean": float(test_stdzd.mean().item()),
                    "norm/test_log1p_alpha_stdzd_std": float(test_stdzd.std(unbiased=False).item()),
                })
                print(
                    f"[norm] Test log1p(alpha): mean={diag['norm/test_log1p_alpha_mean']:.6g}, "
                    f"std={diag['norm/test_log1p_alpha_std']:.6g} | "
                    f"stdzd(mean,std)=({diag['norm/test_log1p_alpha_stdzd_mean']:.6g}, "
                    f"{diag['norm/test_log1p_alpha_stdzd_std']:.6g})"
                )

            # Ensure these derived params/diagnostics get into wandb (initial config update happened earlier)
            try:
                wandb.config.update(
                    {
                        "norm_mode": params.norm_mode,
                        "standard_mean": params.standard_mean,
                        "standard_std": params.standard_std,
                        **diag,
                    },
                    allow_val_change=True,
                )
            except Exception:
                pass
    except Exception as e:
        print(
            f"[norm] Warning: failed to compute global mean/std for log1p(alpha): {e}"
        )

    # Optional: if we are in test stage, pre-load the latest state dict so we can
    # construct the model with normalization settings aligned to the checkpoint.
    latest_state_path = None
    test_state_dict = None
    if getattr(params, "stage", None) == "test":
        try:
            latest_state_path = find_latest_state_path(params.run_name)
            print(f"[test] Using checkpoint: {latest_state_path}")

            # Load state dict on CPU (training currently runs on CPU only)
            test_state_dict = torch.load(latest_state_path, map_location=torch.device("cpu"))

            if not isinstance(test_state_dict, dict):
                raise ValueError(
                    "Loaded checkpoint is not a state dict. "
                    "Expected dict of parameter/buffer tensors (e.g., from model.state_dict())."
                )

            # Mirror the logic from infer_model.load_model_from_config_and_state
            # to align normalization mode with the checkpoint buffers.
            sd_keys = set(test_state_dict.keys())
            if "a_mean" in sd_keys and "a_std" in sd_keys:
                params.norm_mode = "standardize"
                try:
                    a_mean_val = test_state_dict["a_mean"]
                    a_std_val = test_state_dict["a_std"]
                    # Handle both tensors and plain floats
                    a_mean_float = float(
                        a_mean_val.item() if hasattr(a_mean_val, "item") else a_mean_val
                    )
                    a_std_float = float(
                        a_std_val.item() if hasattr(a_std_val, "item") else a_std_val
                    )
                    params.standard_mean = a_mean_float
                    params.standard_std = a_std_float
                except Exception:
                    # If anything goes wrong, just keep norm_mode but skip copying values
                    print(f"[test] warning: couldn't load mean/std from train")
                    pass
            elif "a_min" in sd_keys and "a_max" in sd_keys:
                params.norm_mode = "minmax"
        except Exception as e:
            print(f"[test] Warning: could not prepare test checkpoint: {e}")
            latest_state_path = None
            test_state_dict = None

    # Create data loaders.
    # Note: at this point X_train/X_val/X_test are the exact split tensors we want to feed.
    # Using them explicitly avoids confusion (and makes it easy to insert preprocessing later).
    data_splits = {"train": X_train, "val": X_val, "test": X_test}
    data_loaders = get_data_loaders(
        data_splits, batch_size=params.batch_size, shuffle_train=params.shuffle_train
    )

    train_loader = data_loaders["train"]
    val_loader = data_loaders["val"]
    test_loader = data_loaders["test"]

    # Dynamic model selection based on mode and phase
    if not hasattr(params, "mode"):
        raise ValueError("Parameter 'mode' is missing from config.")
    if not hasattr(params, "phase"):
        raise ValueError("Parameter 'phase' is missing from config.")

    mode = str(params.mode).strip()
    phase = int(params.phase)

    model_class_name = f"Model_{mode}_phase{phase}"
    if not hasattr(models, model_class_name):
        raise ValueError(
            f"Model class '{model_class_name}' not found. Define it or adjust mode/phase "
            f"(mode={mode}, phase={phase}).")
    ModelClass = getattr(models, model_class_name)
    model = ModelClass(params)

    if params.stage == "train":
        # train the model
        trainer(model, train_loader, val_loader, params)
        
        # Unified output directory for all diagnostic plots
        output_dir = f"{params.plot_dir}/newfiles/{params.run_name}"
        
        # plot predictions from saved pickle (evolution over epochs)
        pkl_file = PROJECT_ROOT / params.pred_dir / f"{params.n_vars}D" / f"{params.run_name}.pkl"
        if params.plot_auto:
            ev.plot_pred_only(filepath=pkl_file, params=params, output_dir=output_dir)
        
        # Additional diagnostic plots using the trained model directly
        print("[main] Generating post-training diagnostic plots...")
        device = torch.device("cpu")  # Training runs on CPU
        
        # Enable gradients for computing residuals (need second derivatives)
        torch.set_grad_enabled(True)
        model.eval()
        
        # Build grid: 200x200 points in (x, alpha) space
        n_grid = 200
        log_x_plots = "logx" in params.data_path or "LOG500x" in params.data_path or "log_x" in params.data_path
        max_alpha = (100 if "RED" in params.data_path else 1000)
    
        X_grid, A_grid, inputs = build_grid(
            n=n_grid,
            device=device,
            log_x=log_x_plots,
            log_alpha=True,
            min_alpha=1,
            max_alpha=max_alpha
        )
        
        # Forward pass
        outputs = model(inputs)
        
        # 1. Simple prediction plot: psi(x) vs x
        simple_plot_latest_epoch_only(
            outputs=outputs,
            inputs=inputs,
            run_name=params.run_name,
            output_dir=output_dir,
            log_x=log_x_plots,
            log_alpha=True,
        )
        
        # 2. Colormap of residual^2 over (x, alpha)
        residual_sq = compute_pointwise_residual_sq(outputs, inputs)
        plot_and_save_colormap(
            X_grid=X_grid,
            A_grid=A_grid,
            residual_sq=residual_sq,
            run_name=params.run_name,
            output_dir=output_dir,
            n=n_grid,
            log_x=log_x_plots,
            log_alpha=True,
        )
        
        print("[main] All diagnostic plots complete.")

    elif params.stage == "test":
        # Load the latest checkpoint weights into the already-constructed model.
        if test_state_dict is None or latest_state_path is None:
            raise RuntimeError(
                "Test stage requested, but no valid checkpoint could be loaded. "
                "Ensure weights have been saved for this run (saving_weights/<run_name>/weights_epoch_*)."
            )

        missing, unexpected = model.load_state_dict(test_state_dict, strict=False)
        if missing:
            print(f"[test] Warning: missing keys in state dict: {missing}")
        if unexpected:
            print(f"[test] Warning: unexpected keys in state dict: {unexpected}")

        model.eval()
        torch.set_grad_enabled(False)
        print(
            f"[test] Model loaded with weights from {latest_state_path}. "
            "Ready for prediction or further evaluation."
        )
        # For now, we stop here: the script ends with a fully loaded model.

    else:
        raise ValueError(f"Invalid stage: {params.stage}")

    wandb.finish()