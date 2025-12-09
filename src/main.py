"""
Will run training and inference from here.
"""
import random
from . import models
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

    params = YParams(PROJECT_ROOT / "src/yamls" / f"{yaml}.yaml", "base_config", print_params=True)

    wandb.login()
    wandb.init(
        name=params.run_name,
        group=params.wandb_group,
        project=params.wandb_project,
        entity=params.wandb_entity
    )

    # Log all hyperparameters from config
    # Convert params to dict to ensure all parameters are captured
    config_dict = {}
    for key in dir(params):
        # Skip private attributes and methods
        if not key.startswith('_') and not callable(getattr(params, key, None)):
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

    # Compute global normalization stats on log1p(alpha) across the loaded dataset,
    # before batching, and store on params for the model to use.
    try:
        if int(params.phase) == 2 and str(params.mode).strip() == "hard":
            X_all = None
            if isinstance(data, dict) and all(k in data for k in ("train", "val", "test")):
                X_all = torch.cat([data["train"], data["val"], data["test"]], dim=0)
            else:
                # Fallback if loader structure changes
                X_all = data["train"]
            log_alpha = torch.log1p(X_all[:, 1:2])
            a_mean = float(log_alpha.mean().item())
            a_std = float(log_alpha.std(unbiased=False).item())
            params.norm_mode = "standardize"
            params.standard_mean = a_mean
            params.standard_std = a_std
            print(f"[norm] Global standardize log1p(alpha): mean={a_mean:.6g}, std={a_std:.6g}")
    except Exception as e:
        print(f"[norm] Warning: failed to compute global mean/std for log1p(alpha): {e}")

    # Create data loaders

    data_loaders = get_data_loaders(data, batch_size=params.batch_size,
                                    shuffle_train=params.shuffle_train)

    train_loader = data_loaders["train"]
    val_loader = data_loaders["val"]
    test_loader = data_loaders["test"]

    # Dynamic model selection based on mode and phase
    if not hasattr(params, 'mode'):
        raise ValueError("Parameter 'mode' is missing from config.")
    if not hasattr(params, 'phase'):
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
        # plot predictions
        pkl_file = PROJECT_ROOT / params.pred_dir / f"{params.n_vars}D" / f"{params.run_name}.pkl"
        if params.plot_auto:
            ev.plot_pred_only(filepath=pkl_file, params=params)

    elif params.stage == "test":
        # test the model
        pass
    else:
        raise ValueError(f"Invalid stage: {params.stage}")

    wandb.finish()