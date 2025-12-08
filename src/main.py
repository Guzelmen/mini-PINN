"""
Will run training and inference from here.
"""
import random
import models
from trainer import trainer
from data_utils.loader import load_data, get_data_loaders
import eval_predictions as ev
import torch
import numpy as np
import argparse
import wandb
from YParams import YParams
from utils import PROJECT_ROOT
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
        ev.plot_pred_only(filepath=pkl_file, params=params)

    elif params.stage == "test":
        # test the model
        pass
    else:
        raise ValueError(f"Invalid stage: {params.stage}")

    wandb.finish()
