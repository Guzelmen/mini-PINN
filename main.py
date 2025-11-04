"""
Will run training and inference from here.
"""
import random
from models import Model1, Model2
from trainer import trainer
from data_utils.loader import load_data, get_data_loaders
import torch
import numpy as np
import argparse
import wandb
from YParams import YParams
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    params = YParams("config.yaml", "base_config", print_params=True)

    params.run_name = args.name

    wandb.init(
        name=str(args.name),
        project="mini_pinn",
        entity="guzelmen_msci_project"
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

    data_loaders = get_data_loaders(data, batch_size=params.batch_size)

    train_loader = data_loaders["train"]
    val_loader = data_loaders["val"]
    test_loader = data_loaders["test"]

    if params.mode == "soft":
        model = Model1(params)
    elif params.mode == "hard":
        model = Model2(params)
    else:
        raise ValueError(f"Invalid mode: {params.mode}")

    if params.stage == "train":
        # train the model
        trainer(model, train_loader, val_loader, params)
    elif params.stage == "test":
        # test the model
        pass
    else:
        raise ValueError(f"Invalid stage: {params.stage}")

    wandb.finish()
