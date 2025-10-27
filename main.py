"""
Will run training and inference from here.
"""
import random
from models import *
from trainer import trainer
from data_utils.loader import *
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

    wandb.init(
        name=str(args.name),
        project="mini_pinn",
        entity="guzelmen_msci_project"
    )

    # Log hyperparameters
    wandb.config.update({
        "random_seed": params.random_seed,
        "mode": params.mode,
        "nlayers": params.nlayers,
        "hidden": params.hidden,
        "w0": params.w0,
        "start_lr": params.start_lr,
        "end_lr": params.end_lr,
        "lr_warmup_epochs": params.lr_warmup_epochs,
        "loss_strategy": params.loss_strategy,
        "relative_weights": params.relative_weights,
        "batch_size": params.batch_size,
        "stage": params.stage,
        "epochs": params.epochs,
    })

    print("Logged hyperparameters to wandb config.")

    torch.manual_seed(params.random_seed)
    random.seed(params.random_seed)
    np.random.seed(params.random_seed)

    data = load_data(params)

    # Create data loaders
    from data_utils.loader import get_data_loaders
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
