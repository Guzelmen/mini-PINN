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
    parser.add_argument("--name", type=str, required=True)
    args = parser.parse_args()

    params = YParams("config.yaml", "base_config", print_params=True)

    wandb.init(
        name=args.name,
        project="mini_pinn",
        entity="guzelmen_msci_project"
    )

    torch.manual_seed(params.random_seed)
    random.seed(params.random_seed)
    np.random.seed(params.random_seed)

    data = load_data(params)
    train = data["train"]
    val = data["val"]
    test = data["test"]

    if params.mode == "soft":
        model = Model1()
    elif params.mode == "hard":
        model = Model2()
    else:
        raise ValueError(f"Invalid mode: {params.mode}")

    if params.stage == "train":
        # train the model
        pass
    elif params.stage == "test":
        # test the model
        pass
    else:
        raise ValueError(f"Invalid stage: {params.stage}")

    wandb.finish()
