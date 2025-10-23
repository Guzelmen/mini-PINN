"""
This file contains the trainer for the small PINN.
"""
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from data_utils.loader import *
import time
from tqdm import tqdm
import wandb


def get_warmup_lr_scheduler(optimizer, warmup_epochs, start_lr, end_lr):
    """
    Create a learning rate scheduler with linear warmup.

    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of epochs to warm up
        start_lr: Starting learning rate for warmup
        end_lr: Target learning rate after warmup
        total_epochs: Total number of training epochs

    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return start_lr + (end_lr - start_lr) * epoch / warmup_epochs
        else:
            # Constant learning rate after warmup
            return end_lr

    return LambdaLR(optimizer, lr_lambda)


def trainer(
    model,
    train,
    val,
    params,
):
    epochs = params.epochs

    # Warmup parameters
    warmup_epochs = params.get('warmup_epochs', 0)
    warmup_start_lr = params.get('warmup_start_lr', 0.0)
    warmup_end_lr = params.get('warmup_end_lr', 0)

    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=warmup_end_lr)

    # Create warmup scheduler if warmup is enabled
    if warmup_epochs > 0:
        scheduler = get_warmup_lr_scheduler(
            optimizer, warmup_epochs, warmup_start_lr, warmup_end_lr)
        print(
            f"Using linear warmup: {warmup_epochs} epochs, {warmup_start_lr} -> {warmup_end_lr}")
    else:
        scheduler = None
        print(f"Using constant learning rate: {warmup_end_lr}")

    for ep in range(epochs):
        time_start = time.time()
        model.train()
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = warmup_end_lr

        # Log learning rate to wandb
        wandb.log({"learning_rate": current_lr, "epoch": ep})

        train_loader_iter = tqdm(
            train, desc=f"Epoch {ep+1}/{epochs}", leave=False, ncols=100)

        for batch_idx, data in enumerate(train_loader_iter):
            pass

    return
