"""
This file contains the trainer for the small PINN.
"""
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from data_utils.loader import *
import time
from losses import *
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

    # Initialize LossWeighter
    weighter = LossWeighter(params)

    # Warmup parameters
    warmup_epochs = params.lr_warmup_epochs
    warmup_start_lr = params.start_lr
    warmup_end_lr = params.end_lr

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

        # Get current learning rate (updated at end of prev epoch)
        if scheduler is not None:
            current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = warmup_end_lr

        # Training loop
        epoch_losses = {'residual': [], 'bc_1': [], 'bc_2': [], 'total': []}

        # Collect raw losses for adaptive weighting (update once per epoch)
        raw_losses_for_weighting = {'residual': [], 'bc_1': [], 'bc_2': []}

        train_loader_iter = tqdm(
            train, desc=f"Epoch {ep+1}/{epochs}", leave=False, ncols=100)

        for batch_idx, batch in enumerate(train_loader_iter):
            # Extract x coordinates (first column), r0, Z from batch
            # Data shape: [batch_size, 3] where columns are [x, r0, Z]
            batch_data = batch[0]  # Unpack tuple from DataLoader
            x = batch_data[:, 0:1].clone()  # Extract and clone x coordinates
            x.requires_grad_(True)  # Enable gradients for differentiation
            # r0 = batch_data[:, 1:2]  # Not used for PINN
            # Z = batch_data[:, 2:3]   # Not used for PINN

            # Forward pass
            outputs = model(x)

            # Compute individual losses
            residual_loss = compute_residual_loss(outputs, x)

            if params.mode == "soft":
                bc_1_loss = compute_bc_loss_1(outputs, x)
                bc_2_loss = compute_bc_loss_2(outputs, x)
            else:
                bc_1_loss = torch.tensor(0.0, device=outputs.device)
                bc_2_loss = torch.tensor(0.0, device=outputs.device)

            # Combine losses
            loss_dict = {
                'residual': residual_loss,
                'bc_1': bc_1_loss,
                'bc_2': bc_2_loss
            }

            # For adaptive weighting: collect raw losses, don't update weights yet
            if weighter.strategy == 'adaptive':
                raw_losses_for_weighting['residual'].append(
                    residual_loss.detach())
                raw_losses_for_weighting['bc_1'].append(bc_1_loss.detach())
                raw_losses_for_weighting['bc_2'].append(bc_2_loss.detach())

            total_loss = compute_total_loss(loss_dict, weighter)

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Track losses
            epoch_losses['residual'].append(residual_loss.item())
            epoch_losses['bc_1'].append(bc_1_loss.item())
            epoch_losses['bc_2'].append(bc_2_loss.item())
            epoch_losses['total'].append(total_loss.item())

            # Update progress bar
            train_loader_iter.set_postfix(
                loss=total_loss.item(),
                residual=residual_loss.item(),
                bc1=bc_1_loss.item(),
                bc2=bc_2_loss.item()
            )

        # Update weights once per epoch for adaptive strategy
        if weighter.strategy == 'adaptive':
            avg_losses_for_weighting = {
                'residual': torch.stack(raw_losses_for_weighting['residual']).mean(),
                'bc_1': torch.stack(raw_losses_for_weighting['bc_1']).mean(),
                'bc_2': torch.stack(raw_losses_for_weighting['bc_2']).mean()
            }
            weighter.update_weights(avg_losses_for_weighting)
            # Reset collection for next epoch
            raw_losses_for_weighting = {'residual': [], 'bc_1': [], 'bc_2': []}

        # Update scheduler AFTER all batches (PyTorch best practice)
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

        # Compute average losses for the epoch
        avg_residual = np.mean(epoch_losses['residual'])
        avg_bc1 = np.mean(epoch_losses['bc_1'])
        avg_bc2 = np.mean(epoch_losses['bc_2'])
        avg_total = np.mean(epoch_losses['total'])

        time_end = time.time()
        time_taken = time_end - time_start

        # Log to wandb
        wandb.log({
            "info/learning_rate": current_lr,
            "info/epoch": ep,
            "info/epoch_training_time": time_taken,
            "train/residual_loss": avg_residual,
            "train/bc_1_loss": avg_bc1,
            "train/bc_2_loss": avg_bc2,
            "train/total_loss": avg_total
        })

        # Validation
        if val is not None:
            time_start_val = time.time()
            model.eval()
            val_losses = {'residual': [], 'bc_1': [], 'bc_2': [], 'total': []}

            for batch_idx, batch in enumerate(val):
                if batch_idx % 100 == 0:
                    print(f"Validation batch {batch_idx} of {len(val)}")
                batch_data = batch[0]
                x = batch_data[:, 0:1]
                x.requires_grad_(True)  # Enable gradients for differentiation

                # Forward pass (needs gradients for loss computation)
                outputs = model(x)

                # Compute losses (which require gradients)
                residual_loss = compute_residual_loss(outputs, x)

                if params.mode == "soft":
                    bc_1_loss = compute_bc_loss_1(outputs, x)
                    bc_2_loss = compute_bc_loss_2(outputs, x)
                else:
                    bc_1_loss = torch.tensor(0.0, device=outputs.device)
                    bc_2_loss = torch.tensor(0.0, device=outputs.device)

                val_losses['residual'].append(residual_loss.item())
                val_losses['bc_1'].append(bc_1_loss.item())
                val_losses['bc_2'].append(bc_2_loss.item())

                # Use weighter for consistent total loss (but don't update weights)
                val_loss_dict = {
                    'residual': residual_loss,
                    'bc_1': bc_1_loss,
                    'bc_2': bc_2_loss
                }
                val_total_loss = weighter.get_weighted_loss(val_loss_dict)
                val_losses['total'].append(val_total_loss.item())

            avg_val_residual = np.mean(val_losses['residual'])
            avg_val_bc1 = np.mean(val_losses['bc_1'])
            avg_val_bc2 = np.mean(val_losses['bc_2'])
            avg_val_total = np.mean(val_losses['total'])
            time_end_val = time.time()
            time_taken_val = time_end_val - time_start_val

            wandb.log({
                "val/residual_loss": avg_val_residual,
                "val/bc_1_loss": avg_val_bc1,
                "val/bc_2_loss": avg_val_bc2,
                "val/total_loss": avg_val_total,
                "info/epoch_validation_time": time_taken_val
            })

    return
