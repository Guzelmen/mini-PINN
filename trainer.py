"""
This file contains the trainer for the small PINN.
"""
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import math

import time
import losses
from tqdm import tqdm
import wandb
import pickle


def get_constant_scheduler(optimizer):
    """
    Create a constant learning rate scheduler (no scheduling).

    Args:
        optimizer: PyTorch optimizer

    Returns:
        LambdaLR scheduler (always returns 1.0 multiplier)
    """
    def lr_lambda(_epoch):
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


def get_warmup_linear_scheduler(optimizer, warmup_epochs, start_lr, end_lr):
    """
    Create a learning rate scheduler with linear warmup followed by constant LR.

    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of epochs to warm up
        start_lr: Starting learning rate for warmup
        end_lr: Target learning rate (constant after warmup)

    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup: from start_lr/end_lr to 1.0
            warmup_ratio = epoch / warmup_epochs
            return (start_lr / end_lr) * (1 - warmup_ratio) + 1.0 * warmup_ratio
        else:
            # Constant after warmup
            return 1.0

    return LambdaLR(optimizer, lr_lambda)


def get_warmup_cosine_scheduler(optimizer, warmup_epochs, start_lr,
                                end_lr, min_lr, total_epochs):
    """
    Create a learning rate scheduler with linear warmup followed by cosine decay.

    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of epochs to warm up
        start_lr: Starting learning rate for warmup
        end_lr: Peak learning rate (after warmup, start of cosine decay)
        min_lr: Minimum learning rate for cosine decay
        total_epochs: Total number of training epochs

    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(epoch):
        # Handle edge cases
        if warmup_epochs <= 0:
            # Pure cosine from epoch 0 to total_epochs
            cosine_period = max(1, total_epochs)
            progress = min(max(epoch, 0), cosine_period) / cosine_period
            return (min_lr / end_lr) + (1.0 - min_lr / end_lr) * (1 + math.cos(math.pi * progress)) / 2

        if epoch < warmup_epochs:
            # Linear warmup: from start_lr/end_lr to 1.0
            warmup_ratio = epoch / float(warmup_epochs)
            return (start_lr / end_lr) * (1 - warmup_ratio) + 1.0 * warmup_ratio
        else:
            # Cosine decay: from end_lr down to min_lr
            # Progress from 0 to 1 over (total_epochs - warmup_epochs) epochs
            cosine_epoch = epoch - warmup_epochs
            cosine_period = max(1, total_epochs - warmup_epochs)
            progress = min(max(cosine_epoch, 0), cosine_period) / \
                float(cosine_period)
            # Cosine annealing: goes from 1.0 to (min_lr/end_lr)
            cosine_multiplier = (min_lr / end_lr) + (1.0 - min_lr /
                                                     end_lr) * (1 + math.cos(math.pi * progress)) / 2
            return cosine_multiplier

    return LambdaLR(optimizer, lr_lambda)


def trainer(
    model,
    train,
    val,
    params,
):
    epochs = params.epochs

    # Initialize LossWeighter
    weighter_name = f"LossWeighter_phase{params.phase}"
    Weighterclass = getattr(losses, weighter_name)
    weighter = Weighterclass(params)

    # Learning rate parameters
    # Default to cosine if not specified
    lr_type = params.get('lr_type', 'cos')
    warmup_epochs = params.lr_warmup_epochs
    start_lr = params.start_lr
    end_lr = params.end_lr
    min_lr = params.min_lr

    # Initialize optimizer and scheduler based on lr_type
    if lr_type == "constant":
        constant_lr = params.constant_lr
        optimizer = Adam(model.parameters(), lr=constant_lr)
        scheduler = get_constant_scheduler(optimizer)
        print(f"LR schedule: constant at {constant_lr:.2e}")

    elif lr_type == "linear":
        # Linear warmup then constant
        optimizer = Adam(model.parameters(), lr=end_lr)
        scheduler = get_warmup_linear_scheduler(
            optimizer, warmup_epochs, start_lr, end_lr)
        print(
            f"LR schedule: {warmup_epochs} epochs warmup ({start_lr:.2e} -> {end_lr:.2e}), "
            f"then constant at {end_lr:.2e}")

    elif lr_type == "cos" or lr_type == "cosine":
        # Linear warmup + cosine decay
        optimizer = Adam(model.parameters(), lr=end_lr)
        scheduler = get_warmup_cosine_scheduler(
            optimizer, warmup_epochs, start_lr, end_lr, min_lr, epochs)
        print(
            f"LR schedule: {warmup_epochs} epochs warmup ({start_lr:.2e} -> {end_lr:.2e}), "
            f"then cosine decay to {min_lr:.2e} over {epochs - warmup_epochs} epochs")
    else:
        raise ValueError(
            f"Unknown lr_type: {lr_type}. Choose from: constant, linear, cos/cosine")

    predictions = {}
    # Ensure first epoch uses start_lr for warmup schedules without stepping scheduler before optimizer
    if lr_type in ["linear", "cos", "cosine"] and warmup_epochs > 0:
        optimizer.param_groups[0]["lr"] = start_lr
    for ep in range(1, epochs+1):
        time_start = time.time()
        model.train()

        epoch_x = []
        epoch_alpha = []
        epoch_outputs = []

        # Training loop
        epoch_losses = {'residual': [], 'bc_1': [], 'bc_2': [], 'total': []}

        # Collect raw losses for adaptive weighting (update once per epoch)
        raw_losses_for_weighting = {'residual': [], 'bc_1': [], 'bc_2': []}
        # Collect gradient norms per batch for diagnostics
        epoch_grad_norms = []

        train_loader_iter = tqdm(
            train, desc=f"Epoch {ep}/{epochs}", leave=True, ncols=100)

        for batch in train_loader_iter:
            # Extract x coordinates (first column), r0 from batch
            # Data shape: [batch_size, 2] where columns are [x, r0]
            inputs = batch[0]  # Unpack tuple from DataLoader
            # Ensure inputs require gradients for autograd wrt x
            inputs.requires_grad_(True)
            x = inputs[:, 0:1].clone()
            alpha = inputs[:, 1:2].clone()

            # Forward pass
            outputs = model(inputs)

            if ep % params.save_every == 0:
                # Detach tensors before saving to avoid gradient tracking issues
                epoch_x.append(x.detach())
                epoch_alpha.append(alpha.detach())
                epoch_outputs.append(outputs.detach())

            # Compute individual losses
            residual_name = f"compute_residual_loss_phase{params.phase}"
            residual_fn = getattr(losses, residual_name)
            residual_loss = residual_fn(outputs, inputs, params=params)

            if params.mode == "soft":
                bc1name = f"compute_bc_loss_1_phase{params.phase}"
                bc2name = f"compute_bc_loss_2_phase{params.phase}"
                bc1_fn = getattr(losses, bc1name)
                bc2_fn = getattr(losses, bc2name)
                bc_1_loss = bc1_fn(outputs, inputs)
                bc_2_loss = bc2_fn(outputs, inputs)
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

            totlossname = f"compute_total_loss_phase{params.phase}"
            totloss_fn = getattr(losses, totlossname)
            total_loss = totloss_fn(loss_dict, weighter)

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()

            # Record gradient norm before stepping
            total_grad_sq = 0.0
            num_params_with_grad = 0
            for p in model.parameters():
                if p.grad is not None:
                    g = p.grad.detach()
                    total_grad_sq += float(g.pow(2).sum().item())
                    num_params_with_grad += 1
            if num_params_with_grad > 0:
                epoch_grad_norms.append(total_grad_sq ** 0.5)

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

        if ep % params.save_every == 0:
            # Concatenate tensors first, then convert to numpy
            x_all = torch.cat(epoch_x, dim=0).cpu().numpy()
            alpha_all = torch.cat(epoch_alpha, dim=0).cpu().numpy()
            outputs_all = torch.cat(epoch_outputs, dim=0).cpu().numpy()

            predictions[ep] = {"x": x_all,
                               "r0": alpha_all, "outputs": outputs_all}

            print(
                f"Saved epoch {ep}: {x_all.shape=} {alpha_all.shape=} {outputs_all.shape=}")

        # Training epoch complete - compute averages before printing
        avg_residual = np.mean(epoch_losses['residual'])
        avg_bc1 = np.mean(epoch_losses['bc_1'])
        avg_bc2 = np.mean(epoch_losses['bc_2'])
        avg_total = np.mean(epoch_losses['total'])

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
        scheduler.step()
        # Update for next epoch's logging
        current_lr = optimizer.param_groups[0]['lr']

        time_end = time.time()
        time_taken = time_end - time_start

        # Print epoch summary
        print(
            f"Epoch {ep}/{epochs}: train loss = {avg_total:.6e}, time = {time_taken:.2f} s")

        # In hard mode, log hard-constraint checks using explicit probes at x=0 and x=1
        checks = {}
        if params.mode == "hard":
            alpha_check = torch.tensor([[1e3]])
            x0 = torch.zeros_like(alpha_check)
            x1 = torch.ones_like(alpha_check)

            inp0 = torch.cat([x0, alpha_check], dim=-1).requires_grad_(True)
            inp1 = torch.cat([x1, alpha_check], dim=-1).requires_grad_(True)
            psi0 = model(inp0)
            psi1 = model(inp1)
            # Compute dpsi/dx by differentiating w.r.t. the same tensor used in the forward pass
            dpsi1_dx_full = torch.autograd.grad(
                outputs=psi1,
                inputs=inp1,
                grad_outputs=torch.ones_like(psi1),
                create_graph=False,
                retain_graph=False,
                allow_unused=False
            )[0]
            dpsi1_dx = dpsi1_dx_full[:, 0:1]

            psi0_err = torch.abs(psi0 - 1.0).mean().item()
            psi1_minus_psiprime1 = torch.abs(psi1 - dpsi1_dx).mean().item()
            checks = {
                "checks/psi0_abs_error": psi0_err,
                "checks/psi1_minus_psiprime1_abs": psi1_minus_psiprime1,
            }

        # Gradient norm stats
        grad_stats = {}
        if len(epoch_grad_norms) > 0:
            grad_stats = {
                "grads/epoch_grad_norm_mean": float(np.mean(epoch_grad_norms)),
                "grads/epoch_grad_norm_max": float(np.max(epoch_grad_norms)),
            }

        # Log to wandb
        wandb.log({
            "info/learning_rate": current_lr,
            "info/epoch": ep,
            "info/epoch_training_time": time_taken,
            "train/residual_loss": avg_residual,
            "train/bc_1_loss": avg_bc1,
            "train/bc_2_loss": avg_bc2,
            "train/total_loss": avg_total,
            "weights/residual_weight": weighter.residual_weight,
            "weights/bc_1_weight": weighter.bc_1_weight,
            "weights/bc_2_weight": weighter.bc_2_weight,
            **checks,
            **grad_stats,
        })

        # Validation
        if val is not None:
            time_start_val = time.time()
            model.eval()
            val_losses = {'residual': [], 'bc_1': [], 'bc_2': [], 'total': []}

            for batch in val:
                inputs = batch[0]  # Unpack tuple from DataLoader
                # Ensure inputs require gradients for derivative-based losses
                inputs.requires_grad_(True)

                # Forward pass
                outputs = model(inputs)

                # Compute individual losses
                residual_name = f"compute_residual_loss_phase{params.phase}"
                residual_fn = getattr(losses, residual_name)
                residual_loss = residual_fn(outputs, inputs, params=params)

                if params.mode == "soft":
                    bc1name = f"compute_bc_loss_1_phase{params.phase}"
                    bc2name = f"compute_bc_loss_2_phase{params.phase}"
                    bc1_fn = getattr(losses, bc1name)
                    bc2_fn = getattr(losses, bc2name)
                    bc_1_loss = bc1_fn(outputs, inputs)
                    bc_2_loss = bc2_fn(outputs, inputs)
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

            # Print validation summary
            print(f"  val loss = {avg_val_total:.6e}")

            wandb.log({
                "val/residual_loss": avg_val_residual,
                "val/bc_1_loss": avg_val_bc1,
                "val/bc_2_loss": avg_val_bc2,
                "val/total_loss": avg_val_total,
                "info/epoch_validation_time": time_taken_val
            })

    save_path = f"{params.pred_dir}/{params.n_vars}D/{params.run_name}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(predictions, f)

    print(f"Saved predictions for all selected epochs to {save_path}")

    return
