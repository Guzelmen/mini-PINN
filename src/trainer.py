"""
This file contains the trainer for the small PINN.
"""
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import math
from .utils import sec_deriv_auto, first_deriv_auto, PROJECT_ROOT

import time
from . import losses
from tqdm import tqdm
import wandb
import pickle
import json
import resource


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
                                end_lr, min_lr, total_epochs,
                                cosine_decay_epochs=None):
    """
    Create a learning rate scheduler with linear warmup followed by cosine decay.

    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of epochs to warm up
        start_lr: Starting learning rate for warmup
        end_lr: Peak learning rate (after warmup, start of cosine decay)
        min_lr: Minimum learning rate for cosine decay
        total_epochs: Total number of training epochs
        cosine_decay_epochs: Fixed number of epochs for cosine decay phase.
            If None (default), decay spans (total_epochs - warmup_epochs)
            for backward compatibility. If set, cosine decays over exactly
            this many epochs, then holds at min_lr for the remainder.

    Returns:
        LambdaLR scheduler
    """
    # Determine cosine period: fixed if specified, else fill remaining epochs
    if cosine_decay_epochs is not None:
        _cosine_period = max(1, cosine_decay_epochs)
    else:
        _cosine_period = max(1, total_epochs - warmup_epochs)

    def lr_lambda(epoch):
        # Handle edge cases
        if warmup_epochs <= 0:
            # Pure cosine from epoch 0
            progress = min(max(epoch, 0), _cosine_period) / _cosine_period
            return (min_lr / end_lr) + (1.0 - min_lr / end_lr) * (1 + math.cos(math.pi * progress)) / 2

        if epoch < warmup_epochs:
            # Linear warmup: from start_lr/end_lr to 1.0
            warmup_ratio = epoch / float(warmup_epochs)
            return (start_lr / end_lr) * (1 - warmup_ratio) + 1.0 * warmup_ratio
        else:
            # Cosine decay: from end_lr down to min_lr
            cosine_epoch = epoch - warmup_epochs
            progress = min(max(cosine_epoch, 0), _cosine_period) / \
                float(_cosine_period)
            # Cosine annealing: goes from 1.0 to (min_lr/end_lr)
            # After cosine_period, progress is clamped to 1.0 → holds at min_lr
            cosine_multiplier = (min_lr / end_lr) + (1.0 - min_lr /
                                                     end_lr) * (1 + math.cos(math.pi * progress)) / 2
            return cosine_multiplier

    return LambdaLR(optimizer, lr_lambda)


def get_warmup_cosine_restarts_scheduler(optimizer, warmup_epochs, start_lr,
                                         end_lr, min_lr, restart_period):
    """
    Linear warmup followed by cosine annealing with warm restarts.

    After warmup, the LR follows repeated cosine cycles: each cycle decays
    from end_lr to min_lr over restart_period epochs, then snaps back to
    end_lr for the next cycle.

    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of epochs for linear warmup
        start_lr: Starting LR for warmup
        end_lr: Peak LR (base optimizer LR)
        min_lr: Minimum LR at the bottom of each cosine cycle
        restart_period: Number of epochs per cosine cycle
    """
    _period = max(1, restart_period)

    def lr_lambda(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            warmup_ratio = epoch / float(warmup_epochs)
            return (start_lr / end_lr) * (1 - warmup_ratio) + 1.0 * warmup_ratio
        else:
            # Position within the current cosine cycle
            cosine_epoch = (epoch - warmup_epochs) % _period
            progress = cosine_epoch / float(_period)
            return (min_lr / end_lr) + (1.0 - min_lr / end_lr) * (1 + math.cos(math.pi * progress)) / 2

    return LambdaLR(optimizer, lr_lambda)


def trainer(
    model,
    train,
    val,
    params,
    prelim_time=0.0,
):
    device = getattr(params, "device", torch.device("cpu"))
    epochs = params.epochs

    # Initialize LossWeighter
    if params.phase == 3:
        weighter_name = f"LossWeighter_phase2"
    elif params.phase == 4:
        weighter_name = f"LossWeighter_phase4"
    else:
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
    cosine_decay_epochs = params.get('cosine_decay_epochs', None)
    restart_period = params.get('cosine_restart_period', None)

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
        # Linear warmup + cosine decay (+ optional hold at min_lr)
        optimizer = Adam(model.parameters(), lr=end_lr)
        scheduler = get_warmup_cosine_scheduler(
            optimizer, warmup_epochs, start_lr, end_lr, min_lr, epochs,
            cosine_decay_epochs=cosine_decay_epochs)
        decay_len = cosine_decay_epochs if cosine_decay_epochs is not None else (epochs - warmup_epochs)
        schedule_msg = (
            f"LR schedule: {warmup_epochs} epochs warmup ({start_lr:.2e} -> {end_lr:.2e}), "
            f"then cosine decay to {min_lr:.2e} over {decay_len} epochs")
        if cosine_decay_epochs is not None:
            hold_epochs = max(0, epochs - warmup_epochs - cosine_decay_epochs)
            schedule_msg += f", then hold at {min_lr:.2e} for {hold_epochs} epochs"
        print(schedule_msg)

    elif lr_type == "cosine_restarts":
        # Linear warmup + repeated cosine cycles
        if restart_period is None:
            raise ValueError("lr_type='cosine_restarts' requires cosine_restart_period to be set")
        optimizer = Adam(model.parameters(), lr=end_lr)
        scheduler = get_warmup_cosine_restarts_scheduler(
            optimizer, warmup_epochs, start_lr, end_lr, min_lr, restart_period)
        n_restarts = max(0, (epochs - warmup_epochs) // restart_period - 1)
        print(
            f"LR schedule: {warmup_epochs} epochs warmup ({start_lr:.2e} -> {end_lr:.2e}), "
            f"then cosine restarts ({end_lr:.2e} -> {min_lr:.2e}) every {restart_period} epochs "
            f"(~{n_restarts} restarts over {epochs} epochs)")

    else:
        raise ValueError(
            f"Unknown lr_type: {lr_type}. Choose from: constant, linear, cos/cosine, cosine_restarts")

    predictions = {}
    # Ensure first epoch uses start_lr for warmup schedules without stepping scheduler before optimizer
    if lr_type in ["linear", "cos", "cosine", "cosine_restarts"] and warmup_epochs > 0:
        optimizer.param_groups[0]["lr"] = start_lr
    
    # Initialize m decay strategy if update_m is enabled
    update_m = getattr(params, "update_m", False)
    current_m = getattr(params, "m_loss_m_value", 0.0)  # Initialize current_m for use in loop
    m_decay_interval = None  # Will be set if update_m is True
    m_decay_step = None  # Will be set if update_m is True
    
    if update_m:
        # m_loss_m_value in config is the starting value when update_m is True
        m_decay_interval = getattr(params, "m_decay_interval", 50)  # Default: reduce every 50 epochs
        m_decay_step = getattr(params, "m_decay_step", 0.25)  # Default: reduce by 0.25 each time
        params.m_loss_m_value = current_m  # Set initial value
        print(f"m decay strategy enabled: starting m={current_m:.3f}, reducing by {m_decay_step:.3f} every {m_decay_interval} epochs")
    else:
        # If update_m is False or not set, m_loss_m_value remains constant
        if getattr(params, "use_m_loss", False):
            print(f"m loss enabled with constant m={current_m:.3f}")
    
    # Initialize data weight step-decay schedule
    data_loss_step_epochs = params.get('data_loss_step_epochs', None)
    if (data_loss_step_epochs is not None
            and weighter.hybrid_training
            and weighter.hybrid_strategy == 'fixed'
            and weighter.data_weight > weighter.physics_weight):
        n_steps = max(1, epochs // data_loss_step_epochs)
        data_weight_decrement = (weighter.data_weight - weighter.physics_weight) / n_steps
        print(
            f"Data weight step-decay: {weighter.data_weight:.2f} -> {weighter.physics_weight:.2f} "
            f"in {n_steps} steps (every {data_loss_step_epochs} epochs, "
            f"decrement {data_weight_decrement:.4f})")
    else:
        data_weight_decrement = None

    # ------ Time analysis setup ------
    time_analysis_dict = {}
    time_analysis_dict["prelim_time"] = prelim_time

    for ep in range(1, epochs+1):
        time_start = time.time()
        
        # Update m value if update_m strategy is enabled
        if update_m:
            # Check if it's time to decay m (at epochs that are multiples of m_decay_interval)
            if ep > 0 and ep % m_decay_interval == 0:
                # Reduce m by m_decay_step, but don't go below 0
                new_m = max(0.0, current_m - m_decay_step)
                if new_m != current_m:
                    current_m = new_m
                    params.m_loss_m_value = current_m
                    print(f"Epoch {ep}: Updated m to {current_m:.3f}")
        
        model.train()

        epoch_x = []
        epoch_alpha = []
        epoch_T = []  # For phase 4 temperature
        epoch_outputs = []
        epoch_d2check = []

        # Training loop
        epoch_losses = {'residual': [], 'bc_1': [], 'bc_2': [], 'total': []}
        if getattr(params, "fmt_help", False):
            epoch_losses['fmt'] = []
        if getattr(params, 'hybrid_training', False):
            epoch_losses['data'] = []
            epoch_losses['physics'] = []

        # Collect raw losses for adaptive weighting (update once per epoch)
        raw_losses_for_weighting = {'residual': [], 'bc_1': [], 'bc_2': []}
        if getattr(params, "fmt_help", False):
            raw_losses_for_weighting['fmt'] = []
        if getattr(params, 'hybrid_training', False):
            raw_losses_for_weighting['data'] = []
        # Collect gradient norms per batch for diagnostics
        epoch_grad_norms = []
        clipped_grad_norms = []

        # Per-batch timing accumulators
        batch_times_prefwd = []
        batch_times_fwd = []
        batch_times_loss = []
        batch_times_backprop = []
        batch_times_optim = []

        train_loader_iter = tqdm(
            train, desc=f"Epoch {ep}/{epochs}", leave=True, ncols=100)

        # Check if hybrid training is enabled
        hybrid_training = getattr(params, 'hybrid_training', False)

        for batch in train_loader_iter:
            t_ta_prefwd = time.time()
            # Extract inputs (and targets if present in data)
            # DataLoader returns a tuple. If TensorDataset has 2 tensors, batch is (inputs, targets)
            # If TensorDataset has 1 tensor, batch is (inputs,)
            inputs = batch[0].to(device)  # Unpack tuple from DataLoader
            if params.debug_mode:
                print(f"Batch input shape: {inputs.shape}")
                print(f"Batch length: {len(batch)}")

            # Check if we have targets in the batch
            if len(batch) > 1:
                targets = batch[1].to(device)
                if params.debug_mode:
                    print(f"Batch targets shape: {targets.shape}")
            else:
                targets = None
                if params.debug_mode:
                    print("No targets in batch")
            
            # Clamp x to avoid singularity at x=0
            inputs[:, 0].clamp_(min=1e-6, max=1.0)

            # Ensure inputs require gradients for autograd wrt x
            inputs.requires_grad_(True)
            x = inputs[:, 0:1].clone()
            alpha = inputs[:, 1:2].clone()
            # Extract T for phase 4
            if params.phase == 4:
                T = inputs[:, 2:3].clone()

            # Forward pass
            t_ta_fwd = time.time()
            outputs = model(inputs)
            t_ta_loss = time.time()

            if ep % params.save_every == 0:
                # Detach tensors before saving to avoid gradient tracking issues
                d2outdx2 = sec_deriv_auto(outputs, inputs)
                factor = (x**0.5)/(alpha**(1.5))
                d2check = (d2outdx2 * factor)**(2/3)
                epoch_x.append(x.detach())
                epoch_alpha.append(alpha.detach())
                if params.phase == 4:
                    epoch_T.append(T.detach())
                epoch_outputs.append(outputs.detach())
                epoch_d2check.append(d2check.detach())

            # Compute individual losses
            if params.phase == 3:
                residual_name = f"compute_residual_loss_phase2"
            elif params.phase == 4:
                residual_name = f"compute_residual_loss_phase4"
            else:
                residual_name = f"compute_residual_loss_phase{params.phase}"
            residual_fn = getattr(losses, residual_name)
            residual_loss = residual_fn(outputs, inputs, params=params)

            # Optional FMT helper loss (diff training mode)
            if getattr(params, "fmt_help", False):
                fmt_loss = losses.compute_fmt_loss_phase2(model, inputs, outputs, params)
            else:
                fmt_loss = torch.tensor(0.0, device=outputs.device)

            if params.mode == "soft":
                # Phase 4 reuses phase 1 BC functions (BCs are T-independent)
                if params.phase == 4:
                    bc1name = f"compute_bc_loss_1_phase4"
                    bc2name = f"compute_bc_loss_2_phase4"
                else:
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
            if getattr(params, "fmt_help", False):
                loss_dict['fmt'] = fmt_loss

            # Compute data loss for hybrid training
            if hybrid_training and targets is not None:
                data_loss = losses.compute_data_loss_phase4(outputs, targets, params, val_stage=False)
                loss_dict['data'] = data_loss
            else:
                data_loss = torch.tensor(0.0, device=outputs.device)

            # For adaptive weighting: collect raw losses, don't update weights yet
            if weighter.strategy == 'adaptive' or weighter.hybrid_strategy == 'adaptive':
                raw_losses_for_weighting['residual'].append(
                    residual_loss.detach())
                raw_losses_for_weighting['bc_1'].append(bc_1_loss.detach())
                raw_losses_for_weighting['bc_2'].append(bc_2_loss.detach())
                if getattr(params, "fmt_help", False):
                    raw_losses_for_weighting['fmt'].append(fmt_loss.detach())
                if hybrid_training and targets is not None:
                    raw_losses_for_weighting['data'].append(data_loss.detach())

            
            if params.phase == 3:
                totlossname = f"compute_total_loss_phase2"
            elif params.phase == 4:
                totlossname = f"compute_total_loss_phase4"
            else:
                totlossname = f"compute_total_loss_phase{params.phase}"
            totloss_fn = getattr(losses, totlossname)
            total_loss = totloss_fn(loss_dict, weighter)

            # Backpropagation
            t_ta_bp = time.time()
            optimizer.zero_grad()
            total_loss.backward()

            # Inspect gradients before clipping
            total_grad_norm_before = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_grad_norm_before += param_norm.item() ** 2
            total_grad_norm_before = total_grad_norm_before ** 0.5
            
            # Print if gradient is massive or nan
            if total_grad_norm_before > 100 or math.isnan(total_grad_norm_before) and params.debug_mode is True:
                print(f"[Batch {train_loader_iter.n}] Gradient Norm ALERT: {total_grad_norm_before:.4e}")

            # Record gradient norm before stepping
            t_ta_optim = time.time()
            total_grad_sq = 0.0
            num_params_with_grad = 0
            for p in model.parameters():
                if p.grad is not None:
                    g = p.grad.detach()
                    total_grad_sq += float(g.pow(2).sum().item())
                    num_params_with_grad += 1
            if num_params_with_grad > 0:
                epoch_grad_norms.append(total_grad_sq ** 0.5)

            # Clip gradients to prevent explosion, if activated
            if getattr(params, "grad_clipping", False):
                max_norm = getattr(params, "clip_value", 10.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

            # Sanity check: record grads to ensure they are clipped
            if getattr(params, "grad_clipping", False):
                total_grad_norm_after = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_grad_norm_after += param_norm.item() ** 2
                total_grad_norm_after = total_grad_norm_after ** 0.5
                clipped_grad_norms.append(total_grad_norm_after)

                # Print if clipping was active and made a difference
                if total_grad_norm_before > max_norm and params.debug_mode is True:
                    print(f"[Batch {train_loader_iter.n}] Gradient clipping applied: {total_grad_norm_before:.4e} -> {total_grad_norm_after:.4e}")

            optimizer.step()
            t_ta_batch_end = time.time()

            batch_times_prefwd.append(t_ta_fwd - t_ta_prefwd)
            batch_times_fwd.append(t_ta_loss - t_ta_fwd)
            batch_times_loss.append(t_ta_bp - t_ta_loss)
            batch_times_backprop.append(t_ta_optim - t_ta_bp)
            batch_times_optim.append(t_ta_batch_end - t_ta_optim)

            # Track losses
            epoch_losses['residual'].append(residual_loss.item())
            epoch_losses['bc_1'].append(bc_1_loss.item())
            epoch_losses['bc_2'].append(bc_2_loss.item())
            if getattr(params, "fmt_help", False):
                epoch_losses['fmt'].append(fmt_loss.item())
            if hybrid_training:
                epoch_losses['data'].append(data_loss.item())
                # Physics loss is residual + BCs (weighted)
                physics_loss = weighter.get_physics_loss(loss_dict) if hasattr(weighter, 'get_physics_loss') else residual_loss
                epoch_losses['physics'].append(physics_loss.item() if torch.is_tensor(physics_loss) else physics_loss)
            epoch_losses['total'].append(total_loss.item())

            # Update progress bar
            train_loader_iter.set_postfix(
                loss=total_loss.item(),
                residual=residual_loss.item(),
                bc1=bc_1_loss.item(),
                bc2=bc_2_loss.item(),
                data=data_loss.item() if hybrid_training else 0.0
            )

        # ------ Time analysis: per-epoch statistics (sums across batches) ------
        time_analysis_dict[ep] = {
            "prefwd_sum": float(np.sum(batch_times_prefwd)),
            "fwd_sum": float(np.sum(batch_times_fwd)),
            "loss_sum": float(np.sum(batch_times_loss)),
            "backprop_sum": float(np.sum(batch_times_backprop)),
            "optim_step_sum": float(np.sum(batch_times_optim)),
        }

        if ep % params.save_every == 0:
            # Concatenate tensors first, then convert to numpy
            x_all = torch.cat(epoch_x, dim=0).cpu().numpy()
            alpha_all = torch.cat(epoch_alpha, dim=0).cpu().numpy()
            outputs_all = torch.cat(epoch_outputs, dim=0).cpu().numpy()
            d2check_all = torch.cat(epoch_d2check, dim=0).cpu().numpy()

            predictions[ep] = {"x": x_all,
                               "alpha": alpha_all,
                               "outputs": outputs_all,
                               "d2_check": d2check_all}

            # Include T for phase 4
            if params.phase == 4 and len(epoch_T) > 0:
                T_all = torch.cat(epoch_T, dim=0).cpu().numpy()
                predictions[ep]["T"] = T_all

            print(
                f"Saved epoch {ep} predictions")

        # Training epoch complete - compute averages before printing
        avg_residual = np.mean(epoch_losses['residual'])
        avg_bc1 = np.mean(epoch_losses['bc_1'])
        avg_bc2 = np.mean(epoch_losses['bc_2'])
        avg_total = np.mean(epoch_losses['total'])
        avg_fmt = np.mean(epoch_losses['fmt']) if 'fmt' in epoch_losses else 0.0
        avg_data = np.mean(epoch_losses['data']) if 'data' in epoch_losses and len(epoch_losses['data']) > 0 else 0.0
        print("We have data loss:", 'data' in epoch_losses)
        print("Min and max data loss:", np.min(epoch_losses['data']) if 'data' in epoch_losses and len(epoch_losses['data']) > 0 else 'N/A', np.max(epoch_losses['data']) if 'data' in epoch_losses and len(epoch_losses['data']) > 0 else 'N/A')
        avg_physics = np.mean(epoch_losses['physics']) if 'physics' in epoch_losses and len(epoch_losses['physics']) > 0 else 0.0

        # Update physics sub-loss weights once per epoch for adaptive strategy
        if weighter.strategy == 'adaptive':
            avg_losses_for_weighting = {
                'residual': torch.stack(raw_losses_for_weighting['residual']).mean(),
                'bc_1': torch.stack(raw_losses_for_weighting['bc_1']).mean(),
                'bc_2': torch.stack(raw_losses_for_weighting['bc_2']).mean()
            }
            if getattr(params, "fmt_help", False) and len(raw_losses_for_weighting['fmt']) > 0:
                avg_losses_for_weighting['fmt'] = torch.stack(raw_losses_for_weighting['fmt']).mean()
            weighter.update_weights(avg_losses_for_weighting)

        # Update hybrid weights (physics vs data) - independent of physics sub-loss strategy
        if weighter.hybrid_training and weighter.hybrid_strategy == 'adaptive':
            if len(raw_losses_for_weighting.get('data', [])) > 0:
                hybrid_loss_dict = {}
                hybrid_loss_dict['data'] = torch.stack(raw_losses_for_weighting['data']).mean()
                res_avg = torch.stack(raw_losses_for_weighting['residual']).mean()
                bc1_avg = torch.stack(raw_losses_for_weighting['bc_1']).mean()
                bc2_avg = torch.stack(raw_losses_for_weighting['bc_2']).mean()
                hybrid_loss_dict['physics_total'] = (
                    weighter.residual_weight * res_avg
                    + weighter.bc_1_weight * bc1_avg
                    + weighter.bc_2_weight * bc2_avg
                )
                weighter.update_weights(hybrid_loss_dict)

        # Reset collection for next epoch
        if weighter.strategy == 'adaptive' or weighter.hybrid_strategy == 'adaptive':
            raw_losses_for_weighting = {'residual': [], 'bc_1': [], 'bc_2': []}
            if getattr(params, "fmt_help", False):
                raw_losses_for_weighting['fmt'] = []
            if getattr(params, 'hybrid_training', False):
                raw_losses_for_weighting['data'] = []

        # Step-decay data weight at interval boundaries
        if data_weight_decrement is not None and ep % data_loss_step_epochs == 0:
            new_data_weight = max(weighter.physics_weight,
                                  weighter.data_weight - data_weight_decrement)
            print(f"Epoch {ep}: data_weight {weighter.data_weight:.4f} -> {new_data_weight:.4f}")
            weighter.data_weight = new_data_weight

        # Update scheduler AFTER all batches (PyTorch best practice)
        scheduler.step()
        # Update for next epoch's logging
        current_lr = optimizer.param_groups[0]['lr']

        time_end = time.time()
        time_taken = time_end - time_start
        time_analysis_dict[ep]["epoch_total"] = time_taken

        # Print epoch summary
        print(
            f"Epoch {ep}/{epochs}: train loss = {avg_total:.6e}, time = {time_taken:.2f} s")

        # In hard mode, log hard-constraint checks using explicit probes at x=0 and x=1
        checks = {}
        if params.mode == "hard" and ep % 10 == 0:
            alpha_check = torch.tensor([[1e2]])
            x0 = torch.zeros_like(alpha_check) + 1e-10
            x1 = torch.ones_like(alpha_check)

            if params.phase == 4:
                # Phase 4 has 3 inputs: [x, alpha, T]
                T_check = torch.tensor([[10.0]])  # Use 10 keV as test temperature
                inp0 = torch.cat([x0, alpha_check, T_check], dim=-1).to(device).requires_grad_(True)
                inp1 = torch.cat([x1, alpha_check, T_check], dim=-1).to(device).requires_grad_(True)
            else:
                inp0 = torch.cat([x0, alpha_check], dim=-1).to(device).requires_grad_(True)
                inp1 = torch.cat([x1, alpha_check], dim=-1).to(device).requires_grad_(True)
            psi0 = model(inp0)
            psi1 = model(inp1)

            dpsi1_dx = first_deriv_auto(psi1, inp1)

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
                "grads/clipped_epoch_grad_norm_mean": float(np.mean(clipped_grad_norms)) if len(clipped_grad_norms) > 0 else None,
            }

        # Log to wandb
        is_soft_mode = params.mode == "soft"
        log_dict = {
            "info/learning_rate": current_lr,
            "info/epoch": ep,
            "info/epoch_training_time": time_taken,
            "train/pde_loss": avg_residual,
            **({"train/bc_1_loss": avg_bc1, "train/bc_2_loss": avg_bc2} if is_soft_mode else {}),
            **({"train/fmt_loss": avg_fmt} if 'fmt' in epoch_losses else {}),
            **({"train/data_loss": avg_data, "train/physics_loss": avg_physics} if getattr(params, 'hybrid_training', False) else {}),
            "train/total_loss": avg_total,
            **({"weights/residual_weight": weighter.residual_weight, "weights/bc_1_weight": weighter.bc_1_weight, "weights/bc_2_weight": weighter.bc_2_weight} if is_soft_mode else {}),
            **({"weights/fmt_weight": weighter.fmt_weight} if getattr(params, "fmt_help", False) else {}),
            **({"weights/physics_weight": weighter.physics_weight, "weights/data_weight": weighter.data_weight} if getattr(params, 'hybrid_training', False) else {}),
            **checks,
            **grad_stats,
        }
        # Log m value if m loss is being used
        if getattr(params, "use_m_loss", False):
            log_dict["info/m_loss_m_value"] = getattr(params, "m_loss_m_value", 0.0)
        wandb.log(log_dict)

        # Validation
        if val is not None:
            time_start_val = time.time()
            model.eval()
            val_losses = {'residual': [], 'bc_1': [], 'bc_2': [], 'total': []}
            if getattr(params, "fmt_help", False):
                val_losses['fmt'] = []
            if getattr(params, 'hybrid_training', False):
                val_losses['data'] = []
                val_losses['physics'] = []

            for batch in val:
                # Extract inputs (and targets if present in data)
                # DataLoader returns a tuple. If TensorDataset has 2 tensors, batch is (inputs, targets)
                # If TensorDataset has 1 tensor, batch is (inputs,)
                inputs = batch[0].to(device)  # Unpack tuple from DataLoader

                # Check if we have targets in the batch
                if len(batch) > 1:
                    targets = batch[1].to(device)
                else:
                    targets = None
                
                # Clamp x to avoid singularity at x=0 and ensure x <= 1 for phi_0_transform (same as training)
                inputs[:, 0].clamp_(min=1e-6, max=1.0)
                
                # Ensure inputs require gradients for derivative-based losses
                inputs.requires_grad_(True)

                # Forward pass
                outputs = model(inputs)

                # Compute individual losses
                if params.phase == 3:
                    residual_name = f"compute_residual_loss_phase2"
                elif params.phase == 4:
                    residual_name = f"compute_residual_loss_phase4"
                else:
                    residual_name = f"compute_residual_loss_phase{params.phase}"
                residual_fn = getattr(losses, residual_name)
                residual_loss = residual_fn(outputs, inputs, params=params, val_stage=True)

                if getattr(params, "fmt_help", False):
                    fmt_loss = losses.compute_fmt_loss_phase2(model, inputs, outputs, params)
                else:
                    fmt_loss = torch.tensor(0.0, device=outputs.device)

                if params.mode == "soft":
                    # Phase 4 reuses phase 1 BC functions (BCs are T-independent)
                    if params.phase == 4:
                        bc1name = f"compute_bc_loss_1_phase4"
                        bc2name = f"compute_bc_loss_2_phase4"
                    else:
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
                if getattr(params, "fmt_help", False):
                    val_losses['fmt'].append(fmt_loss.item())

                # Compute data loss for hybrid validation
                if hybrid_training and targets is not None:
                    val_data_loss = losses.compute_data_loss_phase4(outputs, targets, params, val_stage=True)
                    val_losses['data'].append(val_data_loss.item())
                else:
                    val_data_loss = torch.tensor(0.0, device=outputs.device)

                # Compute val total loss: unweighted sum (no hybrid weighting on val)
                val_loss_dict = {
                    'residual': residual_loss,
                    'bc_1': bc_1_loss,
                    'bc_2': bc_2_loss
                }
                if getattr(params, "fmt_help", False):
                    val_loss_dict['fmt'] = fmt_loss

                # Physics loss (residual + BCs) — use weighter for BC weights in soft mode
                val_physics_loss = weighter.get_physics_loss(val_loss_dict) if hasattr(weighter, 'get_physics_loss') else residual_loss
                if hybrid_training and targets is not None:
                    val_losses['physics'].append(val_physics_loss.item() if torch.is_tensor(val_physics_loss) else val_physics_loss)

                # Total = physics + data, unweighted (no physics_weight/data_weight)
                if hybrid_training and targets is not None:
                    val_total_loss = val_physics_loss + val_data_loss
                else:
                    val_total_loss = val_physics_loss
                val_losses['total'].append(val_total_loss.item() if torch.is_tensor(val_total_loss) else val_total_loss)

            avg_val_residual = np.mean(val_losses['residual'])
            avg_val_bc1 = np.mean(val_losses['bc_1'])
            avg_val_bc2 = np.mean(val_losses['bc_2'])
            avg_val_total = np.mean(val_losses['total'])
            avg_val_fmt = np.mean(val_losses['fmt']) if 'fmt' in val_losses else 0.0
            avg_val_data = np.mean(val_losses['data']) if 'data' in val_losses and len(val_losses['data']) > 0 else 0.0
            avg_val_physics = np.mean(val_losses['physics']) if 'physics' in val_losses and len(val_losses['physics']) > 0 else 0.0
            time_end_val = time.time()
            time_taken_val = time_end_val - time_start_val

            # Print validation summary
            print(f"  val loss = {avg_val_total:.6e}")

            wandb.log({
                "val/pde_loss": avg_val_residual,
                **({"val/bc_1_loss": avg_val_bc1, "val/bc_2_loss": avg_val_bc2} if is_soft_mode else {}),
                **({"val/fmt_loss": avg_val_fmt} if 'fmt' in val_losses else {}),
                **({"val/data_loss": avg_val_data, "val/physics_loss": avg_val_physics} if getattr(params, 'hybrid_training', False) else {}),
                "val/total_loss": avg_val_total,
                "info/epoch_validation_time": time_taken_val
            })

        # Periodically save model weights
        save_weights_every = getattr(params, "save_weights_every", None)
        save_weights = getattr(params, "save_weights", False)
        if save_weights is True and isinstance(save_weights_every, int) and save_weights_every > 0 and ep % save_weights_every == 0:
            weights_dir = PROJECT_ROOT / "saving_weights" / f"{params.run_name}"
            weights_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = weights_dir / f"weights_epoch_{ep}"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved weights checkpoint to {ckpt_path}")

    if len(predictions) != 0 and getattr(params, "save_preds", False):
        save_path = PROJECT_ROOT / params.pred_dir / f"phase{params.phase}" / f"{params.run_name}.pkl"
    
        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "wb") as f:
            pickle.dump(predictions, f)

        print(f"Saved predictions for all selected epochs to {save_path}")

    # ------ Time analysis: save results ------
    ta_dir = PROJECT_ROOT / "time_analysis" / f"{params.run_name}"
    ta_dir.mkdir(parents=True, exist_ok=True)

    # Save full nested dict as pickle
    ta_pkl_path = ta_dir / "time_analysis.pkl"
    with open(ta_pkl_path, "wb") as f:
        pickle.dump(time_analysis_dict, f)
    print(f"Saved time analysis data to {ta_pkl_path}")

    # Compute global summary (mean and std of each metric across all epochs)
    # Time unit conversion: raw values are in seconds, convert if requested
    ta_time_unit = getattr(params, "time_unit", "seconds")
    ta_time_scales = {"seconds": 1.0, "ms": 1000.0, "minutes": 1.0 / 60.0}
    ta_scale = ta_time_scales.get(ta_time_unit, 1.0)

    ta_metric_names = ["prefwd", "fwd", "loss", "backprop", "optim_step"]
    ta_summary = {
        "compute_specs": {
            "hpc_nodes": getattr(params, "hpc_nodes", None),
            "hpc_ncpus": getattr(params, "hpc_ncpus", None),
            "hpc_mem": getattr(params, "hpc_mem", None),
            "hpc_ngpus": getattr(params, "hpc_ngpus", None)},
        "compute_used": {
            "device": str(device),
            "torch_num_threads": torch.get_num_threads(),
            "peak_mem_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024), 2)
        },
        "time_unit": ta_time_unit,
        "prelim_time": round(time_analysis_dict["prelim_time"] * ta_scale, 4),
    }
    for ta_metric in ta_metric_names:
        ta_epoch_sums = [time_analysis_dict[ep][f"{ta_metric}_sum"]
                         for ep in range(1, epochs + 1)]
        ta_summary[f"{ta_metric}_global_mean"] = round(float(np.mean(ta_epoch_sums)) * ta_scale, 4)
        ta_summary[f"{ta_metric}_global_std"] = round(float(np.std(ta_epoch_sums)) * ta_scale, 4)

    # epoch_total is a single value per epoch (not per-batch), so average directly
    ta_epoch_totals = [time_analysis_dict[ep]["epoch_total"]
                       for ep in range(1, epochs + 1)]
    ta_summary["epoch_total_global_mean"] = round(float(np.mean(ta_epoch_totals)) * ta_scale, 4)
    ta_summary["epoch_total_global_std"] = round(float(np.std(ta_epoch_totals)) * ta_scale, 4)

    ta_json_path = ta_dir / "time_analysis_summary.json"
    with open(ta_json_path, "w") as f:
        json.dump(ta_summary, f, indent=2)
    print(f"Saved time analysis summary to {ta_json_path}")

    return
