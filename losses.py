import torch


def compute_residual_loss_phase1(outputs, inputs):
    """
    Computes the PDE residual loss.

    The PDE is: d^2(f(x))/dx^2 = 3x
    where f(x) = Psi(x) is the model output

    Args:
        outputs: Model predictions f(x)=Psi(x) (requires_grad=True), shape [batch_size, 1]
        inputs: Input coordinates x (requires_grad=True), shape [batch_size, 1]

    Returns:
        residual_loss: Mean squared residual
    """
    # inputs already has requires_grad=True from trainer, preserve the graph
    # CRITICAL: Keep the original tensor reference - don't create new views
    # The model outputs might be 1D or 2D, but inputs is [batch_size, 1]
    x = inputs  # Keep original reference with shape [batch_size, 1]

    # Outputs from model - handle shape consistently
    if len(outputs.shape) == 2 and outputs.shape[1] == 1:
        f = outputs  # Keep as [batch_size, 1] to match x
    elif len(outputs.shape) == 1:
        f = outputs.unsqueeze(1)  # Convert to [batch_size, 1]
    else:
        f = outputs

    # Compute x*f(x) - element-wise multiplication (broadcasting handles shapes)
    # x: [batch_size, 1], f: [batch_size, 1] or [batch_size]
    # -> g: [batch_size, 1] or [batch_size]
    g = f

    # Compute first derivative: dg/dx
    # CRITICAL: Use the exact same tensor (x/inputs) throughout
    dg_dx_outputs = torch.autograd.grad(
        outputs=g,
        inputs=x,  # Use x (which is inputs reference)
        grad_outputs=torch.ones_like(g),
        create_graph=True,  # Critical: needed for second derivative
        retain_graph=True,  # Keep graph for multiple backward passes
        # Only compute gradients w.r.t. x (not model params)
        only_inputs=True,
        allow_unused=False  # Should not be unused - raise error if disconnected
    )

    if len(dg_dx_outputs) == 0 or dg_dx_outputs[0] is None:
        raise RuntimeError(
            "First derivative (dg_dx) computation failed. "
            "Check that x is properly connected in the computational graph. "
            "Model2 should not use clone().detach() in forward()."
        )
    dg_dx = dg_dx_outputs[0]

    # Compute second derivative: d^2g/dx^2
    # dg_dx depends on x through the graph created by create_graph=True above
    # CRITICAL: Use the same x tensor for second derivative
    d2g_dx2_outputs = torch.autograd.grad(
        outputs=dg_dx,
        inputs=x,  # Use the SAME x tensor
        grad_outputs=torch.ones_like(dg_dx),
        create_graph=True,  # Needed if we want third derivatives later
        retain_graph=True,
        only_inputs=True,  # Only compute gradients w.r.t. x
        allow_unused=False  # Should never be unused if graph is connected
    )

    # Extract gradient w.r.t. x
    if len(d2g_dx2_outputs) == 0 or d2g_dx2_outputs[0] is None:
        raise RuntimeError(
            "Second derivative (d²g/dx²) computation failed. "
            "This indicates the computational graph is disconnected. "
            "Check: 1) x not detached in model forward(), "
            "2) model output properly connected to x, "
            "3) first derivative computation succeeded."
        )
    d2g_dx2 = d2g_dx2_outputs[0]

    # PDE residual: d^2(f)/dx^2 - 3*x = 0

    residual = d2g_dx2 - 3*x

    # Clip residual to prevent extreme values from dominating the loss
    # This helps with numerical stability, especially during early training
    # residual = torch.clamp(residual, min=-1e6, max=1e6)

    residual_loss = torch.mean(residual ** 2)

    return residual_loss


def compute_bc_loss_1_phase1(outputs, inputs):
    """
    Computes boundary condition loss at x=1.

    The old BC was: dV/dx = 0 at x=1
    Given f(x) = A * x * V(x), we can derive:
    V = f/(A*x), so dV/dx = (1/A*x^2) * (x*f' - f)
    Setting dV/dx = 0 at x=1 gives: f'(1) - f(1) = 0

    So the BC for f(x) is: f'(1) - f(1) = 0

    Args:
        outputs: Model predictions f(x), shape [batch_size, 1]
        inputs: Input coordinates x, shape [batch_size, 1]

    Returns:
        bc_loss: Mean squared error at boundary x=1
    """
    # CRITICAL: Keep the original tensor reference - don't create new views
    # The model outputs might be 1D or 2D, but inputs is [batch_size, 1]
    x = inputs  # Keep original reference with shape [batch_size, 1]

    # Outputs from model - handle shape consistently
    if len(outputs.shape) == 2 and outputs.shape[1] == 1:
        f = outputs  # Keep as [batch_size, 1] to match x
    elif len(outputs.shape) == 1:
        f = outputs.unsqueeze(1)  # Convert to [batch_size, 1]
    else:
        f = outputs

    # Find points near x=1 (boundary condition location)
    # Extract values for comparison while preserving graph
    # For shape [batch_size, 1], we need to compare x[:, 0] or x.squeeze() values
    if len(x.shape) == 2 and x.shape[1] == 1:
        x_val = x[:, 0]  # Extract for comparison
    else:
        x_val = x

    # Find points where x is close to 1 (within tolerance)
    tol = 1e-3
    boundary_mask = torch.abs(x_val - 1.0) < tol

    if torch.sum(boundary_mask) == 0:
        # No boundary points found, return zero loss
        return torch.tensor(0.0, device=outputs.device, requires_grad=True)

    # Compute derivative df/dx for ALL points
    # CRITICAL: f depends on the full x tensor (passed to model), not on a subset
    # We must compute grad(f, x) using the full tensors, then extract boundary values
    df_dx_outputs = torch.autograd.grad(
        outputs=f,  # Use full f tensor
        inputs=x,    # Use full x tensor (what f depends on)
        grad_outputs=torch.ones_like(f),
        create_graph=True,  # Needed if higher derivatives are required
        retain_graph=True,
        only_inputs=True,   # Only compute gradients w.r.t. x
        allow_unused=False  # Should not be unused if graph is connected
    )

    if len(df_dx_outputs) == 0 or df_dx_outputs[0] is None:
        raise RuntimeError(
            "BC1 derivative computation failed. "
            "Check that x is properly connected in the computational graph. "
            "Model should not detach x in forward()."
        )
    df_dx = df_dx_outputs[0]  # Shape: [batch_size, 1] or [batch_size]

    # Extract boundary values AFTER computing the gradient
    f_boundary = f[boundary_mask]  # Shape: [n_boundary, 1] or [n_boundary]
    # Extract boundary gradient values
    df_dx_at_boundary = df_dx[boundary_mask]

    # BC constraint: f'(1) - f(1) = 0
    # Ensure shapes match (might need to squeeze/unsqueeze)
    if len(df_dx_at_boundary.shape) != len(f_boundary.shape):
        if len(df_dx_at_boundary.shape) == 2 and df_dx_at_boundary.shape[1] == 1:
            df_dx_at_boundary = df_dx_at_boundary.squeeze(1)
        elif len(f_boundary.shape) == 2 and f_boundary.shape[1] == 1:
            f_boundary = f_boundary.squeeze(1)

    bc_constraint = df_dx_at_boundary - f_boundary
    bc_loss = torch.mean(bc_constraint ** 2)

    return bc_loss


def compute_bc_loss_2_phase1(outputs, inputs):
    """
    Computes boundary condition loss at x=0.

    Psi(0) = 1

    Args:
        outputs: Model predictions f(x), shape [batch_size, 1]
        inputs: Input coordinates x, shape [batch_size, 1]

    Returns:
        bc_loss: Mean squared error at boundary x=0
    """
    # CRITICAL: Keep the original tensor reference - don't create new views
    # The model outputs might be 1D or 2D, but inputs is [batch_size, 1]
    x = inputs  # Keep original reference with shape [batch_size, 1]

    # Outputs from model - handle shape consistently
    if len(outputs.shape) == 2 and outputs.shape[1] == 1:
        f = outputs  # Keep as [batch_size, 1] to match x
    elif len(outputs.shape) == 1:
        f = outputs.unsqueeze(1)  # Convert to [batch_size, 1]
    else:
        f = outputs

    # Find points near x=0 (boundary condition location)
    # Extract values for comparison while preserving graph
    if len(x.shape) == 2 and x.shape[1] == 1:
        x_val = x[:, 0]  # Extract for comparison
    else:
        x_val = x

    # Find points where x is close to 0 (within tolerance)
    tol = 1e-3
    boundary_mask = torch.abs(x_val - 0.0) < tol

    if torch.sum(boundary_mask) == 0:
        # No boundary points found, return zero loss
        return torch.tensor(0.0, device=outputs.device, requires_grad=True)

    # Get boundary points - use indexing to preserve graph connection
    f_boundary = f[boundary_mask]  # Shape: [n_boundary, 1] or [n_boundary]

    # BC constraint: f(0) = 1
    # Handle shape compatibility
    if len(f_boundary.shape) == 2 and f_boundary.shape[1] == 1:
        f_boundary = f_boundary.squeeze(1)

    bc_constraint = f_boundary - 1
    bc_loss = torch.mean(bc_constraint ** 2)

    return bc_loss


class LossWeighter_phase1:
    """Handles different loss weighting strategies."""

    def __init__(self, params):
        """
        Args:
            params: from config.yaml
        """
        self.mode = params.mode  # "soft" or "hard"
        self.strategy = params.loss_strategy  # "fixed" or "adaptive"

        # If hard mode, BCs are enforced in architecture, only residual loss
        if self.mode == "hard":
            self.residual_weight = 1.0
            self.bc_1_weight = 0.0
            self.bc_2_weight = 0.0
        else:  # soft mode
            self.residual_weight = params.relative_weights["residual"]
            self.bc_1_weight = params.relative_weights["bc_1"]
            self.bc_2_weight = params.relative_weights["bc_2"]

    def update_weights(self, loss_dict):
        """
        Update weights based on loss magnitudes (for adaptive strategies).

        Args:
            loss_dict: Dictionary of individual losses
        """
        if self.strategy == 'adaptive' and self.mode == 'soft':
            # Normalize weights to balance loss magnitudes
            # Use inverse loss magnitudes with regularization for stability
            loss_mags = {}

            for key in ["residual", "bc_1", "bc_2"]:
                if key in loss_dict:
                    loss_mag = loss_dict[key].detach().item()
                    # Regularize to avoid extreme weights
                    loss_mags[key] = max(loss_mag, 1e-8)

            # Compute average loss magnitude for normalization
            avg_mag = sum(loss_mags.values()) / \
                len(loss_mags) if loss_mags else 1.0

            # Update weights to be inversely proportional to loss magnitudes
            # Normalized by average to keep weights in reasonable range
            if "residual" in loss_mags:
                self.residual_weight = avg_mag / loss_mags["residual"]
            if "bc_1" in loss_mags:
                self.bc_1_weight = avg_mag / loss_mags["bc_1"]
            if "bc_2" in loss_mags:
                self.bc_2_weight = avg_mag / loss_mags["bc_2"]

    def get_weighted_loss(self, loss_dict):
        """
        Compute weighted sum of losses.

        Args:
            loss_dict: Dictionary of individual losses

        Returns:
            total_loss: Weighted sum
        """
        total = 0.0

        if "residual" in loss_dict:
            total += self.residual_weight * loss_dict["residual"]

        if self.mode == "soft":  # Only add BC losses in soft mode
            if "bc_1" in loss_dict:
                total += self.bc_1_weight * loss_dict["bc_1"]
            if "bc_2" in loss_dict:
                total += self.bc_2_weight * loss_dict["bc_2"]

        return total


def compute_total_loss_phase1(loss_dict, weighter):
    """
    Compute total weighted loss from individual loss components.

    Note: Weights are updated per epoch (not per batch) for adaptive strategy
    to ensure stable training.

    Args:
        loss_dict: Dictionary of individual losses 
        ({'residual': tensor, 'bc_1': tensor, 'bc_2': tensor})
        weighter: LossWeighter instance

    Returns:
        total_loss: Weighted sum of losses
    """
    # Compute weighted total (weights are set externally per epoch for adaptive)
    total_loss = weighter.get_weighted_loss(loss_dict)

    return total_loss


# =========================
# Phase 2 losses
# =========================

def compute_residual_loss_phase2(outputs, inputs, params):
    """
    Nonlinear TF residual for phase 2:
        psi''(x) = C * x^{-1/2} * [psi(x)]^{3/2}

    Changed to:
        psi''(x) * x^(1/2) / C - psi(x)^3/2 = 0

    C depends on r0, and x not in denominator.

    Args:
        outputs: psi(x) predictions, shape [batch, 1] (requires_grad=True)
        inputs: x & alpha, shape [batch, 2] (requires_grad=True)
        params: configs

    Returns:
        residual loss: mean squared residual over the batch
    """
    # Use the same inputs tensor used to compute psi to preserve graph connectivity
    x = inputs[:, 0:1]
    alpha = inputs[:, 1:2]

    # Ensure outputs have shape [batch, 1]
    if len(outputs.shape) == 2 and outputs.shape[1] == 1:
        psi = outputs
    elif len(outputs.shape) == 1:
        psi = outputs.unsqueeze(1)
    else:
        psi = outputs

    # First derivative: compute grad wrt the full inputs, then take x-component
    dpsi_dinputs_out = torch.autograd.grad(
        outputs=psi,
        inputs=inputs,
        grad_outputs=torch.ones_like(psi),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=False
    )
    if len(dpsi_dinputs_out) == 0 or dpsi_dinputs_out[0] is None:
        raise RuntimeError(
            "Phase2: dpsi/dx computation failed; check graph connectivity.")
    dpsi_dx = dpsi_dinputs_out[0][:, 0:1]

    # Second derivative: differentiate dpsi_dx wrt inputs, then take x-component
    d2psi_dinputs_out = torch.autograd.grad(
        outputs=dpsi_dx,
        inputs=inputs,
        grad_outputs=torch.ones_like(dpsi_dx),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=False
    )
    if len(d2psi_dinputs_out) == 0 or d2psi_dinputs_out[0] is None:
        raise RuntimeError(
            "Phase2: d2psi/dx2 computation failed; check graph connectivity.")
    d2psi_dx2 = d2psi_dinputs_out[0][:, 0:1]

    if (alpha < 0).any():
        print("Watch out: alpha (r0/b) is negative")
    c = alpha ** (3/2)

    residual = (d2psi_dx2 * x**0.5) / c - psi**1.5

    if params.loss_type == "mse":
        loss = torch.mean(residual**2)
    else:
        raise ValueError("Need to select an appropiate loss type")

    return loss


def compute_bc_loss_1_phase2(outputs, inputs):
    """
    Enforce robin: psi'(1) = psi(1)
    """
    bc1_phase2 = compute_bc_loss_1_phase1(outputs, inputs)
    return bc1_phase2


def compute_bc_loss_2_phase2(outputs, inputs):
    """
    Computes boundary condition loss at x=0.

    Psi(0) = 1
    """
    bc2_phase2 = compute_bc_loss_2_phase1(outputs, inputs)
    return bc2_phase2


class LossWeighter_phase2:
    """Handles loss weighting for Phase 2 (mirrors Phase 1 behavior)."""

    def __init__(self, params):
        self.mode = params.mode  # "soft" or "hard"
        self.strategy = params.loss_strategy  # "fixed" or "adaptive"

        if self.mode == "hard":
            self.residual_weight = 1.0
            self.bc_1_weight = 0.0
            self.bc_2_weight = 0.0
        else:
            self.residual_weight = params.relative_weights["residual"]
            self.bc_1_weight = params.relative_weights["bc_1"]
            self.bc_2_weight = params.relative_weights["bc_2"]

    def update_weights(self, loss_dict):
        if self.strategy == 'adaptive' and self.mode == 'soft':
            loss_mags = {}
            for key in ["residual", "bc_1", "bc_2"]:
                if key in loss_dict:
                    val = loss_dict[key].detach().item()
                    loss_mags[key] = max(val, 1e-8)
            if not loss_mags:
                return
            avg_mag = sum(loss_mags.values()) / len(loss_mags)
            if "residual" in loss_mags:
                self.residual_weight = avg_mag / loss_mags["residual"]
            if "bc_1" in loss_mags:
                self.bc_1_weight = avg_mag / loss_mags["bc_1"]
            if "bc_2" in loss_mags:
                self.bc_2_weight = avg_mag / loss_mags["bc_2"]

    def get_weighted_loss(self, loss_dict):
        total = 0.0
        if "residual" in loss_dict:
            total += self.residual_weight * loss_dict["residual"]
        if self.mode == "soft":
            if "bc_1" in loss_dict:
                total += self.bc_1_weight * loss_dict["bc_1"]
            if "bc_2" in loss_dict:
                total += self.bc_2_weight * loss_dict["bc_2"]
        return total


def compute_total_loss_phase2(loss_dict, weighter):
    return weighter.get_weighted_loss(loss_dict)
