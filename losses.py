import torch


def compute_residual_loss(outputs, inputs):
    """
    Computes the PDE residual loss.

    The PDE is: 1/x^2 * d^2(x*f(x))/dx^2 = 3/(4*pi)
    where f(x) = Psi(x) is the model output

    Args:
        outputs: Model predictions f(x)=Psi(x) (requires_grad=True), shape [batch_size, 1]
        inputs: Input coordinates x (requires_grad=True), shape [batch_size, 1]

    Returns:
        residual_loss: Mean squared residual
    """
    # inputs already has requires_grad=True from trainer, don't clone it
    x = inputs.squeeze() if len(inputs.shape) > 1 else inputs
    f = outputs.squeeze() if len(outputs.shape) > 1 else outputs

    # Compute x*f(x)
    g = x * f

    # Compute first derivative: dg/dx
    dg_dx = torch.autograd.grad(
        outputs=g,
        inputs=x,
        grad_outputs=torch.ones_like(g),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=False
    )[0]

    # Compute second derivative: d^2g/dx^2
    d2g_dx2 = torch.autograd.grad(
        outputs=dg_dx,
        inputs=x,
        grad_outputs=torch.ones_like(dg_dx),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True
    )
    d2g_dx2 = d2g_dx2[0] if len(
        d2g_dx2) > 0 and d2g_dx2[0] is not None else torch.zeros_like(dg_dx)

    # PDE residual: 1/x^2 * d^2(x*f)/dx^2 - 3/(4*pi) = 0
    x_sq = x ** 2
    # Avoid division by zero for x=0
    x_sq = torch.clamp(x_sq, min=1e-8)

    residual = d2g_dx2 / x_sq - 3 / (4 * torch.pi)
    residual_loss = torch.mean(residual ** 2)

    return residual_loss


def compute_bc_loss_1(outputs, inputs):
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
    x = inputs.clone().detach().requires_grad_(True)
    f = outputs.squeeze()  # f(x), shape: [batch_size]

    # Find points near x=1 (boundary condition location)
    x_val = x.squeeze()

    # Find points where x is close to 1 (within tolerance)
    tol = 1e-3
    boundary_mask = torch.abs(x_val - 1.0) < tol

    if torch.sum(boundary_mask) == 0:
        # No boundary points found, return zero loss
        return torch.tensor(0.0, device=outputs.device, requires_grad=True)

    # Get boundary points
    x_boundary = x[boundary_mask].clone().detach().requires_grad_(True)
    f_boundary = f[boundary_mask]

    # Compute derivative df/dx at boundary
    df_dx_at_boundary = torch.autograd.grad(
        f_boundary,
        x_boundary,
        grad_outputs=torch.ones_like(f_boundary),
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )

    if len(df_dx_at_boundary) > 0 and df_dx_at_boundary[0] is not None:
        df_dx_at_boundary = df_dx_at_boundary[0].squeeze()
    else:
        df_dx_at_boundary = torch.zeros_like(f_boundary)

    # BC constraint: f'(1) - f(1) = 0
    bc_constraint = df_dx_at_boundary - f_boundary
    bc_loss = torch.mean(bc_constraint ** 2)

    return bc_loss


def compute_bc_loss_2(outputs, inputs):
    """
    Computes boundary condition loss at x=0.

    Psi(0) = 1

    Args:
        outputs: Model predictions f(x), shape [batch_size, 1]
        inputs: Input coordinates x, shape [batch_size, 1]

    Returns:
        bc_loss: Mean squared error at boundary x=0
    """
    x = inputs.clone().detach().requires_grad_(True)
    f = outputs.squeeze()  # f(x), shape: [batch_size]

    # Find points near x=0 (boundary condition location)
    x_val = x.squeeze()

    # Find points where x is close to 0 (within tolerance)
    tol = 1e-3
    boundary_mask = torch.abs(x_val - 0.0) < tol

    if torch.sum(boundary_mask) == 0:
        # No boundary points found, return zero loss
        return torch.tensor(0.0, device=outputs.device, requires_grad=True)

    # Get boundary points
    f_boundary = f[boundary_mask]

    # BC constraint: f(0) = 1
    bc_constraint = f_boundary - 1
    bc_loss = torch.mean(bc_constraint ** 2)

    return bc_loss


class LossWeighter:
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


def compute_total_loss(loss_dict, weighter):
    """
    Compute total weighted loss from individual loss components.

    Note: Weights are updated per epoch (not per batch) for adaptive strategy
    to ensure stable training.

    Args:
        loss_dict: Dictionary of individual losses ({'residual': tensor, 'bc_1': tensor, 'bc_2': tensor})
        weighter: LossWeighter instance

    Returns:
        total_loss: Weighted sum of losses
    """
    # Compute weighted total (weights are set externally per epoch for adaptive)
    total_loss = weighter.get_weighted_loss(loss_dict)

    return total_loss
