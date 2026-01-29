"""
Helper functions for derivatives..., whatever is needed.
"""
import torch
from pathlib import Path
from typing import Tuple, Union

# Define project root relative to this file (src/utils.py)
# src/utils.py -> src -> root
PROJECT_ROOT = Path(__file__).parent.parent

def first_deriv_auto(out, inp, var1 = 0):
    df_dinp_out = torch.autograd.grad(
        outputs=out,
        inputs=inp,
        grad_outputs=torch.ones_like(out),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=False
    )
    if len(df_dinp_out) == 0 or df_dinp_out[0] is None:
        raise RuntimeError(
            "Second derivative util: first deriv computation failed; check graph connectivity.")
    if var1 == 0:
        first_deriv = df_dinp_out[0][:, 0:1]
    elif var1 == 1:
        first_deriv = df_dinp_out[0][:, 1:2]

    return first_deriv


def sec_deriv_auto(outputs, inp, var1 = 0, var2 = 0):
    # Ensure outputs have shape [batch, 1]
    if len(outputs.shape) == 2 and outputs.shape[1] == 1:
        out = outputs
    elif len(outputs.shape) == 1:
        out = outputs.unsqueeze(1)
    else:
        out = outputs

    df_dinp_out = torch.autograd.grad(
        outputs=out,
        inputs=inp,
        grad_outputs=torch.ones_like(out),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=False
    )
    if len(df_dinp_out) == 0 or df_dinp_out[0] is None:
        raise RuntimeError(
            "Second derivative util: first deriv computation failed; check graph connectivity.")
    if var1 == 0:
        first_deriv = df_dinp_out[0][:, 0:1]
    elif var1 == 1:
        first_deriv = df_dinp_out[0][:, 1:2]

    d2f_dinp_out = torch.autograd.grad(
        outputs=first_deriv,
        inputs=inp,
        grad_outputs=torch.ones_like(first_deriv),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=False
    )
    if len(d2f_dinp_out) == 0 or d2f_dinp_out[0] is None:
        raise RuntimeError(
            "Second derivative util: second deriv computation failed; check graph connectivity.")
    if var2 == 0:
        sec_deriv = d2f_dinp_out[0][:, 0:1]
    elif var2 == 1:
        sec_deriv = d2f_dinp_out[0][:, 1:2]

    return sec_deriv



def fmt_series(x: torch.Tensor, a2: torch.Tensor, alpha: torch.Tensor = None, **kwargs) -> torch.Tensor:
    """
    Helper FMT series form psi should follow.
    a2 is the initial slope, i.e. d psi / d x at x = 0.
    Implements a truncated series:
        psi_fmt(z; a2) = 1
            + a2 * x^1
            + (4/3) * x^(3/2)
            + (2/5 * a2) * x^(5/2)
            + (1/3) * x^3
            + (3/70 * a2) * x^(7/2)
            + (2/15 * a2) * x^4
            + (2/27 - 1/252 * a2^3) * x^(9/2)
            + (1/175 * a2^2) * x^5
            + (31/1485 * a2 + 1/1056 * a2^4) * x^(11/2)
    All operations are torch-based to keep autograd connectivity.
    Note: If alpha is provided, we compute on z = alpha * x; otherwise use x.
    Shapes:
        x      : [batch, 1] or [batch]
        a2     : [batch, 1] or [batch]
        alpha  : [batch, 1] or [batch] (optional)
    Returns:
        psi_fmt: tensor with same broadcastable shape as inputs
    """
    # Ensure tensors are float and broadcastable
    x = x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
    a2 = a2 if isinstance(a2, torch.Tensor) else torch.tensor(a2, dtype=torch.float32)
    if alpha is not None and not isinstance(alpha, torch.Tensor):
        alpha = torch.tensor(alpha, dtype=torch.float32)

    # Make sure dtypes/devices align
    device = x.device
    dtype = x.dtype
    a2 = a2.to(device=device, dtype=dtype)
    if alpha is not None:
        alpha = alpha.to(device=device, dtype=dtype)

    # Use z = alpha * x if alpha is provided, else z = x (backwards compatible)
    z = (alpha * x) if alpha is not None else x

    # Powers of x (fractional exponents)
    x1   = z.pow(1.0)
    x3_2 = z.pow(1.5)
    x5_2 = z.pow(2.5)
    x3   = z.pow(3.0)
    x7_2 = z.pow(3.5)
    x4   = z.pow(4.0)
    x9_2 = z.pow(4.5)
    x5   = z.pow(5.0)
    x11_2= z.pow(5.5)

    # Coefficients (some depend on a2 and its powers)
    term1 = a2 * x1
    term2 = (4.0/3.0) * x3_2
    # term with zero coefficient omitted
    term4 = (2.0/5.0) * a2 * x5_2
    term5 = (1.0/3.0) * x3
    term6 = (3.0/70.0) * a2.pow(2) * x7_2
    term7 = (2.0/15.0) * a2 * x4
    term8 = (2.0/27.0 - (1.0/252.0) * a2.pow(3)) * x9_2
    term9 = (1.0/175.0) * a2.pow(2) * x5
    term10= ((31.0/1485.0) * a2 + (1.0/1056.0) * a2.pow(4)) * x11_2

    psi_fmt = 1.0 + term1 + term2 + term4 + term5 + term6 + term7 + term8 + term9 + term10
    return psi_fmt

def phi_bc_transform_phase4(x: torch.Tensor, N: torch.Tensor, N_prime: torch.Tensor, params) -> torch.Tensor:
    """
    Phi(x) = x^2 + 1 + xN(x) + x(1 - x)N'(x)
    """
    phi = x**2 + 1 + x * N + x * (1 - x) * N_prime
    if params.debug_mode is True:
        print(f"[During fwd] phi -> Min: {phi.min().item()}, Max: {phi.max().item()}")
    return phi
    
def phi_w_transform(x: torch.Tensor, N: torch.Tensor, N_prime: torch.Tensor, params) -> torch.Tensor:
    w = 0.5 * x**2 + x * (2 - x) * N + x * (1 - x) * N_prime
    if params.debug_mode is True:
        print(f"[During fwd] w -> Min: {w.min().item()}, Max: {w.max().item()}")
    phi = torch.exp(w)
    return phi


def phi_0_transform(x: torch.Tensor, alpha: torch.Tensor, k: Union[torch.Tensor, float, int]) -> torch.Tensor:
    c = alpha.pow(1.5)
    y = x.pow(1.5)
    b = 1-x
    d = b.pow(2)
    expy = torch.exp(- k * c * y)
    phi_0 = 4/3 * c * y * expy * d
    return phi_0