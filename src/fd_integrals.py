"""
Fermi-Dirac integral wrapper for the Thomas-Fermi PINN.

This module provides access to Fermi-Dirac integrals (F_{-1/2}, F_{1/2}, F_{3/2})
using Fukushima's rational approximations, implemented in PyTorch with custom
backward passes for automatic differentiation.

Source: MinimalTFFDintPy repository by Aidan Sherlock
Location: /rds/general/user/gs1622/home/MinimalTFFDintPy

IMPORTANT: The FMT paper (Feynman-Metropolis-Teller) uses UNNORMALIZED Fermi-Dirac
integrals. The original Fukushima implementation divides by Gamma factors. We
remove that normalization here to match FMT conventions:
    - F_{1/2}(η) → (2/3) η^(3/2) as η → ∞  (NOT (4/3√π) η^(3/2))

Physical constants for the temperature-dependent Thomas-Fermi equation:
    - c0 = 1.602e-9 cm = 1.602e-11 m (temperature-dependent length scale coefficient)
    - γ = 0.0899 * Z / T_kV^(3/4)  (reduced potential at origin, Z=1 for hydrogen)
    - λ = α * b * T_kV^(1/4) / c0   (dimensionless length scale ratio)
"""
import sys
import math

# Physical constants
C0_CM = 1.602e-9      # cm, from FMT Eq. 16
C0_M = 1.602e-11      # meters
A0_M = 5.291772105e-11  # Bohr radius in meters
B_M = 0.25 * (4.5 * math.pi**2)**(1/3) * A0_M  # Our length scale b in meters

# Add the external FDint repo to path
FDINT_PATH = '/rds/general/user/gs1622/home/MinimalTFFDintPy'
if FDINT_PATH not in sys.path:
    sys.path.insert(0, FDINT_PATH)

import torch
from torch.autograd import Function

# Import the underlying Fukushima approximation functions (not the final wrapped versions)
from FDint_PyTorch import (
    _fukushima_fdi_m0p5_region_1, _fukushima_fdi_m0p5_region_2,
    _fukushima_fdi_m0p5_region_3, _fukushima_fdi_m0p5_region_4,
    _fukushima_fdi_m0p5_region_5, _fukushima_fdi_m0p5_region_6,
    _fukushima_fdi_m0p5_region_7, _fukushima_fdi_m0p5_region_8,
    _fukushima_fdi_p0p5_region_1, _fukushima_fdi_p0p5_region_2,
    _fukushima_fdi_p0p5_region_3, _fukushima_fdi_p0p5_region_4,
    _fukushima_fdi_p0p5_region_5, _fukushima_fdi_p0p5_region_6,
    _fukushima_fdi_p0p5_region_7, _fukushima_fdi_p0p5_region_8,
    _fukushima_fdi_p1p5_region_1, _fukushima_fdi_p1p5_region_2,
    _fukushima_fdi_p1p5_region_3, _fukushima_fdi_p1p5_region_4,
    _fukushima_fdi_p1p5_region_5, _fukushima_fdi_p1p5_region_6,
    _fukushima_fdi_p1p5_region_7, _fukushima_fdi_p1p5_region_8,
)


def _fermi_dirac_minus_half_unnorm(x):
    """
    Compute F_{-1/2}(x) WITHOUT Gamma function normalization.
    This matches FMT convention.
    """
    x1, x2, x3, x4, x5, x6, x7 = -2.0, 0.0, 2.0, 5.0, 10.0, 20.0, 40.0

    r1 = x < x1
    r2 = (x >= x1) & (x < x2)
    r3 = (x >= x2) & (x < x3)
    r4 = (x >= x3) & (x < x4)
    r5 = (x >= x4) & (x < x5)
    r6 = (x >= x5) & (x < x6)
    r7 = (x >= x6) & (x < x7)
    r8 = x >= x7

    res = torch.zeros_like(x, dtype=x.dtype)

    if r1.any():
        res[r1] = _fukushima_fdi_m0p5_region_1(x[r1])
    if r2.any():
        res[r2] = _fukushima_fdi_m0p5_region_2(x[r2])
    if r3.any():
        res[r3] = _fukushima_fdi_m0p5_region_3(x[r3])
    if r4.any():
        res[r4] = _fukushima_fdi_m0p5_region_4(x[r4])
    if r5.any():
        res[r5] = _fukushima_fdi_m0p5_region_5(x[r5])
    if r6.any():
        res[r6] = _fukushima_fdi_m0p5_region_6(x[r6])
    if r7.any():
        res[r7] = _fukushima_fdi_m0p5_region_7(x[r7])
    if r8.any():
        res[r8] = _fukushima_fdi_m0p5_region_8(x[r8])

    # NO division by Gamma(1/2) = sqrt(pi) - matches FMT convention
    return res


def _fermi_dirac_half_unnorm(x):
    """
    Compute F_{1/2}(x) WITHOUT Gamma function normalization.
    This matches FMT convention: F_{1/2}(η) → (2/3) η^(3/2) as η → ∞
    """
    x1, x2, x3, x4, x5, x6, x7 = -2.0, 0.0, 2.0, 5.0, 10.0, 20.0, 40.0

    r1 = x < x1
    r2 = (x >= x1) & (x < x2)
    r3 = (x >= x2) & (x < x3)
    r4 = (x >= x3) & (x < x4)
    r5 = (x >= x4) & (x < x5)
    r6 = (x >= x5) & (x < x6)
    r7 = (x >= x6) & (x < x7)
    r8 = x >= x7

    res = torch.zeros_like(x, dtype=x.dtype)

    if r1.any():
        res[r1] = _fukushima_fdi_p0p5_region_1(x[r1])
    if r2.any():
        res[r2] = _fukushima_fdi_p0p5_region_2(x[r2])
    if r3.any():
        res[r3] = _fukushima_fdi_p0p5_region_3(x[r3])
    if r4.any():
        res[r4] = _fukushima_fdi_p0p5_region_4(x[r4])
    if r5.any():
        res[r5] = _fukushima_fdi_p0p5_region_5(x[r5])
    if r6.any():
        res[r6] = _fukushima_fdi_p0p5_region_6(x[r6])
    if r7.any():
        res[r7] = _fukushima_fdi_p0p5_region_7(x[r7])
    if r8.any():
        res[r8] = _fukushima_fdi_p0p5_region_8(x[r8])

    # NO division by Gamma(3/2) = 0.5*sqrt(pi) - matches FMT convention
    return res


def _fermi_dirac_three_half_unnorm(x):
    """
    Compute F_{3/2}(x) WITHOUT Gamma function normalization.
    This matches FMT convention.
    """
    x1, x2, x3, x4, x5, x6, x7 = -2.0, 0.0, 2.0, 5.0, 10.0, 20.0, 40.0

    r1 = x < x1
    r2 = (x >= x1) & (x < x2)
    r3 = (x >= x2) & (x < x3)
    r4 = (x >= x3) & (x < x4)
    r5 = (x >= x4) & (x < x5)
    r6 = (x >= x5) & (x < x6)
    r7 = (x >= x6) & (x < x7)
    r8 = x >= x7

    res = torch.zeros_like(x, dtype=x.dtype)

    if r1.any():
        res[r1] = _fukushima_fdi_p1p5_region_1(x[r1])
    if r2.any():
        res[r2] = _fukushima_fdi_p1p5_region_2(x[r2])
    if r3.any():
        res[r3] = _fukushima_fdi_p1p5_region_3(x[r3])
    if r4.any():
        res[r4] = _fukushima_fdi_p1p5_region_4(x[r4])
    if r5.any():
        res[r5] = _fukushima_fdi_p1p5_region_5(x[r5])
    if r6.any():
        res[r6] = _fukushima_fdi_p1p5_region_6(x[r6])
    if r7.any():
        res[r7] = _fukushima_fdi_p1p5_region_7(x[r7])
    if r8.any():
        res[r8] = _fukushima_fdi_p1p5_region_8(x[r8])

    # NO division by Gamma(5/2) = 1.5*0.5*sqrt(pi) - matches FMT convention
    return res


class FermiDiracHalfUnnorm(Function):
    """
    Custom autograd Function for F_{1/2}(η) with proper backward pass.
    Uses the recurrence: d/dη F_{1/2}(η) = F_{-1/2}(η)
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _fermi_dirac_half_unnorm(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # d/dη F_{1/2}(η) = F_{-1/2}(η)
        grad_input = grad_output * _fermi_dirac_minus_half_unnorm(x)
        return grad_input


class FermiDiracThreeHalfUnnorm(Function):
    """
    Custom autograd Function for F_{3/2}(η) with proper backward pass.
    Uses the recurrence: d/dη F_{3/2}(η) = F_{1/2}(η)
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _fermi_dirac_three_half_unnorm(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # d/dη F_{3/2}(η) = F_{1/2}(η)
        grad_input = grad_output * _fermi_dirac_half_unnorm(x)
        return grad_input


# Public API - these are the functions to use
fermi_dirac_half = FermiDiracHalfUnnorm.apply
fermi_dirac_three_half = FermiDiracThreeHalfUnnorm.apply
fermi_dirac_minus_half = _fermi_dirac_minus_half_unnorm  # No custom grad needed (not differentiated through)


def compute_lambda(alpha, T_kV):
    """
    Compute λ = α * b * T_kV^(1/4) / c0

    Args:
        alpha: Tensor of α = r0/b values
        T_kV: Tensor of temperature values in keV

    Returns:
        λ tensor (same shape as inputs after broadcasting)
    """
    return alpha * B_M * (T_kV ** 0.25) / C0_M


def compute_gamma(T_kV, Z=1):
    """
    Compute γ = 0.0899 * Z / T_kV^(3/4)

    Args:
        T_kV: Tensor of temperature values in keV
        Z: Atomic number (default 1 for hydrogen)

    Returns:
        γ tensor (same shape as T_kV)
    """
    return 0.0899 * Z / (T_kV ** 0.75)
