"""
Generate Phase 4 training data using the numerical BVP solver.

This uses the generalised_TF.py solver to produce (x, alpha, T, psi) tuples
where psi is the ground-truth solution to the temperature-dependent
Thomas-Fermi equation.
"""
import sys
import math
import numpy as np
import torch
import wandb
from pathlib import Path
from itertools import product

# Add solver paths - need both the examples dir and the parent for FDint_JAX
SOLVER_REPO = '/rds/general/user/gs1622/home/MinimalTFFDintPy'
SOLVER_EXAMPLES = f'{SOLVER_REPO}/examples'

if SOLVER_REPO not in sys.path:
    sys.path.insert(0, SOLVER_REPO)
if SOLVER_EXAMPLES not in sys.path:
    sys.path.insert(0, SOLVER_EXAMPLES)

from generalised_TF import solve_beta_bvp

# Physical constants (same as fd_integrals.py)
A0_M = 5.291772105e-11  # Bohr radius in meters
B_M = 0.25 * (4.5 * math.pi**2)**(1/3) * A0_M  # Length scale b
C0_M = 1.602e-11  # meters


def generate_solution(alpha, T_kV, n_points=100, tol=1e-5, y_guess=None, w_grid=None, max_nodes=int(2e5)):
    """
    Generate (x, psi) solution for given (alpha, T_kV).
    
    Uses the BVP solver in w-coordinates, then converts back to x-coordinates.
    
    Args:
        alpha: dimensionless parameter r0/b
        T_kV: temperature in keV
        n_points: number of x points to return
        tol: solver tolerance
        y_guess: initial guess for the solution (optional)
        w_grid: mesh to use (optional)
        max_nodes: maximum number of mesh nodes for solve_bvp (default: 2e5)
        
    Returns:
        x: numpy array of x values in (0, 1]
        psi: numpy array of psi(x) values
        dpsi/dx: numpy array of dpsi(x)/dx values
        w_used: numpy array of w values used for the solution
        beta: numpy array of beta values
        dbeta: numpy array of dbeta values
    """
    # Compute solver parameters
    gamma = 0.0899 / (T_kV ** 0.75)
    lam = alpha * B_M * (T_kV ** 0.25) / C0_M
    a = gamma       # β(0) = γ since ψ(0) = 1
    wb = np.sqrt(2 * lam)  # w_max = √(2λ) since w² = 2λx and x_max = 1

    # Solve in w-coordinates
    sol = solve_beta_bvp(a, wb, n_points=n_points, tol=tol, y_guess=y_guess, w_grid=w_grid, max_nodes=max_nodes)

    # Use the mesh actually used for the solution
    if w_grid is not None:
        w_used = w_grid
    else:
        w_used = np.linspace(1e-6, wb, n_points)
    beta = sol.sol(w_used)[0]
    dbeta = sol.sol(w_used)[1]

    # Convert to x-coordinates: x = w²/(2λ)
    x = w_used**2 / (2 * lam)
    # Convert to psi: ψ = β/γ
    psi = beta / gamma
    # Calculate dψ/dx too from dβ/dw
    # dψ/dx = (dβ/dw / γ) * (λ/w)
    w_safe = np.maximum(w_used, 1e-10)
    dpsi_dx = (dbeta/ gamma) * (lam / w_safe)
    

    return x, psi, dpsi_dx, w_used, beta, dbeta


def generate_phase4_solver(
    out_path=None,
    seed=42,
    n_x=160,
    n_alpha=40,
    n_T=40,
    tol=1e-5,
    max_nodes=int(2e5)
):
    """
    Generate Phase 4 dataset using the numerical solver.
    
    Uses same ranges as generate_phase4_small:
        - x: from solver's natural grid (maps to ~[0, 1])
        - alpha: ~1-5 (r0 from 5e-11 to 2.5e-10 m)
        - T: 1-10 keV
        
    Saves dict with 'inputs' [x, alpha, T] and 'targets' [psi].
    Logs to wandb to check progress.
    
    Args:
        out_path: where to save the .pt file (default: data/phase_4_solver.pt)
        seed: random seed (not used for solver, but for consistency)
        n_x: number of x points per (alpha, T) pair
        n_alpha: number of alpha values
        n_T: number of temperature values
        tol: solver tolerance
        max_nodes: maximum number of mesh nodes for solve_bvp (default: 2e5)
    Returns:
        inputs: torch tensor of shape [N, 3] with columns [x, alpha, T]
        targets: torch tensor of shape [N, 2] with psi  and dpsi/dx values
    """
    if out_path is None:
        from ..utils import PROJECT_ROOT
        out_path = PROJECT_ROOT / "data/phase_4_solver_doubletargets_smallrange.pt"
    
    np.random.seed(seed)

    # Initialize wandb
    wandb.login()
    wandb.init(
        name="phase_4_solver_v4_smallrange_newaddedtarget",
        notes=f"Alpha ~1-5 (r0 from 5e-11 to 2.5e-10 m) - {n_alpha} vals, T 1-10 keV - {n_T} vals, x ~0-1 (natural solver grid) - {n_x} vals, tol={tol}. \
        Using previous solutions for initial guess. Max nodes={max_nodes}. Now targets are psi and dpsi/dx, used ffor auxiliary data loss function in training.",
        group="week9feb",
        project="data_generation",
        entity="guzelmen_msci_project",
    )
    
    # alpha: small range (r0 from 5e-11 to 2.5e-10 m -> alpha ~ 1 to 5)
    r0_vals = np.logspace(np.log10(5e-11), np.log10(2.5e-10), n_alpha)
    alpha_vals = r0_vals / B_M
    
    # T: narrow range 1 to 10 keV
    T_vals = np.logspace(np.log10(1.0), np.log10(10.0), n_T)
    
    data = []
    failed = 0
    completed = 0
    
    total = n_alpha * n_T
    print(f"Generating solutions for {n_alpha} alpha × {n_T} T = {total} (α, T) pairs...")
    print(f"  alpha range: [{alpha_vals.min():.3f}, {alpha_vals.max():.3f}]")
    print(f"  T range: [{T_vals.min():.3f}, {T_vals.max():.3f}] keV")
    
    wandb.log({"total_solves_to_do": n_alpha * n_T})
    
    # --- Store previous successful solutions for initial guess reuse ---
    solved = []  # Each entry: {'alpha': ..., 'T': ..., 'w': ..., 'beta': ..., 'dbeta': ...}

    for i, (alpha, T) in enumerate(product(alpha_vals, T_vals)):
            # Compute solver parameters for mesh
        gamma = 0.0899 / (T ** 0.75)
        lam = alpha * B_M * (T ** 0.25) / C0_M
        a = gamma
        wb = np.sqrt(2 * lam)
        w_grid = np.linspace(1e-6, wb, n_x)

        # --- Find closest solved (alpha, T) for initial guess ---
        if solved:
            closest = min(solved, key=lambda s: (s['alpha']-alpha)**2 + (s['T']-T)**2)
            beta_guess = np.interp(w_grid, closest['w'], closest['beta'])
            dbeta_guess = np.interp(w_grid, closest['w'], closest['dbeta'])
            y_guess = np.vstack((beta_guess, dbeta_guess))
        else:
            # Default initial guess as in generalised_TF.py
            beta_guess = a * np.ones_like(w_grid)
            dbeta_guess = (2.0 * a / wb) * (w_grid / wb)
            y_guess = np.vstack((beta_guess, dbeta_guess))

        try:
            # Pass y_guess to solve_beta_bvp (need to add y_guess param to that function)
            x, psi, dpsi_dx, w_used, beta_used, dbeta_used = generate_solution(alpha, T, n_points=n_x, tol=tol, y_guess=y_guess, w_grid=w_grid, max_nodes=max_nodes)
            for xi, psii, dpsii in zip(x, psi, dpsi_dx):
                data.append([xi, alpha, T, psii, dpsii])
            completed += 1
            solved.append({'alpha': alpha, 'T': T, 'w': w_grid, 'beta': beta_used, 'dbeta': dbeta_used})
            wandb.log({"completed_pairs": completed, "attempted_pairs": i + 1, "failed_pairs": failed})
            if completed % 100 == 0:
                print(f"  Completed {completed}/{i + 1} pairs (attempted)...")
        except RuntimeError as e:
            failed += 1
            wandb.log({"failed_pairs": failed, "attempted_pairs": i + 1, "failed_alpha": alpha, "failed_T": T, "completed_pairs": completed})
            print(f"  Failed for alpha={alpha:.3f}, T={T:.3f}: {e}")
    
    if failed > 0:
        print(f"Warning: {failed}/{total} (α, T) pairs failed to solve")
    
    data = np.array(data)
    inputs = torch.tensor(data[:, :3], dtype=torch.float32)  # [x, alpha, T]
    targets = torch.tensor(data[:, 3:5], dtype=torch.float32)  # [psi, dpsi_dx]
    
    # Save
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'inputs': inputs, 'targets': targets}, out_path)
    
    print(f"\nSaved {len(data)} points to {out_path}")
    print(f"  inputs shape: {tuple(inputs.shape)}")
    print(f"  targets shape: {tuple(targets.shape)}")
    print(f"  x range: [{inputs[:, 0].min():.6f}, {inputs[:, 0].max():.6f}]")
    print(f"  psi range: [{targets[:, 0].min():.6f}, {targets[:, 0].max():.6f}]")
    print(f"  dpsi/dx range: [{targets[:, 1].min():.6f}, {targets[:, 1].max():.6f}]")
    
    wandb.finish()

    return inputs, targets


if __name__ == "__main__":
    generate_phase4_solver()
