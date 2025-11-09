## Mini Physics-Informed Neural Network

This is the first task for my MSci Project.

### Problem setup

**Phase 1**:

Functions, classes, variables... with a "phase_1" in their name, refer to the first experiment considered, very simplified Thomas Fermi model, by removing the dependence of the RHS on psi/V.

- Uses only x as input

- PDE residual for the loss: $\frac{d^2 f(x)}{dx^2} = 3x$

- f(x) = $\psi$(x) is the transformed version of V(r) variable we care about in TF, to remove dependencies on other values in the residual

- Boundary Conditions:
    - BC 1: $\psi$(0) = 1
    - BC 2: $\psi$'(1) = $\psi$(1)

**Phase 2**:

Functions, classes, variables... with a "phase_2" in their name, refer to the second experiment considered, less simplified but still no T.


### PINN architecture


### Constraint types

*Created by Guzman Sanchez, 2025.*

