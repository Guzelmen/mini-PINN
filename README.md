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


### How it works

- Data
  - Place your pre-generated tensors under `data/` (not tracked in git). Expected key: `'X'` with shape `[N, 2]` or `[N, 3]` depending on the phase.
  - Use `data_utils/generator.py` if you want to create synthetic datasets.
  - Loading and normalization are handled by `data_utils/loader.py` and `data_utils/normalisation.py`.

- Config
  - All runs are configured via YAMLs in `yamls/`. Pick a file (e.g., `phase_2.yaml`) and adjust hyperparameters.
  - Key fields: `phase`, `mode` (`soft`/`hard`), `stage` (`train`/`test`), `data_path`, `batch_size`, LR schedule, wandb fields.

- Train
  - Run: `python main.py --config phase_2`
  - The script loads `yamls/phase_2.yaml`, builds the model as `Model_{mode}_phase{phase}` from `models.py`, loads data, and trains with `trainer.py`.
  - Weights & Biases logs are enabled if configured.

- Models and losses
  - Architectures live in `models.py` with soft/hard variants by phase.
  - Losses (PDE residual + BCs) live in `losses.py`, also split by phase.

- Predictions and plots
  - During/after training, predictions are saved as `.pkl` files under `predictions/` (not tracked).
  - Plot utilities are in `eval_predictions.py`:
    - `plot_pred_phase1`: phase-1 specific plotting plus polynomial fit.
    - `plot_pred_general`: general plotting; optional fitting of several families.
    - `plot_pred_specific`: fit and plot a single chosen family (e.g., `"poly4"`, `"exp"`).
  - Figures are written to `plot_predictions/` (not tracked).

- Not tracked in git
  - `data/*.pt`, `predictions/*.pkl`, `plot_predictions/*.png`, `wandb/` runs, and other large artifacts are ignored via `.gitignore`.


### Setup

1) Create an environment, then install dependencies:

```
pip install -r requirements.txt
```

2) Put your dataset at the path referenced by `data_path` in the YAML (e.g., `data/64k_x_log_r0.pt`).

3) Launch training:

```
python main.py --config phase_2
```

*Created by Guzman Sanchez, 2025.*

