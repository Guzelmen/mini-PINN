## Physics-Informed Neural Network Solver for Thomas-Fermi

PINN solver for my MSci project, refactored into a simple `src/` package with YAML‑driven configs.


### Overview & layout

- Code lives under `src/` as an importable package. Key files:
  - `src/main.py`: training entrypoint (YAML‑driven).
  - `src/trainer.py`: training loop, logging, saving predictions/weights.
  - `src/models.py`: model architectures, selected via `mode` and `phase`.
  - `src/losses.py`: residual, BC, target loss definitions (by phase).
  - `src/data_utils/`: data loading and preprocessing, targets for T>0 experiments (phase 4 only).
  - `src/eval_predictions.py`: plotting utilities working from saved predictions.
  - `src/fd_integrals.py`: Fermi-Dirac integral wrapper for the Thomas-Fermi PINN. Source: MinimalTFFDintPy repository by Aidan Crilly, accessed locally in this repository (phase 4 only).
- All runs are configured through YAML files in `src/yamls/`.


### Setup

1. Create and activate a Python environment.
2. Install dependencies from the project root:

   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Log in to Weights & Biases once:

   ```bash
   wandb login
   ```

   or set/add `use_wandb = False` in config if you don't want online logging.


### Prepare data

- Place a `.pt` data tensor(s) under `data/` (this directory is not tracked by git).
- Data files supported are either inputs-only (raw tensor) or a dict `{'inputs': ..., 'targets': ...}` with solver targets. The YAML flags `n_vars`, `inp_dim`, `use_solver_data`, `use_deriv_loss`, `hybrid_training`... must match what the file actually contains.
- Point the YAML `data_path` to the file you want to use, for example:

  ```yaml
  data_path: "data/phase_2_final_x_alpha_64k.pt"
  ```


### How to run training

From the project root, call the training entrypoint as a module and pass the YAML stem (without `.yaml`) using `--config`:

```bash
python -m src.main --config CONFIG_NAME
```

This will:
- Load `src/yamls/CONFIG_NAME.yaml` via `YParams`.
- Build the model class from `src/models.py`.
- Load and batch data using `src/data_utils/loader.py`.
- Train using `src/trainer.py`, optionally logging metrics to Weights & Biases.
- Periodically save predictions and (optionally) plots and weight checkpoints, depending on the YAML flags.

You can swap to another experiment just by changing the config name.






*Created by Guzman Sanchez, 2025.*