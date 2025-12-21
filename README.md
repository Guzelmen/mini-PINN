## Mini Physics-Informed Neural Network

Small PINN experiments for my MSci project, refactored into a simple `src/` package with YAML‑driven configs.


### Refactor overview & layout

- Code now lives under `src/` as an importable package:
  - `src/main.py`: training entrypoint (YAML‑driven).
  - `src/trainer.py`: training loop, logging, saving predictions/weights.
  - `src/models.py`: model architectures, selected via `mode` and `phase`.
  - `src/losses.py`: residual and BC loss definitions (by phase).
  - `src/data_utils/`: data loading and preprocessing.
  - `src/eval_predictions.py`: plotting utilities working from saved predictions.
  - `src/infer_model.py`: helper to rebuild a model from a YAML + saved state dict.
  - `src/color_map.py`: builds a 2D grid in \((x,\alpha)\), evaluates a trained phase‑2 model, and saves a residual\(^2\) colormap.
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

   or set `WANDB_MODE=offline` if you don't want online logging.


### Prepare data

- Place a `.pt` data tensor(s) under `data/` (this directory is not tracked by git).
- Point the YAML `data_path` to the file you want to use, for example:

  ```yaml
  data_path: "data/phase_2_final_x_alpha_64k.pt"
  ```


### How to run training

From the project root, call the training entrypoint as a module and pass the YAML stem (without `.yaml`) using `--config`:

```bash
python -m src.main --config phase_2_basic
```

This will:
- Load `src/yamls/phase_2_basic.yaml` via `YParams`.
- Build the model class `Model_{mode}_phase{phase}` from `src/models.py`.
- Load and batch data using `src/data_utils/loader.py`.
- Train using `src/trainer.py`, logging to Weights & Biases.
- Periodically save predictions and (optionally) plots and weight checkpoints, depending on the YAML flags.

You can swap to another experiment just by changing the config name, e.g.:

```bash
python -m src.main --config phase_2_fmt_help
```


### What gets saved where

Exact paths are controlled by YAML keys like `run_name`, `n_vars`, `pred_dir`, `plot_dir`,
`save_weights`, and `save_weights_every`, but the defaults are:

- **Predictions**:
  - During training, every `save_every` epochs the trainer collects predictions and, at the end,
    writes a single pickle file:
    - `predictions/{n_vars}D/{run_name}.pkl`
- **Plots**:
  - If `plot_auto: True` in the YAML, `src/main.py` calls `plot_pred_only` from
    `src/eval_predictions.py` after training, saving:
    - `plot_predictions/{n_vars}D/{run_name}.png`
  - You can also use other helpers in `src/eval_predictions.py` to generate additional plots from a specific prediction file.
- **Weights / checkpoints**:
  - If `save_weights: True` and `save_weights_every` is set, `src/trainer.py` writes state dicts:
    - `saving_weights/{run_name}/weights_epoch_{EPOCH}`


*Created by Guzman Sanchez, 2025.*
