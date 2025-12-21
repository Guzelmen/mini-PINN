"""
Run inference with a loaded model and plot predictions (single plot).
This script:
  - Loads a model via infer_model.load_model_from_config_and_state
  - Reads inputs (x, alpha) from a CSV
  - Runs the model to produce psi(x) predictions
  - Plots psi vs x similarly to eval_predictions.plot_pred_only, but for a single set
"""
import argparse
from pathlib import Path
from typing import Tuple, List, Any
import ast

import numpy as np
import torch
import matplotlib.pyplot as plt

from .utils import PROJECT_ROOT
from .infer_model import load_model_from_config_and_state


def read_inputs_csv(csv_path: str) -> np.ndarray:
    """
    Read inputs from CSV with two columns: x, alpha (no header).
    Returns ndarray of shape [N, 2].
    """
    arr = np.loadtxt(csv_path, delimiter=",")
    if arr.ndim == 1:
        # Single row
        if arr.size != 2:
            raise ValueError("Expected 2 values (x,alpha) in CSV row.")
        arr = arr.reshape(1, 2)
    if arr.shape[1] != 2:
        # Try whitespace-delimited as a fallback
        arr_ws = np.loadtxt(csv_path)
        if arr_ws.ndim == 1 and arr_ws.size == 2:
            arr = arr_ws.reshape(1, 2)
        elif arr_ws.ndim == 2 and arr_ws.shape[1] == 2:
            arr = arr_ws
        else:
            raise ValueError("Input CSV must have exactly 2 columns: x,alpha.")
    return arr


def read_inputs_array_string(array_str: str) -> np.ndarray:
    """
    Parse a 2D array from a string typed in the terminal, e.g.:
      --array '[[0.1, 1e-9], [0.2, 5e-9]]'
    Returns ndarray of shape [N, 2].
    """
    try:
        parsed = ast.literal_eval(array_str)
    except Exception as e:
        raise ValueError(f"Failed to parse --array string: {e}")
    arr = np.array(parsed, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Parsed --array must be a 2D structure with shape [N, 2].")
    return arr


def plot_single_prediction(x: np.ndarray, y: np.ndarray, params, save_path: Path) -> None:
    """
    Plot a single scatter of psi(x) vs x, mirroring eval_predictions.plot_pred_only styling.
    """
    sort_idx = np.argsort(x)
    xs = x[sort_idx]
    ys = y[sort_idx]

    fig, ax = plt.subplots(figsize=(6, 4))

    x_type = getattr(params, "x_type", "normal")
    if x_type == "log":
        ax.set_xscale('log')
        ax.set_xlim(1e-12, 1.0)

    ax.plot(xs, ys, '+', color="blue", ms=0.5, label=r'$\psi(x)$')
    ax.set_xlabel('x' if x_type != "log" else 'x (log scale)')
    ax.set_ylabel(r'$\psi(x)$')
    ax.set_title(f"{params.run_name} (inference)")
    ax.legend(loc='best', framealpha=0.9)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Inference and single-plot prediction.")
    parser.add_argument("--config", required=True, help="YAML config name in src/yamls (without .yaml).")
    parser.add_argument("--state", required=True, help="Path to torch state dict file.")
    parser.add_argument("--input-type", required=True, choices=["single", "csv", "array"],
                        help="Type of input: 'single' for one (x,alpha), 'csv' for CSV file, 'array' for inline 2D array.")
    parser.add_argument("--input-csv", default=None, help="CSV with two columns: x,alpha (no header). Used when --input-type=csv.")
    parser.add_argument("--x", type=float, default=None, help="x value for single input. Used when --input-type=single.")
    parser.add_argument("--alpha", type=float, default=None, help="alpha value for single input. Used when --input-type=single.")
    parser.add_argument("--array", type=str, default=None,
                        help="Inline 2D array string like '[[x1,a1],[x2,a2],...]'. Used when --input-type=array.")
    parser.add_argument("--device", default="cpu", help='Target device, e.g. "cpu" or "cuda:0".')
    parser.add_argument("--out", default=None, help="Optional output PNG path. Defaults under plot_predictions.")
    args = parser.parse_args()

    # Load model/params
    model, params = load_model_from_config_and_state(
        config_name=args.config,
        state_path=args.state,
        device=args.device,
    )

    device = torch.device(args.device)

    if args.input_type == "single":
        if args.x is None or args.alpha is None:
            raise ValueError("--x and --alpha are required when --input-type=single.")
        inputs_np = np.array([[args.x, args.alpha]], dtype=np.float32)
        inputs_t = torch.from_numpy(inputs_np).to(device)
        with torch.no_grad():
            outputs_t = model(inputs_t)
        psi = float(outputs_t.detach().cpu().numpy().reshape(-1)[0])
        print(f"Input: x={args.x:.6g}, alpha={args.alpha:.6g} -> psi(x)={psi:.6g}")
        return

    elif args.input_type == "csv":
        if not args.input_csv:
            raise ValueError("--input-csv is required when --input-type=csv.")
        inputs_np = read_inputs_csv(args.input_csv).astype(np.float32)
    else:  # array
        if not args.array:
            raise ValueError("--array is required when --input-type=array.")
        inputs_np = read_inputs_array_string(args.array).astype(np.float32)

    inputs_t = torch.from_numpy(inputs_np).to(device)
    with torch.no_grad():
        outputs_t = model(inputs_t)
    outputs_np = outputs_t.detach().cpu().numpy().reshape(-1)

    # For csv/array modes: plot the predictions
    x_np = inputs_np[:, 0].reshape(-1)
    if args.out is not None:
        save_path = Path(args.out)
    else:
        save_path = PROJECT_ROOT / params.plot_dir / f"{params.n_vars}D" / f"{params.run_name}_inference.png"
    plot_single_prediction(x_np, outputs_np, params, save_path)
    print(f"Saved inference plot to {save_path}")


if __name__ == "__main__":
    main()


