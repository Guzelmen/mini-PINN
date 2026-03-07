"""
Standalone script to benchmark inference time of a trained PINN model.

Loads model config from wandb and weights from a local checkpoint,
then times forward passes on a real input dataset for both CPU and GPU.

Usage:
    python -m src.time_inference \
        --wandb_run_path "guzelmen_msci_project/mini_pinn/<run_id>" \
        --run_name <run_name> \
        --data_path "data/phase_4_solver_doubletargets_smallrange.pt" \
        [--n_warmup 10] [--n_runs 100]
"""
import sys
import argparse
import time
from types import SimpleNamespace

import torch

from .. import models
from ..utils import PROJECT_ROOT
from .color_map import find_latest_state_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark inference time of a trained PINN model."
    )
    parser.add_argument(
        "--wandb_run_path", required=True,
        help="Full wandb run path: entity/project/run_id",
    )
    parser.add_argument(
        "--run_name", required=True,
        help="Local run name (subfolder in saving_weights/).",
    )
    parser.add_argument(
        "--data_path", required=True,
        help="Path to .pt data file (relative to PROJECT_ROOT or absolute).",
    )
    parser.add_argument("--n_warmup", type=int, default=10, help="Number of warmup forward passes.")
    parser.add_argument("--n_runs", type=int, default=100, help="Number of timed forward passes.")
    return parser.parse_args()


def fetch_config_from_wandb(wandb_run_path: str) -> dict:
    """Fetch the run config dict from the wandb API."""
    try:
        import wandb
        api = wandb.Api()
        wandb_run = api.run(wandb_run_path)
        return dict(wandb_run.config)
    except Exception as e:
        print(f"ERROR: Could not fetch config from wandb ({wandb_run_path}): {e}")
        sys.exit(1)


def build_model(params, state_dict, device):
    """
    Construct the model from params + state_dict, aligning normalization
    buffers with the checkpoint, then load weights.
    """
    sd_keys = set(state_dict.keys())
    if "a_mean" in sd_keys and "a_std" in sd_keys:
        params.norm_mode = "standardize"
        try:
            val = state_dict["a_mean"]
            params.standard_mean = float(val.item() if hasattr(val, "item") else val)
            val = state_dict["a_std"]
            params.standard_std = float(val.item() if hasattr(val, "item") else val)
        except Exception:
            pass
    elif "a_min" in sd_keys and "a_max" in sd_keys:
        params.norm_mode = "minmax"

    if int(params.phase) == 4:
        if "T_mean" in sd_keys and "T_std" in sd_keys:
            try:
                val = state_dict["T_mean"]
                params.T_mean = float(val.item() if hasattr(val, "item") else val)
                val = state_dict["T_std"]
                params.T_std = float(val.item() if hasattr(val, "item") else val)
            except Exception:
                pass

    mode = str(params.mode).strip()
    phase = int(params.phase)
    model_class_name = f"Model_{mode}_phase{phase}"
    if not hasattr(models, model_class_name):
        raise ValueError(
            f"Model class '{model_class_name}' not found in src/models.py "
            f"(mode={mode}, phase={phase})."
        )
    ModelClass = getattr(models, model_class_name)
    model = ModelClass(params)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys in state dict: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys in state dict: {unexpected}")

    model.to(device)
    model.eval()

    # Print loaded model summary
    print(f"\n=== Model loaded: {model_class_name} ===")
    print(f"  mode={mode}, phase={phase}")
    print(f"  hidden={getattr(params, 'hidden', '?')}, nlayers={getattr(params, 'nlayers', '?')}")
    print(f"  activation={getattr(params, 'activation', '?')}, w0={getattr(params, 'w0', '?')}")
    print(f"  inp_dim={getattr(params, 'inp_dim', '?')}, norm_mode={getattr(params, 'norm_mode', '?')}")
    print(f"  add_phi0={getattr(params, 'add_phi0', '?')}, k_val={getattr(params, 'k_val', '?')}")
    if getattr(params, 'norm_mode', None) == "standardize":
        print(f"  standard_mean={getattr(params, 'standard_mean', '?')}, standard_std={getattr(params, 'standard_std', '?')}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  total parameters: {n_params}")
    print()

    return model


def load_inputs(data_path: str, device: torch.device) -> torch.Tensor:
    """Load input tensor from a .pt data file."""
    path = PROJECT_ROOT / data_path if not data_path.startswith("/") else data_path
    data = torch.load(path, map_location="cpu")

    if isinstance(data, dict) and "inputs" in data:
        inputs = data["inputs"]
    elif isinstance(data, dict):
        # Try to find the input tensor
        for key in ["train", "X_train", "x"]:
            if key in data:
                inputs = data[key]
                break
        else:
            raise ValueError(f"Could not find input tensor in {path}. Keys: {list(data.keys())}")
    elif isinstance(data, torch.Tensor):
        inputs = data
    else:
        raise ValueError(f"Unexpected data format in {path}: {type(data)}")

    inputs = inputs.to(device)
    print(f"Loaded inputs: {inputs.shape} from {path}")
    return inputs


def time_on_device(model, inputs, device, device_name, n_warmup, n_runs):
    """Time forward passes on a specific device."""
    use_cuda = device.type == "cuda"

    model = model.to(device)
    inputs_dev = inputs.to(device)

    # Warmup
    for _ in range(n_warmup):
        inp = inputs_dev.detach().requires_grad_(True)
        _ = model(inp)
    if use_cuda:
        torch.cuda.synchronize()

    # Timed runs
    if use_cuda:
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_runs):
        inp = inputs_dev.detach().requires_grad_(True)
        _ = model(inp)
        if use_cuda:
            torch.cuda.synchronize()
    elapsed = (time.time() - start) / n_runs

    n_points = len(inputs_dev)
    throughput = n_points / elapsed

    print(f"\n=== {device_name} Timing ===")
    print(f"  Device: {device}")
    print(f"  Input points: {n_points}")
    print(f"  Warmup runs: {n_warmup}")
    print(f"  Timed runs: {n_runs}")
    print(f"  Mean time per forward pass: {elapsed*1000:.2f} ms")
    print(f"  Throughput: {throughput:.0f} points/s ({throughput/1e6:.2f} M points/s)")
    if inputs_dev.shape[1] >= 3:
        # Count unique (alpha, T) pairs directly from the data
        alpha_T = inputs_dev[:, 1:]
        n_pairs = len(torch.unique(alpha_T, dim=0))
        print(f"  {n_pairs} unique (α,T) pairs, {n_points} total points")

    return elapsed


def main():
    args = parse_args()

    # --- Fetch config from wandb ---
    config = fetch_config_from_wandb(args.wandb_run_path)
    params = SimpleNamespace(**config)
    if not hasattr(params, "debug_mode"):
        params.debug_mode = False

    # --- Load weights ---
    state_path = find_latest_state_path(args.run_name)
    print(f"Loading checkpoint: {state_path}")
    state_dict = torch.load(state_path, map_location="cpu")

    # --- Load inputs ---
    inputs = load_inputs(args.data_path, device=torch.device("cpu"))

    # --- CPU timing ---
    cpu_device = torch.device("cpu")
    params.device = cpu_device
    model = build_model(params, state_dict, cpu_device)
    cpu_time = time_on_device(model, inputs, cpu_device, "CPU", args.n_warmup, args.n_runs)

    # --- GPU timing (if available) ---
    if torch.cuda.is_available():
        gpu_device = torch.device("cuda")
        params.device = gpu_device
        model_gpu = build_model(params, state_dict, gpu_device)
        gpu_time = time_on_device(model_gpu, inputs, gpu_device, "GPU", args.n_warmup, args.n_runs)

        print(f"\n=== Speedup ===")
        print(f"  GPU speedup over CPU: {cpu_time/gpu_time:.1f}x")
    else:
        print("\nNo GPU available, skipping GPU timing.")

    print("\nDone.")


if __name__ == "__main__":
    main()
