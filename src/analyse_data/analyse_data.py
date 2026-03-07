import argparse
import os
import sys
from pathlib import Path
import torch

INPUT_NAMES = ["x", "alpha", "T_kV"]
TARGET_NAMES = ["psi", "dpsi_dx"]
PROJECT_ROOT = Path(__file__).parent.parent


def analyse_dataset(path: str) -> None:
    """Detailed analysis of a .pt dataset with {inputs, targets} structure."""
    if not os.path.exists(path):
        print(f"Error: file not found: {path}")
        return

    obj = torch.load(path, map_location="cpu")
    print(f"{'=' * 60}")
    print(f"Dataset: {path}")
    print(f"{'=' * 60}")

    if not isinstance(obj, dict) or "inputs" not in obj or "targets" not in obj:
        print("Expected dict with 'inputs' and 'targets' keys.")
        print(f"Got: {type(obj)}, keys={list(obj.keys()) if isinstance(obj, dict) else 'N/A'}")
        return

    inputs = obj["inputs"]
    targets = obj["targets"]
    print(f"inputs:  shape={tuple(inputs.shape)}, dtype={inputs.dtype}")
    print(f"targets: shape={tuple(targets.shape)}, dtype={targets.dtype}")
    print(f"Total samples: {inputs.shape[0]}")

    # --- Inputs ---
    print(f"\n{'─' * 40}")
    print("INPUTS")
    print(f"{'─' * 40}")
    n_cols = inputs.shape[1]
    for i in range(n_cols):
        col = inputs[:, i]
        name = INPUT_NAMES[i] if i < len(INPUT_NAMES) else f"col[{i}]"
        vals = col.unique().sort().values
        n_unique = len(vals)

        print(f"\n  [{i}] {name}")
        print(f"      range:   [{float(vals[0]):.6g}, {float(vals[-1]):.6g}]")
        print(f"      unique:  {n_unique}")
        print(f"      mean:    {float(col.mean()):.6g}")

        # First / last values
        n_show = min(6, n_unique)
        first_str = ", ".join(f"{float(v):.6g}" for v in vals[:n_show])
        last_str = ", ".join(f"{float(v):.6g}" for v in vals[-n_show:])
        print(f"      first {n_show}: [{first_str}]")
        print(f"      last  {n_show}: [{last_str}]")

        # Detect spacing
        if n_unique >= 3:
            diffs = vals[1:] - vals[:-1]
            lin_step = float(diffs[0])
            lin_var = float((diffs - diffs.mean()).abs().max())

            ratios = vals[1:] / vals[:-1]
            log_ratio = float(ratios[0])
            log_var = float((ratios - ratios.mean()).abs().max())

            if lin_var / (abs(lin_step) + 1e-30) < 1e-4:
                print(f"      spacing: LINEAR  (step={lin_step:.6g})")
            elif log_var / (abs(log_ratio) + 1e-30) < 1e-4:
                print(f"      spacing: LOG  (ratio={log_ratio:.6g})")
            else:
                print(f"      spacing: IRREGULAR  (lin_var={lin_var:.4g}, log_var={log_var:.4g})")

        # x-specific: count values below 1e-6
        if i == 0:
            n_below = int((col < 1e-6).sum())
            print(f"      x < 1e-6: {n_below} / {len(col)} samples")

    # --- Targets ---
    print(f"\n{'─' * 40}")
    print("TARGETS")
    print(f"{'─' * 40}")
    n_tcols = targets.shape[1]
    for i in range(n_tcols):
        col = targets[:, i]
        name = TARGET_NAMES[i] if i < len(TARGET_NAMES) else f"col[{i}]"
        print(f"\n  [{i}] {name}")
        print(f"      range: [{float(col.min()):.6g}, {float(col.max()):.6g}]")
        print(f"      mean:  {float(col.mean()):.6g}")
        print(f"      std:   {float(col.std()):.6g}")
        # Check for NaN/Inf
        n_nan = int(col.isnan().sum())
        n_inf = int(col.isinf().sum())
        if n_nan or n_inf:
            print(f"      WARNING: {n_nan} NaN, {n_inf} Inf")

    print(f"\n{'=' * 60}\n")


def describe_tensor(name: str, t: torch.Tensor, max_rows: int = 5) -> None:
    print(f"{name}: dtype={t.dtype}, shape={tuple(t.shape)}")
    if t.ndim == 2 and t.shape[1] in (1, 2, 3):
        # Column-wise stats for up to first 3 columns
        for i in range(min(t.shape[1], 3)):
            col = t[:, i]
            print(
                f"  col[{i}] min={float(col.min()):.6g} max={float(col.max()):.6g} mean={float(col.mean()):.6g}")
    # Show a small head preview
    rows = min(t.shape[0], max_rows)
    print(f"first {rows} rows:")
    print(t[:rows])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a .pt dataset under data/")
    parser.add_argument(
        "path",
        metavar="PATH",
        type=str,
        help="Path to .pt file (e.g., 64k_x_r0_log.pt)",
    )
    args = parser.parse_args()

    pt_path = PROJECT_ROOT / "data" / f"{args.path}.pt"
    if not os.path.exists(pt_path):
        print(f"Error: file not found: {pt_path}")
        sys.exit(1)

    obj = torch.load(pt_path, map_location="cpu")
    print(f"Loaded: {pt_path}")

    if isinstance(obj, dict):
        print(f"type: dict with keys = {list(obj.keys())}")
        # Prefer 'X' if present
        if "X" in obj and isinstance(obj["X"], torch.Tensor):
            X = obj["X"]
            describe_tensor("X", X)
            # If 2 or 3 columns, label likely meanings
            if X.ndim == 2:
                if X.shape[1] == 2:
                    print("Detected 2-column dataset: columns likely [x, r0]")
                elif X.shape[1] == 3:
                    print(
                        "Detected 3-column dataset: columns likely [x, r0, Z]")
        # Also print summary of known metadata tensors if present
        for k in ("x", "r0", "Z"):
            if k in obj and isinstance(obj[k], torch.Tensor):
                describe_tensor(k, obj[k])
        # Print simple scalars/ranges if present
        for k in ("x_minmax", "r0_range", "seed"):
            if k in obj:
                print(f"{k}: {obj[k]}")
    elif isinstance(obj, torch.Tensor):
        print("type: torch.Tensor")
        describe_tensor("tensor", obj)
    else:
        print(f"type: {type(obj)} (unrecognized container). Printing repr:")
        print(repr(obj))


if __name__ == "__main__":
    analyse_dataset(str(PROJECT_ROOT / "data" / "phase_4_solver_extended_fine.pt"))
