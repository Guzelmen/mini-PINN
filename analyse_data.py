import argparse
import os
import sys
import torch


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

    pt_path = f"data/{args.path}.pt"
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
    main()
