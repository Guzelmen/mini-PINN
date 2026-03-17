"""
Concatenate part files produced by solver_generator.py --part N into one
final dataset that is identical to what a full single-run would have produced.

Usage:
    python -m src.data_utils.concatenate_parts
    python -m src.data_utils.concatenate_parts --parts 1 2 3 7 8
    python -m src.data_utils.concatenate_parts --files phase_4_solver_mega_fine_part1.pt phase_4_solver_mega_fine_relax_part8.pt

Optional args:
    --parts     explicit list of part numbers to combine (e.g. 1 2 3 7 8)
    --files     explicit list of filenames in data/ to combine (overrides --parts and --base-name)
    --n-parts   number of part files to combine sequentially 1..N (default: 4, ignored if --parts given)
    --out-path  override output path
    --base-name base filename stem, without _partN.pt suffix (default: phase_4_solver_mega_fine)
"""
import argparse
import torch
from pathlib import Path


def concatenate_parts(parts=None, n_parts=4, base_name="phase_4_solver_mega_fine", out_path=None, files=None):
    from ..utils import PROJECT_ROOT

    data_dir = PROJECT_ROOT / "data"
    if out_path is None:
        out_path = data_dir / f"{base_name}.pt"
    else:
        out_path = data_dir / out_path

    all_inputs = []
    all_targets = []

    if files is not None:
        paths = [data_dir / f for f in files]
    else:
        if parts is None:
            parts = list(range(1, n_parts + 1))
        paths = [data_dir / f"{base_name}_part{part}.pt" for part in parts]

    for part_path in paths:
        print(f"Loading {part_path} ...", end=" ")
        d = torch.load(part_path, weights_only=True)
        inputs = d["inputs"]
        targets = d["targets"]
        print(f"{inputs.shape[0]:,} rows")
        all_inputs.append(inputs)
        all_targets.append(targets)

    inputs_cat = torch.cat(all_inputs, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)

    torch.save({"inputs": inputs_cat, "targets": targets_cat}, out_path)

    print(f"\nSaved {inputs_cat.shape[0]:,} rows to {out_path}")
    print(f"  inputs shape:  {tuple(inputs_cat.shape)}")
    print(f"  targets shape: {tuple(targets_cat.shape)}")
    print(f"  x range:       [{inputs_cat[:, 0].min():.6f}, {inputs_cat[:, 0].max():.6f}]")
    print(f"  alpha range:   [{inputs_cat[:, 1].min():.6f}, {inputs_cat[:, 1].max():.6f}]")
    print(f"  T range:       [{inputs_cat[:, 2].min():.6f}, {inputs_cat[:, 2].max():.6f}] keV")
    print(f"  psi range:     [{targets_cat[:, 0].min():.6f}, {targets_cat[:, 0].max():.6f}]")
    print(f"  dpsi/dx range: [{targets_cat[:, 1].min():.6f}, {targets_cat[:, 1].max():.6f}]")

    return inputs_cat, targets_cat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate solver_generator part files")
    parser.add_argument("--parts", type=int, nargs="+", default=None,
                        help="Explicit part numbers to combine (e.g. 1 2 3 7 8)")
    parser.add_argument("--files", type=str, nargs="+", default=None,
                        help="Explicit filenames in data/ to combine (overrides --parts and --base-name)")
    parser.add_argument("--n-parts", type=int, default=4,
                        help="Number of sequential parts 1..N (default: 4, ignored if --parts given)")
    parser.add_argument("--base-name", type=str, default="phase_4_solver_mega_fine",
                        help="Base filename stem (default: phase_4_solver_mega_fine)")
    parser.add_argument("--out-path", type=str, default=None, help="Override output path")
    args = parser.parse_args()
    concatenate_parts(parts=args.parts, n_parts=args.n_parts, base_name=args.base_name, out_path=args.out_path, files=args.files)
