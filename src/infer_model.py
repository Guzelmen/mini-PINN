"""
Utility to build a model from a YAML config and load weights from a state dict.
"""
import argparse
from pathlib import Path
from typing import Tuple

import torch

from . import models
from .YParams import YParams
from .utils import PROJECT_ROOT


def build_model_from_params(params) -> torch.nn.Module:
    """
    Instantiate the model class specified by params.mode and params.phase.
    """
    if not hasattr(params, 'mode'):
        raise ValueError("Parameter 'mode' is missing from config.")
    if not hasattr(params, 'phase'):
        raise ValueError("Parameter 'phase' is missing from config.")

    mode = str(params.mode).strip()
    phase = int(params.phase)
    model_class_name = f"Model_{mode}_phase{phase}"
    if not hasattr(models, model_class_name):
        raise ValueError(
            f"Model class '{model_class_name}' not found. "
            f"Adjust mode/phase (mode={mode}, phase={phase}) or define the class."
        )
    ModelClass = getattr(models, model_class_name)
    return ModelClass(params)


def load_model_from_config_and_state(
    config_name: str,
    state_path: str,
    device: str = "cpu",
) -> Tuple[torch.nn.Module, YParams]:
    """
    Load params from src/yamls/<config_name>.yaml, build the model, and load weights.

    Args:
        config_name: YAML filename stem in src/yamls (without .yaml extension).
        state_path: Path to a state dict saved via torch.save(model.state_dict(), ...).
        device: "cpu" or e.g. "cuda:0".

    Returns:
        (model, params) with model on device and in eval mode.
    """
    yaml_path = PROJECT_ROOT / "src" / "yamls" / f"{config_name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    params = YParams(yaml_path, "base_config", print_params=False)

    # Load state dict first to detect normalization buffers and align model config
    map_location = torch.device(device)
    state_dict = torch.load(Path(state_path), map_location=map_location)
    if not isinstance(state_dict, dict):
        raise ValueError(
            "Loaded checkpoint is not a state dict. "
            "Expected dict of parameter/buffer tensors (e.g., from model.state_dict())."
        )

    # Align params.norm_mode with checkpoint buffers to ensure buffers register then load correctly
    sd_keys = set(state_dict.keys())
    if "a_mean" in sd_keys and "a_std" in sd_keys:
        params.norm_mode = "standardize"
        try:
            params.standard_mean = float(getattr(state_dict["a_mean"], "item", lambda: state_dict["a_mean"])())
            params.standard_std = float(getattr(state_dict["a_std"], "item", lambda: state_dict["a_std"])())
        except Exception:
            pass
    elif "a_min" in sd_keys and "a_max" in sd_keys:
        params.norm_mode = "minmax"

    model = build_model_from_params(params)

    # Load state dict saved with torch.save(model.state_dict(), path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if len(missing) > 0:
        print(f"Warning: missing keys in state dict: {missing}")
    if len(unexpected) > 0:
        print(f"Warning: unexpected keys in state dict: {unexpected}")

    model.to(map_location)
    model.eval()
    torch.set_grad_enabled(False)
    return model, params


def main():
    parser = argparse.ArgumentParser(
        description="Load a model from YAML config and a state dict file."
    )
    parser.add_argument("--config", required=True,
                        help="YAML config name in src/yamls (without .yaml).")
    parser.add_argument("--state", required=True,
                        help="Path to weights file saved via torch.save(model.state_dict(), PATH).")
    parser.add_argument("--device", default="cpu",
                        help='Device to load model onto, e.g. "cpu" or "cuda:0".')
    parser.add_argument("--print-keys", action="store_true",
                        help="Print parameter/buffer keys and shapes from the loaded state dict.")
    args = parser.parse_args()

    model, params = load_model_from_config_and_state(
        config_name=args.config,
        state_path=args.state,
        device=args.device,
    )

    print(f"Loaded model {model.__class__.__name__} "
          f"(mode={params.mode}, phase={params.phase}) "
          f"with weights from {args.state} on device {args.device}.")

    if args.print-keys:
        state_dict = torch.load(Path(args.state), map_location=torch.device(args.device))
        key_shapes = {k: tuple(v.shape) for k, v in state_dict.items() if hasattr(v, "shape")}
        print("State dict keys and shapes:")
        for k, shape in key_shapes.items():
            print(f"  {k}: {shape}")


if __name__ == "__main__":
    main()


