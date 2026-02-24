import torch
import os
from pathlib import Path
from .normalisation import fit_and_transform
from ..utils import PROJECT_ROOT


def load_data_and_norm(params):
    """
    Load the pre-generated data from the PyTorch file, normalize it, and split into train/val/test sets.

    Args:
        params (dict): Parameters for the data loading
        file_path (str, optional): Path to the .pt file containing the data. 
                                  If None, uses relative path from data_utils folder.

    Returns:
        dict: Dictionary containing:
            - 'train': normalized training data tensor
            - 'val': normalized validation data tensor  
            - 'test': normalized test data tensor
            - 'normalizer': fitted SimpleNormalizer object for inverse transforms
    """
    # go to parent of this current folder and then into the data folder
    file_path = PROJECT_ROOT / str(params.data_path)
    file_path = str(file_path)
    print(f"Data file path: {file_path}")

    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")

    # Load the data
    print(f"Loading data from {file_path}...")
    raw_data = torch.load(file_path)

    # Extract the X tensor (supports 2 or 3 columns)
    X = raw_data['X']  # Shape: [N, 2] or [N, 3]
    print(f"Loaded data shape: {X.shape}")

    # Set random seed for reproducible splits
    torch.manual_seed(params.random_seed)

    # Define exact split sizes depending on number of columns
    if X.shape[1] == 2:
        # New 2-column dataset: [x, r0], total 64000
        n_train = 51200  # 100 batches of 512
        n_val = 6400
        n_test = 6400
    else:
        # Legacy 3-column dataset: [x, r0, Z]
        n_train = 25600  # 100 batches of 256
        n_val = 3328     # 13 batches of 256
        n_test = 3328    # 13 batches of 256
    n_total = n_train + n_val + n_test

    # Randomly sample from the 40k available points
    all_indices = torch.randperm(X.shape[0])
    selected_indices = all_indices[:n_total]

    # Split the selected indices
    train_indices = selected_indices[:n_train]
    val_indices = selected_indices[n_train:n_train + n_val]
    test_indices = selected_indices[n_train + n_val:]

    # Split the data
    X_train = X[train_indices]
    X_val = X[val_indices]
    X_test = X[test_indices]

    print(
        f"Data split: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")

    # Normalize the training data and fit the normalizer
    if X.shape[1] == 3:
        print("Fitting normalizer on training data...")
        X_train_normalized, normalizer = fit_and_transform(X_train)
        print("Applying normalization to validation and test sets...")
        X_val_normalized = normalizer.transform(X_val)
        X_test_normalized = normalizer.transform(X_test)
        # Print normalization info
        print("Normalization applied:")
        print(f"  x_mode: {normalizer.cfg['x_mode']}")
        print(f"  r0_mode: {normalizer.cfg['r0_mode']}")
        print(f"  Z_mode: {normalizer.cfg['Z_mode']}")
        print(f"  Z_power: {normalizer.cfg['Z_power']}")
        return {
            'train': X_train_normalized,
            'val': X_val_normalized,
            'test': X_test_normalized,
            'normalizer': normalizer
        }
    else:
        # Simple normalization for 2-column data:
        # - x: identity (already in (0,1))
        # - r0: min-max to [-1,1] based on training stats
        print(
            "Fitting simple normalizer for 2-column data (x identity, r0 minmax[-1,1])...")
        r0_train = X_train[:, 1:2]
        r0_min = float(r0_train.min())
        r0_max = float(r0_train.max())
        denom = (r0_max - r0_min + 1e-12)

        def norm2(Xt):
            x = Xt[:, 0:1]
            r0 = Xt[:, 1:2]
            r0n = 2.0 * (r0 - r0_min) / denom - 1.0
            return torch.cat([x, r0n], dim=1)
        X_train_normalized = norm2(X_train)
        X_val_normalized = norm2(X_val)
        X_test_normalized = norm2(X_test)
        print("Normalization applied (2 columns):")
        print("  x_mode: identity")
        print(
            f"  r0_min: {r0_min:.3e}, r0_max: {r0_max:.3e} (scaled to [-1,1])")
        return {
            'train': X_train_normalized,
            'val': X_val_normalized,
            'test': X_test_normalized,
            'normalizer': dict(type="two_col", x_mode="identity", r0_min=r0_min, r0_max=r0_max)
        }


def load_data(params):
    """
    Load the pre-generated data from the PyTorch file, split into train/val/test sets.
    
    Supports two data formats:
    1. Legacy format: raw tensor with shape [N, 2] or [N, 3]
    2. Solver format: dict with {'inputs': [N, 3], 'targets': [N, 1]}
    
    When use_solver_data=True in params, expects solver format and returns targets.

    Args:
        params (dict): Parameters for the data loading
            - data_path: path to .pt file
            - use_solver_data: bool, if True expects {'inputs', 'targets'} format
            - random_seed: seed for reproducible splits

    Returns:
        dict: Dictionary containing:
            - 'train': training inputs tensor
            - 'val': validation inputs tensor  
            - 'test': test inputs tensor
            - 'train_targets': training targets tensor (if use_solver_data=True)
            - 'val_targets': validation targets tensor (if use_solver_data=True)
            - 'test_targets': test targets tensor (if use_solver_data=True)
    """
    # go to parent of this current folder and then into the data folder
    file_path = PROJECT_ROOT / str(params.data_path)
    file_path = str(file_path)
    print(f"Data file path: {file_path}")

    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")

    # Load the data
    print(f"Loading data from {file_path}...")
    raw_data = torch.load(file_path)

    # Detect data format
    use_solver_data = getattr(params, 'use_solver_data', False)
    
    if use_solver_data or (isinstance(raw_data, dict) and 'inputs' in raw_data and 'targets' in raw_data):
        # Solver format: {'inputs': [N, 3], 'targets': [N, 1 or 2]}
        X = raw_data['inputs']
        Y = raw_data['targets']
        print(f"Loaded solver data: inputs shape {X.shape}, targets shape {Y.shape}")
        has_targets = True
        # Filter out points with x below threshold (these get clamped to wrong values in training)
        if getattr(params, 'filter_x_min', False):
            x_min_thresh = getattr(params, 'x_min_threshold', 1e-6)
            valid_mask = X[:, 0] >= x_min_thresh
            n_before = X.shape[0]
            X = X[valid_mask]
            Y = Y[valid_mask]
            print(f"Filtered x < {x_min_thresh}: {n_before} -> {X.shape[0]} ({n_before - X.shape[0]} removed)")
    else:
        # Legacy format: raw tensor
        X = raw_data  # Shape: [N, 2] or [N, 3]
        Y = None
        print(f"Loaded legacy data shape: {X.shape}")
        has_targets = False

    # Set random seed for reproducible splits
    torch.manual_seed(params.random_seed)

    # Define exact split sizes depending on number of columns
    if X.shape[1] == 2:
        # New 2-column dataset: [x, alpha], total 64000
        n_train = 51200  # 100 batches of 512
        n_val = 6400
        n_test = 6400
    else:
        # 3-column dataset: [x, alpha, T] or [x, r0, Z]
        n_train = 25600  # 100 batches of 256
        n_val = 3328     # 13 batches of 256
        n_test = 3328    # 13 batches of 256
    n_total = n_train + n_val + n_test

    # Randomly sample from the available points
    all_indices = torch.randperm(X.shape[0])
    selected_indices = all_indices[:n_total]

    # Split the selected indices
    train_indices = selected_indices[:n_train]
    val_indices = selected_indices[n_train:n_train + n_val]
    test_indices = selected_indices[n_train + n_val:]

    # Split the data
    X_train = X[train_indices]
    X_val = X[val_indices]
    X_test = X[test_indices]

    print(
        f"Data split: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")

    # Solver data completeness check (for phase 4)
    if use_solver_data:
        expected_total = 256_000
        min_required = int(0.9 * expected_total)
        n_points = X.shape[0]
        if n_points < min_required:
            raise ValueError(f"Solver data has only {n_points} points (<90% of {expected_total}). Aborting.")
        else:
            print(f"Solver data has {n_points} points (>=90% of {expected_total}), proceeding with fixed splits.")
        # Use fixed splits as if we had 256k
        n_train = 204_800  # 80% of 256k
        n_val = 25_600     # 10% of 256k
        n_test = n_points - n_train - n_val
        n_total = n_train + n_val + n_test
        all_indices = torch.randperm(n_points)
        selected_indices = all_indices[:n_total]
        train_indices = selected_indices[:n_train]
        val_indices = selected_indices[n_train:n_train + n_val]
        test_indices = selected_indices[n_train + n_val:]
        X_train = X[train_indices]
        X_val = X[val_indices]
        X_test = X[test_indices]
        result = {"train": X_train, "val": X_val, "test": X_test}
        Y_train = Y[train_indices]
        Y_val = Y[val_indices]
        Y_test = Y[test_indices]
        result['train_targets'] = Y_train
        result['val_targets'] = Y_val
        result['test_targets'] = Y_test
        print(f"  Targets included: train={Y_train.shape}, val={Y_val.shape}, test={Y_test.shape}")
        return result
    else:
        result = {"train": X_train, "val": X_val, "test": X_test}
        
        # Add targets if available
        if has_targets:
            Y_train = Y[train_indices]
            Y_val = Y[val_indices]
            Y_test = Y[test_indices]
            result['train_targets'] = Y_train
            result['val_targets'] = Y_val
            result['test_targets'] = Y_test
            print(f"  Targets included: train={Y_train.shape}, val={Y_val.shape}, test={Y_test.shape}")

        return result


def get_data_loaders(data_dict, batch_size, shuffle_train):
    """
    Convert the data dictionary into PyTorch DataLoaders.

    Args:
        data_dict (dict): Dictionary returned by load_data()
            - Must contain 'train', 'val', 'test' (inputs)
            - May contain 'train_targets', 'val_targets', 'test_targets'
        batch_size (int): Batch size for the DataLoaders
        shuffle_train (bool): Whether to shuffle training data

    Returns:
        dict: Dictionary containing train, val, test DataLoaders
              Each batch is (inputs,) or (inputs, targets) depending on data
    """
    from torch.utils.data import DataLoader, TensorDataset

    # Check if we have targets
    has_targets = 'train_targets' in data_dict

    if has_targets:
        # Create datasets with (inputs, targets) pairs
        train_dataset = TensorDataset(data_dict['train'], data_dict['train_targets'])
        val_dataset = TensorDataset(data_dict['val'], data_dict['val_targets'])
        test_dataset = TensorDataset(data_dict['test'], data_dict['test_targets'])
        print("DataLoaders created with (inputs, targets) pairs")
    else:
        # Create datasets with inputs only
        train_dataset = TensorDataset(data_dict['train'])
        val_dataset = TensorDataset(data_dict['val'])
        test_dataset = TensorDataset(data_dict['test'])
        print("DataLoaders created with inputs only")

    # Create data loaders (pin_memory speeds up CPU→GPU transfer)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle_train, pin_memory=pin)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin)

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'has_targets': has_targets
    }
