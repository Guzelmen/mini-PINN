import torch
import os
from pathlib import Path
from data_utils.normalisation import fit_and_transform


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
    file_path = Path(__file__).parent.parent / str(params.data_path)
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

    Args:
        params (dict): Parameters for the data loading

    Returns:
        dict: Dictionary containing:
            - 'train': training data tensor
            - 'val': validation data tensor  
            - 'test': test data tensor
    """
    # go to parent of this current folder and then into the data folder
    file_path = Path(__file__).parent.parent / str(params.data_path)
    file_path = str(file_path)
    print(f"Data file path: {file_path}")

    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")

    # Load the data
    print(f"Loading data from {file_path}...")
    raw_data = torch.load(file_path)

    # Extract the X tensor (supports 2 or 3 columns)
    X = raw_data  # Shape: [N, 2] or [N, 3]
    print(f"Loaded data shape: {X.shape}")

    # Set random seed for reproducible splits
    torch.manual_seed(params.random_seed)

    # Define exact split sizes depending on number of columns
    if X.shape[1] == 2:
        # New 2-column dataset: [x, alpha], total 64000
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

    return {"train": X_train, "val": X_val, "test": X_test}


def get_data_loaders(data_dict, batch_size, shuffle_train):
    """
    Convert the data dictionary into PyTorch DataLoaders.

    Args:
        data_dict (dict): Dictionary returned by load_data()
        batch_size (int): Batch size for the DataLoaders
        shuffle_train (bool): Whether to shuffle training data

    Returns:
        dict: Dictionary containing train, val, test DataLoaders
    """
    from torch.utils.data import DataLoader, TensorDataset

    # Create datasets
    train_dataset = TensorDataset(data_dict['train'])
    val_dataset = TensorDataset(data_dict['val'])
    test_dataset = TensorDataset(data_dict['test'])

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
