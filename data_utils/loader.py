import torch
import os
from pathlib import Path
from data_utils.normalisation import fit_and_transform


def load_data(params):
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

    # Extract only the X tensor (the actual data with 3 columns: x, r0, Z)
    X = raw_data['X']  # Shape: [40000, 3]
    print(f"Loaded data shape: {X.shape}")

    # Set random seed for reproducible splits
    torch.manual_seed(params.random_seed)

    # Define exact split sizes
    n_train = 25600  # 100 batches of 256
    n_val = 3328     # 13 batches of 256
    n_test = 3328    # 13 batches of 256
    n_total = n_train + n_val + n_test  # 32256 total samples

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
    print("Fitting normalizer on training data...")
    X_train_normalized, normalizer = fit_and_transform(X_train)

    # Apply the same normalization to validation and test sets
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


def get_data_loaders(data_dict, batch_size=32, shuffle_train=True):
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
