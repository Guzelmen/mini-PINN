"""
Analyse .pt data files - inspect shape, contents, statistics, and structure.
"""

import torch
import numpy as np

# ============================================================================
# CONFIGURATION: Set the path to the .pt file you want to analyze
# ============================================================================
DATA_FILE_PATH = "/rds/general/user/gs1622/home/mini PINN/data/phase_4_solver_inp&targets_smallrange.pt"


def analyze_pt_file(file_path):
    """
    Load and analyze a .pt file, providing comprehensive information about its contents.
    
    Args:
        file_path (str): Path to the .pt file to analyze
    """
    print("=" * 80)
    print(f"ANALYZING: {file_path}")
    print("=" * 80)
    
    try:
        # Load the data
        data = torch.load(file_path)
        print(f"\n✓ Successfully loaded file")
        
        # Analyze based on data type
        print(f"\n{'='*80}")
        print("DATA TYPE INFORMATION")
        print(f"{'='*80}")
        print(f"Type: {type(data)}")
        
        if isinstance(data, dict):
            analyze_dict(data)
        elif isinstance(data, torch.Tensor):
            analyze_tensor(data, name="Main Tensor")
        elif isinstance(data, list):
            analyze_list(data)
        elif isinstance(data, tuple):
            analyze_tuple(data)
        else:
            print(f"Data is of type {type(data)}")
            print(f"Value: {data}")
            
    except Exception as e:
        print(f"\n✗ Error loading or analyzing file: {e}")
        import traceback
        traceback.print_exc()


def analyze_dict(data):
    """Analyze dictionary data structure."""
    print(f"\nDictionary with {len(data)} keys:")
    print(f"Keys: {list(data.keys())}")
    
    print(f"\n{'='*80}")
    print("DETAILED ANALYSIS OF EACH KEY")
    print(f"{'='*80}")
    
    for key, value in data.items():
        print(f"\n{'─'*80}")
        print(f"Key: '{key}'")
        print(f"{'─'*80}")
        
        if isinstance(value, torch.Tensor):
            analyze_tensor(value, name=key)
        elif isinstance(value, (list, tuple)):
            print(f"  Type: {type(value)}")
            print(f"  Length: {len(value)}")
            if len(value) > 0:
                print(f"  First element type: {type(value[0])}")
                if isinstance(value[0], torch.Tensor):
                    print(f"  First element shape: {value[0].shape}")
        elif isinstance(value, dict):
            print(f"  Type: Nested dictionary")
            print(f"  Keys: {list(value.keys())}")
        else:
            print(f"  Type: {type(value)}")
            print(f"  Value: {value}")


def analyze_tensor(tensor, name="Tensor"):
    """Analyze a PyTorch tensor."""
    print(f"  Type: torch.Tensor")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    print(f"  Number of elements: {tensor.numel()}")
    
    # Convert to numpy for easier statistics
    if tensor.is_cuda:
        tensor_cpu = tensor.cpu()
    else:
        tensor_cpu = tensor
    
    tensor_np = tensor_cpu.numpy()
    
    print(f"\n  Statistics:")
    print(f"    Min:    {tensor_np.min():.6e}")
    print(f"    Max:    {tensor_np.max():.6e}")
    print(f"    Mean:   {tensor_np.mean():.6e}")
    print(f"    Std:    {tensor_np.std():.6e}")
    print(f"    Median: {np.median(tensor_np):.6e}")
    
    # Check for special values
    has_nan = np.isnan(tensor_np).any()
    has_inf = np.isinf(tensor_np).any()
    
    if has_nan:
        print(f"    ⚠ Contains NaN values: {np.isnan(tensor_np).sum()}")
    if has_inf:
        print(f"    ⚠ Contains Inf values: {np.isinf(tensor_np).sum()}")
    
    # Show a sample of the data
    print(f"\n  Sample of data (first few elements):")
    if tensor.dim() == 1:
        print(f"    {tensor_cpu[:min(5, len(tensor_cpu))].tolist()}")
    elif tensor.dim() == 2:
        print(f"    First row: {tensor_cpu[0, :min(5, tensor.shape[1])].tolist()}")
        if tensor.shape[0] > 1:
            print(f"    Second row: {tensor_cpu[1, :min(5, tensor.shape[1])].tolist()}")
    else:
        print(f"    {tensor_cpu.flatten()[:5].tolist()}")


def analyze_list(data):
    """Analyze list data structure."""
    print(f"\nList with {len(data)} elements")
    print(f"Element types: {[type(item).__name__ for item in data[:5]]}")
    
    if len(data) > 0 and isinstance(data[0], torch.Tensor):
        print(f"\nFirst tensor in list:")
        analyze_tensor(data[0], name="List[0]")


def analyze_tuple(data):
    """Analyze tuple data structure."""
    print(f"\nTuple with {len(data)} elements")
    print(f"Element types: {[type(item).__name__ for item in data]}")
    
    for i, item in enumerate(data):
        print(f"\n  Element {i}:")
        if isinstance(item, torch.Tensor):
            analyze_tensor(item, name=f"Tuple[{i}]")
        else:
            print(f"    Type: {type(item)}")
            print(f"    Value: {item}")


if __name__ == "__main__":
    analyze_pt_file(DATA_FILE_PATH)
