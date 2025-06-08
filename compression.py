"""
Compression and fusion logic (offline SVD simulation)
"""
import torch
from typing import Tuple

def decompose_and_fuse(W_v: torch.Tensor, W_o: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate SVD decomposition and fusion of W_v and W_o matrices
    
    Args:
        W_v: Value projection matrix [d_model, d_head]
        W_o: Output projection matrix [d_model, d_head] 
        rank: Target compression rank
        
    Returns:
        A: Compression matrix [rank, d_head]
        W_fused: Fused output projection [d_model, rank]
    """
    # For barebones testing, we'll simulate the decomposition with random matrices
    # In a real implementation, this would involve SVD of W_v and fusion with W_o
    
    d_model, d_head = W_v.shape
    
    # Simulate compression matrix A (maps from d_head to rank)
    torch.manual_seed(42)  # For reproducible results
    A = torch.randn(rank, d_head) * 0.1
    
    # Simulate fused output projection W_fused (maps from rank to d_model)
    W_fused = torch.randn(d_model, rank) * 0.1
    
    return A, W_fused

def validate_compression_matrices(A: torch.Tensor, W_fused: torch.Tensor, 
                                original_dim: int, compressed_dim: int) -> bool:
    """
    Validate that compression matrices have correct shapes
    
    Args:
        A: Compression matrix
        W_fused: Fused output projection matrix
        original_dim: Original dimension (d_head)
        compressed_dim: Compressed dimension (rank)
        
    Returns:
        True if shapes are valid
    """
    expected_A_shape = (compressed_dim, original_dim)
    expected_W_shape = (512, compressed_dim)  # d_model = 512
    
    A_valid = A.shape == expected_A_shape
    W_valid = W_fused.shape == expected_W_shape
    
    if not A_valid:
        print(f"❌ A matrix shape mismatch: expected {expected_A_shape}, got {A.shape}")
    if not W_valid:
        print(f"❌ W_fused matrix shape mismatch: expected {expected_W_shape}, got {W_fused.shape}")
    
    return A_valid and W_valid
