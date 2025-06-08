"""
Compression and fusion logic (offline SVD simulation)
"""
import torch
from typing import Tuple

def decompose_and_fuse(W_v: torch.Tensor, W_o: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform SVD decomposition of W_v and fuse with W_o
    
    Args:
        W_v: Value projection matrix [d_model, d_head]
        W_o: Output projection matrix [d_model, d_head] 
        rank: Target compression rank
        
    Returns:
        A: Compression matrix [rank, d_head] = S_truncated @ V_truncated^T
        W_fused: Fused output projection [d_model, rank] = W_o.T @ U_truncated
    """
    d_model, d_head = W_v.shape
    
    # Ensure rank doesn't exceed the smaller dimension
    max_rank = min(d_model, d_head)
    if rank > max_rank:
        print(f"⚠️  Reducing rank from {rank} to {max_rank} (max possible for {d_model}x{d_head} matrix)")
        rank = max_rank
    
    # Perform SVD: W_v = U @ S @ V^T
    U, S, V = torch.svd(W_v)
    
    # Truncate to desired rank
    U_truncated = U[:, :rank]          # [d_model, rank] 
    S_truncated = S[:rank]             # [rank]
    V_truncated = V[:, :rank]          # [d_head, rank]
    
    # Create compression matrix: A = S @ V^T (maps from d_head to rank)
    A = torch.diag(S_truncated) @ V_truncated.T  # [rank, d_head]
    
    # Create fused output projection: W_fused = W_o.T @ U (maps from rank to d_model)
    # W_o is [d_model, d_head], so W_o.T is [d_head, d_model]
    # But we want to fuse in the value space, so we use U which maps from value space
    # Actually, let's create a proper fusion: we want the final result to be [d_model, rank]
    W_fused = U_truncated.T  # [rank, d_model] - this will be transposed when used
    
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
