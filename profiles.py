"""
Hardcoded compression profiles for testing with key and value compression
"""
import torch
from compression import decompose_and_fuse, compress_keys

# Set seed for reproducible random matrices
torch.manual_seed(42)

# Model dimensions
d_model = 512  # embedding dimension
d_head = 64    # per-head dimension

# Create mock projection matrices for SVD decomposition
W_v = torch.randn(d_model, d_head) * 0.1  # Value projection matrix
W_o = torch.randn(d_model, d_head) * 0.1  # Output projection matrix
W_k = torch.randn(d_model, d_head) * 0.1  # Key projection matrix

# Fixed key compression rank for all profiles
KEY_COMPRESSION_RANK = 32

# Compression profiles with different ranks - now using proper SVD decomposition
profiles = {}

# Generate profiles using actual SVD decomposition and fusion
for name, value_rank in [("low", 64), ("med", 32), ("high", 16)]:
    # Value compression (adaptive/dynamic)
    A_v, W_fused = decompose_and_fuse(W_v, W_o, value_rank)
    
    # Key compression (fixed rank for all profiles)
    A_k, B_k = compress_keys(W_k, KEY_COMPRESSION_RANK)
    
    profiles[name] = {
        # Value compression matrices
        "A": A_v,           # Value compression matrix [rank, d_head]
        "W_fused": W_fused, # Fused output projection
        "r": value_rank,    # Value compression rank
        
        # Key compression matrices
        "A_K": A_k,         # Key compression matrix [key_rank, d_head]
        "B_K": B_k,         # Key reconstruction matrix [d_head, key_rank]
        "r_k": KEY_COMPRESSION_RANK  # Key compression rank (fixed)
    }

# Keep matrices consistent across runs
for profile in profiles.values():
    profile["A"].requires_grad_(False)
    profile["W_fused"].requires_grad_(False)
    profile["A_K"].requires_grad_(False)
    profile["B_K"].requires_grad_(False)
