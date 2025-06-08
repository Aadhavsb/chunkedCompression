"""
Hardcoded compression profiles for testing
"""
import torch

# Set seed for reproducible random matrices
torch.manual_seed(42)

# Model dimensions
d_model = 512  # embedding dimension
d_head = 64    # per-head dimension

# Compression profiles with different ranks
profiles = {
    "low": {
        "A": torch.randn(64, d_head),           # rank 64
        "W_fused": torch.randn(d_model, 64),   # fused output projection
        "r": 64
    },
    "med": {
        "A": torch.randn(32, d_head),           # rank 32
        "W_fused": torch.randn(d_model, 32),   # fused output projection
        "r": 32
    },
    "high": {
        "A": torch.randn(16, d_head),           # rank 16
        "W_fused": torch.randn(d_model, 16),   # fused output projection
        "r": 16
    }
}

# Keep matrices consistent across runs
for profile in profiles.values():
    profile["A"].requires_grad_(False)
    profile["W_fused"].requires_grad_(False)
