"""
Compression configuration management.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple


@dataclass
class CompressionConfig:
    """Configuration for compression operations."""
    
    # Compression ranks for different profiles
    value_compression_ranks: Dict[str, int] = field(default_factory=lambda: {
        "low": 32,      # High compression, lower quality
        "med": 64,      # Balanced compression and quality  
        "high": 128     # Lower compression, higher quality
    })
    
    # Key compression configuration
    key_compression_rank: int = 128
    
    # Compression strategy
    compression_strategy: str = "adaptive"  # "adaptive", "fixed", "dynamic"
    
    # SVD parameters
    svd_solver: str = "auto"  # "auto", "full", "arpack", "randomized"
    svd_tolerance: float = 1e-6
    
    # Profile building parameters
    profile_layers: Optional[List[int]] = None  # None for all layers
    profile_heads: Optional[List[int]] = None   # None for all heads
    
    # Memory optimization
    use_memory_efficient_svd: bool = True
    chunk_size: int = 1024  # For chunked operations
    
    # Validation parameters
    validate_shapes: bool = True
    validate_compression_ratios: bool = True
    min_compression_ratio: float = 2.0  # Minimum acceptable compression ratio
    
    def __post_init__(self):
        """Post-initialization validation."""
        # Validate compression ranks
        for profile, rank in self.value_compression_ranks.items():
            if rank <= 0:
                raise ValueError(f"Compression rank for {profile} must be positive, got {rank}")
        
        if self.key_compression_rank <= 0:
            raise ValueError(f"Key compression rank must be positive, got {self.key_compression_rank}")
        
        # Validate compression strategy
        valid_strategies = ["adaptive", "fixed", "dynamic"]
        if self.compression_strategy not in valid_strategies:
            raise ValueError(f"Invalid compression strategy: {self.compression_strategy}. "
                           f"Must be one of {valid_strategies}")
        
        # Validate SVD solver
        valid_solvers = ["auto", "full", "arpack", "randomized"]
        if self.svd_solver not in valid_solvers:
            raise ValueError(f"Invalid SVD solver: {self.svd_solver}. "
                           f"Must be one of {valid_solvers}")
    
    @classmethod
    def from_env(cls) -> 'CompressionConfig':
        """Create configuration from environment variables."""
        # Parse compression ranks from environment
        value_ranks = {}
        for profile in ["low", "med", "high"]:
            env_key = f"COMPRESSION_RANK_{profile.upper()}"
            if env_key in os.environ:
                value_ranks[profile] = int(os.environ[env_key])
        
        return cls(
            value_compression_ranks=value_ranks or cls.value_compression_ranks,
            key_compression_rank=int(os.getenv('KEY_COMPRESSION_RANK', cls.key_compression_rank)),
            compression_strategy=os.getenv('COMPRESSION_STRATEGY', cls.compression_strategy),
            svd_solver=os.getenv('SVD_SOLVER', cls.svd_solver),
            use_memory_efficient_svd=os.getenv('MEMORY_EFFICIENT_SVD', 'true').lower() == 'true',
        )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CompressionConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'value_compression_ranks': self.value_compression_ranks,
            'key_compression_rank': self.key_compression_rank,
            'compression_strategy': self.compression_strategy,
            'svd_solver': self.svd_solver,
            'svd_tolerance': self.svd_tolerance,
            'profile_layers': self.profile_layers,
            'profile_heads': self.profile_heads,
            'use_memory_efficient_svd': self.use_memory_efficient_svd,
            'chunk_size': self.chunk_size,
            'validate_shapes': self.validate_shapes,
            'validate_compression_ratios': self.validate_compression_ratios,
            'min_compression_ratio': self.min_compression_ratio,
        }
    
    def get_value_rank(self, profile: str) -> int:
        """Get value compression rank for a specific profile."""
        if profile not in self.value_compression_ranks:
            raise ValueError(f"Unknown compression profile: {profile}. "
                           f"Available profiles: {list(self.value_compression_ranks.keys())}")
        return self.value_compression_ranks[profile]
    
    def get_compression_profile_names(self) -> List[str]:
        """Get list of available compression profile names."""
        return list(self.value_compression_ranks.keys())
    
    def calculate_expected_compression_ratio(self, original_dim: int, profile: str) -> float:
        """
        Calculate expected compression ratio for a given profile.
        
        Args:
            original_dim: Original dimension size
            profile: Compression profile name
            
        Returns:
            Expected compression ratio
        """
        value_rank = self.get_value_rank(profile)
        key_rank = self.key_compression_rank
        
        # Simplified calculation: assume square matrices
        original_params = original_dim * original_dim
        compressed_params = original_dim * (value_rank + key_rank)
        
        return original_params / compressed_params if compressed_params > 0 else 1.0
    
    def validate_profile_compatibility(self, model_hidden_size: int) -> Dict[str, bool]:
        """
        Validate that compression profiles are compatible with model dimensions.
        
        Args:
            model_hidden_size: Hidden size of the model
            
        Returns:
            Dictionary mapping profile names to compatibility status
        """
        compatibility = {}
        
        for profile, rank in self.value_compression_ranks.items():
            # Check if rank is reasonable compared to hidden size
            is_compatible = (
                rank < model_hidden_size and  # Rank should be smaller than input dimension
                rank >= 8 and                 # Minimum reasonable rank
                self.calculate_expected_compression_ratio(model_hidden_size, profile) >= self.min_compression_ratio
            )
            compatibility[profile] = is_compatible
        
        return compatibility