"""
Legacy wrapper to maintain backward compatibility with the original LLaMACompressionProfiles class.
This allows existing code to continue working while using the new modular architecture.
"""
from typing import Dict, Any, Optional
import torch

from ..model.model_loader import LLaMAModelLoader
from ..config.compression_config import CompressionConfig
from .profile_builder import LLaMACompressionProfileBuilder
from .compression_algorithms import SVDCompressionAlgorithm


class LLaMACompressionProfiles:
    """
    Legacy wrapper for backward compatibility with the original LLaMACompressionProfiles class.
    Delegates to the new modular components while maintaining the original API.
    """
    
    def __init__(self, model_loader, layer_idx: int = -1):
        """
        Initialize with the original API for backward compatibility.
        
        Args:
            model_loader: Original model loader or new ModelLoaderInterface
            layer_idx: Layer index to build profiles for
        """
        # Handle both old and new model loader types
        if hasattr(model_loader, 'load_model'):
            # New interface - ensure model is loaded
            if not hasattr(model_loader, 'model') or model_loader.model is None:
                model_loader.load_model()
            self.model_loader = model_loader
        else:
            # Legacy interface - wrap in new interface
            from ..model.model_loader import LLaMAModelLoader
            from ..config.model_config import ModelConfig
            
            # Create config from old loader
            config = ModelConfig(model_path=model_loader.model_path)
            self.model_loader = LLaMAModelLoader(config)
            if not hasattr(self.model_loader, 'model') or self.model_loader.model is None:
                self.model_loader.load_model()
        
        self.layer_idx = layer_idx
        
        # Create compression configuration with aggressive ranks to match original
        self.config = CompressionConfig(
            value_compression_ranks={
                "low": 32,
                "med": 48, 
                "high": 64
            },
            key_compression_rank=32
        )
        
        # Create profile builder
        self.profile_builder = LLaMACompressionProfileBuilder(
            self.model_loader,
            self.config
        )
        
        # Build profiles
        self.profile_builder.build_compression_profiles(layer_idx)
        
        # Expose legacy attributes for backward compatibility
        self._setup_legacy_attributes()
    
    def _setup_legacy_attributes(self):
        """Setup legacy attributes to maintain API compatibility."""
        model_config = self.model_loader.get_model_config()
        
        # Model dimensions
        self.hidden_size = model_config.get_hidden_size()
        self.head_dim = model_config.get_head_dim()
        self.vocab_size = model_config.get_vocab_size()
        self.num_query_heads = model_config.get_num_heads()
        self.num_kv_heads = model_config.get_num_key_value_heads()
        
        # Compression ranks
        self.value_compression_ranks = self.config.value_compression_ranks.copy()
        self.key_compression_rank = self.config.key_compression_rank
        
        # Per-head information
        self.heads_per_kv_head = self.num_query_heads // self.num_kv_heads
        
        # Expose matrices for backward compatibility
        self.key_compression_matrices = self.profile_builder.key_compression_matrices
        self.key_reconstruction_matrices = self.profile_builder.key_reconstruction_matrices
        
        # Profiles
        self.profiles = self.profile_builder.profiles
        
        # Original weights for compatibility
        self.attention_weights = self.model_loader.get_attention_weights(self.layer_idx)
        self.lm_head_weight = self.model_loader.get_language_model_head()
    
    def compress_values(self, values: torch.Tensor, profile_name: str, head_idx: int = 0) -> torch.Tensor:
        """
        Compress value tensors using specified profile and head.
        Maintains original API for backward compatibility.
        """
        return self.profile_builder.compress_values_with_profile(values, profile_name, head_idx)
    
    def compress_keys(self, keys: torch.Tensor, head_idx: int = 0) -> torch.Tensor:
        """
        Compress key tensors using fixed compression rank.
        Maintains original API for backward compatibility.
        """
        return self.profile_builder.compress_keys(keys, 0, head_idx)  # layer_idx=0 (not used)
    
    def reconstruct_keys(self, compressed_keys: torch.Tensor, head_idx: int = 0) -> torch.Tensor:
        """
        Reconstruct key tensors from compressed representation.
        Maintains original API for backward compatibility.
        """
        return self.profile_builder.reconstruct_keys(compressed_keys, head_idx)
    
    def direct_vocab_projection(self, compressed_values: torch.Tensor, profile_name: str, head_idx: int = 0) -> torch.Tensor:
        """
        Direct projection from compressed values to vocabulary logits.
        Maintains original API for backward compatibility.
        """
        if profile_name not in self.profiles:
            raise ValueError(f"Unknown profile: {profile_name}")
        
        W_fused = self.profiles[profile_name]["W_fused"][head_idx]  # [vocab_size, value_rank]
        
        if compressed_values.dim() == 1:
            # Single token: [value_rank] -> [vocab_size]
            return W_fused @ compressed_values
        else:
            # Multiple tokens: [seq_len, value_rank] -> [seq_len, vocab_size]
            return compressed_values @ W_fused.T
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """
        Get compression statistics for all profiles.
        Maintains original API for backward compatibility.
        """
        return self.profile_builder.get_compression_stats(0)  # layer_idx not used
    
    def print_compression_summary(self):
        """
        Print comprehensive compression summary.
        Maintains original API for backward compatibility.
        """
        print(f"\n📊 LLaMA-3 8B Compression Profile Summary")
        print(f"=" * 60)
        print(f"Model: LLaMA-3-8B-Instruct")
        print(f"Hidden size: {self.hidden_size}")
        print(f"Head dimension: {self.head_dim}")
        print(f"Number of heads: {self.num_query_heads}")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Layer used: {self.layer_idx if self.layer_idx >= 0 else 'last'}")
        
        stats = self.get_compression_stats()
        
        for profile_name, profile_stats in stats.items():
            print(f"\n{profile_name.upper()} Compression Profile:")
            print(f"  Value rank: {profile_stats['value_rank']} (per head)")
            print(f"  Key rank: {profile_stats['key_rank']} (per head)")
            print(f"  Query heads: {profile_stats['num_query_heads']}")
            print(f"  KV heads: {profile_stats['num_kv_heads']}")
            print(f"  Value compression: {profile_stats['value_compression_ratio']:.2f}x")
            print(f"  Key compression: {profile_stats['key_compression_ratio']:.2f}x")
            print(f"  Total compression: {profile_stats['total_compression_ratio']:.2f}x")
            print(f"  Memory savings: {profile_stats['memory_savings_percent']:.1f}%")
    
    def _validate_profile_shapes(self, profile_name: str):
        """
        Validate profile shapes.
        Maintains original API for backward compatibility.
        """
        return self.profile_builder.validate_profile_shapes(0, profile_name)
    
    def _build_compression_profiles(self):
        """
        Legacy method - profiles are already built in __init__.
        Kept for API compatibility.
        """
        pass  # Already built in __init__
    
    # Additional legacy methods that might be used by existing code
    def get_profile(self, profile_name: str) -> Dict[str, Any]:
        """Get a specific compression profile."""
        return self.profile_builder.get_profile(profile_name)
    
    def get_available_profiles(self):
        """Get list of available compression profiles."""
        return self.profile_builder.get_available_profiles()
    
    # Properties for backward compatibility
    @property
    def model_config(self):
        """Access to model configuration for backward compatibility."""
        return self.model_loader.get_model_config()
    
    @property
    def algorithm(self):
        """Access to compression algorithm for backward compatibility."""
        return self.profile_builder.algorithm