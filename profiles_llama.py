"""
LLaMA-3 8B Real Compression Profiles
SVD-based compression using actual model weights - NO PLACEHOLDERS
"""
import torch
from typing import Dict, Tuple
from llama_model_loader import LLaMAModelLoader
from compression import decompose_and_fuse, compress_keys

class LLaMACompressionProfiles:
    def __init__(self, model_loader: LLaMAModelLoader, layer_idx: int = -1):
        self.model_loader = model_loader
        self.layer_idx = layer_idx
        
        # Get real model dimensions
        self.hidden_size = model_loader.hidden_size
        self.head_dim = model_loader.head_dim
        self.vocab_size = model_loader.vocab_size
        
        # Compression ranks for value tensors (adaptive)
        self.value_compression_ranks = {
            "low": 64,      # Low compression, high quality
            "med": 128,     # Medium compression
            "high": 256     # High compression
        }
        
        # Fixed key compression rank
        self.key_compression_rank = 128
        
        # Extract real model weights
        self.attention_weights = model_loader.get_attention_weights(layer_idx)
        self.lm_head_weight = model_loader.get_language_model_head()
        
        # Build compression profiles
        self.profiles = {}
        self._build_compression_profiles()
        
    def _build_compression_profiles(self):
        """Build real compression profiles using actual LLaMA weights"""
        print(f"🔧 Building compression profiles using real LLaMA-3 weights...")
        print(f"   Using layer {self.layer_idx if self.layer_idx >= 0 else 'last'}")
        
        # Extract real weight matrices
        W_V = self.attention_weights["W_V"]  # [hidden_size, hidden_size]
        W_O = self.attention_weights["W_O"]  # [hidden_size, hidden_size]
        W_K = self.attention_weights["W_K"]  # [hidden_size, hidden_size]
        W_LM_HEAD = self.lm_head_weight      # [vocab_size, hidden_size]
        
        print(f"   Real weight shapes: W_V{W_V.shape}, W_O{W_O.shape}, W_K{W_K.shape}")
        print(f"   LM head shape: {W_LM_HEAD.shape}")
        
        # Project to head dimension for single-head compression
        # In multi-head attention, we work with per-head projections
        W_V_head = W_V[:, :self.head_dim]  # [hidden_size, head_dim]
        W_K_head = W_K[:, :self.head_dim]  # [hidden_size, head_dim]
        W_O_head = W_O[:self.head_dim, :]  # [head_dim, hidden_size]
        
        # Build key compression matrices (fixed rank for all profiles)
        print(f"   Building key compression (fixed rank {self.key_compression_rank})...")
        A_K, B_K = compress_keys(W_K_head, self.key_compression_rank)
        
        # Build value compression profiles (adaptive ranks)
        for profile_name, value_rank in self.value_compression_ranks.items():
            print(f"   Building {profile_name} profile (value rank {value_rank})...")
            
            # Value compression using real SVD
            A_V, W_fused_attn = decompose_and_fuse(W_V_head, W_O_head.T, value_rank)
            
            # Create fused output projection for language modeling
            # We need to fuse the attention output with the language model head
            U, S, V = torch.svd(W_V_head)
            U_truncated = U[:, :value_rank]  # [head_dim, value_rank]
            
            # For the fused LM projection, we need to go through the attention output first
            # Path: compressed_value -> U_truncated -> W_O -> W_LM_HEAD
            # First fuse attention output projection
            W_attn_to_hidden = W_O_head @ U_truncated  # [hidden_size, value_rank]
            
            # Then fuse with language model head
            W_fused_lm = W_LM_HEAD @ W_attn_to_hidden  # [vocab_size, value_rank]
            
            self.profiles[profile_name] = {
                # Value compression matrices
                "A_V": A_V,                    # [value_rank, head_dim] - compression matrix
                "W_fused": W_fused_lm,         # [vocab_size, value_rank] - fused LM projection
                "value_rank": value_rank,      # Value compression rank
                
                # Key compression matrices (shared across all profiles)
                "A_K": A_K,                    # [key_rank, head_dim] - key compression
                "B_K": B_K,                    # [head_dim, key_rank] - key reconstruction
                "key_rank": self.key_compression_rank,  # Key compression rank
                
                # Original matrices for validation
                "W_V_original": W_V_head,      # Original value projection
                "W_K_original": W_K_head,      # Original key projection
                "W_O_original": W_O_head.T     # Original output projection
            }
            
            # Validate matrix shapes
            self._validate_profile_shapes(profile_name)
            
        print(f"✅ Built {len(self.profiles)} compression profiles")
        
    def _validate_profile_shapes(self, profile_name: str):
        """Validate all matrix shapes for a profile"""
        profile = self.profiles[profile_name]
        value_rank = profile["value_rank"]
        key_rank = profile["key_rank"]
        
        # Expected shapes
        expected_shapes = {
            "A_V": (value_rank, self.head_dim),
            "W_fused": (self.vocab_size, value_rank),
            "A_K": (key_rank, self.head_dim),
            "B_K": (self.head_dim, key_rank)
        }
        
        errors = []
        for matrix_name, expected_shape in expected_shapes.items():
            actual_shape = profile[matrix_name].shape
            if actual_shape != expected_shape:
                errors.append(f"{matrix_name}: expected {expected_shape}, got {actual_shape}")
        
        if errors:
            raise ValueError(f"Shape validation failed for {profile_name}: {errors}")
        
        print(f"     ✅ {profile_name}: A_V{profile['A_V'].shape}, W_fused{profile['W_fused'].shape}")
        print(f"         A_K{profile['A_K'].shape}, B_K{profile['B_K'].shape}")
    
    def compress_values(self, values: torch.Tensor, profile_name: str) -> torch.Tensor:
        """
        Compress value tensors using specified profile
        
        Args:
            values: Value tensor [seq_len, head_dim] or [head_dim]
            profile_name: Compression profile ("low", "med", "high")
            
        Returns:
            Compressed values [seq_len, value_rank] or [value_rank]
        """
        if profile_name not in self.profiles:
            raise ValueError(f"Unknown profile: {profile_name}")
        
        A_V = self.profiles[profile_name]["A_V"]
        
        if values.dim() == 1:
            # Single token: [head_dim] -> [value_rank]
            return A_V @ values
        else:
            # Multiple tokens: [seq_len, head_dim] -> [seq_len, value_rank]
            return values @ A_V.T
    
    def compress_keys(self, keys: torch.Tensor) -> torch.Tensor:
        """
        Compress key tensors using fixed compression rank
        
        Args:
            keys: Key tensor [seq_len, head_dim] or [head_dim]
            
        Returns:
            Compressed keys [seq_len, key_rank] or [key_rank]
        """
        # Use A_K from any profile (they're all the same)
        A_K = self.profiles["med"]["A_K"]
        
        if keys.dim() == 1:
            # Single token: [head_dim] -> [key_rank]
            return A_K @ keys
        else:
            # Multiple tokens: [seq_len, head_dim] -> [seq_len, key_rank]
            return keys @ A_K.T
    
    def reconstruct_keys(self, compressed_keys: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct keys from compressed representation on-the-fly
        
        Args:
            compressed_keys: Compressed keys [seq_len, key_rank] or [key_rank]
            
        Returns:
            Reconstructed keys [seq_len, head_dim] or [head_dim]
        """
        # Use B_K from any profile (they're all the same)
        B_K = self.profiles["med"]["B_K"]
        
        if compressed_keys.dim() == 1:
            # Single token: [key_rank] -> [head_dim]
            return B_K @ compressed_keys
        else:
            # Multiple tokens: [seq_len, key_rank] -> [seq_len, head_dim]
            return compressed_keys @ B_K.T
    
    def decode_to_logits(self, compressed_values: torch.Tensor, profile_name: str) -> torch.Tensor:
        """
        Decode compressed values directly to vocabulary logits using fused matrix
        
        Args:
            compressed_values: Compressed values [seq_len, value_rank] or [value_rank]
            profile_name: Compression profile
            
        Returns:
            Logits [seq_len, vocab_size] or [vocab_size]
        """
        if profile_name not in self.profiles:
            raise ValueError(f"Unknown profile: {profile_name}")
        
        W_fused = self.profiles[profile_name]["W_fused"]
        
        if compressed_values.dim() == 1:
            # Single token: [value_rank] -> [vocab_size]
            return W_fused @ compressed_values
        else:
            # Multiple tokens: [seq_len, value_rank] -> [seq_len, vocab_size]
            return compressed_values @ W_fused.T
    
    def get_compression_stats(self) -> Dict[str, Dict[str, float]]:
        """Get compression statistics for all profiles"""
        stats = {}
        
        for profile_name, profile in self.profiles.items():
            value_rank = profile["value_rank"]
            key_rank = profile["key_rank"]
            
            # Calculate memory savings
            original_value_params = self.hidden_size * self.head_dim + self.vocab_size * self.hidden_size
            compressed_value_params = value_rank * self.head_dim + self.vocab_size * value_rank
            value_compression_ratio = original_value_params / compressed_value_params
            
            original_key_params = self.hidden_size * self.head_dim
            compressed_key_params = key_rank * self.head_dim + self.head_dim * key_rank
            key_compression_ratio = original_key_params / compressed_key_params
            
            stats[profile_name] = {
                "value_rank": value_rank,
                "key_rank": key_rank,
                "value_compression_ratio": value_compression_ratio,
                "key_compression_ratio": key_compression_ratio,
                "total_compression_ratio": (original_value_params + original_key_params) / 
                                         (compressed_value_params + compressed_key_params)
            }
        
        return stats
    
    def print_compression_summary(self):
        """Print comprehensive compression summary"""
        print(f"\n📊 LLaMA-3 8B Compression Profile Summary")
        print(f"=" * 60)
        print(f"Model: LLaMA-3-8B-Instruct")
        print(f"Hidden size: {self.hidden_size}")
        print(f"Head dimension: {self.head_dim}")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Layer used: {self.layer_idx if self.layer_idx >= 0 else 'last'}")
        
        stats = self.get_compression_stats()
        
        for profile_name, profile_stats in stats.items():
            print(f"\n{profile_name.upper()} Compression Profile:")
            print(f"  Value rank: {profile_stats['value_rank']}")
            print(f"  Key rank: {profile_stats['key_rank']}")
            print(f"  Value compression: {profile_stats['value_compression_ratio']:.2f}x")
            print(f"  Key compression: {profile_stats['key_compression_ratio']:.2f}x")
            print(f"  Total compression: {profile_stats['total_compression_ratio']:.2f}x")
