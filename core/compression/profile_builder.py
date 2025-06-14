"""
Compression profile builder for LLaMA models.
Handles the creation of compression profiles from model weights.
"""
import torch
from typing import Dict, List, Optional, Tuple, Any
from ..interfaces.compression_interface import CompressionProfileInterface
from ..interfaces.model_interface import ModelLoaderInterface
from ..config.compression_config import CompressionConfig
from .compression_algorithms import SVDCompressionAlgorithm


class LLaMACompressionProfileBuilder(CompressionProfileInterface):
    """
    Builds compression profiles for LLaMA models using SVD-based compression.
    Handles Grouped Query Attention (GQA) architecture with per-head decomposition.
    """
    
    def __init__(self, 
                 model_loader: ModelLoaderInterface,
                 compression_config: Optional[CompressionConfig] = None,
                 compression_algorithm: Optional[SVDCompressionAlgorithm] = None):
        """
        Initialize the profile builder.
        
        Args:
            model_loader: Model loader implementing ModelLoaderInterface
            compression_config: Compression configuration
            compression_algorithm: Compression algorithm instance
        """
        self.model_loader = model_loader
        self.config = compression_config or CompressionConfig()
        self.algorithm = compression_algorithm or SVDCompressionAlgorithm(
            solver=self.config.svd_solver,
            tolerance=self.config.svd_tolerance
        )
        
        # Model configuration
        self.model_config = model_loader.get_model_config()
        
        # Cached profiles and matrices
        self.profiles: Dict[str, Dict[str, Any]] = {}
        self.key_compression_matrices: List[torch.Tensor] = []
        self.key_reconstruction_matrices: List[torch.Tensor] = []
        
        # GQA configuration
        self.num_query_heads = self.model_config.get_num_heads()
        self.num_kv_heads = self.model_config.get_num_key_value_heads()
        self.head_dim = self.model_config.get_head_dim()
        self.heads_per_kv_head = self.num_query_heads // self.num_kv_heads
        
        print(f"🔧 Initialized compression profile builder:")
        print(f"   Query heads: {self.num_query_heads}")
        print(f"   KV heads: {self.num_kv_heads}")
        print(f"   Head dimension: {self.head_dim}")
        print(f"   Architecture: {self.model_config.get_architecture_type()}")
    
    def build_compression_profiles(self, layer_idx: int = -1) -> None:
        """
        Build compression profiles for a specific layer.
        
        Args:
            layer_idx: Layer index to build profiles for (-1 for last layer)
        """
        print(f"🔧 Building compression profiles for layer {layer_idx}...")
        
        # Extract model weights
        attention_weights = self.model_loader.get_attention_weights(layer_idx)
        lm_head_weight = self.model_loader.get_language_model_head()
        
        # Validate weight shapes
        self._validate_attention_weights(attention_weights)
        
        # Build key compression matrices (shared across all profiles)
        self._build_key_compression_matrices(attention_weights["W_K"])
        
        # Build value compression profiles for each compression level
        for profile_name in self.config.get_compression_profile_names():
            self._build_value_compression_profile(
                profile_name,
                attention_weights,
                lm_head_weight
            )
            
            # Validate the built profile
            if self.config.validate_shapes:
                self.validate_profile_shapes(layer_idx, profile_name)
        
        print(f"✅ Built {len(self.profiles)} compression profiles")
        self._print_profile_summary()
    
    def _validate_attention_weights(self, attention_weights: Dict[str, torch.Tensor]) -> None:
        """Validate extracted attention weights have expected shapes."""
        expected_q_size = self.num_query_heads * self.head_dim
        expected_kv_size = self.num_kv_heads * self.head_dim
        
        W_Q, W_K, W_V, W_O = attention_weights["W_Q"], attention_weights["W_K"], attention_weights["W_V"], attention_weights["W_O"]
        
        if W_Q.shape[0] != expected_q_size:
            raise ValueError(f"W_Q shape mismatch: expected {expected_q_size}, got {W_Q.shape[0]}")
        if W_K.shape[0] != expected_kv_size:
            raise ValueError(f"W_K shape mismatch: expected {expected_kv_size}, got {W_K.shape[0]}")
        if W_V.shape[0] != expected_kv_size:
            raise ValueError(f"W_V shape mismatch: expected {expected_kv_size}, got {W_V.shape[0]}")
        
        print(f"   ✅ Attention weight shapes validated for GQA architecture")
    
    def _build_key_compression_matrices(self, W_K: torch.Tensor) -> None:
        """Build key compression matrices for all KV heads."""
        print(f"   Building key compression (rank {self.config.key_compression_rank})...")
        
        # Reshape for per-head processing
        W_K_heads = W_K.reshape(self.num_kv_heads, self.head_dim, -1)  # [num_kv_heads, head_dim, hidden_size]
        
        self.key_compression_matrices = []
        self.key_reconstruction_matrices = []
        
        for head_idx in range(self.num_kv_heads):
            W_K_head = W_K_heads[head_idx]  # [head_dim, hidden_size]
            
            # Transpose for compression algorithm which expects [hidden_size, head_dim]
            W_K_head_T = W_K_head.T  # [hidden_size, head_dim]
            
            # Perform SVD compression
            compression_matrix, reconstruction_matrix = self.algorithm.perform_svd_compression(
                W_K_head_T, 
                self.config.key_compression_rank
            )
            
            # compression_matrix: [key_rank, hidden_size]
            # reconstruction_matrix: [hidden_size, key_rank]
            # We need A_K: [key_rank, head_dim] and B_K: [head_dim, key_rank]
            
            A_K = compression_matrix @ W_K_head_T  # [key_rank, head_dim]
            B_K = reconstruction_matrix.T  # [head_dim, key_rank]
            
            self.key_compression_matrices.append(A_K)
            self.key_reconstruction_matrices.append(B_K)
        
        print(f"     ✅ Built key compression for {self.num_kv_heads} KV heads")
    
    def _build_value_compression_profile(self, 
                                       profile_name: str,
                                       attention_weights: Dict[str, torch.Tensor],
                                       lm_head_weight: torch.Tensor) -> None:
        """Build value compression profile for a specific compression level."""
        print(f"   Building {profile_name} profile...")
        
        value_rank = self.config.get_value_rank(profile_name)
        
        W_V = attention_weights["W_V"]  # [num_kv_heads * head_dim, hidden_size]
        W_O = attention_weights["W_O"]  # [hidden_size, hidden_size]
        
        # Reshape for per-head processing
        W_V_heads = W_V.reshape(self.num_kv_heads, self.head_dim, -1)  # [num_kv_heads, head_dim, hidden_size]
        
        A_V_heads = []
        W_fused_heads = []
        
        # Process each value head and corresponding query heads
        for kv_head_idx in range(self.num_kv_heads):
            W_V_head = W_V_heads[kv_head_idx]  # [head_dim, hidden_size]
            
            # Process all query heads that correspond to this KV head
            for local_q_idx in range(self.heads_per_kv_head):
                global_q_idx = kv_head_idx * self.heads_per_kv_head + local_q_idx
                
                # Extract output projection for this query head
                start_idx = global_q_idx * self.head_dim
                end_idx = start_idx + self.head_dim
                W_O_head = W_O[:, start_idx:end_idx]  # [hidden_size, head_dim]
                
                # Perform SVD compression with fused output projection
                A_V_head, W_fused_head = self.algorithm.perform_value_svd_with_fusion(
                    W_V_head,
                    W_O_head,
                    lm_head_weight,
                    value_rank
                )
                
                A_V_heads.append(A_V_head)
                W_fused_heads.append(W_fused_head)
        
        # Stack matrices
        A_V_all = torch.stack(A_V_heads, dim=0)      # [num_query_heads, value_rank, head_dim]
        W_fused_all = torch.stack(W_fused_heads, dim=0)  # [num_query_heads, vocab_size, value_rank]
        
        # Prepare key matrices (shared across profiles)
        A_K_all = torch.stack(self.key_compression_matrices, dim=0)   # [num_kv_heads, key_rank, head_dim]
        B_K_all = torch.stack(self.key_reconstruction_matrices, dim=0)  # [num_kv_heads, head_dim, key_rank]
        
        # Store profile
        actual_value_rank = A_V_all.shape[1]
        self.profiles[profile_name] = {
            # Value compression matrices
            "A_V": A_V_all,
            "W_fused": W_fused_all,
            "value_rank": actual_value_rank,
            
            # Key compression matrices
            "A_K": A_K_all,
            "B_K": B_K_all,
            "key_rank": self.config.key_compression_rank,
            
            # Architecture information
            "num_query_heads": self.num_query_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            
            # Original matrices for validation (first head only)
            "W_V_original": W_V_heads[0],
            "W_K_original": W_K.reshape(self.num_kv_heads, self.head_dim, -1)[0].T,
            "W_O_original": W_O[:, :self.head_dim],
        }
        
        print(f"     ✅ {profile_name}: A_V{A_V_all.shape}, W_fused{W_fused_all.shape}")
    
    def compress_values(self, values: torch.Tensor, layer_idx: int, head_idx: int = 0) -> torch.Tensor:
        """
        Compress value tensors using the default compression profile.
        
        Args:
            values: Value tensor [seq_len, head_dim] or [head_dim]
            layer_idx: Layer index (for interface compatibility)
            head_idx: Attention head index
            
        Returns:
            Compressed values
        """
        # Use medium compression as default
        return self.compress_values_with_profile(values, "med", head_idx)
    
    def compress_values_with_profile(self, 
                                   values: torch.Tensor, 
                                   profile_name: str, 
                                   head_idx: int = 0) -> torch.Tensor:
        """
        Compress value tensors using specified profile.
        
        Args:
            values: Value tensor [seq_len, head_dim] or [head_dim]
            profile_name: Compression profile name
            head_idx: Attention head index
            
        Returns:
            Compressed values
        """
        if profile_name not in self.profiles:
            raise ValueError(f"Unknown profile: {profile_name}")
        
        if head_idx >= self.num_query_heads:
            raise ValueError(f"Head index {head_idx} out of range for {self.num_query_heads} heads")
        
        A_V = self.profiles[profile_name]["A_V"][head_idx]  # [value_rank, head_dim]
        
        if values.dim() == 1:
            # Single token: [head_dim] -> [value_rank]
            return A_V @ values
        else:
            # Multiple tokens: [seq_len, head_dim] -> [seq_len, value_rank]
            return values @ A_V.T
    
    def compress_keys(self, keys: torch.Tensor, layer_idx: int, head_idx: int = 0) -> torch.Tensor:
        """
        Compress key tensors using fixed compression rank.
        
        Args:
            keys: Key tensor [seq_len, head_dim] or [head_dim]
            layer_idx: Layer index (for interface compatibility)
            head_idx: Query head index
            
        Returns:
            Compressed keys
        """
        if head_idx >= self.num_query_heads:
            raise ValueError(f"Head index {head_idx} out of range for {self.num_query_heads} heads")
        
        # Map query head to corresponding KV head
        kv_head_idx = head_idx // self.heads_per_kv_head
        A_K = self.key_compression_matrices[kv_head_idx]  # [key_rank, head_dim]
        
        if keys.dim() == 1:
            # Single token: [head_dim] -> [key_rank]
            return A_K @ keys
        else:
            # Multiple tokens: [seq_len, head_dim] -> [seq_len, key_rank]
            return keys @ A_K.T
    
    def reconstruct_keys(self, compressed_keys: torch.Tensor, head_idx: int = 0) -> torch.Tensor:
        """
        Reconstruct key tensors from compressed representation.
        
        Args:
            compressed_keys: Compressed keys [seq_len, key_rank] or [key_rank]
            head_idx: Query head index
            
        Returns:
            Reconstructed keys
        """
        if head_idx >= self.num_query_heads:
            raise ValueError(f"Head index {head_idx} out of range for {self.num_query_heads} heads")
        
        # Map query head to corresponding KV head
        kv_head_idx = head_idx // self.heads_per_kv_head
        B_K = self.key_reconstruction_matrices[kv_head_idx]  # [head_dim, key_rank]
        
        # Validate dimensions
        expected_key_rank = B_K.shape[1]
        actual_key_rank = compressed_keys.shape[-1]
        
        if actual_key_rank != expected_key_rank:
            raise ValueError(f"Key rank mismatch: expected {expected_key_rank}, got {actual_key_rank}")
        
        if compressed_keys.dim() == 1:
            # Single token: [key_rank] -> [head_dim]
            return B_K @ compressed_keys
        else:
            # Multiple tokens: [seq_len, key_rank] -> [seq_len, head_dim]
            return compressed_keys @ B_K.T
    
    def get_compression_stats(self, layer_idx: int) -> Dict[str, Any]:
        """
        Get compression statistics for all profiles.
        
        Args:
            layer_idx: Layer index (for interface compatibility)
            
        Returns:
            Dictionary containing compression statistics
        """
        stats = {}
        
        for profile_name, profile in self.profiles.items():
            value_rank = profile["value_rank"]
            key_rank = profile["key_rank"]
            
            # Calculate parameter counts
            original_value_params = self.num_kv_heads * self.head_dim * self.model_config.get_hidden_size()
            original_key_params = self.num_kv_heads * self.head_dim * self.model_config.get_hidden_size()
            original_output_params = self.model_config.get_hidden_size() * self.model_config.get_hidden_size()
            
            compressed_value_params = self.num_query_heads * (value_rank * self.head_dim + self.model_config.get_vocab_size() * value_rank)
            compressed_key_params = self.num_kv_heads * (key_rank * self.head_dim + self.head_dim * key_rank)
            
            # Calculate compression ratios
            value_compression_ratio = original_value_params / (compressed_value_params - self.num_query_heads * self.model_config.get_vocab_size() * value_rank)
            key_compression_ratio = original_key_params / compressed_key_params
            total_compression_ratio = (original_value_params + original_key_params + original_output_params) / (compressed_value_params + compressed_key_params)
            
            stats[profile_name] = {
                "value_rank": value_rank,
                "key_rank": key_rank,
                "num_query_heads": self.num_query_heads,
                "num_kv_heads": self.num_kv_heads,
                "value_compression_ratio": value_compression_ratio,
                "key_compression_ratio": key_compression_ratio,
                "total_compression_ratio": total_compression_ratio,
                "memory_savings_percent": (1 - 1/total_compression_ratio) * 100
            }
        
        return stats
    
    def validate_profile_shapes(self, layer_idx: int, profile_name: str = None) -> bool:
        """
        Validate compression profile shapes.
        
        Args:
            layer_idx: Layer index (for interface compatibility)
            profile_name: Profile name to validate (None for all profiles)
            
        Returns:
            True if shapes are valid, False otherwise
        """
        profiles_to_validate = [profile_name] if profile_name else list(self.profiles.keys())
        
        for name in profiles_to_validate:
            if name not in self.profiles:
                print(f"Warning: Profile {name} not found")
                return False
            
            profile = self.profiles[name]
            value_rank = profile["value_rank"]
            key_rank = profile["key_rank"]
            
            expected_shapes = {
                "A_V": (self.num_query_heads, value_rank, self.head_dim),
                "W_fused": (self.num_query_heads, self.model_config.get_vocab_size(), value_rank),
                "A_K": (self.num_kv_heads, key_rank, self.head_dim),
                "B_K": (self.num_kv_heads, self.head_dim, key_rank)
            }
            
            for matrix_name, expected_shape in expected_shapes.items():
                actual_shape = profile[matrix_name].shape
                if actual_shape != expected_shape:
                    print(f"Shape validation failed for {name}.{matrix_name}: expected {expected_shape}, got {actual_shape}")
                    return False
        
        return True
    
    def _print_profile_summary(self) -> None:
        """Print summary of built compression profiles."""
        print(f"\n📊 Compression Profile Summary:")
        print(f"   Profiles built: {list(self.profiles.keys())}")
        
        stats = self.get_compression_stats(0)  # Layer index not used in current implementation
        
        for profile_name, profile_stats in stats.items():
            print(f"   {profile_name.upper()}: {profile_stats['total_compression_ratio']:.2f}x compression, "
                  f"{profile_stats['memory_savings_percent']:.1f}% memory savings")
    
    def get_profile(self, profile_name: str) -> Dict[str, Any]:
        """Get a specific compression profile."""
        if profile_name not in self.profiles:
            raise ValueError(f"Profile {profile_name} not found. Available: {list(self.profiles.keys())}")
        return self.profiles[profile_name]
    
    def get_available_profiles(self) -> List[str]:
        """Get list of available compression profiles."""
        return list(self.profiles.keys())