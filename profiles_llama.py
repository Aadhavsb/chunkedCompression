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
        self.num_query_heads = model_loader.num_attention_heads  # 32 for LLaMA-3 8B
        self.num_kv_heads = model_loader.num_key_value_heads    # 8 for LLaMA-3 8B (GQA)
        
        # Compression ranks for value tensors (adaptive) - limited by head_dim=128
        self.value_compression_ranks = {
            "low": 64,      # Low compression, high quality
            "med": 96,      # Medium compression  
            "high": 128     # High compression (max possible for head_dim=128)
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
        """Build real compression profiles using actual LLaMA weights with per-head decomposition"""
        print(f"🔧 Building compression profiles using real LLaMA-3 weights...")
        print(f"   Using layer {self.layer_idx if self.layer_idx >= 0 else 'last'}")
        
        # Extract real weight matrices
        W_Q = self.attention_weights["W_Q"]  # [hidden_size, hidden_size] - all query heads
        W_V = self.attention_weights["W_V"]  # [num_kv_heads * head_dim, hidden_size] - fewer value heads
        W_O = self.attention_weights["W_O"]  # [hidden_size, hidden_size]
        W_K = self.attention_weights["W_K"]  # [num_kv_heads * head_dim, hidden_size] - fewer key heads
        W_LM_HEAD = self.lm_head_weight      # [vocab_size, hidden_size]
        
        print(f"   Real weight shapes:")
        print(f"     W_Q: {W_Q.shape} (query: {self.num_query_heads} heads)")
        print(f"     W_K: {W_K.shape} (key: {self.num_kv_heads} heads)")
        print(f"     W_V: {W_V.shape} (value: {self.num_kv_heads} heads)")
        print(f"     W_O: {W_O.shape}")
        print(f"   LM head shape: {W_LM_HEAD.shape}")
        
        # Calculate expected dimensions for GQA
        expected_q_size = self.num_query_heads * self.head_dim  # 32 * 128 = 4096
        expected_kv_size = self.num_kv_heads * self.head_dim    # 8 * 128 = 1024
        
        # Verify dimensions match GQA architecture
        assert W_Q.shape[0] == expected_q_size, f"W_Q shape mismatch: expected {expected_q_size}, got {W_Q.shape[0]}"
        assert W_V.shape[0] == expected_kv_size, f"W_V shape mismatch: expected {expected_kv_size}, got {W_V.shape[0]}"
        assert W_K.shape[0] == expected_kv_size, f"W_K shape mismatch: expected {expected_kv_size}, got {W_K.shape[0]}"
        
        print(f"   GQA architecture: {self.num_query_heads} query heads, {self.num_kv_heads} key/value heads")
        
        # 1️⃣ Per-head slicing for GQA architecture
        # Query heads: reshape for all 32 heads
        W_Q_heads = W_Q.reshape(self.num_query_heads, self.head_dim, self.hidden_size)  # [32, 128, 4096]
        
        # Key/Value heads: reshape for 8 heads only (GQA)
        W_V_heads = W_V.reshape(self.num_kv_heads, self.head_dim, self.hidden_size)    # [8, 128, 4096]
        W_K_heads = W_K.reshape(self.num_kv_heads, self.head_dim, self.hidden_size)    # [8, 128, 4096]
        
        # Output projection: split by query head output dimensions (since output has 32 head dims)
        # W_O projects from [32 * 128] back to [4096]
        W_O_heads = []
        for head_idx in range(self.num_query_heads):
            # Extract output projection for this head: [hidden_size, head_dim]
            W_O_head = W_O[:, head_idx * self.head_dim:(head_idx + 1) * self.head_dim]
            W_O_heads.append(W_O_head)
        W_O_heads = torch.stack(W_O_heads, dim=0)  # [32, hidden_size, head_dim]
        
        # 2️⃣ Build key compression matrices per head (fixed rank for all profiles)
        print(f"   Building key compression (fixed rank {self.key_compression_rank})...")
        A_K_heads = []
        B_K_heads = []
        
        for head_idx in range(self.num_kv_heads):  # Only 8 key heads in GQA
            W_K_head = W_K_heads[head_idx]  # [head_dim, hidden_size]
            # Transpose for compress_keys function which expects [hidden_size, head_dim]
            W_K_head_T = W_K_head.T  # [hidden_size, head_dim]
            A_K_head, B_K_head = compress_keys(W_K_head_T, self.key_compression_rank)
            A_K_heads.append(A_K_head)
            B_K_heads.append(B_K_head)
        
        A_K_all = torch.stack(A_K_heads, dim=0)  # [num_kv_heads, key_rank, head_dim]
        B_K_all = torch.stack(B_K_heads, dim=0)  # [num_kv_heads, head_dim, key_rank]
        
        # 3️⃣ Build value compression profiles per head (adaptive ranks)
        for profile_name, value_rank in self.value_compression_ranks.items():
            print(f"   Building {profile_name} profile (value rank {value_rank})...")
            
            A_V_heads = []
            W_fused_heads = []
            
            # Process each value head individually (8 heads for GQA)
            for head_idx in range(self.num_kv_heads):
                W_V_head = W_V_heads[head_idx]  # [head_dim, hidden_size]
                
                # For GQA, we need to map this value head to corresponding query heads
                # Each value head serves multiple query heads (32/8 = 4 query heads per value head)
                heads_per_kv = self.num_query_heads // self.num_kv_heads  # 4
                
                # Process query heads that correspond to this value head
                for local_q_idx in range(heads_per_kv):
                    global_q_idx = head_idx * heads_per_kv + local_q_idx
                    W_O_head = W_O_heads[global_q_idx]  # [hidden_size, head_dim]
                    
                    # Convert to float32 for SVD (bfloat16 not supported)
                    W_V_f32 = W_V_head.float()  # [head_dim, hidden_size]
                    U, S, V = torch.svd(W_V_f32, some=True)
                    
                    # Ensure we don't exceed available dimensions
                    actual_rank = min(value_rank, U.shape[1], S.shape[0])
                    if actual_rank < value_rank:
                        print(f"     Warning: Requested rank {value_rank} reduced to {actual_rank} (limited by dimensions)")
                    
                    # Truncate to actual rank
                    U_truncated = U[:, :actual_rank].to(W_V_head.dtype)   # [head_dim, actual_rank]
                    S_truncated = S[:actual_rank].to(W_V_head.dtype)      # [actual_rank]
                    
                    # Create compression matrix A_V [actual_rank, head_dim]
                    # This transforms: value [head_dim] -> compressed_value [actual_rank]
                    A_V_head = U_truncated.T  # [actual_rank, head_dim]
                    A_V_heads.append(A_V_head)
                    
                    # Create fused output projection for this query head
                    # compressed_value [actual_rank] -> logits [vocab_size]
                    # W_LM_HEAD @ W_O_head @ U_truncated @ diag(S_truncated)
                    # [vocab_size, hidden_size] @ [hidden_size, head_dim] @ [head_dim, actual_rank] @ [actual_rank, actual_rank]
                    W_fused_head = W_LM_HEAD @ W_O_head @ U_truncated @ torch.diag(S_truncated)  # [vocab_size, actual_rank]
                    W_fused_heads.append(W_fused_head)
            
            # 4️⃣ Aggregate per-head profiles (back to 32 query heads)
            A_V_all = torch.stack(A_V_heads, dim=0)      # [num_query_heads, actual_rank, head_dim]
            W_fused_all = torch.stack(W_fused_heads, dim=0)  # [num_query_heads, vocab_size, actual_rank]
            
            # Get actual rank from the first matrix (all should be the same)
            actual_value_rank = A_V_all.shape[1]
            
            self.profiles[profile_name] = {
                # Value compression matrices (per query head for output compatibility)
                "A_V": A_V_all,                    # [num_query_heads, actual_rank, head_dim] - compression matrix
                "W_fused": W_fused_all,            # [num_query_heads, vocab_size, actual_rank] - fused LM projection
                "value_rank": actual_value_rank,   # Actual value compression rank (may be less than requested)
                
                # Key compression matrices (per kv head, shared across profiles)
                "A_K": A_K_all,                    # [num_kv_heads, key_rank, head_dim] - key compression
                "B_K": B_K_all,                    # [num_kv_heads, head_dim, key_rank] - key reconstruction
                "key_rank": self.key_compression_rank,  # Key compression rank
                
                # Multi-head information
                "num_query_heads": self.num_query_heads,  # Number of query heads (32)
                "num_kv_heads": self.num_kv_heads,        # Number of key/value heads (8)
                "head_dim": self.head_dim,         # Dimension per head
                
                # Original matrices for validation (first head only for compatibility)
                "W_V_original": W_V_heads[0],      # Original value projection [head_dim, hidden_size]
                "W_K_original": W_K_heads[0].T,    # Original key projection [hidden_size, head_dim] 
                "W_O_original": W_O_heads[0]       # Original output projection [hidden_size, head_dim]
            }
            
            # Validate matrix shapes
            self._validate_profile_shapes(profile_name)
            
        print(f"✅ Built {len(self.profiles)} compression profiles")
        
    def _validate_profile_shapes(self, profile_name: str):
        """Validate all matrix shapes for a profile with GQA per-head decomposition"""
        profile = self.profiles[profile_name]
        value_rank = profile["value_rank"]
        key_rank = profile["key_rank"]
        num_query_heads = profile["num_query_heads"]  # 32
        num_kv_heads = profile["num_kv_heads"]        # 8
        
        # Expected shapes for GQA per-head matrices
        expected_shapes = {
            "A_V": (num_query_heads, value_rank, self.head_dim),        # Per query head value compression
            "W_fused": (num_query_heads, self.vocab_size, value_rank),  # Per query head fused LM projection
            "A_K": (num_kv_heads, key_rank, self.head_dim),             # Per kv head key compression
            "B_K": (num_kv_heads, self.head_dim, key_rank)              # Per kv head key reconstruction
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
        print(f"         GQA matrices validated: {num_query_heads} query heads, {num_kv_heads} kv heads")
    
    def compress_values(self, values: torch.Tensor, profile_name: str, head_idx: int = 0) -> torch.Tensor:
        """
        Compress value tensors using specified profile and head
        
        Args:
            values: Value tensor [seq_len, head_dim] or [head_dim]
            profile_name: Compression profile ("low", "med", "high")
            head_idx: Attention head index (0 to num_heads-1)
            
        Returns:
            Compressed values [seq_len, value_rank] or [value_rank]
        """
        if profile_name not in self.profiles:
            raise ValueError(f"Unknown profile: {profile_name}")
        
        A_V = self.profiles[profile_name]["A_V"][head_idx]  # [value_rank, head_dim]
        
        if values.dim() == 1:
            # Single token: [head_dim] -> [value_rank]
            return A_V @ values
        else:
            # Multiple tokens: [seq_len, head_dim] -> [seq_len, value_rank]
            return values @ A_V.T
    
    def compress_keys(self, keys: torch.Tensor, head_idx: int = 0) -> torch.Tensor:
        """
        Compress key tensors using fixed compression rank for specified head
        
        Args:
            keys: Key tensor [seq_len, head_dim] or [head_dim]
            head_idx: Query head index (0 to num_query_heads-1)
            
        Returns:
            Compressed keys [seq_len, key_rank] or [key_rank]
        """
        # Map query head to corresponding key/value head (GQA)
        heads_per_kv = self.num_query_heads // self.num_kv_heads  # 4
        kv_head_idx = head_idx // heads_per_kv
        
        # Use A_K from any profile (they're all the same across profiles)
        A_K = self.profiles["med"]["A_K"][kv_head_idx]  # [key_rank, head_dim]
        
        if keys.dim() == 1:
            # Single token: [head_dim] -> [key_rank]
            return A_K @ keys
        else:
            # Multiple tokens: [seq_len, head_dim] -> [seq_len, key_rank]
            return keys @ A_K.T
    
    def reconstruct_keys(self, compressed_keys: torch.Tensor, head_idx: int = 0) -> torch.Tensor:
        """
        Reconstruct key tensors from compressed representation for specified head
        
        Args:
            compressed_keys: Compressed keys [seq_len, key_rank] or [key_rank]
            head_idx: Query head index (0 to num_query_heads-1)
            
        Returns:
            Reconstructed keys [seq_len, head_dim] or [head_dim]
        """
        # Map query head to corresponding key/value head (GQA)
        heads_per_kv = self.num_query_heads // self.num_kv_heads  # 4
        kv_head_idx = head_idx // heads_per_kv
        
        # Use B_K from any profile (they're all the same across profiles)
        B_K = self.profiles["med"]["B_K"][kv_head_idx]  # [head_dim, key_rank]
        
        if compressed_keys.dim() == 1:
            # Single token: [key_rank] -> [head_dim]
            return B_K @ compressed_keys
        else:
            # Multiple tokens: [seq_len, key_rank] -> [seq_len, head_dim]
            return compressed_keys @ B_K.T
    
    def decode_to_logits(self, compressed_values: torch.Tensor, profile_name: str, head_idx: int = 0) -> torch.Tensor:
        """
        Decode compressed values directly to vocabulary logits using fused matrix for specified head
        
        Args:
            compressed_values: Compressed values [seq_len, value_rank] or [value_rank]
            profile_name: Compression profile
            head_idx: Query head index (0 to num_query_heads-1)
            
        Returns:
            Logits [seq_len, vocab_size] or [vocab_size]
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
    
    def get_compression_stats(self) -> Dict[str, Dict[str, float]]:
        """Get compression statistics for all profiles with GQA per-head accounting"""
        stats = {}
        
        for profile_name, profile in self.profiles.items():
            value_rank = profile["value_rank"]
            key_rank = profile["key_rank"]
            num_query_heads = profile["num_query_heads"]  # 32
            num_kv_heads = profile["num_kv_heads"]        # 8
            
            # Calculate memory savings with GQA architecture
            # Original parameters - query heads use full dimensions, kv heads are shared
            original_value_params_per_kv_head = self.head_dim * self.hidden_size
            original_key_params_per_kv_head = self.head_dim * self.hidden_size
            original_output_params = self.vocab_size * self.hidden_size  # LM head (shared)
            
            # Compressed parameters - value compression per query head, key compression per kv head
            compressed_value_params_per_query_head = value_rank * self.head_dim + self.vocab_size * value_rank
            compressed_key_params_per_kv_head = key_rank * self.head_dim + self.head_dim * key_rank
            
            # Total parameters
            original_value_params_total = original_value_params_per_kv_head * num_kv_heads
            original_key_params_total = original_key_params_per_kv_head * num_kv_heads
            original_total = original_value_params_total + original_key_params_total + original_output_params
            
            compressed_value_params_total = compressed_value_params_per_query_head * num_query_heads
            compressed_key_params_total = compressed_key_params_per_kv_head * num_kv_heads
            compressed_total = compressed_value_params_total + compressed_key_params_total
            
            # Compression ratios
            value_compression_ratio = original_value_params_total / (compressed_value_params_total - self.vocab_size * value_rank * num_query_heads)
            key_compression_ratio = original_key_params_total / compressed_key_params_total
            total_compression_ratio = original_total / compressed_total
            
            stats[profile_name] = {
                "value_rank": value_rank,
                "key_rank": key_rank,
                "num_query_heads": num_query_heads,
                "num_kv_heads": num_kv_heads,
                "value_compression_ratio": value_compression_ratio,
                "key_compression_ratio": key_compression_ratio,
                "total_compression_ratio": total_compression_ratio,
                "memory_savings_percent": (1 - 1/total_compression_ratio) * 100
            }
        
        return stats
    
    def print_compression_summary(self):
        """Print comprehensive compression summary with per-head details"""
        print(f"\n📊 LLaMA-3 8B Compression Profile Summary")
        print(f"=" * 60)
        print(f"Model: LLaMA-3-8B-Instruct")
        print(f"Hidden size: {self.hidden_size}")
        print(f"Head dimension: {self.head_dim}")
        print(f"Number of heads: {self.model_loader.num_attention_heads}")
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
