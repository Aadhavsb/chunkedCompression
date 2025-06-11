"""
Barebones transformer decoder with fused attention and compression
"""
import torch
import torch.nn.functional as F
from typing import List
from kv_cache import KVCache
from profiles import profiles

class BareDecoder:
    def __init__(self, d_model: int = 512, d_head: int = 64, num_groups: int = 1):
        self.d_model = d_model
        self.d_head = d_head
        self.num_groups = num_groups
        
        # Initialize KV cache
        self.kv_cache = KVCache()
        
        # Mock query projection (for attention)
        torch.manual_seed(42)
        self.W_q = torch.randn(d_model, d_head) * 0.1
        
    def forward(self, X: torch.Tensor, compression_map: List[str]) -> torch.Tensor:
        """
        Forward pass through the decoder
        
        Args:
            X: Token embeddings [T, d_model]
            compression_map: List of compression options per token
            
        Returns:
            Output tensor [T, d_model]
        """
        T, d_model = X.shape
        
        print(f"\n🔄 Forward pass: {T} tokens, {d_model}D embeddings")
        
        # Step 1: Compress and store in KV cache
        self._compress_and_cache(X, compression_map)
        
        # Step 2: Perform attention over compressed cache
        outputs = self._attention_over_cache(X, compression_map)
        
        print(f"✅ Forward pass complete. Output shape: {outputs.shape}")
        return outputs
    
    def _compress_and_cache(self, X: torch.Tensor, compression_map: List[str]):
        """Compress both keys and values and store in KV cache"""
        print("🗜️  Compressing keys and values and caching...")
        
        for t, (x_t, option) in enumerate(zip(X, compression_map)):
            # Get compression profile
            profile = profiles[option]
            A_v = profile["A"]      # Value compression matrix [r_v, d_head]
            A_k = profile["A_K"]    # Key compression matrix [r_k, d_head]
            
            # Project embedding to head dimension (simulating K and V projections)
            # In real transformer: k_t = x_t @ W_k, v_t = x_t @ W_v
            k_t = x_t[:self.d_head]  # [d_head] - simulated key
            v_t = x_t[:self.d_head]  # [d_head] - simulated value
            
            # Compress keys and values
            h_k = A_k @ k_t  # [r_k] - compressed key
            h_v = A_v @ v_t  # [r_v] - compressed value
            
            # Store both in cache (using group_id=0 for simplicity)
            self.kv_cache.append(token_idx=t, group_id=0, h_v=h_v, h_k=h_k, option=option)
        
        # Print cache stats
        cache_keys = self.kv_cache.get_all_keys()
        print(f"   Cached {self.kv_cache.size()} KV pairs across {len(cache_keys)} option groups")
        for group_id, option in cache_keys:
            cached_values = self.kv_cache.retrieve_values(group_id, option)
            cached_keys = self.kv_cache.retrieve_keys(group_id, option)
            print(f"     Group {group_id}, {option}: V{cached_values.shape}, K{cached_keys.shape}")
    
    def _attention_over_cache(self, X: torch.Tensor, compression_map: List[str]) -> torch.Tensor:
        """Perform attention over compressed cache with on-the-fly key reconstruction"""
        print("🎯 Computing attention over compressed cache with key reconstruction...")
        
        T = X.shape[0]
        outputs = torch.zeros(T, self.d_model)
        
        # Get unique cache keys
        cache_keys = self.kv_cache.get_all_keys()
        
        for t, (x_t, option) in enumerate(zip(X, compression_map)):
            # Compute query
            q_t = x_t @ self.W_q  # [d_head]
            
            # Collect attention outputs from all cached groups
            attention_output = torch.zeros(self.d_model)
            
            for group_id, cached_option in cache_keys:
                # Get cached compressed vectors for this group/option
                cached_values = self.kv_cache.retrieve_values(group_id, cached_option)  # [T_cached, r_v]
                cached_keys = self.kv_cache.retrieve_keys(group_id, cached_option)      # [T_cached, r_k]
                
                if cached_values.numel() == 0 or cached_keys.numel() == 0:
                    continue
                
                # Get compression profile
                profile = profiles[cached_option]
                A_v = profile["A"]         # Value compression matrix [r_v, d_head]
                W_fused = profile["W_fused"]  # Fused output projection [rank, d_model]
                B_k = profile["B_K"]       # Key reconstruction matrix [d_head, r_k]
                
                # Reconstruct full keys on-the-fly: K = H_K @ B_K^T
                reconstructed_keys = cached_keys @ B_k.T  # [T_cached, d_head]
                
                # Compute attention scores: scores = q @ K^T
                scores = q_t @ reconstructed_keys.T  # [T_cached]
                attn_weights = F.softmax(scores, dim=0)  # [T_cached]
                
                # Weighted sum of cached value latents (stay in compressed space)
                context = attn_weights @ cached_values  # [r_v]
                
                # Apply fused output projection (direct from compressed to output)
                group_output = W_fused.T @ context  # [d_model]
                attention_output += group_output
            
            outputs[t] = attention_output
        
        print(f"   Attention computed for {T} tokens with key reconstruction")
        return outputs
    
    def clear_cache(self):
        """Clear the KV cache"""
        self.kv_cache.clear()
