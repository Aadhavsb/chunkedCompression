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
        """Compress tokens and store in KV cache"""
        print("🗜️  Compressing tokens and caching...")
        
        for t, (x_t, option) in enumerate(zip(X, compression_map)):
            # Get compression profile
            profile = profiles[option]
            A = profile["A"]  # [r_opt, d_head]
            
            # Mock value projection: project embedding to head dimension
            # In real transformer: v_t = x_t @ W_v, but we'll use first d_head dims
            v_t = x_t[:self.d_head]  # [d_head]
            
            # Compress: h_t = A @ v_t
            h_t = A @ v_t  # [r_opt]
            
            # Store in cache (using group_id=0 for simplicity)
            self.kv_cache.append(token_idx=t, group_id=0, h_t=h_t, option=option)
        
        # Print cache stats
        cache_keys = self.kv_cache.get_all_keys()
        print(f"   Cached {self.kv_cache.size()} latent vectors across {len(cache_keys)} option groups")
        for group_id, option in cache_keys:
            cached_tensor = self.kv_cache.retrieve(group_id, option)
            print(f"     Group {group_id}, {option}: {cached_tensor.shape}")
    
    def _attention_over_cache(self, X: torch.Tensor, compression_map: List[str]) -> torch.Tensor:
        """Perform attention over compressed cache and apply fused projection"""
        print("🎯 Computing attention over compressed cache...")
        
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
                # Get cached latent vectors for this group/option
                cached_latents = self.kv_cache.retrieve(group_id, cached_option)  # [T_cached, r_opt]
                
                if cached_latents.numel() == 0:
                    continue
                
                # Simple attention: dot product with query (projected to latent space)
                profile = profiles[cached_option]
                A = profile["A"]  # [r_opt, d_head]
                W_fused = profile["W_fused"]  # [d_model, r_opt]
                
                # Project query to latent space: q_compressed = A @ q_t
                q_compressed = A @ q_t  # [r_opt]
                
                # Attention scores: scores = cached_latents @ q_compressed
                scores = cached_latents @ q_compressed  # [T_cached]
                attn_weights = F.softmax(scores, dim=0)  # [T_cached]
                
                # Weighted sum of cached latents
                context = attn_weights @ cached_latents  # [r_opt]
                
                # Apply fused output projection
                group_output = W_fused @ context  # [d_model]
                attention_output += group_output
            
            outputs[t] = attention_output
        
        print(f"   Attention computed for {T} tokens")
        return outputs
    
    def clear_cache(self):
        """Clear the KV cache"""
        self.kv_cache.clear()
