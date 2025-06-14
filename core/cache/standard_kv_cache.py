"""
Standard Uncompressed KV Cache - Baseline for Comparison
Traditional transformer KV cache with no compression
"""
import torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import time
import sys

class StandardKVCache:
    """
    Standard uncompressed KV cache for baseline comparison
    Stores full K/V vectors without any compression
    """
    def __init__(self):
        # Storage: (layer_idx, head_idx) -> list of (token_idx, vector) tuples
        self.k_cache: Dict[Tuple[int, int], List[Tuple[int, torch.Tensor]]] = defaultdict(list)
        self.v_cache: Dict[Tuple[int, int], List[Tuple[int, torch.Tensor]]] = defaultdict(list)
        
        # Performance tracking
        self.cache_stats = {
            "total_tokens": 0,
            "storage_time": 0.0,
            "retrieval_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        print(f"🗃️  Initialized Standard (uncompressed) KV Cache")
    
    def append(self, layer_idx: int, head_idx: int, token_idx: int, 
               k_vector: torch.Tensor, v_vector: torch.Tensor):
        """
        Store full uncompressed key-value pairs
        
        Args:
            layer_idx: Transformer layer index
            head_idx: Attention head index
            token_idx: Token position index
            k_vector: Full key vector [head_dim]
            v_vector: Full value vector [head_dim]
        """
        start_time = time.time()
        
        cache_key = (layer_idx, head_idx)
        
        # Store full vectors with token index
        self.k_cache[cache_key].append((token_idx, k_vector.clone()))
        self.v_cache[cache_key].append((token_idx, v_vector.clone()))
        
        self.cache_stats["total_tokens"] += 1
        self.cache_stats["storage_time"] += time.time() - start_time
    
    def get_keys(self, layer_idx: int, head_idx: int, 
                 start_idx: int = 0, end_idx: Optional[int] = None) -> torch.Tensor:
        """
        Retrieve key vectors for given range
        
        Args:
            layer_idx: Transformer layer index
            head_idx: Attention head index
            start_idx: Start token index (inclusive)
            end_idx: End token index (exclusive), None for all
            
        Returns:
            Stacked key vectors [seq_len, head_dim]
        """
        start_time = time.time()
        
        cache_key = (layer_idx, head_idx)
        
        if cache_key not in self.k_cache or len(self.k_cache[cache_key]) == 0:
            self.cache_stats["cache_misses"] += 1
            return torch.empty(0, 0)
        
        self.cache_stats["cache_hits"] += 1
        
        # Filter by token index range
        if end_idx is None:
            end_idx = float('inf')
        
        filtered_keys = [
            k_vec for token_idx, k_vec in self.k_cache[cache_key]
            if start_idx <= token_idx < end_idx
        ]
        
        self.cache_stats["retrieval_time"] += time.time() - start_time
        
        if len(filtered_keys) == 0:
            return torch.empty(0, 0)
        
        return torch.stack(filtered_keys, dim=0)  # [seq_len, head_dim]
    
    def get_values(self, layer_idx: int, head_idx: int,
                   start_idx: int = 0, end_idx: Optional[int] = None) -> torch.Tensor:
        """
        Retrieve value vectors for given range
        
        Args:
            layer_idx: Transformer layer index
            head_idx: Attention head index
            start_idx: Start token index (inclusive)
            end_idx: End token index (exclusive), None for all
            
        Returns:
            Stacked value vectors [seq_len, head_dim]
        """
        start_time = time.time()
        
        cache_key = (layer_idx, head_idx)
        
        if cache_key not in self.v_cache or len(self.v_cache[cache_key]) == 0:
            self.cache_stats["cache_misses"] += 1
            return torch.empty(0, 0)
        
        self.cache_stats["cache_hits"] += 1
        
        # Filter by token index range
        if end_idx is None:
            end_idx = float('inf')
        
        filtered_values = [
            v_vec for token_idx, v_vec in self.v_cache[cache_key]
            if start_idx <= token_idx < end_idx
        ]
        
        self.cache_stats["retrieval_time"] += time.time() - start_time
        
        if len(filtered_values) == 0:
            return torch.empty(0, 0)
        
        return torch.stack(filtered_values, dim=0)  # [seq_len, head_dim]
    
    def get_all_kv(self, layer_idx: int, head_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get all cached K/V pairs for a layer/head
        
        Returns:
            Tuple of (all_keys, all_values) each [seq_len, head_dim]
        """
        keys = self.get_keys(layer_idx, head_idx)
        values = self.get_values(layer_idx, head_idx)
        return keys, values
    
    def get_cache_groups(self) -> List[Tuple[int, int]]:
        """Get all cached layer/head combinations"""
        return list(set(self.k_cache.keys()) | set(self.v_cache.keys()))
    
    def get_sequence_length(self, layer_idx: int, head_idx: int) -> int:
        """Get sequence length for specific cache group"""
        cache_key = (layer_idx, head_idx)
        return len(self.k_cache.get(cache_key, []))
    
    def clear_cache(self):
        """Clear all cached data"""
        self.k_cache.clear()
        self.v_cache.clear()
        
        # Reset stats except timing
        self.cache_stats["total_tokens"] = 0
        self.cache_stats["cache_hits"] = 0
        self.cache_stats["cache_misses"] = 0
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Calculate detailed memory usage statistics"""
        total_key_elements = 0
        total_value_elements = 0
        total_entries = 0
        
        # Count all tensor elements
        for cache_key in self.k_cache:
            for token_idx, k_tensor in self.k_cache[cache_key]:
                total_key_elements += k_tensor.numel()
                total_entries += 1
        
        for cache_key in self.v_cache:
            for token_idx, v_tensor in self.v_cache[cache_key]:
                total_value_elements += v_tensor.numel()
        
        # Calculate memory usage (assuming float16 = 2 bytes per element)
        key_memory_mb = (total_key_elements * 2) / (1024 * 1024)
        value_memory_mb = (total_value_elements * 2) / (1024 * 1024)
        total_memory_mb = key_memory_mb + value_memory_mb
        
        # Python object overhead (rough estimate)
        object_overhead_mb = (total_entries * sys.getsizeof(tuple()) * 2) / (1024 * 1024)
        
        return {
            "key_memory_mb": key_memory_mb,
            "value_memory_mb": value_memory_mb,
            "total_memory_mb": total_memory_mb,
            "object_overhead_mb": object_overhead_mb,
            "total_with_overhead_mb": total_memory_mb + object_overhead_mb,
            "total_key_elements": total_key_elements,
            "total_value_elements": total_value_elements,
            "total_entries": total_entries
        }
    
    def get_cache_efficiency_stats(self) -> Dict[str, float]:
        """Get cache performance statistics"""
        total_requests = self.cache_stats["cache_hits"] + self.cache_stats["cache_misses"]
        hit_rate = self.cache_stats["cache_hits"] / max(total_requests, 1)
        
        return {
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "avg_storage_time_ms": self.cache_stats["storage_time"] * 1000 / max(self.cache_stats["total_tokens"], 1),
            "avg_retrieval_time_ms": self.cache_stats["retrieval_time"] * 1000 / max(total_requests, 1)
        }
    
    def print_cache_stats(self):
        """Print comprehensive cache statistics"""
        memory_usage = self.get_memory_usage()
        efficiency_stats = self.get_cache_efficiency_stats()
        
        print(f"\n🗃️  Standard KV Cache Statistics")
        print(f"=" * 50)
        print(f"Total tokens cached: {self.cache_stats['total_tokens']}")
        print(f"Cache groups: {len(self.get_cache_groups())}")
        print(f"Cache hits: {self.cache_stats['cache_hits']}")
        print(f"Cache misses: {self.cache_stats['cache_misses']}")
        print(f"Hit rate: {efficiency_stats['hit_rate']:.2%}")
        
        print(f"\nMemory Usage:")
        print(f"  Keys: {memory_usage['key_memory_mb']:.2f} MB")
        print(f"  Values: {memory_usage['value_memory_mb']:.2f} MB")
        print(f"  Object overhead: {memory_usage['object_overhead_mb']:.2f} MB")
        print(f"  Total: {memory_usage['total_with_overhead_mb']:.2f} MB")
        
        print(f"\nPerformance:")
        print(f"  Avg storage time: {efficiency_stats['avg_storage_time_ms']:.3f} ms/token")
        print(f"  Avg retrieval time: {efficiency_stats['avg_retrieval_time_ms']:.3f} ms/request")
        
        print(f"\nData Statistics:")
        print(f"  Total entries: {memory_usage['total_entries']}")
        print(f"  Key elements: {memory_usage['total_key_elements']:,}")
        print(f"  Value elements: {memory_usage['total_value_elements']:,}")
        
        # Per-group breakdown
        print(f"\nCache Groups:")
        for layer_idx, head_idx in sorted(self.get_cache_groups()):
            seq_len = self.get_sequence_length(layer_idx, head_idx)
            print(f"  Layer {layer_idx}, Head {head_idx}: {seq_len} tokens")


class StandardAttentionComputation:
    """
    Standard attention computation using uncompressed KV cache
    For baseline comparison against compressed attention
    """
    
    def __init__(self, model_loader, kv_cache: StandardKVCache):
        self.model_loader = model_loader
        self.kv_cache = kv_cache
        self.attention_stats = {
            "total_computations": 0,
            "total_computation_time": 0.0,
            "total_flops": 0
        }
    
    def compute_standard_attention(self, 
                                 query: torch.Tensor,
                                 layer_idx: int,
                                 head_idx: int,
                                 token_idx: int,
                                 hidden_state: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute standard attention with uncompressed KV cache
        
        Args:
            query: Query vector [head_dim]
            layer_idx: Transformer layer index
            head_idx: Attention head index
            token_idx: Current token position
            hidden_state: Current hidden state [hidden_dim]
            
        Returns:
            Tuple of (context_vector, computation_stats)
        """
        start_time = time.time()
        
        # Get attention weights from model
        attention_weights = self.model_loader.get_attention_weights(layer_idx)
        
        # Compute full K and V for current token
        W_K = attention_weights["W_K"][head_idx]  # [head_dim, hidden_dim]
        W_V = attention_weights["W_V"][head_idx]  # [head_dim, hidden_dim]
        
        current_k = W_K @ hidden_state  # [head_dim]
        current_v = W_V @ hidden_state  # [head_dim]
        
        # Store in standard cache
        self.kv_cache.append(layer_idx, head_idx, token_idx, current_k, current_v)
        
        # Retrieve all cached K/V for this head
        all_keys, all_values = self.kv_cache.get_all_kv(layer_idx, head_idx)
        
        if all_keys.numel() == 0:
            # First token, return zero context
            head_dim = current_k.shape[0]
            return torch.zeros(head_dim), {"computation_time": time.time() - start_time}
        
        # Standard attention computation
        # scores = Q @ K^T  [seq_len]
        attention_scores = query @ all_keys.T
        
        # Apply softmax
        attention_probs = torch.softmax(attention_scores, dim=-1)
        
        # Context = attention_probs @ V  [head_dim]
        context_vector = attention_probs @ all_values
        
        # Calculate computation stats
        computation_time = time.time() - start_time
        seq_len = all_keys.shape[0]
        head_dim = all_keys.shape[1]
        
        # FLOP count: Q@K^T + softmax + attn@V
        flops = seq_len * head_dim + seq_len + seq_len * head_dim
        
        self.attention_stats["total_computations"] += 1
        self.attention_stats["total_computation_time"] += computation_time
        self.attention_stats["total_flops"] += flops
        
        computation_stats = {
            "computation_time": computation_time,
            "sequence_length": seq_len,
            "flops": flops,
            "memory_accessed_mb": (all_keys.numel() + all_values.numel()) * 2 / (1024 * 1024)
        }
        
        return context_vector, computation_stats
    
    def get_attention_stats(self) -> Dict[str, float]:
        """Get accumulated attention computation statistics"""
        total_comps = max(self.attention_stats["total_computations"], 1)
        
        return {
            "total_computations": self.attention_stats["total_computations"],
            "avg_computation_time_ms": self.attention_stats["total_computation_time"] * 1000 / total_comps,
            "total_flops": self.attention_stats["total_flops"],
            "avg_flops_per_computation": self.attention_stats["total_flops"] / total_comps
        }
