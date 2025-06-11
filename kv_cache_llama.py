"""
Production LLaMA KV Cache with Real Compression
Stores compressed K/V representations with on-the-fly reconstruction
"""
import torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import time

class LLaMAKVCache:
    def __init__(self, enable_compression: bool = True):
        self.enable_compression = enable_compression
        
        # Compressed storage: (layer_idx, head_idx, profile) -> tensors
        self.compressed_values: Dict[Tuple[int, int, str], List[torch.Tensor]] = defaultdict(list)
        self.compressed_keys: Dict[Tuple[int, int, str], List[torch.Tensor]] = defaultdict(list)
        
        # Metadata storage
        self.token_indices: Dict[Tuple[int, int, str], List[int]] = defaultdict(list)
        self.cache_stats = {
            "total_tokens": 0,
            "compression_time": 0.0,
            "reconstruction_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        print(f"🗃️  Initialized LLaMA KV Cache (compression: {'enabled' if enable_compression else 'disabled'})")
    
    def store_compressed_kv(self, 
                          layer_idx: int,
                          head_idx: int, 
                          compressed_keys: torch.Tensor,
                          compressed_values: torch.Tensor,
                          token_idx: int,
                          compression_profile: str):
        """
        Store compressed key-value pairs
        
        Args:
            layer_idx: Transformer layer index
            head_idx: Attention head index
            compressed_keys: Compressed keys [key_rank] 
            compressed_values: Compressed values [value_rank]
            token_idx: Token position index
            compression_profile: Profile name ("low", "med", "high")
        """
        start_time = time.time()
        
        cache_key = (layer_idx, head_idx, compression_profile)
        
        self.compressed_keys[cache_key].append(compressed_keys)
        self.compressed_values[cache_key].append(compressed_values)
        self.token_indices[cache_key].append(token_idx)
        
        self.cache_stats["total_tokens"] += 1
        self.cache_stats["compression_time"] += time.time() - start_time
    
    def retrieve_compressed_kv(self, 
                             layer_idx: int, 
                             head_idx: int,
                             compression_profile: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve all compressed K/V pairs for given layer/head/profile
        
        Args:
            layer_idx: Transformer layer index
            head_idx: Attention head index  
            compression_profile: Profile name
            
        Returns:
            Tuple of (stacked_compressed_keys, stacked_compressed_values)
        """
        cache_key = (layer_idx, head_idx, compression_profile)
        
        if cache_key not in self.compressed_keys or len(self.compressed_keys[cache_key]) == 0:
            self.cache_stats["cache_misses"] += 1
            return torch.empty(0, 0), torch.empty(0, 0)
        
        self.cache_stats["cache_hits"] += 1
        
        # Stack all compressed vectors
        stacked_keys = torch.stack(self.compressed_keys[cache_key], dim=0)  # [seq_len, key_rank]
        stacked_values = torch.stack(self.compressed_values[cache_key], dim=0)  # [seq_len, value_rank]
        
        return stacked_keys, stacked_values
    
    def get_cache_groups(self) -> List[Tuple[int, int, str]]:
        """Get all cached layer/head/profile combinations"""
        return list(self.compressed_keys.keys())
    
    def get_sequence_length(self, layer_idx: int, head_idx: int, compression_profile: str) -> int:
        """Get sequence length for specific cache group"""
        cache_key = (layer_idx, head_idx, compression_profile)
        return len(self.compressed_keys.get(cache_key, []))
    
    def clear_cache(self):
        """Clear all cached data"""
        self.compressed_keys.clear()
        self.compressed_values.clear()
        self.token_indices.clear()
        
        # Reset stats except timing
        self.cache_stats["total_tokens"] = 0
        self.cache_stats["cache_hits"] = 0
        self.cache_stats["cache_misses"] = 0
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Calculate memory usage statistics"""
        total_key_elements = 0
        total_value_elements = 0
        
        for cache_key in self.compressed_keys:
            for tensor in self.compressed_keys[cache_key]:
                total_key_elements += tensor.numel()
            for tensor in self.compressed_values[cache_key]:
                total_value_elements += tensor.numel()
        
        # Assuming float16 (2 bytes per element)
        key_memory_mb = (total_key_elements * 2) / (1024 * 1024)
        value_memory_mb = (total_value_elements * 2) / (1024 * 1024)
        total_memory_mb = key_memory_mb + value_memory_mb
        
        return {
            "key_memory_mb": key_memory_mb,
            "value_memory_mb": value_memory_mb,
            "total_memory_mb": total_memory_mb,
            "total_key_elements": total_key_elements,
            "total_value_elements": total_value_elements
        }
    
    def print_cache_stats(self):
        """Print comprehensive cache statistics"""
        memory_usage = self.get_memory_usage()
        
        print(f"\n🗃️  LLaMA KV Cache Statistics")
        print(f"=" * 50)
        print(f"Total tokens cached: {self.cache_stats['total_tokens']}")
        print(f"Cache groups: {len(self.compressed_keys)}")
        print(f"Cache hits: {self.cache_stats['cache_hits']}")
        print(f"Cache misses: {self.cache_stats['cache_misses']}")
        
        if self.cache_stats['cache_hits'] + self.cache_stats['cache_misses'] > 0:
            hit_rate = self.cache_stats['cache_hits'] / (self.cache_stats['cache_hits'] + self.cache_stats['cache_misses'])
            print(f"Hit rate: {hit_rate:.2%}")
        
        print(f"\nMemory Usage:")
        print(f"  Keys: {memory_usage['key_memory_mb']:.2f} MB")
        print(f"  Values: {memory_usage['value_memory_mb']:.2f} MB")
        print(f"  Total: {memory_usage['total_memory_mb']:.2f} MB")
        
        print(f"\nTiming:")
        print(f"  Compression time: {self.cache_stats['compression_time']:.4f}s")
        print(f"  Reconstruction time: {self.cache_stats['reconstruction_time']:.4f}s")
        
        # Per-group breakdown
        print(f"\nCache Groups:")
        for cache_key in sorted(self.compressed_keys.keys()):
            layer_idx, head_idx, profile = cache_key
            seq_len = len(self.compressed_keys[cache_key])
            print(f"  Layer {layer_idx}, Head {head_idx}, {profile}: {seq_len} tokens")


class StandardKVCache:
    """
    Standard uncompressed KV cache for baseline comparison
    """
    def __init__(self):
        # Standard storage: (layer_idx, head_idx) -> tensors
        self.keys: Dict[Tuple[int, int], List[torch.Tensor]] = defaultdict(list)
        self.values: Dict[Tuple[int, int], List[torch.Tensor]] = defaultdict(list)
        self.token_indices: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        
        self.cache_stats = {
            "total_tokens": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        print(f"🗃️  Initialized Standard (uncompressed) KV Cache")
    
    def store_kv(self, 
                layer_idx: int,
                head_idx: int,
                keys: torch.Tensor,
                values: torch.Tensor, 
                token_idx: int):
        """
        Store uncompressed key-value pairs
        
        Args:
            layer_idx: Transformer layer index
            head_idx: Attention head index
            keys: Keys [head_dim]
            values: Values [head_dim] 
            token_idx: Token position index
        """
        cache_key = (layer_idx, head_idx)
        
        self.keys[cache_key].append(keys)
        self.values[cache_key].append(values)
        self.token_indices[cache_key].append(token_idx)
        
        self.cache_stats["total_tokens"] += 1
    
    def retrieve_kv(self, layer_idx: int, head_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve all K/V pairs for given layer/head
        
        Returns:
            Tuple of (stacked_keys, stacked_values)
        """
        cache_key = (layer_idx, head_idx)
        
        if cache_key not in self.keys or len(self.keys[cache_key]) == 0:
            self.cache_stats["cache_misses"] += 1
            return torch.empty(0, 0), torch.empty(0, 0)
        
        self.cache_stats["cache_hits"] += 1
        
        stacked_keys = torch.stack(self.keys[cache_key], dim=0)
        stacked_values = torch.stack(self.values[cache_key], dim=0)
        
        return stacked_keys, stacked_values
    
    def clear_cache(self):
        """Clear all cached data"""
        self.keys.clear()
        self.values.clear() 
        self.token_indices.clear()
        self.cache_stats["total_tokens"] = 0
        self.cache_stats["cache_hits"] = 0
        self.cache_stats["cache_misses"] = 0
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Calculate memory usage for uncompressed cache"""
        total_elements = 0
        
        for cache_key in self.keys:
            for tensor in self.keys[cache_key]:
                total_elements += tensor.numel()
            for tensor in self.values[cache_key]:
                total_elements += tensor.numel()
        
        # Assuming float16 (2 bytes per element)
        total_memory_mb = (total_elements * 2) / (1024 * 1024)
        
        return {
            "total_memory_mb": total_memory_mb,
            "total_elements": total_elements
        }
    
    def print_cache_stats(self):
        """Print cache statistics"""
        memory_usage = self.get_memory_usage()
        
        print(f"\n🗃️  Standard KV Cache Statistics")
        print(f"=" * 50)
        print(f"Total tokens cached: {self.cache_stats['total_tokens']}")
        print(f"Cache groups: {len(self.keys)}")
        print(f"Cache hits: {self.cache_stats['cache_hits']}")
        print(f"Cache misses: {self.cache_stats['cache_misses']}")
        print(f"Total memory: {memory_usage['total_memory_mb']:.2f} MB")
