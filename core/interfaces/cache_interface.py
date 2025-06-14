"""
Abstract interfaces for KV cache operations.
"""
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class KVCacheInterface(ABC):
    """Abstract interface for KV cache operations."""
    
    @abstractmethod
    def store_kv_cache(self, layer_idx: int, head_idx: int, 
                      keys: 'torch.Tensor', values: 'torch.Tensor',
                      **kwargs) -> None:
        """
        Store key-value cache for a specific layer and head.
        
        Args:
            layer_idx: Layer index
            head_idx: Head index
            keys: Key tensor
            values: Value tensor
            **kwargs: Additional storage parameters
        """
        pass
    
    @abstractmethod
    def retrieve_kv_cache(self, layer_idx: int, head_idx: int,
                         **kwargs) -> Tuple['torch.Tensor', 'torch.Tensor']:
        """
        Retrieve key-value cache for a specific layer and head.
        
        Args:
            layer_idx: Layer index
            head_idx: Head index
            **kwargs: Additional retrieval parameters
            
        Returns:
            Tuple of (keys, values)
        """
        pass
    
    @abstractmethod
    def clear_cache(self, layer_idx: Optional[int] = None, 
                   head_idx: Optional[int] = None) -> None:
        """
        Clear cache for specified layer/head or all if not specified.
        
        Args:
            layer_idx: Layer index to clear (None for all layers)
            head_idx: Head index to clear (None for all heads)
        """
        pass
    
    @abstractmethod
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary containing memory usage metrics
        """
        pass
    
    @abstractmethod
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Dictionary containing cache performance metrics
        """
        pass
    
    @abstractmethod
    def has_cache(self, layer_idx: int, head_idx: int) -> bool:
        """
        Check if cache exists for specific layer and head.
        
        Args:
            layer_idx: Layer index
            head_idx: Head index
            
        Returns:
            True if cache exists, False otherwise
        """
        pass


class CacheMetricsInterface(ABC):
    """Abstract interface for cache metrics and monitoring."""
    
    @abstractmethod
    def track_cache_hit(self, layer_idx: int, head_idx: int) -> None:
        """Track cache hit."""
        pass
    
    @abstractmethod
    def track_cache_miss(self, layer_idx: int, head_idx: int) -> None:
        """Track cache miss."""
        pass
    
    @abstractmethod
    def get_hit_rate(self, layer_idx: Optional[int] = None) -> float:
        """
        Get cache hit rate.
        
        Args:
            layer_idx: Layer index (None for overall rate)
            
        Returns:
            Hit rate as percentage
        """
        pass
    
    @abstractmethod
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        pass