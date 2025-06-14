"""
Abstract interfaces for compression operations.
"""
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class CompressionProfileInterface(ABC):
    """Abstract interface for compression profile operations."""
    
    @abstractmethod
    def build_compression_profiles(self, layer_idx: int) -> None:
        """
        Build compression profiles for a specific layer.
        
        Args:
            layer_idx: Layer index to build profiles for
        """
        pass
    
    @abstractmethod
    def compress_values(self, values: 'torch.Tensor', layer_idx: int, head_idx: int = 0) -> 'torch.Tensor':
        """
        Compress value tensors using the compression profile.
        
        Args:
            values: Value tensor to compress
            layer_idx: Layer index
            head_idx: Head index for multi-head attention
            
        Returns:
            Compressed value tensor
        """
        pass
    
    @abstractmethod
    def compress_keys(self, keys: 'torch.Tensor', layer_idx: int, head_idx: int = 0) -> 'torch.Tensor':
        """
        Compress key tensors using the compression profile.
        
        Args:
            keys: Key tensor to compress
            layer_idx: Layer index
            head_idx: Head index for multi-head attention
            
        Returns:
            Compressed key tensor
        """
        pass
    
    @abstractmethod
    def get_compression_stats(self, layer_idx: int) -> Dict[str, Any]:
        """
        Get compression statistics for a layer.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Dictionary containing compression statistics
        """
        pass
    
    @abstractmethod
    def validate_profile_shapes(self, layer_idx: int) -> bool:
        """
        Validate compression profile shapes for a layer.
        
        Args:
            layer_idx: Layer index to validate
            
        Returns:
            True if shapes are valid, False otherwise
        """
        pass


class CompressionAlgorithmInterface(ABC):
    """Abstract interface for compression algorithms."""
    
    @abstractmethod
    def perform_svd_compression(self, weight_matrix: 'torch.Tensor', rank: int) -> Tuple['torch.Tensor', 'torch.Tensor']:
        """
        Perform SVD-based compression on weight matrix.
        
        Args:
            weight_matrix: Weight matrix to compress
            rank: Target compression rank
            
        Returns:
            Tuple of (compression_matrix, reconstruction_matrix)
        """
        pass
    
    @abstractmethod
    def compute_compression_ratio(self, original_shape: Tuple[int, ...], compressed_shapes: Tuple[Tuple[int, ...], ...]) -> float:
        """
        Compute compression ratio.
        
        Args:
            original_shape: Shape of original tensor
            compressed_shapes: Shapes of compressed tensors
            
        Returns:
            Compression ratio
        """
        pass