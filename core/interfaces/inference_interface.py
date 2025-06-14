"""
Abstract interfaces for inference operations.
"""
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class InferenceInterface(ABC):
    """Abstract interface for inference operations."""
    
    @abstractmethod
    def run_inference(self, input_text: str, **kwargs) -> 'torch.Tensor':
        """
        Run inference on input text.
        
        Args:
            input_text: Input text to process
            **kwargs: Additional inference parameters
            
        Returns:
            Model output tensor
        """
        pass
    
    @abstractmethod
    def compute_attention(self, query: 'torch.Tensor', keys: 'torch.Tensor', 
                         values: 'torch.Tensor', layer_idx: int, **kwargs) -> 'torch.Tensor':
        """
        Compute attention mechanism.
        
        Args:
            query: Query tensor
            keys: Key tensor
            values: Value tensor
            layer_idx: Layer index
            **kwargs: Additional attention parameters
            
        Returns:
            Attention output tensor
        """
        pass
    
    @abstractmethod
    def get_inference_stats(self) -> Dict[str, Any]:
        """
        Get inference performance statistics.
        
        Returns:
            Dictionary containing inference metrics
        """
        pass


class BenchmarkInterface(ABC):
    """Abstract interface for benchmarking operations."""
    
    @abstractmethod
    def run_benchmark(self, **kwargs) -> Dict[str, Any]:
        """
        Run comprehensive benchmark.
        
        Args:
            **kwargs: Benchmark parameters
            
        Returns:
            Dictionary containing benchmark results
        """
        pass
    
    @abstractmethod
    def run_compression_benchmark(self, **kwargs) -> Dict[str, Any]:
        """
        Run compression-specific benchmark.
        
        Args:
            **kwargs: Benchmark parameters
            
        Returns:
            Dictionary containing compression benchmark results
        """
        pass
    
    @abstractmethod
    def calculate_quality_metrics(self, compressed_output: 'torch.Tensor', 
                                 baseline_output: 'torch.Tensor') -> Dict[str, float]:
        """
        Calculate quality metrics comparing compressed vs baseline output.
        
        Args:
            compressed_output: Output from compressed model
            baseline_output: Output from baseline model
            
        Returns:
            Dictionary containing quality metrics (MSE, cosine similarity, etc.)
        """
        pass
    
    @abstractmethod
    def calculate_efficiency_metrics(self, **kwargs) -> Dict[str, float]:
        """
        Calculate efficiency metrics.
        
        Args:
            **kwargs: Parameters for efficiency calculation
            
        Returns:
            Dictionary containing efficiency metrics
        """
        pass


class MetricsInterface(ABC):
    """Abstract interface for metrics collection and analysis."""
    
    @abstractmethod
    def collect_memory_metrics(self) -> Dict[str, float]:
        """Collect memory usage metrics."""
        pass
    
    @abstractmethod
    def collect_timing_metrics(self) -> Dict[str, float]:
        """Collect timing metrics."""
        pass
    
    @abstractmethod
    def collect_quality_metrics(self, output1: 'torch.Tensor', output2: 'torch.Tensor') -> Dict[str, float]:
        """Collect quality comparison metrics."""
        pass
    
    @abstractmethod
    def aggregate_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple metrics into summary statistics."""
        pass