"""
Core interfaces for the chunked compression system.
"""

from .model_interface import ModelLoaderInterface, ModelConfigInterface
from .compression_interface import CompressionProfileInterface, CompressionAlgorithmInterface
from .cache_interface import KVCacheInterface, CacheMetricsInterface
from .inference_interface import InferenceInterface, BenchmarkInterface, MetricsInterface
from .data_interface import DatasetInterface, DataProcessorInterface, DataValidatorInterface

__all__ = [
    # Model interfaces
    'ModelLoaderInterface',
    'ModelConfigInterface',
    
    # Compression interfaces
    'CompressionProfileInterface',
    'CompressionAlgorithmInterface',
    
    # Cache interfaces
    'KVCacheInterface',
    'CacheMetricsInterface',
    
    # Inference interfaces
    'InferenceInterface',
    'BenchmarkInterface',
    'MetricsInterface',
    
    # Data interfaces
    'DatasetInterface',
    'DataProcessorInterface',
    'DataValidatorInterface',
]