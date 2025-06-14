"""
Core components for the chunked compression system.

This package provides a modular, well-structured implementation of the LLaMA
compression system with proper interfaces, configuration management, and
separation of concerns.

Main modules:
- config: Configuration management for models, compression, and inference
- interfaces: Abstract base classes defining component interfaces
- model: Model loading and configuration wrapper
- compression: Compression algorithms and profile builders
- cache: KV cache implementations (planned)
- inference: Inference and benchmarking components (planned)
- utils: Utility functions and classes
"""

# Import configuration (always available)
from .config import ModelConfig, CompressionConfig, InferenceConfig, BenchmarkConfig

# Import interfaces (always available)
from .interfaces import (
    ModelLoaderInterface,
    ModelConfigInterface,
    CompressionProfileInterface,
    CompressionAlgorithmInterface,
    KVCacheInterface,
    CacheMetricsInterface,
    InferenceInterface,
    BenchmarkInterface,
    MetricsInterface,
    DatasetInterface,
    DataProcessorInterface,
    DataValidatorInterface,
)

# Import utilities (handle missing dependencies gracefully)
from .utils import MemoryManager

# Try to import heavy components (may fail if torch/transformers not available)
try:
    from .model import LLaMAModelLoader, ModelConfigWrapper
    from .compression import LLaMACompressionProfileBuilder, SVDCompressionAlgorithm
    HEAVY_COMPONENTS_AVAILABLE = True
except ImportError:
    HEAVY_COMPONENTS_AVAILABLE = False
    # These will be None if not available
    LLaMAModelLoader = None
    ModelConfigWrapper = None
    LLaMACompressionProfileBuilder = None
    SVDCompressionAlgorithm = None

__version__ = "0.2.0"
__author__ = "LLaMA Compression Research Team"

# Dynamic __all__ based on what's available
__all__ = [
    # Configuration classes (always available)
    'ModelConfig',
    'CompressionConfig', 
    'InferenceConfig',
    'BenchmarkConfig',
    
    # Utilities (always available)
    'MemoryManager',
    
    # Interfaces (always available, for type hints and custom implementations)
    'ModelLoaderInterface',
    'ModelConfigInterface',
    'CompressionProfileInterface',
    'CompressionAlgorithmInterface',
    'KVCacheInterface',
    'CacheMetricsInterface',
    'InferenceInterface',
    'BenchmarkInterface',
    'MetricsInterface',
    'DatasetInterface',
    'DataProcessorInterface',
    'DataValidatorInterface',
    
    # Availability flag
    'HEAVY_COMPONENTS_AVAILABLE',
]

# Add heavy components if available
if HEAVY_COMPONENTS_AVAILABLE:
    __all__.extend([
        'LLaMAModelLoader',
        'ModelConfigWrapper',
        'LLaMACompressionProfileBuilder',
        'SVDCompressionAlgorithm',
    ])