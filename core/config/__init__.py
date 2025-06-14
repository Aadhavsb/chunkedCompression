"""
Configuration management for the chunked compression system.
"""

from .model_config import ModelConfig
from .compression_config import CompressionConfig
from .inference_config import InferenceConfig, BenchmarkConfig

__all__ = [
    'ModelConfig',
    'CompressionConfig', 
    'InferenceConfig',
    'BenchmarkConfig',
]