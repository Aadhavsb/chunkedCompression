"""
Model loading and configuration components.

File Organization:
- model_loader.py: Contains LLaMAModelLoader (main class for all use cases)
- llama_loader.py: Contains LLaMA3Loader (cluster-specific memory optimization utility)
- legacy/llama_model_loader.py: Original implementation (preserved for reference)

LLaMAModelLoader is the primary interface with backward compatibility properties.
LLaMA3Loader is an optional cluster utility for memory-efficient loading.
"""

from .model_loader import LLaMAModelLoader
from .model_config_wrapper import ModelConfigWrapper

# Import cluster utilities (optional)
try:
    from .llama_loader import LLaMA3Loader
    
    __all__ = [
        'LLaMAModelLoader',        # Primary model loader (use this)
        'ModelConfigWrapper',      # Configuration wrapper
        'LLaMA3Loader',           # Cluster utility (optional)
    ]
except ImportError:
    # Handle case where dependencies are not available
    __all__ = [
        'LLaMAModelLoader',        # Primary model loader
        'ModelConfigWrapper',      # Configuration wrapper
    ]