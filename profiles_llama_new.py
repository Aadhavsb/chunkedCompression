"""
New modular implementation of LLaMA compression profiles.
This file provides the new implementation while maintaining backward compatibility.

For new code, import from core.compression instead:
    from core.compression import LLaMACompressionProfileBuilder
    from core.model import LLaMAModelLoader
    from core.config import ModelConfig, CompressionConfig
"""

# Try to import components, but handle gracefully if torch is missing
try:
    # Import the legacy wrapper for backward compatibility
    from core.compression.legacy_wrapper import LLaMACompressionProfiles
    
    # For users who want to use the new modular approach
    from core.compression import LLaMACompressionProfileBuilder, SVDCompressionAlgorithm
    from core.model import LLaMAModelLoader
    from core.config import ModelConfig, CompressionConfig
    
    # Re-export the original class name for backward compatibility
    __all__ = ['LLaMACompressionProfiles']
    
    # Export new components for advanced users
    __all__.extend([
        'LLaMACompressionProfileBuilder',
        'SVDCompressionAlgorithm', 
        'LLaMAModelLoader',
        'ModelConfig',
        'CompressionConfig'
    ])
    
    COMPONENTS_AVAILABLE = True
    
except ImportError as e:
    print(f"Warning: Some components unavailable due to missing dependencies: {e}")
    print("This is normal if torch/transformers are not installed in this environment.")
    print("Components will be available when running in the proper environment (cluster/singularity).")
    
    # Provide minimal interface for basic testing
    __all__ = []
    COMPONENTS_AVAILABLE = False

# Usage examples in docstring
"""
USAGE EXAMPLES:

## Legacy usage (maintains old API):
```python
from llama_model_loader import LLaMAModelLoader  # old import
from profiles_llama_new import LLaMACompressionProfiles

# This continues to work exactly as before
model_loader = LLaMAModelLoader()
profiles = LLaMACompressionProfiles(model_loader)
compressed_values = profiles.compress_values(values, "med", head_idx=0)
```

## New modular usage (recommended for new code):
```python
from core.model import LLaMAModelLoader
from core.config import ModelConfig, CompressionConfig
from core.compression import LLaMACompressionProfileBuilder

# More flexible and configurable
model_config = ModelConfig.from_env()  # Load from environment
compression_config = CompressionConfig(
    value_compression_ranks={"low": 32, "med": 64, "high": 128}
)

model_loader = LLaMAModelLoader(model_config)
model_loader.load_model()

profile_builder = LLaMACompressionProfileBuilder(
    model_loader, 
    compression_config
)
profile_builder.build_compression_profiles(layer_idx=0)

# Use with explicit profile specification
compressed_values = profile_builder.compress_values_with_profile(
    values, "med", head_idx=0
)
```

## Migration path:
1. Replace import: `from profiles_llama import` -> `from profiles_llama_new import`
2. Code continues to work unchanged
3. When ready, migrate to new API for better configurability and modularity
"""