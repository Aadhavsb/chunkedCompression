"""
Legacy components from the original implementation.

These files are kept for reference and backward compatibility but should not be 
used in new code. Use the modular components in the core/ package instead.

Files:
- llama_model_loader.py: Original model loader (replaced by core.model.LLaMAModelLoader)
- profiles_llama.py: Original compression profiles (replaced by core.compression.LLaMACompressionProfileBuilder)
- compression.py: Original compression utilities (replaced by core.compression.SVDCompressionAlgorithm)

Migration Guide:
- Old: from llama_model_loader import LLaMAModelLoader
- New: from core.model import LLaMAModelLoader

- Old: from profiles_llama import LLaMACompressionProfiles  
- New: from core.compression import LLaMACompressionProfileBuilder

- Old: from compression import decompose_and_fuse
- New: from core.compression import SVDCompressionAlgorithm
"""

# Note: These files have heavy dependencies and are not imported by default
# Import them explicitly if needed for backward compatibility

__all__ = []  # No automatic exports to avoid dependency issues