"""
Inference components for the chunked compression system.
"""

# Import legacy inference implementations (moved from root)
try:
    from .llama_inference import LLaMACompressionInference
    from .llama_full_forward import *  # Various forward pass utilities
    from .compressed_autoregressive_decoder import *  # Decoder components
    
    __all__ = [
        'LLaMACompressionInference',
        # Additional exports from other modules will be included automatically
    ]
except ImportError:
    # Handle case where dependencies are not available
    __all__ = []