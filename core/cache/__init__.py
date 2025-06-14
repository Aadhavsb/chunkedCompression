"""
KV cache implementations for the chunked compression system.
"""

# Import legacy cache implementations (moved from root)
try:
    from .kv_cache_llama import LLaMAKVCache
    from .standard_kv_cache import StandardKVCache
    
    __all__ = [
        'LLaMAKVCache',
        'StandardKVCache',
    ]
except ImportError:
    # Handle case where torch dependencies are not available
    __all__ = []