"""
Data handling components for the chunked compression system.
"""

# Import legacy data handling implementations (moved from root)
try:
    from .dataset_llama import LLaMADatasetHandler
    
    __all__ = [
        'LLaMADatasetHandler',
    ]
except ImportError:
    # Handle case where dependencies are not available
    __all__ = []