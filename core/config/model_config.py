"""
Model configuration management.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for model loading and management."""
    
    # Model paths and identifiers
    model_path: str = "/mnt/vstor/CSE_ECSE_GXD234/Meta-Llama-3-8B-Instruct"
    model_name: str = "llama-3-8b-instruct"
    cache_dir: Optional[str] = None
    
    # Hardware configuration
    device: str = "auto"  # "auto", "cuda", "cpu"
    dtype: str = "bfloat16"  # "bfloat16", "float16", "float32"
    
    # Memory management
    low_memory_mode: bool = False
    max_memory: Optional[Dict[str, str]] = None
    offload_folder: Optional[str] = None
    
    # Model loading options
    trust_remote_code: bool = False
    use_auth_token: Optional[str] = None
    revision: str = "main"
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Set cache directory if not specified
        if self.cache_dir is None:
            self.cache_dir = str(Path.home() / ".cache" / "chunked_compression")
        
        # Create cache directory if it doesn't exist
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Validate model path
        if not os.path.exists(self.model_path):
            print(f"Warning: Model path {self.model_path} does not exist")
    
    @classmethod
    def from_env(cls) -> 'ModelConfig':
        """Create configuration from environment variables."""
        return cls(
            model_path=os.getenv('LLAMA_MODEL_PATH', cls.model_path),
            model_name=os.getenv('MODEL_NAME', cls.model_name),
            cache_dir=os.getenv('CACHE_DIR'),
            device=os.getenv('DEVICE', cls.device),
            dtype=os.getenv('DTYPE', cls.dtype),
            low_memory_mode=os.getenv('LOW_MEMORY_MODE', 'false').lower() == 'true',
        )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model_path': self.model_path,
            'model_name': self.model_name,
            'cache_dir': self.cache_dir,
            'device': self.device,
            'dtype': self.dtype,
            'low_memory_mode': self.low_memory_mode,
            'max_memory': self.max_memory,
            'offload_folder': self.offload_folder,
            'trust_remote_code': self.trust_remote_code,
            'use_auth_token': self.use_auth_token,
            'revision': self.revision,
        }
    
    def get_device_config(self) -> str:
        """Get appropriate device configuration."""
        if self.device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.device
    
    def get_dtype_config(self):
        """Get appropriate dtype configuration."""
        try:
            import torch
            dtype_mapping = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            return dtype_mapping.get(self.dtype, torch.bfloat16)
        except ImportError:
            return self.dtype