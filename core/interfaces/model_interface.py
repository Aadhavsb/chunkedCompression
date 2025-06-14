"""
Abstract interfaces for model loading and management.
"""
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class ModelLoaderInterface(ABC):
    """Abstract interface for model loading operations."""
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model and tokenizer."""
        pass
    
    @abstractmethod
    def get_attention_weights(self, layer_idx: int) -> Dict[str, 'torch.Tensor']:
        """
        Get attention weights for a specific layer.
        
        Args:
            layer_idx: Layer index to extract weights from
            
        Returns:
            Dictionary containing W_Q, W_K, W_V, W_O weight matrices
        """
        pass
    
    @abstractmethod
    def extract_hidden_states(self, input_text: str, max_length: int = 50) -> 'torch.Tensor':
        """
        Extract hidden states from model forward pass.
        
        Args:
            input_text: Input text to process
            max_length: Maximum sequence length
            
        Returns:
            Hidden states tensor
        """
        pass
    
    @abstractmethod
    def get_model_config(self) -> Any:
        """Get model configuration object."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up model resources."""
        pass


class ModelConfigInterface(ABC):
    """Abstract interface for model configuration."""
    
    @abstractmethod
    def get_num_layers(self) -> int:
        """Get number of transformer layers."""
        pass
    
    @abstractmethod
    def get_num_heads(self) -> int:
        """Get number of attention heads."""
        pass
    
    @abstractmethod
    def get_head_dim(self) -> int:
        """Get attention head dimension."""
        pass
    
    @abstractmethod
    def get_hidden_size(self) -> int:
        """Get hidden state size."""
        pass