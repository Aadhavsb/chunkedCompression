"""
Model configuration wrapper that implements the ModelConfigInterface.
"""
from typing import Any
from ..interfaces.model_interface import ModelConfigInterface


class ModelConfigWrapper(ModelConfigInterface):
    """
    Wrapper around the actual model configuration that implements our interface.
    Provides a consistent API regardless of the underlying model configuration structure.
    """
    
    def __init__(self, model_config: Any):
        """
        Initialize the wrapper with the actual model configuration.
        
        Args:
            model_config: The actual model configuration object from transformers
        """
        self.config = model_config
        self._cache = {}  # Cache computed values for efficiency
        
    def get_num_layers(self) -> int:
        """Get number of transformer layers."""
        if 'num_layers' not in self._cache:
            self._cache['num_layers'] = getattr(
                self.config, 'num_hidden_layers', 
                getattr(self.config, 'n_layer', 32)
            )
        return self._cache['num_layers']
    
    def get_num_heads(self) -> int:
        """Get number of attention heads (query heads)."""
        if 'num_heads' not in self._cache:
            self._cache['num_heads'] = getattr(
                self.config, 'num_attention_heads',
                getattr(self.config, 'n_head', 32)
            )
        return self._cache['num_heads']
    
    def get_num_key_value_heads(self) -> int:
        """Get number of key-value heads (for GQA/MQA)."""
        if 'num_kv_heads' not in self._cache:
            # For GQA models, use num_key_value_heads if available
            # Otherwise, assume same as query heads (standard MHA)
            self._cache['num_kv_heads'] = getattr(
                self.config, 'num_key_value_heads',
                self.get_num_heads()
            )
        return self._cache['num_kv_heads']
    
    def get_head_dim(self) -> int:
        """Get attention head dimension."""
        if 'head_dim' not in self._cache:
            # Calculate head dimension from hidden size and number of heads
            hidden_size = self.get_hidden_size()
            num_heads = self.get_num_heads()
            self._cache['head_dim'] = hidden_size // num_heads
        return self._cache['head_dim']
    
    def get_hidden_size(self) -> int:
        """Get hidden state size."""
        if 'hidden_size' not in self._cache:
            self._cache['hidden_size'] = getattr(
                self.config, 'hidden_size',
                getattr(self.config, 'd_model', 4096)
            )
        return self._cache['hidden_size']
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        if 'vocab_size' not in self._cache:
            self._cache['vocab_size'] = getattr(
                self.config, 'vocab_size',
                getattr(self.config, 'n_vocab', 128256)
            )
        return self._cache['vocab_size']
    
    def get_intermediate_size(self) -> int:
        """Get intermediate (feedforward) layer size."""
        if 'intermediate_size' not in self._cache:
            self._cache['intermediate_size'] = getattr(
                self.config, 'intermediate_size',
                getattr(self.config, 'n_inner', 4 * self.get_hidden_size())
            )
        return self._cache['intermediate_size']
    
    def get_max_position_embeddings(self) -> int:
        """Get maximum position embeddings."""
        if 'max_pos_embeddings' not in self._cache:
            self._cache['max_pos_embeddings'] = getattr(
                self.config, 'max_position_embeddings',
                getattr(self.config, 'n_positions', 2048)
            )
        return self._cache['max_pos_embeddings']
    
    def get_rope_theta(self) -> float:
        """Get RoPE theta parameter."""
        if 'rope_theta' not in self._cache:
            self._cache['rope_theta'] = getattr(
                self.config, 'rope_theta', 500000.0
            )
        return self._cache['rope_theta']
    
    def is_gqa(self) -> bool:
        """Check if the model uses Grouped Query Attention."""
        return self.get_num_key_value_heads() < self.get_num_heads()
    
    def is_mqa(self) -> bool:
        """Check if the model uses Multi-Query Attention."""
        return self.get_num_key_value_heads() == 1
    
    def get_query_groups(self) -> int:
        """Get number of query groups per key-value head."""
        return self.get_num_heads() // self.get_num_key_value_heads()
    
    def get_architecture_type(self) -> str:
        """Get the attention architecture type."""
        if self.is_mqa():
            return "MQA"  # Multi-Query Attention
        elif self.is_gqa():
            return "GQA"  # Grouped Query Attention
        else:
            return "MHA"  # Multi-Head Attention
    
    def get_total_parameters(self) -> int:
        """Estimate total number of parameters."""
        if 'total_params' not in self._cache:
            # Rough estimation based on common transformer architecture
            hidden_size = self.get_hidden_size()
            vocab_size = self.get_vocab_size()
            num_layers = self.get_num_layers()
            intermediate_size = self.get_intermediate_size()
            
            # Embedding parameters
            embedding_params = vocab_size * hidden_size
            
            # Attention parameters per layer
            # Q: hidden_size * hidden_size
            # K, V: num_kv_heads * head_dim * hidden_size (for GQA)
            # O: hidden_size * hidden_size
            attention_params_per_layer = (
                hidden_size * hidden_size +  # Q projection
                self.get_num_key_value_heads() * self.get_head_dim() * hidden_size +  # K projection
                self.get_num_key_value_heads() * self.get_head_dim() * hidden_size +  # V projection
                hidden_size * hidden_size  # O projection
            )
            
            # FFN parameters per layer (up_proj, gate_proj, down_proj for LLaMA)
            ffn_params_per_layer = 3 * hidden_size * intermediate_size
            
            # Layer norm parameters per layer (pre-attention, pre-ffn)
            layernorm_params_per_layer = 2 * hidden_size
            
            total_layer_params = num_layers * (
                attention_params_per_layer + 
                ffn_params_per_layer + 
                layernorm_params_per_layer
            )
            
            # Final layer norm and LM head
            final_params = hidden_size + vocab_size * hidden_size
            
            self._cache['total_params'] = embedding_params + total_layer_params + final_params
        
        return self._cache['total_params']
    
    def get_config_dict(self) -> dict:
        """Get all configuration as a dictionary."""
        return {
            'num_layers': self.get_num_layers(),
            'num_heads': self.get_num_heads(),
            'num_key_value_heads': self.get_num_key_value_heads(),
            'head_dim': self.get_head_dim(),
            'hidden_size': self.get_hidden_size(),
            'vocab_size': self.get_vocab_size(),
            'intermediate_size': self.get_intermediate_size(),
            'max_position_embeddings': self.get_max_position_embeddings(),
            'rope_theta': self.get_rope_theta(),
            'architecture_type': self.get_architecture_type(),
            'query_groups': self.get_query_groups(),
            'estimated_total_parameters': self.get_total_parameters(),
        }
    
    def __repr__(self) -> str:
        """String representation of the model configuration."""
        config = self.get_config_dict()
        return f"ModelConfigWrapper({config})"