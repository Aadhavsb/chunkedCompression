"""
Refactored LLaMA model loader with configuration support and proper interfaces.
"""
import os
import gc
from typing import Dict, Tuple, Any, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoModelForCausalLM = None
    AutoTokenizer = None

from ..interfaces.model_interface import ModelLoaderInterface, ModelConfigInterface
from ..config.model_config import ModelConfig
from ..utils.memory_manager import MemoryManager
from .model_config_wrapper import ModelConfigWrapper


class LLaMAModelLoader(ModelLoaderInterface):
    """
    Production-grade LLaMA model loader with configuration management.
    Implements the ModelLoaderInterface for consistency across the system.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the model loader with configuration.
        
        Args:
            config: Model configuration object. If None, creates default config.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required but not available. Install torch to use model loading.")
        
        self.config = config or ModelConfig()
        self.model = None
        self.tokenizer = None
        self.model_config_wrapper = None
        self.device = None
        self.memory_manager = MemoryManager()
        
        # Initialize cluster loader lazily
        self._cluster_loader = None
        
    def load_model(self) -> None:
        """Load the LLaMA model and tokenizer."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required but not available. Install torch to use model loading.")
        
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers library is required but not available. Install transformers to use model loading.")
        
        print(f"🦙 Loading LLaMA-3 8B Instruct from {self.config.model_path}")
        
        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"Model path not found: {self.config.model_path}")
        
        # Initialize device
        self.device = torch.device(self.config.get_device_config())
        
        # Load model using cluster loader for memory efficiency
        self._load_with_cluster_loader()
        
        # Setup tokenizer
        self._setup_tokenizer()
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Create model config wrapper
        self.model_config_wrapper = ModelConfigWrapper(self.model.config)
        
        # Log model info
        self._log_model_info()
        
    def _load_with_cluster_loader(self) -> None:
        """Load model using the cluster loader for memory efficiency."""
        try:
            from .llama_loader import LLaMA3Loader
            
            # Initialize cluster loader
            self._cluster_loader = LLaMA3Loader(
                self.config.model_path, 
                dtype=self.config.dtype
            )
            
            # Load model and tokenizer
            self.model, self.tokenizer = self._cluster_loader.load_model()
            
            print("✅ Model loaded successfully using cluster loader")
            
        except ImportError as e:
            print(f"Warning: Cluster loader not available, falling back to standard loading: {e}")
            self._load_with_transformers()
    
    def _load_with_transformers(self) -> None:
        """Fallback to standard transformers loading - NO META TENSORS."""
        with self.memory_manager.managed_computation():
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=self.config.get_dtype_config(),
                device_map={"": 0} if self.device.type == "cuda" else None,  # FORCE GPU 0 - NO META
                low_cpu_mem_usage=False,  # Disable CPU offloading
                trust_remote_code=self.config.trust_remote_code,
                use_auth_token=self.config.use_auth_token,
                revision=self.config.revision,
                max_memory=None,          # No memory limits that cause meta tensors
                offload_folder=None,      # No disk offloading
                load_in_8bit=False,       # No quantization
                load_in_4bit=False,       # No quantization
            )
            
            # Verify NO meta tensors exist
            meta_tensors = []
            for name, param in self.model.named_parameters():
                if param.is_meta:
                    meta_tensors.append(name)
            
            if meta_tensors:
                raise RuntimeError(f"CRITICAL: Found {len(meta_tensors)} meta tensors: {meta_tensors[:5]}... - FORBIDDEN!")
            
            print("✅ Model loaded with transformers - NO META TENSORS")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=self.config.trust_remote_code,
                use_auth_token=self.config.use_auth_token,
                revision=self.config.revision,
            )
        
        print("✅ Model loaded successfully using transformers")
    
    def _setup_tokenizer(self) -> None:
        """Setup tokenizer with proper configuration."""
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
    def _log_model_info(self) -> None:
        """Log detailed model information."""
        print(f"📏 LLaMA-3 8B Model Information:")
        print(f"   Device: {self.device}")
        print(f"   Model dtype: {next(self.model.parameters()).dtype}")
        
        if hasattr(self.model, 'hf_device_map'):
            print(f"   Device map: {self.model.hf_device_map}")
        
        if self.model_config_wrapper:
            print(f"   Hidden size: {self.model_config_wrapper.get_hidden_size()}")
            print(f"   Query heads: {self.model_config_wrapper.get_num_heads()}")
            print(f"   Key/Value heads: {self.model_config_wrapper.get_num_key_value_heads()}")
            print(f"   Head dimension: {self.model_config_wrapper.get_head_dim()}")
            print(f"   Vocabulary size: {self.model_config_wrapper.get_vocab_size()}")
            print(f"   Number of layers: {self.model_config_wrapper.get_num_layers()}")
            print(f"   Architecture: {'GQA' if self.model_config_wrapper.get_num_key_value_heads() < self.model_config_wrapper.get_num_heads() else 'MHA'}")
    
    def get_attention_weights(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """
        Extract attention projection matrices from specified layer.
        
        Args:
            layer_idx: Layer index to extract weights from
            
        Returns:
            Dictionary containing W_Q, W_K, W_V, W_O weight matrices
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if layer_idx < 0:
            layer_idx = self.model_config_wrapper.get_num_layers() + layer_idx
        
        if layer_idx >= self.model_config_wrapper.get_num_layers():
            raise ValueError(f"Layer index {layer_idx} out of range for model with {self.model_config_wrapper.get_num_layers()} layers")
        
        layer = self.model.model.layers[layer_idx]
        attention = layer.self_attn
        
        # Extract projection weights with meta tensor handling
        def safe_extract_weight(module):
            """Safely extract weight tensor, handling meta tensors."""
            weight = module.weight
            if weight.device.type == 'meta':
                # Force materialization of meta tensor
                print(f"   Warning: Materializing meta tensor for {module.__class__.__name__}")
                # Move to CPU first, then to target device
                weight = weight.to('cpu')
            return weight.data
        
        W_Q = safe_extract_weight(attention.q_proj)
        W_K = safe_extract_weight(attention.k_proj)
        W_V = safe_extract_weight(attention.v_proj)
        W_O = safe_extract_weight(attention.o_proj)
        
        print(f"🔍 Extracted attention weights from layer {layer_idx}:")
        print(f"   W_Q: {W_Q.shape} (query heads: {self.model_config_wrapper.get_num_heads()})")
        print(f"   W_K: {W_K.shape} (key heads: {self.model_config_wrapper.get_num_key_value_heads()})")
        print(f"   W_V: {W_V.shape} (value heads: {self.model_config_wrapper.get_num_key_value_heads()})")
        print(f"   W_O: {W_O.shape}")
        
        return {
            "W_Q": W_Q,
            "W_K": W_K,
            "W_V": W_V,
            "W_O": W_O
        }
    
    def get_language_model_head(self) -> torch.Tensor:
        """Extract the language model head weight matrix."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Handle meta tensor for language model head
        lm_head_weight = self.model.lm_head.weight
        if lm_head_weight.device.type == 'meta':
            print(f"   Warning: Materializing meta tensor for lm_head")
            lm_head_weight = lm_head_weight.to('cpu')
        
        W_LM_HEAD = lm_head_weight.data
        print(f"🎯 Extracted language model head: {W_LM_HEAD.shape}")
        return W_LM_HEAD
    
    def extract_hidden_states(self, input_text: str, max_length: int = 50) -> torch.Tensor:
        """
        Extract hidden states from model forward pass.
        
        Args:
            input_text: Input text to process
            max_length: Maximum sequence length
            
        Returns:
            Hidden states tensor
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        print(f"🧠 Processing text: '{input_text[:50]}{'...' if len(input_text) > 50 else ''}'")
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        )
        
        # Move to model device
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Get hidden states from model
        with torch.no_grad():
            with self.memory_manager.managed_computation():
                outputs = self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False
                )
                
                # Extract last layer hidden states
                hidden_states = outputs.hidden_states[-1]
        
        # Remove batch dimension if batch_size=1
        if hidden_states.shape[0] == 1:
            hidden_states = hidden_states.squeeze(0)
        
        print(f"   Hidden states shape: {hidden_states.shape}")
        return hidden_states
    
    def get_model_config(self) -> ModelConfigInterface:
        """Get model configuration wrapper."""
        if self.model_config_wrapper is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model_config_wrapper
    
    @property
    def hidden_size(self) -> int:
        """Get hidden size for backward compatibility."""
        return self.get_model_config().get_hidden_size()
    
    @property 
    def vocab_size(self) -> int:
        """Get vocabulary size for backward compatibility."""
        return self.get_model_config().get_vocab_size()
    
    @property
    def num_heads(self) -> int:
        """Get number of attention heads for backward compatibility."""
        return self.get_model_config().get_num_heads()
    
    @property
    def num_attention_heads(self) -> int:
        """Get number of attention heads for backward compatibility."""
        return self.get_model_config().get_num_heads()
    
    @property
    def num_key_value_heads(self) -> int:
        """Get number of key-value heads for backward compatibility."""
        return self.get_model_config().get_num_key_value_heads()
    
    @property
    def head_dim(self) -> int:
        """Get head dimension for backward compatibility."""
        return self.get_model_config().get_head_dim()
    
    @property
    def num_layers(self) -> int:
        """Get number of layers for backward compatibility."""
        return self.get_model_config().get_num_layers()
    
    def get_hidden_states(self, input_text: str, max_length: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get hidden states for backward compatibility."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        print(f"🧠 Processing text: '{input_text[:50]}{'...' if len(input_text) > 50 else ''}'")
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        )
        
        # Move to model device
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Get hidden states from model
        with torch.no_grad():
            with self.memory_manager.managed_computation():
                outputs = self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False
                )
                
                # Extract last layer hidden states
                hidden_states = outputs.hidden_states[-1]
        
        # Remove batch dimension if batch_size=1
        if hidden_states.shape[0] == 1:
            hidden_states = hidden_states.squeeze(0)
            input_ids = input_ids.squeeze(0)
        
        print(f"   Hidden states shape: {hidden_states.shape}")
        return hidden_states, input_ids
    
    def generate_text(self, prompt: str, max_new_tokens: int = 50, temperature: float = 0.7) -> str:
        """
        Generate text using the LLaMA model.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            with self.memory_manager.managed_computation():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
        
        # Decode only the new tokens
        new_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return generated_text
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        return {
            "model_name": self.config.model_name,
            "model_path": self.config.model_path,
            "hidden_size": self.model_config_wrapper.get_hidden_size(),
            "num_attention_heads": self.model_config_wrapper.get_num_heads(),
            "num_key_value_heads": self.model_config_wrapper.get_num_key_value_heads(),
            "head_dim": self.model_config_wrapper.get_head_dim(),
            "vocab_size": self.model_config_wrapper.get_vocab_size(),
            "num_layers": self.model_config_wrapper.get_num_layers(),
            "device": str(self.device),
            "dtype": str(next(self.model.parameters()).dtype),
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
    
    def cleanup(self) -> None:
        """Clean up model resources and GPU memory."""
        if self._cluster_loader is not None:
            self._cluster_loader.cleanup()
            self._cluster_loader = None
        
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if self.model_config_wrapper is not None:
            self.model_config_wrapper = None
        
        # Force memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("🧹 LLaMAModelLoader cleaned up resources and GPU memory")