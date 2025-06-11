"""
LLaMA-3 8B Instruct Model Loader
Real model loading with no placeholders or dummy data
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from typing import Dict, Tuple, Any

class LLaMAModelLoader:
    def __init__(self, model_path: str = "/mnt/vstor/CSE_ECSE_GXD234/Meta-Llama-3-8B-Instruct"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.config = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._load_model()
        self._extract_dimensions()
        
    def _load_model(self):
        """Load LLaMA-3 8B Instruct model and tokenizer"""
        print(f"🦙 Loading LLaMA-3 8B Instruct from {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path not found: {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=True
        )
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with automatic device mapping for cluster GPUs
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=torch.float16,  # Use half precision for memory efficiency
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        self.model.eval()  # Set to evaluation mode
        self.config = self.model.config
        
        print(f"✅ Model loaded successfully")
        print(f"   Device map: {self.model.hf_device_map if hasattr(self.model, 'hf_device_map') else 'single device'}")
        print(f"   Model dtype: {next(self.model.parameters()).dtype}")
        
    def _extract_dimensions(self):
        """Extract real model dimensions from LLaMA config"""
        self.hidden_size = self.config.hidden_size  # 4096 for LLaMA-3-8B
        self.num_attention_heads = self.config.num_attention_heads  # 32
        self.head_dim = self.hidden_size // self.num_attention_heads  # 128
        self.vocab_size = self.config.vocab_size  # 128256
        self.num_layers = self.config.num_hidden_layers  # 32
        self.intermediate_size = self.config.intermediate_size  # 14336
        
        print(f"📏 LLaMA-3 8B Model Dimensions:")
        print(f"   Hidden size: {self.hidden_size}")
        print(f"   Attention heads: {self.num_attention_heads}")
        print(f"   Head dimension: {self.head_dim}")
        print(f"   Vocabulary size: {self.vocab_size}")
        print(f"   Number of layers: {self.num_layers}")
        
    def get_attention_weights(self, layer_idx: int = -1) -> Dict[str, torch.Tensor]:
        """
        Extract real attention projection matrices from specified layer
        
        Args:
            layer_idx: Layer index (-1 for last layer)
            
        Returns:
            Dictionary with W_Q, W_K, W_V, W_O matrices
        """
        if layer_idx == -1:
            layer_idx = self.num_layers - 1
            
        layer = self.model.model.layers[layer_idx]
        attention = layer.self_attn
        
        # Extract projection weights
        W_Q = attention.q_proj.weight.data  # [hidden_size, hidden_size]
        W_K = attention.k_proj.weight.data  # [hidden_size, hidden_size]  
        W_V = attention.v_proj.weight.data  # [hidden_size, hidden_size]
        W_O = attention.o_proj.weight.data  # [hidden_size, hidden_size]
        
        print(f"🔍 Extracted attention weights from layer {layer_idx}:")
        print(f"   W_Q: {W_Q.shape}")
        print(f"   W_K: {W_K.shape}")
        print(f"   W_V: {W_V.shape}")
        print(f"   W_O: {W_O.shape}")
        
        return {
            "W_Q": W_Q,
            "W_K": W_K,
            "W_V": W_V,
            "W_O": W_O
        }
    
    def get_language_model_head(self) -> torch.Tensor:
        """Extract the language model head weight matrix"""
        W_LM_HEAD = self.model.lm_head.weight.data  # [vocab_size, hidden_size]
        
        print(f"🎯 Extracted language model head: {W_LM_HEAD.shape}")
        return W_LM_HEAD
    
    def get_hidden_states(self, text: str, max_length: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get real hidden states from LLaMA model for given text
        
        Args:
            text: Input text to process
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (hidden_states, input_ids)
        """
        print(f"🧠 Processing text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        )
        
        # Move to model device
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask', None)
        
        # Get hidden states from model
        with torch.no_grad():
            outputs = self.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False  # Don't use KV cache for extraction
            )
            
            # Extract last layer hidden states
            hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
            
        # Remove batch dimension if batch_size=1
        if hidden_states.shape[0] == 1:
            hidden_states = hidden_states.squeeze(0)  # [seq_len, hidden_size]
            input_ids = input_ids.squeeze(0)  # [seq_len]
        
        print(f"   Hidden states: {hidden_states.shape}")
        print(f"   Input tokens: {input_ids.shape}")
        print(f"   Tokens: {self.tokenizer.decode(input_ids)}")
        
        return hidden_states, input_ids
    
    def generate_text(self, prompt: str, max_new_tokens: int = 50, temperature: float = 0.7) -> str:
        """
        Generate text using the LLaMA model
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        new_tokens = outputs[0][inputs.input_ids.shape[-1]:]
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return generated_text
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        return {
            "model_name": "LLaMA-3-8B-Instruct",
            "model_path": self.model_path,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "head_dim": self.head_dim,
            "vocab_size": self.vocab_size,
            "num_layers": self.num_layers,
            "device": str(self.device),
            "dtype": str(next(self.model.parameters()).dtype),
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
