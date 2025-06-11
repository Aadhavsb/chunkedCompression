"""
Memory-Safe LLaMA-3 8B Cluster Loader
Designed for HPC environments with GPU clusters
Avoids OOM errors and loads directly onto GPUs
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LLaMA3Loader:
    def __init__(self, model_path, dtype="bfloat16"):
        self.model_path = model_path
        self.dtype = self._get_torch_dtype(dtype)
        self.tokenizer = None
        self.model = None

    def _get_torch_dtype(self, dtype):
        """Convert string dtype to torch dtype"""
        if dtype == "bfloat16":
            return torch.bfloat16
        elif dtype == "float16":
            return torch.float16
        elif dtype == "fp32":
            return torch.float32
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    def load_model(self):
        """Load model safely onto GPU with memory optimization"""
        print(f"🦙 Loading tokenizer from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        print(f"🦙 Loading model from {self.model_path} onto GPUs...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            device_map="auto"   # CRITICAL: safe cluster load
        )
        print("✅ Model fully loaded into GPU memory.")
        return self.model, self.tokenizer

    def get_model_info(self):
        """Get basic model information"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        config = self.model.config
        return {
            "hidden_size": config.hidden_size,
            "num_attention_heads": config.num_attention_heads,
            "num_hidden_layers": config.num_hidden_layers,
            "vocab_size": config.vocab_size,
            "intermediate_size": getattr(config, 'intermediate_size', None),
            "model_type": config.model_type
        }

    def cleanup(self):
        """Clean up GPU memory"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("🧹 GPU memory cleaned up")
