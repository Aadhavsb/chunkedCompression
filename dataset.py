"""
Dataset loading and tokenization for WikiText-2
"""
import torch
from typing import Tuple, List
from transformers import AutoTokenizer

class WikiTextDataset:
    def __init__(self, max_tokens: int = 128):
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Mock embedding layer (for testing)
        self.d_model = 512
        self.vocab_size = self.tokenizer.vocab_size
        torch.manual_seed(42)  # For reproducible embeddings
        self.embedding = torch.nn.Embedding(self.vocab_size, self.d_model)
        
    def load_sample_text(self) -> str:
        """Load a sample text for testing (mock WikiText-2 content)"""
        # Using a simple mock text instead of actual WikiText-2 for barebones testing
        sample_text = """
        The quick brown fox jumps over the lazy dog. This is a sample text to test 
        the chunked compression system. We need enough tokens to demonstrate the 
        compression profiles working across different chunks. The system will assign 
        different compression ratios to different parts of this sequence. Some tokens 
        will use low compression, others medium, and some high compression ratios.
        This allows us to test the fused output projection and attention mechanisms
        over compressed latent representations in the key-value cache.
        """
        return sample_text.strip()
    
    def get_tokenized_batch(self) -> Tuple[List[int], torch.Tensor]:
        """
        Get tokenized text and embeddings
        
        Returns:
            tokens: List of token IDs
            embeddings: Tensor of shape [T, d_model]
        """
        text = self.load_sample_text()
        
        # Tokenize
        encoded = self.tokenizer.encode(text, add_special_tokens=True)
        
        # Truncate to max_tokens
        tokens = encoded[:self.max_tokens]
        
        # Convert to tensor and embed
        token_tensor = torch.tensor(tokens, dtype=torch.long)
        embeddings = self.embedding(token_tensor)
        
        print(f"📝 Loaded {len(tokens)} tokens from sample text")
        print(f"   Embedding shape: {embeddings.shape}")
        
        return tokens, embeddings
