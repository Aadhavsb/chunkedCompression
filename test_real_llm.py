"""
Test script with real LLM (GPT-2) for chunked compression
"""
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from compression import decompose_and_fuse
import os

class ChunkedCompressionTester:
    def __init__(self):
        print("🚀 Initializing Real LLM Compression Test")
        
        # Load GPT-2 small for testing
        print("📥 Loading GPT-2 model...")
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model.eval()
        
        # Model dimensions
        self.d_model = 768  # GPT-2 hidden size
        self.d_head = 64    # Typical attention head dimension
        self.vocab_size = 50257  # GPT-2 vocab size
        
        # Compression ranks
        self.ranks = {"low": 32, "med": 64, "high": 128}
        
        # Initialize compression profiles
        self.profiles = {}
        self._setup_compression_profiles()
        
    def _setup_compression_profiles(self):
        """Create real SVD-based compression profiles"""
        print("🔧 Setting up compression profiles with real SVD...")
        
        # Create mock W_v and W_o matrices (simulating attention weights)
        torch.manual_seed(42)
        W_v = torch.randn(self.d_model, self.d_head) * 0.02
        W_o = torch.randn(self.d_model, self.d_head) * 0.02
        
        # Use the actual output projection from GPT-2's last layer
        W0 = self.model.lm_head.weight.T  # [d_model, vocab_size]
        
        for name, rank in self.ranks.items():
            print(f"   Creating {name} profile (rank {rank})...")
            
            # Get SVD-based compression matrices
            A, W_fused_attn = decompose_and_fuse(W_v, W_o, rank)
            
            # Create fused output projection: W0 @ B.T where B comes from SVD
            # Here B is essentially the U matrix from SVD, so W_fused_final = W0 @ U[:, :rank]
            U, S, V = torch.svd(W_v)
            B = U[:, :rank]  # [d_model, rank]
            W_fused_final = W0 @ B  # [vocab_size, rank]
            
            self.profiles[name] = {
                "A": A,  # [rank, d_head] - compression matrix
                "W_fused": W_fused_final,  # [vocab_size, rank] - fused output projection
                "r": rank
            }
            
            print(f"     A: {A.shape}, W_fused: {W_fused_final.shape}")
    
    def get_real_hidden_states(self, text: str, max_length: int = 32):
        """Get real hidden states from GPT-2"""
        print(f"🧠 Getting hidden states for: '{text[:50]}...'")
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True)
        input_ids = inputs['input_ids']
        
        # Get hidden states from GPT-2
        with torch.no_grad():
            outputs = self.model.transformer(input_ids, output_hidden_states=True)
            # Use hidden states from the last layer
            hidden_states = outputs.hidden_states[-1].squeeze(0)  # [seq_len, d_model]
        
        print(f"   Hidden states shape: {hidden_states.shape}")
        return hidden_states, input_ids.squeeze(0)
        
    def compress_chunk(self, hidden_states: torch.Tensor, compression_option: str):
        """Compress a chunk of hidden states"""
        profile = self.profiles[compression_option]
        A = profile["A"]  # [rank, d_head]
        
        # Project to head dimension (simulate attention values)
        values = hidden_states[:, :self.d_head]  # [seq_len, d_head]
        
        # Compress: H = values @ A.T  (each token gets compressed)
        compressed = values @ A.T  # [seq_len, rank]
        
        print(f"🗜️  Compressed {hidden_states.shape} -> {compressed.shape} using {compression_option}")
        return compressed
    
    def decode_step(self, compressed_latents: torch.Tensor, compression_option: str):
        """Decode compressed latents to logits using fused matrices"""
        profile = self.profiles[compression_option]
        W_fused = profile["W_fused"]  # [vocab_size, rank]
        
        # Apply fused projection: logits = compressed @ W_fused.T
        logits = compressed_latents @ W_fused.T  # [seq_len, vocab_size]
        
        print(f"🎯 Decoded {compressed_latents.shape} -> {logits.shape} using {compression_option}")
        return logits
    
    def run_pipeline_test(self):
        """Run the full compression/decompression pipeline"""
        print("\n" + "="*60)
        print("🔬 RUNNING FULL PIPELINE TEST")
        print("="*60)
        
        # Get real hidden states
        text = "The quick brown fox jumps over the lazy dog. This is a test sentence for compression."
        hidden_states, tokens = self.get_real_hidden_states(text)
        
        # Test different compression levels
        results = {}
        
        for option in ["low", "med", "high"]:
            print(f"\n--- Testing {option} compression (rank {self.ranks[option]}) ---")
            
            # Compress
            compressed = self.compress_chunk(hidden_states, option)
            
            # Decode
            logits = self.decode_step(compressed, option)
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
            predicted_text = self.tokenizer.decode(predictions)
            
            # Calculate compression ratio
            original_size = hidden_states.numel() * 4  # float32 bytes
            compressed_size = compressed.numel() * 4
            compression_ratio = original_size / compressed_size
            
            results[option] = {
                "compressed_shape": compressed.shape,
                "logits_shape": logits.shape,
                "compression_ratio": compression_ratio,
                "predicted_text": predicted_text[:100] + "..."
            }
            
            print(f"   Compression ratio: {compression_ratio:.2f}x")
            print(f"   Predicted text: {predicted_text[:80]}...")
        
        return results
    
    def validate_matrices(self):
        """Validate all compression matrices have correct shapes"""
        print("\n🔍 Validating compression matrices...")
        
        for option, profile in self.profiles.items():
            A = profile["A"]
            W_fused = profile["W_fused"]
            rank = profile["r"]
            
            # Check shapes
            A_correct = A.shape == (rank, self.d_head)
            W_correct = W_fused.shape == (self.vocab_size, rank)
            
            status = "✅" if (A_correct and W_correct) else "❌"
            print(f"   {option}: {status} A{A.shape}, W_fused{W_fused.shape}")
            
            if not A_correct:
                print(f"      Expected A: ({rank}, {self.d_head})")
            if not W_correct:
                print(f"      Expected W_fused: ({self.vocab_size}, {rank})")

def main():
    """Main test function"""
    # Create tester
    tester = ChunkedCompressionTester()
    
    # Validate matrices
    tester.validate_matrices()
    
    # Run pipeline test
    results = tester.run_pipeline_test()
    
    # Print summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    
    for option, result in results.items():
        print(f"\n{option.upper()} compression:")
        print(f"  Compression ratio: {result['compression_ratio']:.2f}x")
        print(f"  Compressed shape: {result['compressed_shape']}")
        print(f"  Output shape: {result['logits_shape']}")
        print(f"  Sample prediction: {result['predicted_text'][:60]}...")

if __name__ == "__main__":
    main()
