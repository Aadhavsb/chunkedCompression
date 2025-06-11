"""
Test script with real LLM (GPT-2) for chunked compression
"""
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModel, AutoTokenizer
from compression import decompose_and_fuse
import numpy as np
import os

class ChunkedCompressionTester:
    def __init__(self):
        print("🚀 Initializing Real LLM Compression Test")
        
        # Load GPT-2 small for testing
        print("📥 Loading GPT-2 model...")
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        
        # Also load the base model for hidden states extraction
        self.base_model = AutoModel.from_pretrained('gpt2', output_hidden_states=True)
        self.base_model.eval()
        
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
        """Create real SVD-based compression profiles using actual GPT-2 attention weights"""
        print("🔧 Setting up compression profiles with real GPT-2 attention matrices...")
        
        # Extract REAL attention matrices from GPT-2's first layer
        first_layer = self.model.transformer.h[0]
        
        # Get actual attention projection matrices
        # GPT-2 uses combined QKV projection, so we'll extract parts of it
        qkv_weight = first_layer.attn.c_attn.weight  # [d_model, 3*d_model]
        
        # Split into Q, K, V portions (each is d_model x d_model)
        d_model = self.d_model
        q_proj = qkv_weight[:, :d_model]           # [d_model, d_model] 
        k_proj = qkv_weight[:, d_model:2*d_model]  # [d_model, d_model]
        v_proj = qkv_weight[:, 2*d_model:]         # [d_model, d_model]
        
        # Project to head dimension (simulating single head)
        W_q = q_proj[:, :self.d_head]  # [d_model, d_head]
        W_k = k_proj[:, :self.d_head]  # [d_model, d_head] 
        W_v = v_proj[:, :self.d_head]  # [d_model, d_head]
        
        # Get actual output projection matrix
        W_o = first_layer.attn.c_proj.weight[:self.d_head, :]  # [d_head, d_model] -> transpose to [d_model, d_head]
        W_o = W_o.T
        
        # Use the actual language model head from GPT-2
        W0 = self.model.lm_head.weight.T  # [d_model, vocab_size]
        
        # Fixed key compression rank for all profiles
        key_compression_rank = 32
        
        for name, value_rank in self.ranks.items():
            print(f"   Creating {name} profile (value rank {value_rank}, key rank {key_compression_rank})...")
            
            # Value compression using real W_v and W_o
            A_v, W_fused_attn = decompose_and_fuse(W_v, W_o, value_rank)
            
            # Key compression using real W_k
            from compression import compress_keys
            A_k, B_k = compress_keys(W_k, key_compression_rank)
            
            # Create proper fused output projection for language modeling
            # We want: compressed_latents @ W_fused.T -> logits [seq_len, vocab_size]
            # So W_fused should be [vocab_size, rank]
            
            # Get the U matrix from value SVD for proper fusion with LM head
            U, S, V = torch.svd(W_v)
            U_truncated = U[:, :value_rank]  # [d_model, value_rank]
            
            # Fuse with the language model head: W0.T @ U_truncated
            # W0 is [d_model, vocab_size], so W0.T is [vocab_size, d_model]
            # U_truncated is [d_model, value_rank]
            # Result: [vocab_size, value_rank]
            W_fused_final = W0.T @ U_truncated  # [vocab_size, value_rank]
            
            self.profiles[name] = {
                # Value compression matrices
                "A": A_v,           # Value compression matrix [value_rank, d_head]
                "W_fused": W_fused_final,  # Fused output projection [vocab_size, value_rank]
                "r": value_rank,    # Value compression rank
                
                # Key compression matrices
                "A_K": A_k,         # Key compression matrix [key_rank, d_head]
                "B_K": B_k,         # Key reconstruction matrix [d_head, key_rank]
                "r_k": key_compression_rank  # Key compression rank (fixed)
            }
            
            print(f"     A_v: {A_v.shape}, W_fused: {W_fused_final.shape}")
            print(f"     A_k: {A_k.shape}, B_k: {B_k.shape}")
    
    def get_real_hidden_states(self, text: str, max_length: int = 32):
        """Get real hidden states from GPT-2"""
        print(f"🧠 Getting hidden states for: '{text[:50]}...'")
        
        # Tokenize using proper GPT-2 tokenizer
        inputs = self.tokenizer(text, return_tensors='pt', max_length=max_length, 
                               truncation=True, padding=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask', None)
        
        # Get hidden states from GPT-2 base model
        with torch.no_grad():
            if attention_mask is not None:
                outputs = self.base_model(input_ids, attention_mask=attention_mask)
            else:
                outputs = self.base_model(input_ids)
            
            # Use hidden states from the last layer
            hidden_states = outputs.hidden_states[-1].squeeze(0)  # [seq_len, d_model]
        
        print(f"   Hidden states shape: {hidden_states.shape}")
        print(f"   Tokens: {self.tokenizer.decode(input_ids.squeeze(0))}")
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
    
    def get_ground_truth_logits(self, input_ids: torch.Tensor):
        """Get ground truth logits from original GPT-2"""
        with torch.no_grad():
            outputs = self.model(input_ids.unsqueeze(0))
            logits = outputs.logits.squeeze(0)  # [seq_len, vocab_size]
        return logits
    
    def calculate_perplexity(self, logits: torch.Tensor, target_ids: torch.Tensor):
        """Calculate perplexity of predictions"""
        # Shift logits and targets for next-token prediction
        shift_logits = logits[:-1, :].contiguous()
        shift_labels = target_ids[1:].contiguous()
        
        # Calculate cross-entropy loss
        loss = F.cross_entropy(shift_logits, shift_labels, reduction='mean')
        perplexity = torch.exp(loss)
        
        return perplexity.item(), loss.item()
    
    def autoregressive_decode(self, prompt: str, max_new_tokens: int = 10, compression_option: str = "med"):
        """Implement autoregressive decoding with compression"""
        print(f"\n🔄 Starting autoregressive decoding with {compression_option} compression...")
        print(f"Prompt: '{prompt}'")
        
        # Tokenize initial prompt
        inputs = self.tokenizer(prompt, return_tensors='pt')
        input_ids = inputs['input_ids'].squeeze(0)  # [prompt_len]
        
        generated_tokens = input_ids.clone()
        all_compressed_states = []
        
        for step in range(max_new_tokens):
            print(f"\n--- Generation Step {step + 1} ---")
            
            # Get hidden states for current sequence
            hidden_states, _ = self.get_real_hidden_states(
                self.tokenizer.decode(generated_tokens), 
                max_length=len(generated_tokens) + 1
            )
            
            # Compress the last token's hidden state
            last_hidden = hidden_states[-1:, :]  # [1, d_model]
            compressed = self.compress_chunk(last_hidden, compression_option)
            all_compressed_states.append(compressed)
            
            # Decode to get logits
            logits = self.decode_step(compressed, compression_option)  # [1, vocab_size]
            
            # Sample next token (greedy decoding)
            next_token_id = torch.argmax(logits[0, :]).item()
            next_token = self.tokenizer.decode([next_token_id])
            
            print(f"   Generated token: '{next_token}' (ID: {next_token_id})")
            
            # Append to sequence
            generated_tokens = torch.cat([generated_tokens, torch.tensor([next_token_id])])
            
            # Stop if we hit EOS
            if next_token_id == self.tokenizer.eos_token_id:
                break
        
        generated_text = self.tokenizer.decode(generated_tokens)
        print(f"\n✅ Generated text: '{generated_text}'")
        
        return generated_text, generated_tokens, all_compressed_states
    
    def run_pipeline_test(self):
        """Run the full compression/decompression pipeline"""
        print("\n" + "="*60)
        print("🔬 RUNNING FULL PIPELINE TEST")
        print("="*60)
        
        # Test with WikiText-style data
        test_texts = [
            "The history of science begins with ancient civilizations.",
            "Machine learning models require large amounts of training data.",
            "The quick brown fox jumps over the lazy dog repeatedly."
        ]
        
        all_results = {}
        
        for text_idx, text in enumerate(test_texts):
            print(f"\n--- Test Text {text_idx + 1}: '{text[:40]}...' ---")
            
            # Get real hidden states
            hidden_states, tokens = self.get_real_hidden_states(text)
            
            # Get ground truth logits for comparison
            ground_truth_logits = self.get_ground_truth_logits(tokens)
            
            text_results = {}
            
            # Test different compression levels
            for option in ["low", "med", "high"]:
                print(f"\n🔸 Testing {option} compression (rank {self.ranks[option]})")
                
                # Compress
                compressed = self.compress_chunk(hidden_states, option)
                
                # Decode
                logits = self.decode_step(compressed, option)
                
                # Calculate metrics
                perplexity, loss = self.calculate_perplexity(logits, tokens)
                gt_perplexity, gt_loss = self.calculate_perplexity(ground_truth_logits, tokens)
                
                # Token accuracy
                predictions = torch.argmax(logits, dim=-1)
                targets = tokens[1:]  # Shift for next-token prediction
                pred_shifted = predictions[:-1]
                accuracy = (pred_shifted == targets).float().mean().item()
                
                # Compression ratio
                original_size = hidden_states.numel() * 4  # float32 bytes
                compressed_size = compressed.numel() * 4
                compression_ratio = original_size / compressed_size
                
                # Predicted text
                predicted_text = self.tokenizer.decode(predictions)
                
                text_results[option] = {
                    "compressed_shape": compressed.shape,
                    "logits_shape": logits.shape,
                    "compression_ratio": compression_ratio,
                    "perplexity": perplexity,
                    "loss": loss,
                    "accuracy": accuracy,
                    "predicted_text": predicted_text[:100] + "...",
                    "ground_truth_perplexity": gt_perplexity,
                    "ground_truth_loss": gt_loss
                }
                
                print(f"   Compression ratio: {compression_ratio:.2f}x")
                print(f"   Perplexity: {perplexity:.2f} (vs GT: {gt_perplexity:.2f})")
                print(f"   Loss: {loss:.4f} (vs GT: {gt_loss:.4f})")
                print(f"   Accuracy: {accuracy:.2%}")
                print(f"   Predicted: {predicted_text[:60]}...")
            
            all_results[f"text_{text_idx}"] = text_results
        
        return all_results
    
    def run_autoregressive_test(self):
        """Test autoregressive generation with compression"""
        print("\n" + "="*60)
        print("🤖 RUNNING AUTOREGRESSIVE GENERATION TEST")
        print("="*60)
        
        test_prompts = [
            "The future of artificial intelligence",
            "In the beginning was",
            "Scientists have discovered"
        ]
        
        generation_results = {}
        
        for prompt_idx, prompt in enumerate(test_prompts):
            print(f"\n--- Prompt {prompt_idx + 1}: '{prompt}' ---")
            
            prompt_results = {}
            
            # Test different compression levels for generation
            for option in ["low", "med", "high"]:
                print(f"\n🔸 Generating with {option} compression...")
                
                generated_text, generated_tokens, compressed_states = self.autoregressive_decode(
                    prompt, max_new_tokens=8, compression_option=option
                )
                
                # Calculate compression stats
                total_original_size = sum(
                    state.shape[0] * self.d_model * 4 for state in compressed_states
                )
                total_compressed_size = sum(
                    state.numel() * 4 for state in compressed_states
                )
                avg_compression_ratio = total_original_size / total_compressed_size if total_compressed_size > 0 else 0
                
                prompt_results[option] = {
                    "generated_text": generated_text,
                    "generated_tokens": generated_tokens.tolist(),
                    "num_compressed_states": len(compressed_states),
                    "avg_compression_ratio": avg_compression_ratio
                }
                
                print(f"   Generated: '{generated_text}'")
                print(f"   Avg compression: {avg_compression_ratio:.2f}x")
            
            generation_results[f"prompt_{prompt_idx}"] = prompt_results
        
        return generation_results
    
    def validate_matrices(self):
        """Validate all compression matrices have correct shapes"""
        print("\n🔍 Validating compression matrices...")
        
        for option, profile in self.profiles.items():
            A_v = profile["A"]
            W_fused = profile["W_fused"]
            A_k = profile["A_K"]
            B_k = profile["B_K"]
            value_rank = profile["r"]
            key_rank = profile["r_k"]
            
            # Check value compression shapes
            A_v_correct = A_v.shape == (value_rank, self.d_head)
            W_correct = W_fused.shape == (self.vocab_size, value_rank)
            
            # Check key compression shapes
            A_k_correct = A_k.shape == (key_rank, self.d_head)
            B_k_correct = B_k.shape == (self.d_head, key_rank)
            
            all_correct = A_v_correct and W_correct and A_k_correct and B_k_correct
            status = "✅" if all_correct else "❌"
            
            print(f"   {option}: {status}")
            print(f"     Value: A_v{A_v.shape}, W_fused{W_fused.shape}")
            print(f"     Key: A_k{A_k.shape}, B_k{B_k.shape}")
            
            if not A_v_correct:
                print(f"      Expected A_v: ({value_rank}, {self.d_head})")
            if not W_correct:
                print(f"      Expected W_fused: ({self.vocab_size}, {value_rank})")
            if not A_k_correct:
                print(f"      Expected A_k: ({key_rank}, {self.d_head})")
            if not B_k_correct:
                print(f"      Expected B_k: ({self.d_head}, {key_rank})")

def main():
    """Main test function"""
    print("🚀 STARTING COMPREHENSIVE REAL LLM COMPRESSION TEST")
    print("="*70)
    
    # Create tester
    tester = ChunkedCompressionTester()
    
    # Validate matrices
    tester.validate_matrices()
    
    # Run compression pipeline test
    print("\n" + "🔬" * 20)
    pipeline_results = tester.run_pipeline_test()
    
    # Run autoregressive generation test
    print("\n" + "🤖" * 20)
    generation_results = tester.run_autoregressive_test()
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("📊 COMPREHENSIVE SUMMARY")
    print("="*70)
    
    # Pipeline results summary
    print("\n📈 COMPRESSION PIPELINE RESULTS:")
    for text_key, text_results in pipeline_results.items():
        print(f"\n{text_key.upper()}:")
        for option, result in text_results.items():
            print(f"  {option.upper()}:")
            print(f"    Compression: {result['compression_ratio']:.2f}x")
            print(f"    Perplexity: {result['perplexity']:.2f} (vs GT: {result['ground_truth_perplexity']:.2f})")
            print(f"    Accuracy: {result['accuracy']:.2%}")
            print(f"    Loss: {result['loss']:.4f} (vs GT: {result['ground_truth_loss']:.4f})")
    
    # Generation results summary
    print("\n🤖 AUTOREGRESSIVE GENERATION RESULTS:")
    for prompt_key, prompt_results in generation_results.items():
        print(f"\n{prompt_key.upper()}:")
        for option, result in prompt_results.items():
            print(f"  {option.upper()}:")
            print(f"    Generated: '{result['generated_text'][:80]}...'")
            print(f"    Avg compression: {result['avg_compression_ratio']:.2f}x")
            print(f"    Tokens generated: {result['num_compressed_states']}")
    
    # Calculate overall metrics
    print("\n📊 OVERALL PERFORMANCE:")
    all_perplexities = {}
    all_accuracies = {}
    all_compression_ratios = {}
    
    for text_results in pipeline_results.values():
        for option, result in text_results.items():
            if option not in all_perplexities:
                all_perplexities[option] = []
                all_accuracies[option] = []
                all_compression_ratios[option] = []
            
            all_perplexities[option].append(result['perplexity'])
            all_accuracies[option].append(result['accuracy'])
            all_compression_ratios[option].append(result['compression_ratio'])
    
    for option in ["low", "med", "high"]:
        avg_perplexity = np.mean(all_perplexities[option])
        avg_accuracy = np.mean(all_accuracies[option])
        avg_compression = np.mean(all_compression_ratios[option])
        
        print(f"\n{option.upper()} compression averages:")
        print(f"  Perplexity: {avg_perplexity:.2f}")
        print(f"  Accuracy: {avg_accuracy:.2%}")
        print(f"  Compression: {avg_compression:.2f}x")
    
    print("\n✅ COMPREHENSIVE TESTING COMPLETE!")
    print("🎯 The system now uses real GPT-2 hidden states, proper tokenization,")
    print("   autoregressive decoding, and comprehensive evaluation metrics!")
    
    return pipeline_results, generation_results

if __name__ == "__main__":
    main()
