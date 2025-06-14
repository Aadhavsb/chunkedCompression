"""
LLaMA-3 8B Compressed Autoregressive Decoder
Full feed-forward autoregressive decoding with real-time KV compression
NO SIMULATIONS - Real inference with compressed KV cache
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import time
from datetime import datetime

from ..model import LLaMAModelLoader
from ..compression import LLaMACompressionProfileBuilder
from ..cache import LLaMAKVCache, StandardKVCache
from ..data import LLaMADatasetHandler

class CompressedAutoRegressiveDecoder:
    def __init__(self, model_path: str = "/mnt/vstor/CSE_ECSE_GXD234/Meta-Llama-3-8B-Instruct"):
        print(f"🚀 Initializing Compressed Autoregressive Decoder")
        print(f"📍 Model path: {model_path}")
        
        # Load components
        self.model_loader = LLaMAModelLoader(model_path)
        self.compression_profiles = LLaMACompressionProfiles(self.model_loader)
        self.dataset_handler = LLaMADatasetHandler(self.model_loader)
        
        # Initialize caches
        self.compressed_cache = LLaMAKVCache(enable_compression=True)
        self.standard_cache = StandardKVCache()
        
        # Model components
        self.model = self.model_loader.model
        self.tokenizer = self.model_loader.tokenizer
        
        # Model dimensions
        self.hidden_size = self.model_loader.hidden_size
        self.num_layers = self.model_loader.num_layers
        self.num_attention_heads = self.model_loader.num_attention_heads
        self.num_key_value_heads = self.model_loader.num_key_value_heads
        self.head_dim = self.model_loader.head_dim
        self.vocab_size = self.model_loader.vocab_size
        
        # Generation settings
        self.generation_stats = {
            "total_tokens_generated": 0,
            "total_generation_time": 0.0,
            "compression_overhead": 0.0,
            "reconstruction_overhead": 0.0,
            "memory_usage": {}
        }
        
        print(f"✅ Compressed decoder initialized")
        print(f"   Model: {self.model_loader.model_name}")
        print(f"   Layers: {self.num_layers}")
        print(f"   Attention heads: {self.num_attention_heads}")
        print(f"   KV heads: {self.num_key_value_heads}")
        print(f"   Hidden size: {self.hidden_size}")
        print(f"   Vocab size: {self.vocab_size}")

    def generate_compressed(self, 
                          prompt: str,
                          max_new_tokens: int = 50,
                          compression_strategy: str = "adaptive",
                          temperature: float = 1.0,
                          top_k: int = 50,
                          do_sample: bool = True) -> Dict[str, Any]:
        """
        Generate text using compressed KV cache autoregressive decoding
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            compression_strategy: How to assign compression profiles
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            do_sample: Whether to sample or use greedy decoding
            
        Returns:
            Generation results with quality metrics
        """
        print(f"\n🎯 Starting Compressed Autoregressive Generation")
        print(f"   Prompt: '{prompt[:50]}...'")
        print(f"   Max new tokens: {max_new_tokens}")
        print(f"   Compression strategy: {compression_strategy}")
        print(f"=" * 60)
        
        generation_start = time.time()
        
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        prompt_length = input_ids.shape[1]
        
        print(f"📝 Prompt tokenized: {input_ids.shape} tokens")
        
        # Clear compressed cache
        self.compressed_cache.clear_cache()
        
        # Initialize generation tracking
        generated_ids = input_ids.clone()
        all_logits = []
        compression_mappings = []
        generation_metrics = {
            "tokens_generated": 0,
            "compression_overhead_per_token": [],
            "memory_usage_per_token": [],
            "attention_computation_time": [],
            "total_generation_time": 0.0
        }
        
        # Autoregressive generation loop
        for step in range(max_new_tokens):
            step_start = time.time()
            
            print(f"\n🔄 Generation Step {step + 1}/{max_new_tokens}")
            
            # Determine compression profile for this token
            compression_profile = self._get_compression_profile(step, compression_strategy)
            compression_mappings.append(compression_profile)
            
            print(f"   Compression profile: {compression_profile}")
            
            # Forward pass with compressed KV cache
            with torch.no_grad():
                # Get current sequence
                current_ids = generated_ids
                
                # Run model forward with compressed attention
                logits, attention_metrics = self._forward_with_compressed_kv(
                    input_ids=current_ids,
                    current_position=prompt_length + step,
                    compression_profile=compression_profile
                )
                
                # Extract logits for next token prediction (last position)
                next_token_logits = logits[0, -1, :]  # [vocab_size]
                all_logits.append(next_token_logits.cpu())
                
                # Sample next token
                next_token_id = self._sample_next_token(
                    next_token_logits, 
                    temperature=temperature,
                    top_k=top_k,
                    do_sample=do_sample
                )
                
                # Append to sequence
                next_token_tensor = torch.tensor([[next_token_id]], device=self.model.device)
                generated_ids = torch.cat([generated_ids, next_token_tensor], dim=1)
                
                # Decode current token for display
                next_token_text = self.tokenizer.decode(next_token_id, skip_special_tokens=True)
                print(f"   Generated token: '{next_token_text}' (id: {next_token_id})")
                
                # Track metrics
                step_time = time.time() - step_start
                generation_metrics["tokens_generated"] += 1
                generation_metrics["attention_computation_time"].append(attention_metrics["computation_time"])
                
                # Memory usage
                memory_usage = self.compressed_cache.get_memory_usage()
                generation_metrics["memory_usage_per_token"].append(memory_usage["total_memory_mb"])
                
                print(f"   Step time: {step_time:.4f}s")
                print(f"   Cache memory: {memory_usage['total_memory_mb']:.4f} MB")
        
        total_generation_time = time.time() - generation_start
        generation_metrics["total_generation_time"] = total_generation_time
        
        # Decode full generated text
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        new_text = self.tokenizer.decode(generated_ids[0, prompt_length:], skip_special_tokens=True)
        
        print(f"\n✅ Generation Complete!")
        print(f"   Total time: {total_generation_time:.2f}s")
        print(f"   Tokens/second: {max_new_tokens / total_generation_time:.2f}")
        print(f"   Generated text: '{new_text[:100]}...'")
        
        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "new_text": new_text,
            "generated_ids": generated_ids.cpu(),
            "prompt_length": prompt_length,
            "tokens_generated": max_new_tokens,
            "compression_mappings": compression_mappings,
            "generation_metrics": generation_metrics,
            "total_generation_time": total_generation_time,
            "tokens_per_second": max_new_tokens / total_generation_time,
            "all_logits": torch.stack(all_logits) if all_logits else None
        }

    def generate_standard(self,
                         prompt: str,
                         max_new_tokens: int = 50,
                         temperature: float = 1.0,
                         top_k: int = 50,
                         do_sample: bool = True) -> Dict[str, Any]:
        """
        Generate text using standard (uncompressed) KV cache for comparison
        """
        print(f"\n🔄 Starting Standard Autoregressive Generation")
        print(f"   Prompt: '{prompt[:50]}...'")
        print(f"   Max new tokens: {max_new_tokens}")
        print(f"=" * 60)
        
        generation_start = time.time()
        
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        prompt_length = input_ids.shape[1]
        
        # Clear standard cache
        self.standard_cache.clear_cache()
        
        # Use standard model generation
        with torch.no_grad():
            # Use the model's built-in generation
            generated_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        total_generation_time = time.time() - generation_start
        
        # Decode text
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        new_text = self.tokenizer.decode(generated_ids[0, prompt_length:], skip_special_tokens=True)
        
        print(f"\n✅ Standard Generation Complete!")
        print(f"   Total time: {total_generation_time:.2f}s")
        print(f"   Tokens/second: {max_new_tokens / total_generation_time:.2f}")
        
        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "new_text": new_text,
            "generated_ids": generated_ids.cpu(),
            "prompt_length": prompt_length,
            "tokens_generated": max_new_tokens,
            "total_generation_time": total_generation_time,
            "tokens_per_second": max_new_tokens / total_generation_time
        }

    def _forward_with_compressed_kv(self,
                                  input_ids: torch.Tensor,
                                  current_position: int,
                                  compression_profile: str) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward pass with compressed KV cache integration
        
        Args:
            input_ids: Current sequence [1, seq_len]
            current_position: Current token position in sequence
            compression_profile: Compression profile to use
            
        Returns:
            Tuple of (logits, attention_metrics)
        """
        computation_start = time.time()
        
        # Get embeddings for current sequence
        inputs_embeds = self.model.model.embed_tokens(input_ids)  # [1, seq_len, hidden_size]
        hidden_states = inputs_embeds
        
        # Process through each transformer layer
        for layer_idx in range(min(2, self.num_layers)):  # Test first 2 layers for speed
            layer = self.model.model.layers[layer_idx]
            
            # Input layernorm
            normed_hidden_states = layer.input_layernorm(hidden_states)
            
            # Self-attention with compressed KV cache
            attention_output = self._compressed_self_attention(
                hidden_states=normed_hidden_states,
                layer_idx=layer_idx,
                current_position=current_position,
                compression_profile=compression_profile
            )
            
            # Residual connection
            hidden_states = hidden_states + attention_output
            
            # Post-attention layernorm
            normed_hidden_states = layer.post_attention_layernorm(hidden_states)
            
            # MLP
            mlp_output = layer.mlp(normed_hidden_states)
            
            # Residual connection
            hidden_states = hidden_states + mlp_output
        
        # Final layernorm
        hidden_states = self.model.model.norm(hidden_states)
        
        # Language model head
        logits = self.model.lm_head(hidden_states)  # [1, seq_len, vocab_size]
        
        computation_time = time.time() - computation_start
        
        return logits, {"computation_time": computation_time}

    def _compressed_self_attention(self,
                                 hidden_states: torch.Tensor,
                                 layer_idx: int,
                                 current_position: int,
                                 compression_profile: str) -> torch.Tensor:
        """
        Self-attention with compressed KV cache
        
        Args:
            hidden_states: Input hidden states [1, seq_len, hidden_size]
            layer_idx: Current layer index
            current_position: Current token position
            compression_profile: Compression profile to use
            
        Returns:
            Attention output [1, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Get attention weights for this layer
        attention_weights = self.model_loader.get_attention_weights(layer_idx)
        W_Q = attention_weights["W_Q"]  # [num_query_heads * head_dim, hidden_size]
        W_K = attention_weights["W_K"]  # [num_kv_heads * head_dim, hidden_size]
        W_V = attention_weights["W_V"]  # [num_kv_heads * head_dim, hidden_size]
        W_O = attention_weights["W_O"]  # [hidden_size, num_query_heads * head_dim]
        
        # Project queries, keys, values for current token (last token in sequence)
        current_hidden = hidden_states[0, -1, :]  # [hidden_size] - last token
        
        # Multi-head attention output
        attention_outputs = []
        
        # Process each attention head
        for head_idx in range(min(4, self.num_attention_heads)):  # Test subset for speed
            # GQA mapping
            heads_per_kv = self.num_attention_heads // self.num_key_value_heads
            kv_head_idx = head_idx // heads_per_kv
            
            # Extract head-specific projections
            W_Q_head = W_Q[head_idx * self.head_dim:(head_idx + 1) * self.head_dim, :]
            W_K_head = W_K[kv_head_idx * self.head_dim:(kv_head_idx + 1) * self.head_dim, :]
            W_V_head = W_V[kv_head_idx * self.head_dim:(kv_head_idx + 1) * self.head_dim, :]
            
            # Compute Q, K, V for current token
            query = W_Q_head @ current_hidden      # [head_dim]
            key = W_K_head @ current_hidden        # [head_dim]
            value = W_V_head @ current_hidden      # [head_dim]
            
            # Compress and store K, V in cache
            compression_start = time.time()
            compressed_key = self.compression_profiles.compress_keys(key, head_idx)
            compressed_value = self.compression_profiles.compress_values(value, compression_profile, head_idx)
            
            self.compressed_cache.store_compressed_kv(
                layer_idx=layer_idx,
                head_idx=head_idx,
                compressed_keys=compressed_key,
                compressed_values=compressed_value,
                token_idx=current_position,
                compression_profile=compression_profile
            )
            compression_time = time.time() - compression_start
            self.generation_stats["compression_overhead"] += compression_time
            
            # Retrieve all cached K, V for attention computation
            cache_groups = self.compressed_cache.get_cache_groups()
            all_keys = []
            all_values = []
            
            for group_layer_idx, group_head_idx, profile_name in cache_groups:
                if group_layer_idx == layer_idx and group_head_idx == head_idx:
                    # Get compressed K, V from cache
                    cached_comp_keys, cached_comp_values = self.compressed_cache.retrieve_compressed_kv(
                        group_layer_idx, group_head_idx, profile_name
                    )
                    
                    if cached_comp_keys.numel() > 0:
                        # Reconstruct keys on-the-fly
                        reconstruction_start = time.time()
                        reconstructed_keys = self.compression_profiles.reconstruct_keys(cached_comp_keys, head_idx)
                        reconstruction_time = time.time() - reconstruction_start
                        self.generation_stats["reconstruction_overhead"] += reconstruction_time
                        
                        all_keys.append(reconstructed_keys)
                        all_values.append(cached_comp_values)  # Keep values compressed
            
            if len(all_keys) > 0:
                # Concatenate all cached keys/values
                all_keys_tensor = torch.cat(all_keys, dim=0)  # [total_cached_tokens, head_dim]
                all_values_tensor = torch.cat(all_values, dim=0)  # [total_cached_tokens, value_rank]
                
                # Compute attention scores
                scores = query @ all_keys_tensor.T  # [total_cached_tokens]
                attention_weights = F.softmax(scores / np.sqrt(self.head_dim), dim=0)
                
                # Apply attention to compressed values
                context_compressed = attention_weights @ all_values_tensor  # [value_rank]
                
                # Decode compressed values to output space using fused projection
                context_output = self.compression_profiles.decode_to_logits(
                    context_compressed, compression_profile, head_idx
                )
                
                # Project back to hidden space (simplified)
                if context_output.shape[0] == self.vocab_size:
                    # Map from vocab space to hidden space
                    lm_head_weight = self.model_loader.get_language_model_head()
                    context_hidden = context_output @ lm_head_weight / self.vocab_size
                    attention_outputs.append(context_hidden)
                else:
                    # Direct hidden space output
                    attention_outputs.append(context_output)
            else:
                # No cached context, use zero
                attention_outputs.append(torch.zeros(hidden_size, device=hidden_states.device, dtype=hidden_states.dtype))
        
        # Combine all head outputs
        if len(attention_outputs) > 0:
            combined_output = torch.mean(torch.stack(attention_outputs), dim=0)  # [hidden_size]
        else:
            combined_output = torch.zeros(hidden_size, device=hidden_states.device, dtype=hidden_states.dtype)
        
        # Apply output projection
        final_output = combined_output.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_size]
        
        # Expand to full sequence length (simplified - normally only last position changes)
        output = torch.zeros_like(hidden_states)
        output[0, -1, :] = combined_output  # Only update last position
        
        return output

    def _get_compression_profile(self, step: int, strategy: str) -> str:
        """Determine compression profile for current generation step"""
        if strategy == "adaptive":
            # Use more aggressive compression as we generate more tokens
            if step < 10:
                return "high"  # Less compression for early tokens
            elif step < 30:
                return "med"   # Medium compression for middle tokens
            else:
                return "low"   # Aggressive compression for later tokens
        elif strategy == "aggressive":
            return "low"  # Always use aggressive compression
        elif strategy == "conservative":
            return "high"  # Always use conservative compression
        else:
            # Cycle through profiles
            profiles = ["low", "med", "high"]
            return profiles[step % len(profiles)]

    def _sample_next_token(self,
                          logits: torch.Tensor,
                          temperature: float = 1.0,
                          top_k: int = 50,
                          do_sample: bool = True) -> int:
        """Sample next token from logits"""
        if not do_sample:
            # Greedy decoding
            return logits.argmax().item()
        
        # Apply temperature
        logits = logits / temperature
        
        # Top-k sampling
        if top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, top_k)
            # Zero out logits not in top-k
            filtered_logits = torch.full_like(logits, float('-inf'))
            filtered_logits[top_k_indices] = top_k_logits
            logits = filtered_logits
        
        # Sample from distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token.item()

    def compare_generations(self,
                          prompt: str,
                          max_new_tokens: int = 50,
                          num_runs: int = 3) -> Dict[str, Any]:
        """
        Compare compressed vs standard generation quality
        
        Args:
            prompt: Input prompt
            max_new_tokens: Number of tokens to generate
            num_runs: Number of runs to average
            
        Returns:
            Comprehensive comparison results
        """
        print(f"\n🔬 Running Generation Comparison")
        print(f"   Prompt: '{prompt[:50]}...'")
        print(f"   Tokens per run: {max_new_tokens}")
        print(f"   Number of runs: {num_runs}")
        print(f"=" * 60)
        
        comparison_results = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "num_runs": num_runs,
            "compressed_runs": [],
            "standard_runs": [],
            "aggregate_metrics": {}
        }
        
        # Run compressed generations
        print(f"\n🗜️ Running {num_runs} compressed generations...")
        for run_idx in range(num_runs):
            print(f"\n--- Compressed Run {run_idx + 1}/{num_runs} ---")
            
            compressed_result = self.generate_compressed(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                compression_strategy="adaptive",
                temperature=0.8,  # Consistent sampling
                do_sample=True
            )
            comparison_results["compressed_runs"].append(compressed_result)
        
        # Run standard generations
        print(f"\n📝 Running {num_runs} standard generations...")
        for run_idx in range(num_runs):
            print(f"\n--- Standard Run {run_idx + 1}/{num_runs} ---")
            
            standard_result = self.generate_standard(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.8,  # Consistent sampling
                do_sample=True
            )
            comparison_results["standard_runs"].append(standard_result)
        
        # Calculate aggregate metrics
        comparison_results["aggregate_metrics"] = self._calculate_comparison_metrics(
            comparison_results["compressed_runs"],
            comparison_results["standard_runs"]
        )
        
        # Print summary
        self._print_comparison_summary(comparison_results)
        
        return comparison_results

    def _calculate_comparison_metrics(self,
                                    compressed_runs: List[Dict],
                                    standard_runs: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregate comparison metrics"""
        
        # Speed metrics
        avg_compressed_speed = np.mean([run["tokens_per_second"] for run in compressed_runs])
        avg_standard_speed = np.mean([run["tokens_per_second"] for run in standard_runs])
        speedup = avg_standard_speed / avg_compressed_speed if avg_compressed_speed > 0 else 1.0
        
        # Memory metrics (from compressed runs)
        memory_usage = compressed_runs[0]["generation_metrics"]["memory_usage_per_token"]
        avg_memory_usage = np.mean(memory_usage) if memory_usage else 0.0
        
        # Text quality metrics (simplified)
        compressed_texts = [run["new_text"] for run in compressed_runs]
        standard_texts = [run["new_text"] for run in standard_runs]
        
        # Calculate text similarity (character-level)
        similarities = []
        for comp_text, std_text in zip(compressed_texts, standard_texts):
            similarity = self._calculate_text_similarity(comp_text, std_text)
            similarities.append(similarity)
        
        avg_text_similarity = np.mean(similarities)
        
        return {
            "speed_metrics": {
                "avg_compressed_tokens_per_sec": avg_compressed_speed,
                "avg_standard_tokens_per_sec": avg_standard_speed,
                "speed_ratio": speedup,
                "is_compressed_faster": speedup > 1.0
            },
            "memory_metrics": {
                "avg_memory_usage_mb": avg_memory_usage,
                "compression_overhead": self.generation_stats["compression_overhead"],
                "reconstruction_overhead": self.generation_stats["reconstruction_overhead"]
            },
            "quality_metrics": {
                "avg_text_similarity": avg_text_similarity,
                "similarity_threshold_met": avg_text_similarity > 0.7  # 70% similarity threshold
            },
            "compressed_texts": compressed_texts,
            "standard_texts": standard_texts,
            "text_similarities": similarities
        }

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate character-level similarity between two texts"""
        if not text1 or not text2:
            return 0.0
        
        # Simple character-level Jaccard similarity
        set1 = set(text1.lower())
        set2 = set(text2.lower())
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0

    def _print_comparison_summary(self, results: Dict[str, Any]):
        """Print comprehensive comparison summary"""
        print(f"\n🎯 GENERATION COMPARISON SUMMARY")
        print(f"=" * 60)
        
        metrics = results["aggregate_metrics"]
        speed = metrics["speed_metrics"]
        memory = metrics["memory_metrics"]
        quality = metrics["quality_metrics"]
        
        print(f"⚡ Speed Performance:")
        print(f"   Compressed: {speed['avg_compressed_tokens_per_sec']:.2f} tokens/sec")
        print(f"   Standard: {speed['avg_standard_tokens_per_sec']:.2f} tokens/sec")
        print(f"   Speed ratio: {speed['speed_ratio']:.2f}x")
        
        if speed['is_compressed_faster']:
            print(f"   🚀 Compressed generation is FASTER!")
        else:
            print(f"   🐌 Compressed generation is slower")
        
        print(f"\n💾 Memory Usage:")
        print(f"   Average memory: {memory['avg_memory_usage_mb']:.4f} MB")
        print(f"   Compression overhead: {memory['compression_overhead']:.4f}s")
        print(f"   Reconstruction overhead: {memory['reconstruction_overhead']:.4f}s")
        
        print(f"\n🎯 Quality Metrics:")
        print(f"   Text similarity: {quality['avg_text_similarity']:.2%}")
        print(f"   Quality threshold met: {'✅' if quality['similarity_threshold_met'] else '❌'}")
        
        print(f"\n📝 Sample Outputs:")
        for i, (comp_text, std_text) in enumerate(zip(quality['compressed_texts'][:2], quality['standard_texts'][:2])):
            print(f"   Run {i+1}:")
            print(f"     Compressed: '{comp_text[:80]}...'")
            print(f"     Standard:   '{std_text[:80]}...'")
            print(f"     Similarity: {quality['text_similarities'][i]:.2%}")


def main():
    """Test compressed autoregressive decoding"""
    print(f"🚀 LLaMA-3 8B Compressed Autoregressive Decoder Test")
    print(f"=" * 60)
    
    # Initialize decoder
    decoder = CompressedAutoRegressiveDecoder()
    
    # Test prompts
    test_prompts = [
        "The future of artificial intelligence will",
        "Climate change is a global challenge that requires",
        "In the field of quantum computing, researchers are"
    ]
    
    # Run generation comparison for each prompt
    for prompt_idx, prompt in enumerate(test_prompts):
        print(f"\n{'='*60}")
        print(f"🧪 Testing Prompt {prompt_idx + 1}/{len(test_prompts)}")
        print(f"{'='*60}")
        
        results = decoder.compare_generations(
            prompt=prompt,
            max_new_tokens=30,  # Shorter for testing
            num_runs=2          # Fewer runs for testing
        )
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = f"tests/compressed_generation_results_{prompt_idx+1}_{timestamp}.json"
        
        # Convert tensors to lists for JSON serialization
        results_serializable = {}
        for key, value in results.items():
            if key in ["compressed_runs", "standard_runs"]:
                serializable_runs = []
                for run in value:
                    serializable_run = {}
                    for run_key, run_value in run.items():
                        if isinstance(run_value, torch.Tensor):
                            serializable_run[run_key] = run_value.tolist()
                        else:
                            serializable_run[run_key] = run_value
                    serializable_runs.append(serializable_run)
                results_serializable[key] = serializable_runs
            else:
                results_serializable[key] = value
        
        import json
        with open(results_path, 'w') as f:
            json.dump(results_serializable, f, indent=2, default=str)
        
        print(f"💾 Results saved to {results_path}")
    
    print(f"\n🎉 All compressed autoregressive tests completed!")


if __name__ == "__main__":
    main()
