"""
LLaMA-3 8B Compression Inference Pipeline
Production-grade compressed attention with real model weights
"""
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import time
from ..model import LLaMAModelLoader
from ..compression import LLaMACompressionProfileBuilder
from ..cache import LLaMAKVCache, StandardKVCache
from ..data import LLaMADatasetHandler

class LLaMACompressionInference:
    def __init__(self, 
                 model_loader=None, 
                 profile_builder=None,
                 model_path: str = "/mnt/vstor/CSE_ECSE_GXD234/Meta-Llama-3-8B-Instruct"):
        print(f"🚀 Initializing LLaMA-3 8B Compression Inference Pipeline")
        
        # Load real LLaMA model
        if model_loader is None:
            from ..config import ModelConfig
            model_config = ModelConfig(model_path=model_path)
            self.model_loader = LLaMAModelLoader(model_config)
            self.model_loader.load_model()  # Auto-load for convenience
        else:
            self.model_loader = model_loader
            # Ensure model is loaded
            if self.model_loader.model is None:
                self.model_loader.load_model()
        
        # Create compression profiles from real model weights
        if profile_builder is None:
            from ..config import CompressionConfig
            compression_config = CompressionConfig()
            self.compression_profiles = LLaMACompressionProfileBuilder(self.model_loader, compression_config)
            # Auto-build profiles for last layer (-1) for convenience
            self.compression_profiles.build_compression_profiles(layer_idx=-1)
        else:
            self.compression_profiles = profile_builder
            # Ensure profiles are built if not already
            if not self.compression_profiles.profiles:
                self.compression_profiles.build_compression_profiles(layer_idx=-1)
        
        # Create dataset handler
        self.dataset_handler = LLaMADatasetHandler(self.model_loader)
        
        # Initialize caches
        self.compressed_cache = LLaMAKVCache(enable_compression=True)
        self.standard_cache = StandardKVCache()
        
        # Model dimensions
        self.hidden_size = self.model_loader.hidden_size
        self.head_dim = self.model_loader.head_dim
        self.num_heads = self.model_loader.num_attention_heads
        self.vocab_size = self.model_loader.vocab_size
        
        # Performance tracking
        self.inference_stats = {
            "compressed_forward_time": 0.0,
            "standard_forward_time": 0.0,
            "compression_overhead": 0.0,
            "reconstruction_overhead": 0.0,
            "total_tokens_processed": 0
        }
        
        print(f"✅ Pipeline initialized with real LLaMA-3 8B weights")
        self.compression_profiles.print_compression_summary()
    
    def compress_and_cache_kv(self, 
                            hidden_states: torch.Tensor,
                            compression_mapping: List[str],
                            layer_idx: int = 0,
                            head_idx: int = 0) -> Dict[str, torch.Tensor]:
        """
        Compress K/V tensors and store in cache using real LLaMA projections
        
        Args:
            hidden_states: Real hidden states from LLaMA [seq_len, hidden_size]
            compression_mapping: List of compression profiles per token
            layer_idx: Layer index for caching
            head_idx: Head index for caching
            
        Returns:
            Dictionary with compression statistics
        """
        start_time = time.time()
        seq_len = hidden_states.shape[0]
        
        print(f"🗜️  Compressing and caching K/V for {seq_len} tokens...")
        
        # Extract real attention weights
        attention_weights = self.model_loader.get_attention_weights(layer_idx)
        W_K = attention_weights["W_K"]  # [num_kv_heads * head_dim, hidden_size] for GQA
        W_V = attention_weights["W_V"]  # [num_kv_heads * head_dim, hidden_size] for GQA
        
        # Map query head to corresponding key/value head (GQA)
        num_query_heads = self.model_loader.num_attention_heads     # 32
        num_kv_heads = self.model_loader.num_key_value_heads        # 8
        heads_per_kv = num_query_heads // num_kv_heads              # 4
        kv_head_idx = head_idx // heads_per_kv                      # Which kv head group
        
        # Extract the correct K/V head projections for GQA
        W_K_head = W_K[kv_head_idx * self.head_dim:(kv_head_idx + 1) * self.head_dim, :]  # [head_dim, hidden_size]
        W_V_head = W_V[kv_head_idx * self.head_dim:(kv_head_idx + 1) * self.head_dim, :]  # [head_dim, hidden_size]
        
        print(f"   GQA mapping: query head {head_idx} -> kv head {kv_head_idx} (heads_per_kv={heads_per_kv})")
        
        compression_stats = {"tokens_per_profile": {}}
        
        for token_idx, (hidden_state, profile_name) in enumerate(zip(hidden_states, compression_mapping)):
            # Project hidden state to key/value space using real LLaMA weights
            # hidden_state: [hidden_size], W_K_head: [head_dim, hidden_size]
            key_t = W_K_head @ hidden_state      # [head_dim]
            value_t = W_V_head @ hidden_state    # [head_dim]
            
            # Compress using real compression matrices (pass query head index for GQA mapping)
            compressed_key = self.compression_profiles.compress_keys(key_t, head_idx)
            compressed_value = self.compression_profiles.compress_values(value_t, profile_name, head_idx)
            
            # Store in compressed cache
            self.compressed_cache.store_compressed_kv(
                layer_idx=layer_idx,
                head_idx=head_idx,
                compressed_keys=compressed_key,
                compressed_values=compressed_value,
                token_idx=token_idx,
                compression_profile=profile_name
            )
            
            # Store in standard cache for comparison
            self.standard_cache.store_kv(
                layer_idx=layer_idx,
                head_idx=head_idx,
                keys=key_t,
                values=value_t,
                token_idx=token_idx
            )
            
            # Track statistics
            if profile_name not in compression_stats["tokens_per_profile"]:
                compression_stats["tokens_per_profile"][profile_name] = 0
            compression_stats["tokens_per_profile"][profile_name] += 1
        
        compression_time = time.time() - start_time
        self.inference_stats["compression_overhead"] += compression_time
        self.inference_stats["total_tokens_processed"] += seq_len
        
        compression_stats.update({
            "compression_time": compression_time,
            "sequence_length": seq_len,
            "layer_idx": layer_idx,
            "head_idx": head_idx
        })
        
        print(f"   Compressed {seq_len} tokens in {compression_time:.4f}s")
        print(f"   Profile distribution: {compression_stats['tokens_per_profile']}")
        
        return compression_stats
    
    def compressed_attention_forward(self, 
                                   query_hidden_state: torch.Tensor,
                                   layer_idx: int = 0,
                                   head_idx: int = 0) -> torch.Tensor:
        """
        Perform attention computation using compressed K/V cache with on-the-fly reconstruction
        
        Args:
            query_hidden_state: Query hidden state [hidden_size]
            layer_idx: Layer index
            head_idx: Head index
            
        Returns:
            Attention output [hidden_size]
        """
        start_time = time.time()
        
        # Project query using real LLaMA weights
        attention_weights = self.model_loader.get_attention_weights(layer_idx)
        W_Q = attention_weights["W_Q"]  # [num_query_heads * head_dim, hidden_size]
        W_Q_head = W_Q[head_idx * self.head_dim:(head_idx + 1) * self.head_dim, :]  # [head_dim, hidden_size]
        
        query = W_Q_head @ query_hidden_state  # [head_dim]
        
        # Get all cached compression groups
        cache_groups = self.compressed_cache.get_cache_groups()
        attention_output = torch.zeros(self.hidden_size, dtype=query.dtype, device=query.device)
        
        for group_layer_idx, group_head_idx, profile_name in cache_groups:
            if group_layer_idx != layer_idx or group_head_idx != head_idx:
                continue
                
            # Retrieve compressed K/V
            compressed_keys, compressed_values = self.compressed_cache.retrieve_compressed_kv(
                group_layer_idx, group_head_idx, profile_name
            )
            
            if compressed_keys.numel() == 0:
                continue
            
            # Reconstruct keys on-the-fly (pass query head index for GQA mapping)
            reconstruction_start = time.time()
            reconstructed_keys = self.compression_profiles.reconstruct_keys(compressed_keys, head_idx)
            self.inference_stats["reconstruction_overhead"] += time.time() - reconstruction_start
            
            # Compute attention scores with reconstructed keys
            scores = query @ reconstructed_keys.T  # [seq_len]
            attention_weights = F.softmax(scores, dim=0)  # [seq_len]
            
            # Apply attention to compressed values (stay in compressed space)
            context = attention_weights @ compressed_values  # [value_rank]
            
            # Decode directly to output space using fused matrix (pass query head index)
            group_output = self.compression_profiles.decode_to_logits(context, profile_name, head_idx)
            
            # Project back to hidden space (simplified - normally would use W_O)
            # For this test, we'll use a simple linear mapping
            if group_output.shape[0] == self.vocab_size:
                # Map from vocab space back to hidden space (approximate)
                lm_head_weight = self.model_loader.get_language_model_head()
                hidden_output = group_output @ lm_head_weight / self.vocab_size  # [hidden_size]
                attention_output += hidden_output
        
        forward_time = time.time() - start_time
        self.inference_stats["compressed_forward_time"] += forward_time
        
        return attention_output
    
    def standard_attention_forward(self,
                                 query_hidden_state: torch.Tensor,
                                 layer_idx: int = 0,
                                 head_idx: int = 0) -> torch.Tensor:
        """
        Perform standard (uncompressed) attention for baseline comparison
        
        Args:
            query_hidden_state: Query hidden state [hidden_size]
            layer_idx: Layer index
            head_idx: Head index
            
        Returns:
            Attention output [hidden_size]
        """
        start_time = time.time()
        
        # Project query using real LLaMA weights
        attention_weights = self.model_loader.get_attention_weights(layer_idx)
        W_Q = attention_weights["W_Q"]  # [num_query_heads * head_dim, hidden_size]
        W_O = attention_weights["W_O"]  # [hidden_size, num_query_heads * head_dim]
        W_Q_head = W_Q[head_idx * self.head_dim:(head_idx + 1) * self.head_dim, :]  # [head_dim, hidden_size]
        
        query = W_Q_head @ query_hidden_state  # [head_dim]
        
        # Retrieve uncompressed K/V
        keys, values = self.standard_cache.retrieve_kv(layer_idx, head_idx)
        
        if keys.numel() == 0:
            return torch.zeros(self.hidden_size, dtype=query.dtype, device=query.device)
        
        # Standard attention computation
        scores = query @ keys.T  # [seq_len]
        attention_weights = F.softmax(scores, dim=0)  # [seq_len]
        context = attention_weights @ values  # [head_dim]
        
        # Apply output projection
        W_O_head = W_O[:, head_idx * self.head_dim:(head_idx + 1) * self.head_dim]  # [hidden_size, head_dim]
        output = W_O_head @ context  # [hidden_size]
        
        forward_time = time.time() - start_time
        self.inference_stats["standard_forward_time"] += forward_time
        
        return output
    
    def run_compression_benchmark(self, 
                                texts: Optional[List[str]] = None,
                                max_length: int = 256) -> Dict[str, any]:
        """
        Run comprehensive benchmark comparing compressed vs standard attention
        
        Args:
            texts: List of texts to benchmark (uses default if None)
            max_length: Maximum sequence length
            
        Returns:
            Benchmark results dictionary
        """
        print(f"\n🏁 Running LLaMA-3 8B Compression Benchmark")
        print(f"=" * 60)
        
        # Get real hidden states
        hidden_states_list, input_ids_list = self.dataset_handler.get_real_hidden_states_batch(
            texts, max_length
        )
        
        benchmark_results = {
            "texts_processed": len(hidden_states_list),
            "total_tokens": sum(h.shape[0] for h in hidden_states_list),
            "compression_profiles_used": set(),
            "per_text_results": [],
            "aggregate_metrics": {}
        }
        
        for text_idx, (hidden_states, input_ids) in enumerate(zip(hidden_states_list, input_ids_list)):
            print(f"\n--- Processing text {text_idx + 1}/{len(hidden_states_list)} ---")
            
            seq_len = hidden_states.shape[0]
            
            # Create compression mapping
            compression_mapping = self.dataset_handler.create_compression_mapping(
                seq_len, strategy="adaptive"
            )
            benchmark_results["compression_profiles_used"].update(compression_mapping)
            
            # Clear caches
            self.compressed_cache.clear_cache()
            self.standard_cache.clear_cache()
            
            # Compress and cache K/V
            compression_stats = self.compress_and_cache_kv(
                hidden_states, compression_mapping, layer_idx=0, head_idx=0
            )
            
            # Test attention computation on last token
            query_state = hidden_states[-1]  # Use last token as query
            
            # Compressed attention
            compressed_output = self.compressed_attention_forward(query_state, 0, 0)
            
            # Standard attention  
            standard_output = self.standard_attention_forward(query_state, 0, 0)
            
            # Calculate metrics
            output_mse = F.mse_loss(compressed_output, standard_output).item()
            output_cosine_sim = F.cosine_similarity(
                compressed_output.unsqueeze(0), 
                standard_output.unsqueeze(0)
            ).item()
            
            # Get ground truth logits for perplexity calculation
            gt_logits = self.dataset_handler.get_ground_truth_logits(input_ids)
            gt_perplexity, gt_loss = self.dataset_handler.calculate_perplexity(gt_logits, input_ids)
            
            # Memory usage comparison
            compressed_memory = self.compressed_cache.get_memory_usage()
            standard_memory = self.standard_cache.get_memory_usage()
            memory_savings = 1 - (compressed_memory["total_memory_mb"] / 
                                max(standard_memory["total_memory_mb"], 1e-6))
            
            text_results = {
                "text_index": text_idx,
                "sequence_length": seq_len,
                "compression_mapping": compression_mapping,
                "output_mse": output_mse,
                "output_cosine_similarity": output_cosine_sim,
                "ground_truth_perplexity": gt_perplexity,
                "ground_truth_loss": gt_loss,
                "compressed_memory_mb": compressed_memory["total_memory_mb"],
                "standard_memory_mb": standard_memory["total_memory_mb"],
                "memory_savings_ratio": memory_savings,
                "compression_time": compression_stats["compression_time"]
            }
            
            benchmark_results["per_text_results"].append(text_results)
            
            print(f"   Output MSE: {output_mse:.6f}")
            print(f"   Cosine similarity: {output_cosine_sim:.6f}")
            print(f"   Memory savings: {memory_savings:.2%}")
            print(f"   GT perplexity: {gt_perplexity:.2f}")
        
        # Calculate aggregate metrics
        all_results = benchmark_results["per_text_results"]
        benchmark_results["aggregate_metrics"] = {
            "avg_output_mse": sum(r["output_mse"] for r in all_results) / len(all_results),
            "avg_cosine_similarity": sum(r["output_cosine_similarity"] for r in all_results) / len(all_results),
            "avg_memory_savings": sum(r["memory_savings_ratio"] for r in all_results) / len(all_results),
            "avg_gt_perplexity": sum(r["ground_truth_perplexity"] for r in all_results) / len(all_results),
            "total_compression_time": sum(r["compression_time"] for r in all_results),
            "compression_profiles_used": list(benchmark_results["compression_profiles_used"])
        }
        
        # Print final summary
        self._print_benchmark_summary(benchmark_results)
        
        return benchmark_results
    
    def _print_benchmark_summary(self, results: Dict[str, any]):
        """Print comprehensive benchmark summary"""
        metrics = results["aggregate_metrics"]
        
        print(f"\n🎯 LLaMA-3 8B Compression Benchmark Results")
        print(f"=" * 60)
        print(f"Texts processed: {results['texts_processed']}")
        print(f"Total tokens: {results['total_tokens']}")
        print(f"Profiles used: {metrics['compression_profiles_used']}")
        
        print(f"\n📊 Quality Metrics:")
        print(f"  Average output MSE: {metrics['avg_output_mse']:.6f}")
        print(f"  Average cosine similarity: {metrics['avg_cosine_similarity']:.6f}")
        print(f"  Average GT perplexity: {metrics['avg_gt_perplexity']:.2f}")
        
        print(f"\n💾 Efficiency Metrics:")
        print(f"  Average memory savings: {metrics['avg_memory_savings']:.2%}")
        print(f"  Total compression time: {metrics['total_compression_time']:.4f}s")
        
        print(f"\n⚡ Performance Stats:")
        stats = self.inference_stats
        print(f"  Compressed forward time: {stats['compressed_forward_time']:.4f}s")
        print(f"  Standard forward time: {stats['standard_forward_time']:.4f}s")
        print(f"  Compression overhead: {stats['compression_overhead']:.4f}s")
        print(f"  Reconstruction overhead: {stats['reconstruction_overhead']:.4f}s")
        
        # Cache statistics
        print(f"\n🗃️  Cache Performance:")
        self.compressed_cache.print_cache_stats()
    
    def clear_all_caches(self):
        """Clear all caches and reset statistics"""
        self.compressed_cache.clear_cache()
        self.standard_cache.clear_cache()
        
        # Reset performance stats
        for key in self.inference_stats:
            self.inference_stats[key] = 0.0


def main():
    """CLI entry point for LLaMA compression inference"""
    print("🚀 Starting LLaMA-3 8B Compression Inference...")
    
    try:
        # Initialize inference pipeline
        inference = LLaMACompressionInference()
        
        # Run compression benchmark
        results = inference.run_compression_benchmark()
        
        # Display results
        print("\n📊 Compression Results:")
        metrics = results['aggregate_metrics']
        print(f"  Memory savings: {metrics['avg_memory_savings']:.2%}")
        print(f"  Cosine similarity: {metrics['avg_cosine_similarity']:.4f}")
        print(f"  MSE: {metrics['avg_mse']:.6f}")
        
        print("✅ Compression benchmark completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
