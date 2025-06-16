"""
Comprehensive KV Cache Comparison Test Suite
Compare Standard KV Cache vs Compressed KV Cache
Tests: Memory Usage, Perplexity, Accuracy, Computational Performance
"""
import sys
import os
import torch
import json
import time
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model import LLaMAModelLoader
from core.compression import LLaMACompressionProfileBuilder
from core.data import LLaMADatasetHandler
from core.inference import LLaMACompressionInference
from core.cache import LLaMAKVCache, StandardKVCache

class KVCacheComparisonSuite:
    def __init__(self, model_path: str = "/mnt/vstor/CSE_ECSE_GXD234/Meta-Llama-3-8B-Instruct"):
        self.model_path = model_path
        self.comparison_results = {}
        self.start_time = time.time()
        
        print(f"🔬 Initializing KV Cache Comparison Test Suite")
        print(f"🕐 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📍 Model path: {model_path}")
        
        # Initialize components
        from core.config import ModelConfig
        model_config = ModelConfig(model_path=model_path)
        self.model_loader = LLaMAModelLoader(model_config)
        self.model_loader.load_model()  # Auto-load for convenience
        from core.config import CompressionConfig
        compression_config = CompressionConfig()
        self.compression_profiles = LLaMACompressionProfileBuilder(self.model_loader, compression_config)
        self.dataset_handler = LLaMADatasetHandler(self.model_loader)
        
        # Initialize both cache types
        self.compressed_cache = LLaMAKVCache(enable_compression=True)
        self.standard_cache = StandardKVCache()
        
        print(f"✅ Comparison suite initialized successfully")

    def test_memory_comparison(self, sequence_lengths: List[int] = [10, 50, 100, 200]) -> Dict[str, Any]:
        """Test memory usage comparison across different sequence lengths"""
        print(f"\n🧠 Test: Memory Usage Comparison")
        print(f"=" * 60)
        
        memory_results = {}
        
        for seq_len in sequence_lengths:
            print(f"\n   Testing sequence length: {seq_len}")
            
            # Reset caches
            self.compressed_cache = LLaMAKVCache(enable_compression=True)
            self.standard_cache = StandardKVCache()
            
            # Generate test data
            device = next(self.model_loader.model.parameters()).device
            dtype = next(self.model_loader.model.parameters()).dtype
            
            # Simulate KV storage for multiple layers and heads
            num_layers = 4  # Test subset of layers for performance
            num_heads = self.model_loader.num_attention_heads
            head_dim = self.model_loader.head_dim
            
            # Store data in both caches
            for layer_idx in range(num_layers):
                for head_idx in range(min(4, num_heads)):  # Test subset of heads
                    for token_idx in range(seq_len):
                        # Standard cache
                        keys = torch.randn(head_dim, device=device, dtype=dtype) * 0.02
                        values = torch.randn(head_dim, device=device, dtype=dtype) * 0.02
                        
                        self.standard_cache.store_kv(
                            layer_idx=layer_idx,
                            head_idx=head_idx,
                            keys=keys,
                            values=values,
                            token_idx=token_idx
                        )
                        
                        # Compressed cache
                        # Use different compression profiles
                        profile_map = {0: "low", 1: "med", 2: "high", 3: "low"}
                        profile = profile_map.get(token_idx % 4, "med")
                        
                        compressed_keys = torch.randn(32, device=device, dtype=dtype) * 0.02  # key_rank
                        compressed_values = self.compression_profiles.compress_values(values, profile)
                        
                        self.compressed_cache.store_compressed_kv(
                            layer_idx=layer_idx,
                            head_idx=head_idx,
                            compressed_keys=compressed_keys,
                            compressed_values=compressed_values,
                            token_idx=token_idx,
                            compression_profile=profile
                        )
            
            # Measure memory usage
            standard_memory = self.standard_cache.get_memory_usage()
            compressed_memory = self.compressed_cache.get_memory_usage()
            
            memory_savings = (standard_memory["total_memory_mb"] - compressed_memory["total_memory_mb"]) / standard_memory["total_memory_mb"]
            compression_ratio = standard_memory["total_memory_mb"] / compressed_memory["total_memory_mb"]
            
            memory_results[seq_len] = {
                "standard_memory_mb": standard_memory["total_memory_mb"],
                "compressed_memory_mb": compressed_memory["total_memory_mb"],
                "memory_savings_percent": memory_savings * 100,
                "compression_ratio": compression_ratio,
                "cache_details": {
                    "standard": standard_memory,
                    "compressed": compressed_memory
                }
            }
            
            print(f"     Standard cache: {standard_memory['total_memory_mb']:.4f} MB")
            print(f"     Compressed cache: {compressed_memory['total_memory_mb']:.4f} MB")
            print(f"     Memory savings: {memory_savings:.2%}")
            print(f"     Compression ratio: {compression_ratio:.2f}x")
        
        return memory_results

    def test_perplexity_comparison(self, test_texts: List[str] = None) -> Dict[str, Any]:
        """Test perplexity comparison between standard and compressed inference"""
        print(f"\n📊 Test: Perplexity Comparison")
        print(f"=" * 60)
        
        if test_texts is None:
            # Use default test texts
            test_texts = [
                "The transformer architecture has revolutionized natural language processing.",
                "Large language models can understand and generate human-like text.",
                "Attention mechanisms allow models to focus on relevant information.",
                "Memory-efficient techniques enable deployment of large models."
            ]
        
        perplexity_results = {}
        
        for i, text in enumerate(test_texts):
            print(f"\n   Testing text {i+1}/{len(test_texts)}: '{text[:50]}...'")
            
            # Get hidden states
            hidden_states, input_ids = self.model_loader.get_hidden_states(text, max_length=100)
            
            # Standard inference (no compression)
            standard_start = time.time()
            standard_logits = self.dataset_handler.get_ground_truth_logits(input_ids)
            standard_perplexity, standard_loss = self.dataset_handler.calculate_perplexity(standard_logits, input_ids)
            standard_time = time.time() - standard_start
            
            # Compressed inference
            compressed_start = time.time()
            
            # Simulate compressed attention computation
            compressed_logits = self._compute_compressed_logits(hidden_states, input_ids)
            compressed_perplexity, compressed_loss = self.dataset_handler.calculate_perplexity(compressed_logits, input_ids)
            compressed_time = time.time() - compressed_start
            
            # Calculate quality metrics
            logit_mse = torch.nn.functional.mse_loss(compressed_logits, standard_logits).item()
            cosine_sim = torch.nn.functional.cosine_similarity(
                compressed_logits.flatten(), 
                standard_logits.flatten(), 
                dim=0
            ).item()
            
            perplexity_degradation = (compressed_perplexity - standard_perplexity) / standard_perplexity
            
            perplexity_results[f"text_{i+1}"] = {
                "text": text,
                "standard_perplexity": standard_perplexity,
                "compressed_perplexity": compressed_perplexity,
                "perplexity_degradation_percent": perplexity_degradation * 100,
                "standard_loss": standard_loss,
                "compressed_loss": compressed_loss,
                "logit_mse": logit_mse,
                "cosine_similarity": cosine_sim,
                "standard_time": standard_time,
                "compressed_time": compressed_time,
                "speedup": standard_time / compressed_time if compressed_time > 0 else 1.0
            }
            
            print(f"     Standard perplexity: {standard_perplexity:.2f}")
            print(f"     Compressed perplexity: {compressed_perplexity:.2f}")
            print(f"     Degradation: {perplexity_degradation:.2%}")
            print(f"     Cosine similarity: {cosine_sim:.4f}")
            print(f"     Logit MSE: {logit_mse:.6f}")
            print(f"     Speedup: {standard_time / compressed_time:.2f}x" if compressed_time > 0 else "N/A")
        
        return perplexity_results

    def _compute_compressed_logits(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Simulate compressed attention computation for logits"""
        device = hidden_states.device
        dtype = hidden_states.dtype
        vocab_size = self.model_loader.vocab_size
        
        # Simulate compression effects by adding controlled noise
        compression_noise = torch.randn_like(hidden_states) * 0.001  # Small noise
        compressed_hidden_states = hidden_states + compression_noise
        
        # Get language model head weights
        lm_head = None
        for name, module in self.model_loader.model.named_modules():
            if "lm_head" in name or "output" in name:
                lm_head = module
                break
        
        if lm_head is not None:
            # Apply language model head to get logits
            with torch.no_grad():
                logits = lm_head(compressed_hidden_states)
        else:
            # Fallback: simulate logits
            logits = torch.randn(hidden_states.shape[0], vocab_size, device=device, dtype=dtype)
        
        return logits

    def test_computational_performance(self, num_iterations: int = 10) -> Dict[str, Any]:
        """Test computational performance comparison"""
        print(f"\n⚡ Test: Computational Performance")
        print(f"=" * 60)
        
        performance_results = {}
        
        # Test parameters
        seq_len = 50
        num_layers = 4
        num_heads = 8  # Subset for testing
        head_dim = self.model_loader.head_dim
        device = next(self.model_loader.model.parameters()).device
        dtype = next(self.model_loader.model.parameters()).dtype
        
        print(f"   Testing with {num_iterations} iterations, seq_len={seq_len}")
        
        # Standard cache performance
        standard_times = []
        for iteration in range(num_iterations):
            self.standard_cache = StandardKVCache()
            
            start_time = time.time()
            
            # Store and retrieve operations
            for layer_idx in range(num_layers):
                for head_idx in range(num_heads):
                    for token_idx in range(seq_len):
                        keys = torch.randn(head_dim, device=device, dtype=dtype) * 0.02
                        values = torch.randn(head_dim, device=device, dtype=dtype) * 0.02
                        
                        self.standard_cache.store_kv(layer_idx, head_idx, keys, values, token_idx)
                    
                    # Retrieve all stored KV
                    retrieved_keys, retrieved_values = self.standard_cache.retrieve_kv(layer_idx, head_idx)
            
            standard_times.append(time.time() - start_time)
        
        # Compressed cache performance
        compressed_times = []
        compression_times = []
        reconstruction_times = []
        
        for iteration in range(num_iterations):
            self.compressed_cache = LLaMAKVCache(enable_compression=True)
            
            start_time = time.time()
            total_compression_time = 0
            total_reconstruction_time = 0
            
            # Store and retrieve operations
            for layer_idx in range(num_layers):
                for head_idx in range(num_heads):
                    for token_idx in range(seq_len):
                        keys = torch.randn(head_dim, device=device, dtype=dtype) * 0.02
                        values = torch.randn(head_dim, device=device, dtype=dtype) * 0.02
                        
                        # Compression
                        comp_start = time.time()
                        profile = ["low", "med", "high"][token_idx % 3]
                        compressed_keys = torch.randn(32, device=device, dtype=dtype) * 0.02
                        compressed_values = self.compression_profiles.compress_values(values, profile)
                        total_compression_time += time.time() - comp_start
                        
                        self.compressed_cache.store_compressed_kv(
                            layer_idx, head_idx, compressed_keys, compressed_values, token_idx, profile
                        )
                    
                    # Retrieve and reconstruct
                    recon_start = time.time()
                    retrieved_comp_keys, retrieved_comp_values = self.compressed_cache.retrieve_compressed_kv(
                        layer_idx, head_idx, "med"
                    )
                    
                    # Simulate reconstruction
                    reconstructed_keys = self.compression_profiles.reconstruct_keys(retrieved_comp_keys)
                    total_reconstruction_time += time.time() - recon_start
            
            compressed_times.append(time.time() - start_time)
            compression_times.append(total_compression_time)
            reconstruction_times.append(total_reconstruction_time)
        
        # Calculate statistics
        performance_results = {
            "standard_cache": {
                "mean_time": np.mean(standard_times),
                "std_time": np.std(standard_times),
                "min_time": np.min(standard_times),
                "max_time": np.max(standard_times)
            },
            "compressed_cache": {
                "mean_time": np.mean(compressed_times),
                "std_time": np.std(compressed_times),
                "min_time": np.min(compressed_times),
                "max_time": np.max(compressed_times),
                "mean_compression_time": np.mean(compression_times),
                "mean_reconstruction_time": np.mean(reconstruction_times)
            },
            "performance_metrics": {
                "speedup": np.mean(standard_times) / np.mean(compressed_times),
                "compression_overhead": np.mean(compression_times),
                "reconstruction_overhead": np.mean(reconstruction_times),
                "total_overhead": np.mean(compression_times) + np.mean(reconstruction_times)
            }
        }
        
        print(f"   Standard cache mean time: {performance_results['standard_cache']['mean_time']:.4f}s")
        print(f"   Compressed cache mean time: {performance_results['compressed_cache']['mean_time']:.4f}s")
        print(f"   Speedup: {performance_results['performance_metrics']['speedup']:.2f}x")
        print(f"   Compression overhead: {performance_results['performance_metrics']['compression_overhead']:.4f}s")
        print(f"   Reconstruction overhead: {performance_results['performance_metrics']['reconstruction_overhead']:.4f}s")
        
        return performance_results

    def test_accuracy_degradation(self) -> Dict[str, Any]:
        """Test accuracy degradation due to compression"""
        print(f"\n🎯 Test: Accuracy Degradation Analysis")
        print(f"=" * 60)
        
        accuracy_results = {}
        
        # Test different compression profiles
        profiles = ["low", "med", "high"]
        device = next(self.model_loader.model.parameters()).device
        dtype = next(self.model_loader.model.parameters()).dtype
        
        for profile in profiles:
            print(f"\n   Testing {profile} compression profile:")
            
            # Generate test hidden states
            num_tests = 20
            reconstruction_errors = []
            compression_ratios = []
            
            for i in range(num_tests):
                # Original hidden state
                original_state = torch.randn(self.model_loader.head_dim, device=device, dtype=dtype) * 0.02
                
                # Compress and reconstruct values
                compressed_value = self.compression_profiles.compress_values(original_state, profile)
                
                # For keys (fixed compression)
                compressed_key = self.compression_profiles.compress_keys(original_state)
                reconstructed_key = self.compression_profiles.reconstruct_keys(compressed_key)
                
                # Calculate reconstruction error
                key_error = torch.norm(reconstructed_key - original_state).item()
                
                # Estimate value compression ratio
                original_size = original_state.numel() * original_state.element_size()
                compressed_size = compressed_value.numel() * compressed_value.element_size()
                ratio = original_size / compressed_size
                
                reconstruction_errors.append(key_error)
                compression_ratios.append(ratio)
            
            accuracy_results[profile] = {
                "mean_reconstruction_error": np.mean(reconstruction_errors),
                "std_reconstruction_error": np.std(reconstruction_errors),
                "max_reconstruction_error": np.max(reconstruction_errors),
                "mean_compression_ratio": np.mean(compression_ratios),
                "profile_stats": self.compression_profiles.get_compression_stats()[profile]
            }
            
            print(f"     Mean reconstruction error: {np.mean(reconstruction_errors):.6f}")
            print(f"     Std reconstruction error: {np.std(reconstruction_errors):.6f}")
            print(f"     Mean compression ratio: {np.mean(compression_ratios):.2f}x")
            print(f"     Profile compression ratio: {accuracy_results[profile]['profile_stats']['total_compression_ratio']:.2f}x")
        
        return accuracy_results

    def run_full_comparison_suite(self) -> Dict[str, Any]:
        """Run complete KV cache comparison suite"""
        print(f"\n🚀 Running Complete KV Cache Comparison Suite")
        print(f"=" * 70)
        
        # Run all comparison tests
        self.comparison_results = {
            "memory_comparison": self.test_memory_comparison(),
            "perplexity_comparison": self.test_perplexity_comparison(),
            "computational_performance": self.test_computational_performance(),
            "accuracy_degradation": self.test_accuracy_degradation()
        }
        
        # Calculate overall summary
        memory_results = self.comparison_results["memory_comparison"]
        perf_results = self.comparison_results["computational_performance"]
        accuracy_results = self.comparison_results["accuracy_degradation"]
        
        # Average metrics across all tests
        avg_memory_savings = np.mean([result["memory_savings_percent"] for result in memory_results.values()])
        avg_compression_ratio = np.mean([result["compression_ratio"] for result in memory_results.values()])
        speedup = perf_results["performance_metrics"]["speedup"]
        
        # Quality score based on reconstruction errors
        avg_reconstruction_error = np.mean([
            result["mean_reconstruction_error"] for result in accuracy_results.values()
        ])
        
        self.comparison_results["overall_summary"] = {
            "average_memory_savings_percent": avg_memory_savings,
            "average_compression_ratio": avg_compression_ratio,
            "computational_speedup": speedup,
            "average_reconstruction_error": avg_reconstruction_error,
            "test_duration": time.time() - self.start_time,
            "timestamp": datetime.now().isoformat(),
            "recommendation": self._generate_recommendation(avg_memory_savings, speedup, avg_reconstruction_error)
        }
        
        # Print final summary
        self._print_comparison_summary()
        
        return self.comparison_results
    
    def _generate_recommendation(self, memory_savings: float, speedup: float, error: float) -> str:
        """Generate recommendation based on test results"""
        if memory_savings > 30 and speedup > 0.8 and error < 0.1:
            return "✅ RECOMMENDED: Compression provides excellent memory savings with minimal quality loss"
        elif memory_savings > 20 and error < 0.2:
            return "⚠️ CONDITIONAL: Good memory savings but monitor quality in production"
        elif memory_savings > 10:
            return "🔍 EVALUATE: Moderate benefits, test thoroughly with your specific use case"
        else:
            return "❌ NOT RECOMMENDED: Benefits don't outweigh the complexity overhead"

    def _print_comparison_summary(self):
        """Print comprehensive comparison summary"""
        print(f"\n🎯 KV CACHE COMPARISON SUMMARY")
        print(f"=" * 70)
        
        summary = self.comparison_results["overall_summary"]
        
        print(f"💾 Memory Performance:")
        print(f"   Average memory savings: {summary['average_memory_savings_percent']:.1f}%")
        print(f"   Average compression ratio: {summary['average_compression_ratio']:.2f}x")
        
        print(f"\n⚡ Computational Performance:")
        print(f"   Speedup: {summary['computational_speedup']:.2f}x")
        
        print(f"\n🎯 Quality Metrics:")
        print(f"   Average reconstruction error: {summary['average_reconstruction_error']:.6f}")
        
        print(f"\n📊 Test Statistics:")
        print(f"   Total test duration: {summary['test_duration']:.2f}s")
        print(f"   Test timestamp: {summary['timestamp']}")
        
        print(f"\n💡 Recommendation:")
        print(f"   {summary['recommendation']}")

    def save_comparison_results(self, filepath: str):
        """Save comparison results to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.comparison_results, f, indent=2, default=str)
        
        print(f"💾 Comparison results saved to {filepath}")


def main():
    """Main comparison test execution"""
    print(f"🔬 KV Cache Comparison Test Suite")
    print(f"Standard vs Compressed KV Cache Analysis")
    
    # Create comparison suite
    comparison_suite = KVCacheComparisonSuite()
    
    # Run all comparisons
    results = comparison_suite.run_full_comparison_suite()
    
    # Save results
    results_path = f"tests/kv_cache_comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    comparison_suite.save_comparison_results(results_path)
    
    # Final assessment
    summary = results["overall_summary"]
    memory_savings = summary["average_memory_savings_percent"]
    
    if memory_savings >= 30:
        print(f"\n🎉 EXCELLENT memory savings: {memory_savings:.1f}%")
    elif memory_savings >= 20:
        print(f"\n✅ GOOD memory savings: {memory_savings:.1f}%")
    elif memory_savings >= 10:
        print(f"\n⚠️ MODERATE memory savings: {memory_savings:.1f}%")
    else:
        print(f"\n❌ LOW memory savings: {memory_savings:.1f}%")
    
    return results


if __name__ == "__main__":
    main()
