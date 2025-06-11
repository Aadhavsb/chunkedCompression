"""
Comprehensive LLaMA-3 8B Compression Test Suite
Production-grade testing with real model weights and data
"""
import sys
import os
import torch
import json
from typing import Dict, List, Any
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_model_loader import LLaMAModelLoader
from profiles_llama import LLaMACompressionProfiles  
from dataset_llama import LLaMADatasetHandler
from llama_inference import LLaMACompressionInference
from kv_cache_llama import LLaMAKVCache, StandardKVCache

class LLaMACompressionTestSuite:
    def __init__(self, model_path: str = "/mnt/vstor/CSE_ECSE_GXD234/Meta-Llama-3-8B-Instruct"):
        self.model_path = model_path
        self.test_results = {}
        self.start_time = time.time()
        
        print(f"🧪 Initializing LLaMA-3 8B Compression Test Suite")
        print(f"🕐 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📍 Model path: {model_path}")
        
        # Initialize components
        try:
            self.inference_pipeline = LLaMACompressionInference(model_path)
            print(f"✅ Test suite initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize test suite: {e}")
            raise
    
    def test_model_loading(self) -> Dict[str, Any]:
        """Test 1: Verify real LLaMA model loading"""
        print(f"\n🔄 Test 1: Model Loading Verification")
        print(f"=" * 50)
        
        model_loader = self.inference_pipeline.model_loader
        
        # Test basic model properties
        model_info = model_loader.get_model_info()
        
        tests = {
            "model_loaded": model_loader.model is not None,
            "tokenizer_loaded": model_loader.tokenizer is not None,
            "correct_hidden_size": model_info["hidden_size"] == 4096,
            "correct_vocab_size": model_info["vocab_size"] > 100000,
            "correct_num_heads": model_info["num_attention_heads"] == 32,
            "model_in_eval_mode": not model_loader.model.training
        }
        
        # Test model inference
        try:
            test_text = "The capital of France is"
            hidden_states, input_ids = model_loader.get_hidden_states(test_text, max_length=20)
            
            tests.update({
                "hidden_states_shape_correct": hidden_states.shape[1] == 4096,
                "tokenization_working": len(input_ids) > 0,
                "hidden_states_non_zero": hidden_states.abs().sum() > 0
            })
            
            print(f"   Sample text: '{test_text}'")
            print(f"   Hidden states: {hidden_states.shape}")
            print(f"   Input tokens: {len(input_ids)}")
            
        except Exception as e:
            print(f"   ❌ Model inference failed: {e}")
            tests["model_inference_working"] = False
        
        # Summary
        passed = sum(tests.values())
        total = len(tests)
        
        print(f"\n   Results: {passed}/{total} tests passed")
        for test_name, passed in tests.items():
            status = "✅" if passed else "❌"
            print(f"     {status} {test_name}")
        
        self.test_results["test_model_loading"] = {
            "passed": passed,
            "total": total,
            "success_rate": passed / total,
            "details": tests,
            "model_info": model_info
        }
        
        return self.test_results["test_model_loading"]
    
    def test_compression_profiles(self) -> Dict[str, Any]:
        """Test 2: Verify real compression profiles from LLaMA weights"""
        print(f"\n🔄 Test 2: Compression Profiles Verification")
        print(f"=" * 50)
        
        profiles = self.inference_pipeline.compression_profiles
        
        tests = {}
        
        # Test profile existence and structure
        expected_profiles = ["low", "med", "high"]
        for profile_name in expected_profiles:
            profile_exists = profile_name in profiles.profiles
            tests[f"{profile_name}_profile_exists"] = profile_exists
            
            if profile_exists:
                profile = profiles.profiles[profile_name]
                required_keys = ["A_V", "W_fused", "A_K", "B_K", "value_rank", "key_rank"]
                
                for key in required_keys:
                    tests[f"{profile_name}_{key}_exists"] = key in profile
                
                # Test matrix shapes
                if all(key in profile for key in required_keys):
                    value_rank = profile["value_rank"]
                    key_rank = profile["key_rank"]
                    
                    tests[f"{profile_name}_A_V_shape"] = profile["A_V"].shape == (value_rank, profiles.head_dim)
                    tests[f"{profile_name}_W_fused_shape"] = profile["W_fused"].shape == (profiles.vocab_size, value_rank)
                    tests[f"{profile_name}_A_K_shape"] = profile["A_K"].shape == (key_rank, profiles.head_dim)
                    tests[f"{profile_name}_B_K_shape"] = profile["B_K"].shape == (profiles.head_dim, key_rank)
        
        # Test compression/decompression functionality
        try:
            test_hidden_state = torch.randn(profiles.head_dim) * 0.02
            
            for profile_name in expected_profiles:
                # Test value compression
                compressed_value = profiles.compress_values(test_hidden_state, profile_name)
                tests[f"{profile_name}_value_compression_works"] = compressed_value.numel() > 0
                
                # Test key compression/reconstruction
                compressed_key = profiles.compress_keys(test_hidden_state)
                reconstructed_key = profiles.reconstruct_keys(compressed_key)
                
                reconstruction_error = torch.norm(reconstructed_key - test_hidden_state).item()
                tests[f"{profile_name}_key_reconstruction_error_reasonable"] = reconstruction_error < 1.0
                
                print(f"   {profile_name}: value {test_hidden_state.shape} -> {compressed_value.shape}")
                print(f"   {profile_name}: key reconstruction error = {reconstruction_error:.6f}")
        
        except Exception as e:
            print(f"   ❌ Compression functionality test failed: {e}")
            tests["compression_functionality_working"] = False
        
        # Test compression statistics
        try:
            compression_stats = profiles.get_compression_stats()
            tests["compression_stats_available"] = len(compression_stats) == 3
            
            for profile_name, stats in compression_stats.items():
                compression_ratio = stats["total_compression_ratio"]
                tests[f"{profile_name}_compression_ratio_reasonable"] = 1.5 <= compression_ratio <= 50.0
                print(f"   {profile_name}: total compression ratio = {compression_ratio:.2f}x")
        
        except Exception as e:
            print(f"   ❌ Compression statistics test failed: {e}")
        
        # Summary
        passed = sum(tests.values())
        total = len(tests)
        
        print(f"\n   Results: {passed}/{total} tests passed")
        for test_name, passed in tests.items():
            status = "✅" if passed else "❌"
            print(f"     {status} {test_name}")
        
        self.test_results["test_compression_profiles"] = {
            "passed": passed,
            "total": total,
            "success_rate": passed / total,
            "details": tests
        }
        
        return self.test_results["test_compression_profiles"]
    
    def test_real_hidden_states(self) -> Dict[str, Any]:
        """Test 3: Verify real hidden state processing"""
        print(f"\n🔄 Test 3: Real Hidden States Processing")
        print(f"=" * 50)
        
        dataset_handler = self.inference_pipeline.dataset_handler
        
        tests = {}
        
        try:
            # Test single text processing
            test_text = "Large language models have revolutionized natural language processing."
            hidden_states, input_ids = dataset_handler.model_loader.get_hidden_states(test_text)
            
            tests.update({
                "hidden_states_correct_shape": len(hidden_states.shape) == 2,
                "hidden_states_correct_dim": hidden_states.shape[1] == 4096,
                "input_ids_correct_length": len(input_ids) == hidden_states.shape[0],
                "hidden_states_non_zero": hidden_states.abs().sum() > 0,
                "hidden_states_reasonable_magnitude": 0.001 < hidden_states.abs().mean() < 1.0
            })
            
            # Test statistical analysis
            stats = dataset_handler.analyze_hidden_states(hidden_states)
            tests.update({
                "stats_has_mean": "mean" in stats,
                "stats_has_std": "std" in stats,
                "stats_has_l2_norm": "l2_norm" in stats,
                "stats_reasonable_values": -1.0 < stats["mean"] < 1.0 and stats["std"] > 0
            })
            
            print(f"   Text: '{test_text}'")
            print(f"   Hidden states: {hidden_states.shape}")
            print(f"   Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
            print(f"   L2 norm: {stats['l2_norm']:.2f}")
            
            # Test batch processing
            batch_hidden_states, batch_input_ids = dataset_handler.get_real_hidden_states_batch(
                texts=None, max_length=100  # Use default WikiText samples
            )
            
            tests.update({
                "batch_processing_works": len(batch_hidden_states) > 0,
                "batch_all_correct_dim": all(h.shape[1] == 4096 for h in batch_hidden_states),
                "batch_all_non_empty": all(h.shape[0] > 0 for h in batch_hidden_states)
            })
            
            print(f"   Batch processing: {len(batch_hidden_states)} texts")
            for i, h in enumerate(batch_hidden_states):
                print(f"     Text {i+1}: {h.shape}")
            
            # Test ground truth logits
            gt_logits = dataset_handler.get_ground_truth_logits(input_ids)
            perplexity, loss = dataset_handler.calculate_perplexity(gt_logits, input_ids)
            
            tests.update({
                "ground_truth_logits_correct_shape": gt_logits.shape == (len(input_ids), dataset_handler.model_loader.vocab_size),
                "perplexity_reasonable": 1.0 < perplexity < 1000.0,
                "loss_reasonable": 0.0 < loss < 10.0
            })
            
            print(f"   Ground truth perplexity: {perplexity:.2f}")
            print(f"   Ground truth loss: {loss:.4f}")
        
        except Exception as e:
            print(f"   ❌ Hidden states processing failed: {e}")
            tests["hidden_states_processing_working"] = False
        
        # Summary
        passed = sum(tests.values())
        total = len(tests)
        
        print(f"\n   Results: {passed}/{total} tests passed")
        
        self.test_results["test_real_hidden_states"] = {
            "passed": passed,
            "total": total,
            "success_rate": passed / total,
            "details": tests
        }
        
        return self.test_results["test_real_hidden_states"]
    
    def test_kv_cache_operations(self) -> Dict[str, Any]:
        """Test 4: Verify KV cache operations"""
        print(f"\n🔄 Test 4: KV Cache Operations")
        print(f"=" * 50)
        
        tests = {}
        
        try:
            # Test compressed cache
            compressed_cache = LLaMAKVCache(enable_compression=True)
            
            # Create test data
            test_compressed_keys = torch.randn(5, 128) * 0.1    # [seq_len, key_rank]
            test_compressed_values = torch.randn(5, 64) * 0.1   # [seq_len, value_rank]
            
            # Test storing and retrieving
            for i in range(5):
                compressed_cache.store_compressed_kv(
                    layer_idx=0, head_idx=0,
                    compressed_keys=test_compressed_keys[i],
                    compressed_values=test_compressed_values[i],
                    token_idx=i,
                    compression_profile="med"
                )
            
            retrieved_keys, retrieved_values = compressed_cache.retrieve_compressed_kv(0, 0, "med")
            
            tests.update({
                "compressed_cache_store_retrieve": torch.allclose(retrieved_keys, test_compressed_keys, atol=1e-6),
                "compressed_cache_correct_shapes": (retrieved_keys.shape == test_compressed_keys.shape and 
                                                  retrieved_values.shape == test_compressed_values.shape),
                "compressed_cache_sequence_length": compressed_cache.get_sequence_length(0, 0, "med") == 5
            })
            
            # Test standard cache
            standard_cache = StandardKVCache()
            
            test_keys = torch.randn(5, 128) * 0.1    # [seq_len, head_dim]
            test_values = torch.randn(5, 128) * 0.1  # [seq_len, head_dim]
            
            for i in range(5):
                standard_cache.store_kv(
                    layer_idx=0, head_idx=0,
                    keys=test_keys[i],
                    values=test_values[i],
                    token_idx=i
                )
            
            retrieved_std_keys, retrieved_std_values = standard_cache.retrieve_kv(0, 0)
            
            tests.update({
                "standard_cache_store_retrieve": torch.allclose(retrieved_std_keys, test_keys, atol=1e-6),
                "standard_cache_correct_shapes": (retrieved_std_keys.shape == test_keys.shape and
                                                retrieved_std_values.shape == test_values.shape)
            })
            
            # Test memory usage calculations
            compressed_memory = compressed_cache.get_memory_usage()
            standard_memory = standard_cache.get_memory_usage()
            
            tests.update({
                "compressed_memory_calculation": compressed_memory["total_memory_mb"] > 0,
                "standard_memory_calculation": standard_memory["total_memory_mb"] > 0,
                "memory_savings": compressed_memory["total_memory_mb"] < standard_memory["total_memory_mb"]
            })
            
            print(f"   Compressed cache memory: {compressed_memory['total_memory_mb']:.4f} MB")
            print(f"   Standard cache memory: {standard_memory['total_memory_mb']:.4f} MB")
            print(f"   Memory savings: {(1 - compressed_memory['total_memory_mb']/standard_memory['total_memory_mb']):.2%}")
        
        except Exception as e:
            print(f"   ❌ KV cache operations failed: {e}")
            tests["kv_cache_operations_working"] = False
        
        # Summary
        passed = sum(tests.values())
        total = len(tests)
        
        print(f"\n   Results: {passed}/{total} tests passed")
        
        self.test_results["test_kv_cache_operations"] = {
            "passed": passed,
            "total": total,
            "success_rate": passed / total,
            "details": tests
        }
        
        return self.test_results["test_kv_cache_operations"]
    
    def test_end_to_end_inference(self) -> Dict[str, Any]:
        """Test 5: End-to-end compression inference pipeline"""
        print(f"\n🔄 Test 5: End-to-End Inference Pipeline")
        print(f"=" * 50)
        
        tests = {}
        
        try:
            # Run compressed benchmark on small dataset
            test_texts = [
                "The transformer architecture has revolutionized AI.",
                "Attention mechanisms allow models to focus on relevant information."
            ]
            
            benchmark_results = self.inference_pipeline.run_compression_benchmark(
                texts=test_texts, max_length=50
            )
            
            # Verify benchmark results structure
            required_keys = ["texts_processed", "total_tokens", "per_text_results", "aggregate_metrics"]
            for key in required_keys:
                tests[f"benchmark_has_{key}"] = key in benchmark_results
            
            # Verify metrics quality
            if "aggregate_metrics" in benchmark_results:
                metrics = benchmark_results["aggregate_metrics"]
                
                tests.update({
                    "cosine_similarity_reasonable": metrics.get("avg_cosine_similarity", 0) > 0.8,
                    "output_mse_reasonable": metrics.get("avg_output_mse", float('inf')) < 1.0,
                    "memory_savings_positive": metrics.get("avg_memory_savings", 0) > 0,
                    "perplexity_reasonable": 1.0 < metrics.get("avg_gt_perplexity", 0) < 100.0
                })
                
                print(f"   Texts processed: {benchmark_results['texts_processed']}")
                print(f"   Total tokens: {benchmark_results['total_tokens']}")
                print(f"   Avg cosine similarity: {metrics.get('avg_cosine_similarity', 0):.4f}")
                print(f"   Avg output MSE: {metrics.get('avg_output_mse', 0):.6f}")
                print(f"   Avg memory savings: {metrics.get('avg_memory_savings', 0):.2%}")
                print(f"   Avg perplexity: {metrics.get('avg_gt_perplexity', 0):.2f}")
            
            # Test individual components
            if len(benchmark_results.get("per_text_results", [])) > 0:
                first_result = benchmark_results["per_text_results"][0]
                
                tests.update({
                    "compression_mapping_exists": "compression_mapping" in first_result,
                    "memory_comparison_available": "compressed_memory_mb" in first_result and "standard_memory_mb" in first_result,
                    "timing_data_available": "compression_time" in first_result
                })
        
        except Exception as e:
            print(f"   ❌ End-to-end inference failed: {e}")
            tests["end_to_end_inference_working"] = False
        
        # Summary
        passed = sum(tests.values())
        total = len(tests)
        
        print(f"\n   Results: {passed}/{total} tests passed")
        
        self.test_results["test_end_to_end_inference"] = {
            "passed": passed,
            "total": total,
            "success_rate": passed / total,
            "details": tests
        }
        
        return self.test_results["test_end_to_end_inference"]
    
    def run_full_test_suite(self) -> Dict[str, Any]:
        """Run complete test suite"""
        print(f"\n🚀 Running Complete LLaMA-3 8B Compression Test Suite")
        print(f"=" * 70)
        
        test_methods = [
            self.test_model_loading,
            self.test_compression_profiles,
            self.test_real_hidden_states,
            self.test_kv_cache_operations,
            self.test_end_to_end_inference
        ]
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                test_name = test_method.__name__
                print(f"\n❌ {test_name} failed with exception: {e}")
                self.test_results[test_name] = {
                    "passed": 0,
                    "total": 1,
                    "success_rate": 0.0,
                    "error": str(e)
                }
        
        # Calculate overall results
        total_passed = sum(result.get("passed", 0) for result in self.test_results.values())
        total_tests = sum(result.get("total", 0) for result in self.test_results.values())
        overall_success_rate = total_passed / max(total_tests, 1)
        
        self.test_results["overall_summary"] = {
            "total_passed": total_passed,
            "total_tests": total_tests,
            "success_rate": overall_success_rate,
            "test_duration": time.time() - self.start_time,
            "timestamp": datetime.now().isoformat()
        }
        
        # Print final summary
        print(f"\n🎯 FINAL TEST SUITE RESULTS")
        print(f"=" * 70)
        print(f"Total tests passed: {total_passed}/{total_tests}")
        print(f"Overall success rate: {overall_success_rate:.2%}")
        print(f"Test duration: {time.time() - self.start_time:.2f} seconds")
        
        # Per-test summary
        for test_name, result in self.test_results.items():
            if test_name == "overall_summary":
                continue
            
            success_rate = result.get("success_rate", 0)
            status = "✅" if success_rate >= 0.8 else "⚠️" if success_rate >= 0.5 else "❌"
            print(f"{status} {test_name}: {result.get('passed', 0)}/{result.get('total', 0)} ({success_rate:.1%})")
        
        return self.test_results
    
    def save_test_results(self, filepath: str):
        """Save test results to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        print(f"💾 Test results saved to {filepath}")


def main():
    """Main test execution"""
    print(f"🦙 LLaMA-3 8B Compression Test Suite")
    print(f"Testing with REAL model weights - NO PLACEHOLDERS")
    
    # Create test suite
    test_suite = LLaMACompressionTestSuite()
    
    # Run all tests
    results = test_suite.run_full_test_suite()
    
    # Save results
    results_path = f"tests/llama_compression_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    test_suite.save_test_results(results_path)
    
    # Final status
    overall_success = results["overall_summary"]["success_rate"]
    if overall_success >= 0.8:
        print(f"\n🎉 Test suite PASSED with {overall_success:.1%} success rate")
    else:
        print(f"\n⚠️ Test suite FAILED with {overall_success:.1%} success rate")
    
    return results


if __name__ == "__main__":
    main()
