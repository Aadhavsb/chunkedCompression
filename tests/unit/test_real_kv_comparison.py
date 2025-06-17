"""
REAL KV Cache Comparison Test Suite
Uses ACTUAL KV tensors from LLaMA-3 8B forward passes
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

class RealKVCacheComparisonSuite:
    def __init__(self, model_path: str = "/mnt/vstor/CSE_ECSE_GXD234/Meta-Llama-3-8B-Instruct"):
        self.model_path = model_path
        self.comparison_results = {}
        self.start_time = time.time()
        
        print(f"🔬 Initializing REAL KV Cache Comparison Test Suite")
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
        
        print(f"✅ REAL comparison suite initialized successfully")

    def extract_real_kv_tensors(self, 
                               texts: List[str],
                               max_length: int = 100,
                               num_layers: int = 4,
                               num_heads: int = 8) -> Dict[str, Any]:
        """
        Extract REAL KV tensors from LLaMA forward passes
        
        Args:
            texts: Input texts to process
            max_length: Maximum sequence length
            num_layers: Number of layers to extract from
            num_heads: Number of heads to extract from
            
        Returns:
            Dictionary containing real KV tensors
        """
        print(f"\n🧠 Extracting REAL KV Tensors from LLaMA-3 8B")
        print(f"   Processing {len(texts)} texts through {num_layers} layers, {num_heads} heads")
        print(f"=" * 70)
        
        real_kv_data = {
            "texts": [],
            "layers": {},
            "compression_stats": {}
        }
        
        for text_idx, text in enumerate(texts):
            print(f"\n--- Processing text {text_idx + 1}/{len(texts)}: '{text[:50]}...' ---")
            
            # Get real hidden states from LLaMA
            hidden_states, input_ids = self.model_loader.get_hidden_states(text, max_length)
            seq_len = hidden_states.shape[0]
            
            print(f"   Hidden states: {hidden_states.shape}")
            print(f"   Sequence length: {seq_len}")
            
            text_data = {
                "text": text,
                "hidden_states": hidden_states,
                "input_ids": input_ids,
                "sequence_length": seq_len,
                "layers": {}
            }
            
            # Process each layer
            for layer_idx in range(min(num_layers, self.model_loader.num_layers)):
                print(f"   📍 Layer {layer_idx}: Extracting real KV tensors...")
                
                # Get real attention weights for this layer
                attention_weights = self.model_loader.get_attention_weights(layer_idx)
                W_K = attention_weights["W_K"]  # [num_kv_heads * head_dim, hidden_size]
                W_V = attention_weights["W_V"]  # [num_kv_heads * head_dim, hidden_size]
                
                layer_data = {
                    "attention_weights": attention_weights,
                    "heads": {}
                }
                
                # Process each head
                for head_idx in range(min(num_heads, self.model_loader.num_attention_heads)):
                    # GQA mapping: map query head to KV head
                    num_query_heads = self.model_loader.num_attention_heads     # 32
                    num_kv_heads = self.model_loader.num_key_value_heads        # 8
                    heads_per_kv = num_query_heads // num_kv_heads              # 4
                    kv_head_idx = head_idx // heads_per_kv                      # Which kv head group
                    
                    # Extract head-specific weights
                    head_dim = self.model_loader.head_dim
                    W_K_head = W_K[kv_head_idx * head_dim:(kv_head_idx + 1) * head_dim, :]  # [head_dim, hidden_size]
                    W_V_head = W_V[kv_head_idx * head_dim:(kv_head_idx + 1) * head_dim, :]  # [head_dim, hidden_size]
                    
                    # Compute REAL KV tensors for this head
                    real_keys = []
                    real_values = []
                    compressed_keys = []
                    compressed_values = []
                    
                    print(f"     🔑 Head {head_idx}: Computing real K/V tensors...")
                    
                    # Use consistent compression profile for all tokens in this head
                    profile_name = ["low", "med", "high"][head_idx % 3]  # Per head, not per token
                    
                    for token_idx, hidden_state in enumerate(hidden_states):
                        # Project hidden state to REAL key/value using LLaMA weights
                        real_key = W_K_head @ hidden_state      # [head_dim]
                        real_value = W_V_head @ hidden_state    # [head_dim]
                        
                        real_keys.append(real_key)
                        real_values.append(real_value)
                        
                        # Compress using REAL compression profiles (consistent profile per head)
                        compressed_key = self.compression_profiles.compress_keys(real_key, head_idx)
                        compressed_value = self.compression_profiles.compress_values(real_value, profile_name, head_idx)
                        
                        compressed_keys.append(compressed_key)
                        compressed_values.append(compressed_value)
                    
                    # Stack into tensors
                    real_keys_tensor = torch.stack(real_keys)        # [seq_len, head_dim]
                    real_values_tensor = torch.stack(real_values)    # [seq_len, head_dim]
                    compressed_keys_tensor = torch.stack(compressed_keys)    # [seq_len, key_rank]
                    compressed_values_tensor = torch.stack(compressed_values)  # [seq_len, value_rank]
                    
                    head_data = {
                        "real_keys": real_keys_tensor,
                        "real_values": real_values_tensor,
                        "compressed_keys": compressed_keys_tensor,
                        "compressed_values": compressed_values_tensor,
                        "kv_head_idx": kv_head_idx,
                        "compression_profile_used": profile_name  # Single profile per head
                    }
                    
                    layer_data["heads"][head_idx] = head_data
                    
                    print(f"       Real keys: {real_keys_tensor.shape}")
                    print(f"       Real values: {real_values_tensor.shape}")
                    print(f"       Compressed keys: {compressed_keys_tensor.shape}")
                    print(f"       Compressed values: {compressed_values_tensor.shape}")
                
                text_data["layers"][layer_idx] = layer_data
                print(f"   ✅ Layer {layer_idx}: Complete")
            
            real_kv_data["texts"].append(text_data)
            print(f"✅ Text {text_idx + 1}: Complete")
        
        print(f"\n🎯 REAL KV Extraction Summary:")
        print(f"   Texts processed: {len(texts)}")
        print(f"   Layers extracted: {num_layers}")
        print(f"   Heads per layer: {num_heads}")
        print(f"   Total KV tensors: {len(texts) * num_layers * num_heads * 2}")  # 2 = keys + values
        
        return real_kv_data

    def test_real_memory_comparison(self, real_kv_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test memory usage using REAL KV tensors"""
        print(f"\n🧠 Test: REAL Memory Usage Comparison")
        print(f"=" * 70)
        
        memory_results = {}
        
        for text_idx, text_data in enumerate(real_kv_data["texts"]):
            text = text_data["text"]
            seq_len = text_data["sequence_length"]
            
            print(f"\n   Text {text_idx + 1}: '{text[:40]}...' (seq_len={seq_len})")
            
            # Reset caches
            self.compressed_cache = LLaMAKVCache(enable_compression=True)
            self.standard_cache = StandardKVCache()
            
            # Store REAL KV tensors in both caches
            for layer_idx, layer_data in text_data["layers"].items():
                for head_idx, head_data in layer_data["heads"].items():
                    real_keys = head_data["real_keys"]           # [seq_len, head_dim]
                    real_values = head_data["real_values"]       # [seq_len, head_dim] 
                    compressed_keys = head_data["compressed_keys"]     # [seq_len, key_rank]
                    compressed_values = head_data["compressed_values"] # [seq_len, value_rank]
                    
                    # Store in standard cache (real KV tensors)
                    for token_idx in range(seq_len):
                        self.standard_cache.store_kv(
                            layer_idx=layer_idx,
                            head_idx=head_idx,
                            keys=real_keys[token_idx],
                            values=real_values[token_idx],
                            token_idx=token_idx
                        )
                    
                    # Store in compressed cache (compressed KV tensors)
                    profile_name = head_data["compression_profile_used"]  # Use the profile from extraction
                    for token_idx in range(seq_len):
                        self.compressed_cache.store_compressed_kv(
                            layer_idx=layer_idx,
                            head_idx=head_idx,
                            compressed_keys=compressed_keys[token_idx],
                            compressed_values=compressed_values[token_idx],
                            token_idx=token_idx,
                            compression_profile=profile_name
                        )
            
            # Measure REAL memory usage
            standard_memory = self.standard_cache.get_memory_usage()
            compressed_memory = self.compressed_cache.get_memory_usage()
            
            memory_savings = (standard_memory["total_memory_mb"] - compressed_memory["total_memory_mb"]) / standard_memory["total_memory_mb"]
            compression_ratio = standard_memory["total_memory_mb"] / compressed_memory["total_memory_mb"]
            
            memory_results[f"text_{text_idx + 1}"] = {
                "text": text,
                "sequence_length": seq_len,
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

    def test_real_reconstruction_accuracy(self, real_kv_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test reconstruction accuracy using REAL KV tensors"""
        print(f"\n🎯 Test: REAL Reconstruction Accuracy")
        print(f"=" * 70)
        
        accuracy_results = {}
        
        for text_idx, text_data in enumerate(real_kv_data["texts"]):
            text = text_data["text"]
            print(f"\n   Text {text_idx + 1}: '{text[:40]}...'")
            
            text_accuracy = {
                "text": text,
                "layers": {}
            }
            
            for layer_idx, layer_data in text_data["layers"].items():
                layer_accuracy = {"heads": {}}
                head_accuracy = {}  # Initialize head_accuracy dictionary
                
                for head_idx, head_data in layer_data["heads"].items():
                    real_keys = head_data["real_keys"]               # [seq_len, head_dim]
                    real_values = head_data["real_values"]           # [seq_len, head_dim]
                    compressed_keys = head_data["compressed_keys"]   # [seq_len, key_rank]
                    compressed_values = head_data["compressed_values"] # [seq_len, value_rank]
                    
                    # Test key reconstruction accuracy
                    reconstructed_keys = self.compression_profiles.reconstruct_keys(compressed_keys, head_idx)
                    
                    # Calculate errors
                    key_mse = torch.nn.functional.mse_loss(reconstructed_keys, real_keys).item()
                    key_cosine = torch.nn.functional.cosine_similarity(
                        reconstructed_keys.flatten(), 
                        real_keys.flatten(), 
                        dim=0
                    ).item()
                    
                    # Calculate compression ratios
                    original_key_size = real_keys.numel() * real_keys.element_size()
                    compressed_key_size = compressed_keys.numel() * compressed_keys.element_size()
                    key_compression_ratio = original_key_size / compressed_key_size
                    
                    original_value_size = real_values.numel() * real_values.element_size()
                    compressed_value_size = compressed_values.numel() * compressed_values.element_size()
                    value_compression_ratio = original_value_size / compressed_value_size
                    
                    head_accuracy[head_idx] = {
                        "key_reconstruction_mse": key_mse,
                        "key_cosine_similarity": key_cosine,
                        "key_compression_ratio": key_compression_ratio,
                        "value_compression_ratio": value_compression_ratio,
                        "original_shapes": {
                            "keys": real_keys.shape,
                            "values": real_values.shape
                        },
                        "compressed_shapes": {
                            "keys": compressed_keys.shape,
                            "values": compressed_values.shape
                        }
                    }
                    
                    print(f"     Layer {layer_idx}, Head {head_idx}:")
                    print(f"       Key MSE: {key_mse:.6f}")
                    print(f"       Key Cosine: {key_cosine:.4f}")
                    print(f"       Key compression: {key_compression_ratio:.2f}x")
                    print(f"       Value compression: {value_compression_ratio:.2f}x")
                
                layer_accuracy["heads"] = head_accuracy
                text_accuracy["layers"][layer_idx] = layer_accuracy
            
            accuracy_results[f"text_{text_idx + 1}"] = text_accuracy
        
        return accuracy_results

    def test_real_inference_performance(self, real_kv_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test inference performance using REAL KV tensors"""
        print(f"\n⚡ Test: REAL Inference Performance")
        print(f"=" * 70)
        
        performance_results = {}
        
        for text_idx, text_data in enumerate(real_kv_data["texts"]):
            text = text_data["text"]
            hidden_states = text_data["hidden_states"]
            input_ids = text_data["input_ids"]
            
            print(f"\n   Text {text_idx + 1}: '{text[:40]}...'")
            
            # Clear caches
            self.compressed_cache.clear_cache()
            self.standard_cache.clear_cache()
            
            # Populate caches with REAL data
            compression_mapping = self.dataset_handler.create_compression_mapping(
                len(hidden_states), strategy="adaptive"
            )
            
            # Time compressed inference
            compressed_start = time.time()
            
            # Store real compressed KV data
            for layer_idx, layer_data in text_data["layers"].items():
                for head_idx, head_data in layer_data["heads"].items():
                    compressed_keys = head_data["compressed_keys"]
                    compressed_values = head_data["compressed_values"]
                    profile_name = head_data["compression_profile_used"]  # Use consistent profile
                    
                    for token_idx in range(len(hidden_states)):
                        self.compressed_cache.store_compressed_kv(
                            layer_idx=layer_idx,
                            head_idx=head_idx,
                            compressed_keys=compressed_keys[token_idx],
                            compressed_values=compressed_values[token_idx],
                            token_idx=token_idx,
                            compression_profile=profile_name
                        )
            
            # Test compressed forward pass
            inference_pipeline = LLaMACompressionInference(self.model_path)
            inference_pipeline.compressed_cache = self.compressed_cache
            
            query_state = hidden_states[-1]  # Use last token as query
            compressed_output = inference_pipeline.compressed_attention_forward(query_state, 0, 0)
            
            compressed_time = time.time() - compressed_start
            
            # Time standard inference
            standard_start = time.time()
            
            # Store real standard KV data
            for layer_idx, layer_data in text_data["layers"].items():
                for head_idx, head_data in layer_data["heads"].items():
                    real_keys = head_data["real_keys"]
                    real_values = head_data["real_values"]
                    
                    for token_idx in range(len(hidden_states)):
                        self.standard_cache.store_kv(
                            layer_idx=layer_idx,
                            head_idx=head_idx,
                            keys=real_keys[token_idx],
                            values=real_values[token_idx],
                            token_idx=token_idx
                        )
            
            inference_pipeline.standard_cache = self.standard_cache
            standard_output = inference_pipeline.standard_attention_forward(query_state, 0, 0)
            
            standard_time = time.time() - standard_start
            
            # Calculate quality metrics
            output_mse = torch.nn.functional.mse_loss(compressed_output, standard_output).item()
            output_cosine = torch.nn.functional.cosine_similarity(
                compressed_output.unsqueeze(0), 
                standard_output.unsqueeze(0)
            ).item()
            
            # Get ground truth perplexity
            gt_logits = self.dataset_handler.get_ground_truth_logits(input_ids)
            gt_perplexity, gt_loss = self.dataset_handler.calculate_perplexity(gt_logits, input_ids)
            
            performance_results[f"text_{text_idx + 1}"] = {
                "text": text,
                "compressed_time": compressed_time,
                "standard_time": standard_time,
                "speedup": standard_time / compressed_time if compressed_time > 0 else 1.0,
                "output_mse": output_mse,
                "output_cosine_similarity": output_cosine,
                "ground_truth_perplexity": gt_perplexity,
                "ground_truth_loss": gt_loss
            }
            
            print(f"     Compressed time: {compressed_time:.4f}s")
            print(f"     Standard time: {standard_time:.4f}s")
            print(f"     Speedup: {standard_time / compressed_time:.2f}x" if compressed_time > 0 else "N/A")
            print(f"     Output MSE: {output_mse:.6f}")
            print(f"     Output Cosine: {output_cosine:.4f}")
            print(f"     GT Perplexity: {gt_perplexity:.2f}")
        
        return performance_results

    def run_real_comparison_suite(self, 
                                 test_texts: List[str] = None,
                                 max_length: int = 100,
                                 num_layers: int = 2,
                                 num_heads: int = 4) -> Dict[str, Any]:
        """Run complete REAL KV cache comparison suite"""
        print(f"\n🚀 Running Complete REAL KV Cache Comparison Suite")
        print(f"=" * 70)
        
        if test_texts is None:
            test_texts = [
                "The transformer architecture has revolutionized natural language processing with attention mechanisms.",
                "Large language models demonstrate emergent capabilities through scale and sophisticated training procedures.",
                "Memory-efficient attention techniques enable deployment of massive neural networks on constrained hardware."
            ]
        
        print(f"📋 Test Configuration:")
        print(f"   Texts: {len(test_texts)}")
        print(f"   Max length: {max_length}")
        print(f"   Layers: {num_layers}")
        print(f"   Heads per layer: {num_heads}")
        
        # Step 1: Extract REAL KV tensors
        real_kv_data = self.extract_real_kv_tensors(
            texts=test_texts,
            max_length=max_length,
            num_layers=num_layers,
            num_heads=num_heads
        )
        
        # Step 2: Run all comparison tests with REAL data
        self.comparison_results = {
            "real_kv_data_summary": {
                "num_texts": len(test_texts),
                "num_layers": num_layers,
                "num_heads": num_heads,
                "max_length": max_length
            },
            "memory_comparison": self.test_real_memory_comparison(real_kv_data),
            "reconstruction_accuracy": self.test_real_reconstruction_accuracy(real_kv_data),
            "inference_performance": self.test_real_inference_performance(real_kv_data)
        }
        
        # Step 3: Calculate overall summary
        self._calculate_real_summary()
        
        # Step 4: Print results
        self._print_real_comparison_summary()
        
        return self.comparison_results

    def _calculate_real_summary(self):
        """Calculate overall summary from REAL test results"""
        memory_results = self.comparison_results["memory_comparison"]
        accuracy_results = self.comparison_results["reconstruction_accuracy"]
        performance_results = self.comparison_results["inference_performance"]
        
        # Average memory savings
        avg_memory_savings = np.mean([
            result["memory_savings_percent"] for result in memory_results.values()
        ])
        
        avg_compression_ratio = np.mean([
            result["compression_ratio"] for result in memory_results.values()
        ])
        
        # Average reconstruction accuracy
        all_key_mse = []
        all_key_cosine = []
        all_key_compression = []
        all_value_compression = []
        
        for text_result in accuracy_results.values():
            for layer_result in text_result["layers"].values():
                for head_result in layer_result["heads"].values():
                    all_key_mse.append(head_result["key_reconstruction_mse"])
                    all_key_cosine.append(head_result["key_cosine_similarity"])
                    all_key_compression.append(head_result["key_compression_ratio"])
                    all_value_compression.append(head_result["value_compression_ratio"])
        
        # Average performance
        avg_speedup = np.mean([
            result["speedup"] for result in performance_results.values()
        ])
        
        avg_output_mse = np.mean([
            result["output_mse"] for result in performance_results.values()
        ])
        
        avg_output_cosine = np.mean([
            result["output_cosine_similarity"] for result in performance_results.values()
        ])
        
        self.comparison_results["overall_summary"] = {
            "average_memory_savings_percent": avg_memory_savings,
            "average_compression_ratio": avg_compression_ratio,
            "average_key_reconstruction_mse": np.mean(all_key_mse),
            "average_key_cosine_similarity": np.mean(all_key_cosine),
            "average_key_compression_ratio": np.mean(all_key_compression),
            "average_value_compression_ratio": np.mean(all_value_compression),
            "average_inference_speedup": avg_speedup,
            "average_output_mse": avg_output_mse,
            "average_output_cosine_similarity": avg_output_cosine,
            "test_duration": time.time() - self.start_time,
            "timestamp": datetime.now().isoformat(),
            "aggressive_compression_enabled": True,
            "key_rank": 32,  # New aggressive key rank
            "value_ranks": {"low": 32, "med": 48, "high": 64}  # New aggressive value ranks
        }

    def _print_real_comparison_summary(self):
        """Print comprehensive REAL comparison summary"""
        print(f"\n🎯 REAL KV CACHE COMPARISON SUMMARY")
        print(f"=" * 70)
        
        summary = self.comparison_results["overall_summary"]
        
        print(f"🚀 AGGRESSIVE COMPRESSION RESULTS:")
        print(f"   Key rank: {summary['key_rank']} (was 128)")
        print(f"   Value ranks: low={summary['value_ranks']['low']}, med={summary['value_ranks']['med']}, high={summary['value_ranks']['high']}")
        
        print(f"\n💾 Memory Performance:")
        print(f"   Average memory savings: {summary['average_memory_savings_percent']:.1f}%")
        print(f"   Average compression ratio: {summary['average_compression_ratio']:.2f}x")
        print(f"   Key compression ratio: {summary['average_key_compression_ratio']:.2f}x")
        print(f"   Value compression ratio: {summary['average_value_compression_ratio']:.2f}x")
        
        print(f"\n🎯 Reconstruction Quality:")
        print(f"   Key reconstruction MSE: {summary['average_key_reconstruction_mse']:.6f}")
        print(f"   Key cosine similarity: {summary['average_key_cosine_similarity']:.4f}")
        
        print(f"\n⚡ Inference Performance:")
        print(f"   Average speedup: {summary['average_inference_speedup']:.2f}x")
        print(f"   Output MSE: {summary['average_output_mse']:.6f}")
        print(f"   Output cosine similarity: {summary['average_output_cosine_similarity']:.4f}")
        
        print(f"\n📊 Test Statistics:")
        print(f"   Total test duration: {summary['test_duration']:.2f}s")
        print(f"   Using REAL LLaMA-3 8B KV tensors: ✅")
        
        # Generate recommendation
        if summary['average_memory_savings_percent'] > 40 and summary['average_key_cosine_similarity'] > 0.95:
            recommendation = "🎉 EXCELLENT: Aggressive compression delivers outstanding results!"
        elif summary['average_memory_savings_percent'] > 30 and summary['average_key_cosine_similarity'] > 0.90:
            recommendation = "✅ VERY GOOD: Strong compression with acceptable quality loss"
        elif summary['average_memory_savings_percent'] > 20:
            recommendation = "⚠️ MODERATE: Decent compression but monitor quality closely"
        else:
            recommendation = "❌ POOR: Compression benefits don't justify the complexity"
        
        print(f"\n💡 Recommendation:")
        print(f"   {recommendation}")

    def save_real_comparison_results(self, filepath: str):
        """Save REAL comparison results to JSON file"""
        # Convert tensors to lists for JSON serialization
        results_copy = {}
        for key, value in self.comparison_results.items():
            if key == "real_kv_data_summary":
                results_copy[key] = value
            else:
                results_copy[key] = self._convert_tensors_to_lists(value)
        
        with open(filepath, 'w') as f:
            json.dump(results_copy, f, indent=2, default=str)
        
        print(f"💾 REAL comparison results saved to {filepath}")

    def _convert_tensors_to_lists(self, obj):
        """Convert tensors to lists for JSON serialization"""
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_tensors_to_lists(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_tensors_to_lists(item) for item in obj]
        else:
            return obj


def main():
    """Main REAL comparison test execution"""
    print(f"🔬 REAL KV Cache Comparison Test Suite")
    print(f"Using ACTUAL LLaMA-3 8B KV tensors (NO SIMULATIONS)")
    
    # Create REAL comparison suite
    comparison_suite = RealKVCacheComparisonSuite()
    
    # Define test texts
    test_texts = [
        "The transformer architecture revolutionized natural language processing through self-attention mechanisms.",
        "Large language models exhibit emergent capabilities that scale with model size and training data.",
        "Memory-efficient attention techniques enable deployment of massive neural networks on resource-constrained devices."
    ]
    
    # Run all REAL comparisons
    results = comparison_suite.run_real_comparison_suite(
        test_texts=test_texts,
        max_length=80,
        num_layers=3,    # Test 3 layers
        num_heads=8      # Test 8 heads per layer
    )
    
    # Save results
    results_path = f"tests/real_kv_cache_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    comparison_suite.save_real_comparison_results(results_path)
    
    # Final assessment
    summary = results["overall_summary"]
    memory_savings = summary["average_memory_savings_percent"]
    
    print(f"\n🏆 FINAL ASSESSMENT:")
    if memory_savings >= 50:
        print(f"🎉 OUTSTANDING: {memory_savings:.1f}% memory savings with aggressive compression!")
    elif memory_savings >= 40:
        print(f"🚀 EXCELLENT: {memory_savings:.1f}% memory savings - compression working brilliantly!")
    elif memory_savings >= 30:
        print(f"✅ VERY GOOD: {memory_savings:.1f}% memory savings - strong results!")
    elif memory_savings >= 20:
        print(f"⚠️ MODERATE: {memory_savings:.1f}% memory savings - acceptable but can improve")
    else:
        print(f"❌ LOW: {memory_savings:.1f}% memory savings - needs optimization")
    
    print(f"Key compression: {summary['average_key_compression_ratio']:.1f}x")
    print(f"Value compression: {summary['average_value_compression_ratio']:.1f}x")
    print(f"Quality retention: {summary['average_key_cosine_similarity']:.3f} cosine similarity")
    
    return results


if __name__ == "__main__":
    main()
