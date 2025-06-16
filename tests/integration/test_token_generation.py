"""
Real Token Generation Test with Compressed KV Cache
Compare original tokens vs generated tokens using REAL LLaMA-3 8B
"""
import sys
import os
import torch
import time
from typing import List, Dict, Any
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.model import LLaMAModelLoader
from core.compression import LLaMACompressionProfileBuilder
from core.cache import LLaMAKVCache, StandardKVCache
from core.data import LLaMADatasetHandler
from core.inference import CompressedAutoregressiveDecoder

class RealTokenGenerationTest:
    def __init__(self, model_path: str = "/mnt/vstor/CSE_ECSE_GXD234/Meta-Llama-3-8B-Instruct"):
        self.model_path = model_path
        
        print(f"🚀 Initializing Real Token Generation Test")
        print(f"🕐 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📍 Model path: {model_path}")
        
        # Initialize components
        self.model_loader = LLaMAModelLoader(model_path)
        from core.config import CompressionConfig
        compression_config = CompressionConfig()
        self.compression_profiles = LLaMACompressionProfileBuilder(self.model_loader, compression_config)
        self.dataset_handler = LLaMADatasetHandler(self.model_loader)
        
        # Get tokenizer
        self.tokenizer = self.model_loader.tokenizer
        
        # Initialize decoder
        self.compressed_decoder = CompressedAutoRegressiveDecoder(
            self.model_loader, self.compression_profiles, self.tokenizer
        )
        
        print(f"✅ Token generation test initialized successfully")

    def decode_tokens(self, token_ids: List[int]) -> List[str]:
        """Convert token IDs to readable tokens"""
        return [self.tokenizer.decode([token_id]) for token_id in token_ids]

    def test_token_generation_comparison(self, 
                                       prompt: str,
                                       max_new_tokens: int = 20,
                                       temperature: float = 0.7,
                                       top_k: int = 50) -> Dict[str, Any]:
        """
        Generate tokens with both compressed and standard methods
        
        Args:
            prompt: Input prompt text
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            
        Returns:
            Dictionary with generation results and token comparisons
        """
        print(f"\n🎯 Token Generation Comparison Test")
        print(f"=" * 60)
        print(f"Prompt: '{prompt}'")
        print(f"Max new tokens: {max_new_tokens}")
        print(f"Temperature: {temperature}")
        print(f"Top-k: {top_k}")
        
        # Tokenize the prompt
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        print(f"\n📝 Prompt tokens: {prompt_tokens}")
        print(f"Prompt token count: {len(prompt_tokens)}")
        
        # Show prompt tokens as text
        prompt_token_texts = self.decode_tokens(prompt_tokens)
        print(f"Prompt tokens as text: {prompt_token_texts}")
        
        results = {
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "prompt_token_texts": prompt_token_texts,
            "max_new_tokens": max_new_tokens,
            "compressed_generation": {},
            "standard_generation": {},
            "comparison": {}
        }
        
        # 1. Generate with COMPRESSED KV cache
        print(f"\n🗜️  Generating with COMPRESSED KV cache...")
        compressed_start = time.time()
        
        try:
            compressed_output = self.compressed_decoder.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                compression_strategy="adaptive"
            )
            
            compressed_time = time.time() - compressed_start
            compressed_tokens = compressed_output["output_ids"]
            compressed_new_tokens = compressed_tokens[len(prompt_tokens):]
            compressed_new_token_texts = self.decode_tokens(compressed_new_tokens)
            compressed_full_text = self.tokenizer.decode(compressed_tokens, skip_special_tokens=True)
            
            results["compressed_generation"] = {
                "success": True,
                "generation_time": compressed_time,
                "full_tokens": compressed_tokens,
                "new_tokens": compressed_new_tokens,
                "new_token_texts": compressed_new_token_texts,
                "full_text": compressed_full_text,
                "tokens_generated": len(compressed_new_tokens),
                "cache_stats": compressed_output.get("cache_stats", {}),
                "compression_mapping": compressed_output.get("compression_mapping", [])
            }
            
            print(f"   ✅ Compressed generation successful")
            print(f"   Time: {compressed_time:.3f}s")
            print(f"   Tokens generated: {len(compressed_new_tokens)}")
            print(f"   New tokens: {compressed_new_tokens}")
            print(f"   New token texts: {compressed_new_token_texts}")
            
        except Exception as e:
            print(f"   ❌ Compressed generation failed: {e}")
            results["compressed_generation"] = {
                "success": False,
                "error": str(e),
                "generation_time": time.time() - compressed_start
            }
        
        # 2. Generate with STANDARD (uncompressed) method
        print(f"\n📝 Generating with STANDARD method...")
        standard_start = time.time()
        
        try:
            # Use the model's built-in generation
            with torch.no_grad():
                input_ids = torch.tensor([prompt_tokens], device=self.model_loader.device)
                
                # Generate with standard LLaMA
                standard_outputs = self.model_loader.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                standard_tokens = standard_outputs[0].cpu().tolist()
                standard_time = time.time() - standard_start
                standard_new_tokens = standard_tokens[len(prompt_tokens):]
                standard_new_token_texts = self.decode_tokens(standard_new_tokens)
                standard_full_text = self.tokenizer.decode(standard_tokens, skip_special_tokens=True)
            
            results["standard_generation"] = {
                "success": True,
                "generation_time": standard_time,
                "full_tokens": standard_tokens,
                "new_tokens": standard_new_tokens,
                "new_token_texts": standard_new_token_texts,
                "full_text": standard_full_text,
                "tokens_generated": len(standard_new_tokens)
            }
            
            print(f"   ✅ Standard generation successful")
            print(f"   Time: {standard_time:.3f}s")
            print(f"   Tokens generated: {len(standard_new_tokens)}")
            print(f"   New tokens: {standard_new_tokens}")
            print(f"   New token texts: {standard_new_token_texts}")
            
        except Exception as e:
            print(f"   ❌ Standard generation failed: {e}")
            results["standard_generation"] = {
                "success": False,
                "error": str(e),
                "generation_time": time.time() - standard_start
            }
        
        # 3. Compare the results
        if results["compressed_generation"]["success"] and results["standard_generation"]["success"]:
            self._compare_generations(results)
        
        return results

    def _compare_generations(self, results: Dict[str, Any]):
        """Compare compressed vs standard generation results"""
        print(f"\n🔍 Comparing Generation Results")
        print(f"=" * 60)
        
        compressed = results["compressed_generation"]
        standard = results["standard_generation"]
        
        # Token-level comparison
        compressed_new = compressed["new_tokens"]
        standard_new = standard["new_tokens"]
        
        min_length = min(len(compressed_new), len(standard_new))
        
        # Calculate token-level accuracy
        matching_tokens = sum(1 for i in range(min_length) 
                            if compressed_new[i] == standard_new[i])
        token_accuracy = matching_tokens / min_length if min_length > 0 else 0
        
        # Performance comparison
        speedup = standard["generation_time"] / compressed["generation_time"] if compressed["generation_time"] > 0 else 1.0
        
        # Text similarity (simple word overlap)
        compressed_words = set(compressed["full_text"].lower().split())
        standard_words = set(standard["full_text"].lower().split())
        
        if len(compressed_words.union(standard_words)) > 0:
            text_overlap = len(compressed_words.intersection(standard_words)) / len(compressed_words.union(standard_words))
        else:
            text_overlap = 0.0
        
        results["comparison"] = {
            "token_accuracy": token_accuracy,
            "matching_tokens": matching_tokens,
            "total_tokens_compared": min_length,
            "speedup": speedup,
            "text_word_overlap": text_overlap,
            "length_difference": abs(len(compressed_new) - len(standard_new)),
            "compressed_length": len(compressed_new),
            "standard_length": len(standard_new)
        }
        
        print(f"Token Accuracy: {token_accuracy:.2%} ({matching_tokens}/{min_length})")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Text Word Overlap: {text_overlap:.2%}")
        print(f"Length difference: {abs(len(compressed_new) - len(standard_new))} tokens")

    def display_token_comparison(self, results: Dict[str, Any]):
        """Display detailed token-by-token comparison"""
        print(f"\n📊 DETAILED TOKEN COMPARISON")
        print(f"=" * 80)
        
        if not (results["compressed_generation"]["success"] and results["standard_generation"]["success"]):
            print("❌ Cannot compare - one or both generations failed")
            return
        
        # Get the data
        prompt_tokens = results["prompt_tokens"]
        prompt_token_texts = results["prompt_token_texts"]
        
        compressed_new = results["compressed_generation"]["new_tokens"]
        compressed_new_texts = results["compressed_generation"]["new_token_texts"]
        
        standard_new = results["standard_generation"]["new_tokens"]
        standard_new_texts = results["standard_generation"]["new_token_texts"]
        
        # Display prompt tokens
        print(f"\n🎯 ORIGINAL PROMPT TOKENS:")
        print(f"{'Position':<8} {'Token ID':<10} {'Token Text':<20}")
        print(f"-" * 50)
        for i, (token_id, token_text) in enumerate(zip(prompt_tokens, prompt_token_texts)):
            print(f"{i:<8} {token_id:<10} '{token_text}'")
        
        # Display generated tokens side by side
        print(f"\n🆚 GENERATED TOKENS COMPARISON:")
        print(f"{'Pos':<4} {'Compressed ID':<12} {'Compressed Text':<20} {'Standard ID':<12} {'Standard Text':<20} {'Match':<6}")
        print(f"-" * 90)
        
        max_length = max(len(compressed_new), len(standard_new))
        
        for i in range(max_length):
            # Get compressed token
            if i < len(compressed_new):
                comp_id = compressed_new[i]
                comp_text = compressed_new_texts[i] if i < len(compressed_new_texts) else "N/A"
            else:
                comp_id = "N/A"
                comp_text = "N/A"
            
            # Get standard token
            if i < len(standard_new):
                std_id = standard_new[i]
                std_text = standard_new_texts[i] if i < len(standard_new_texts) else "N/A"
            else:
                std_id = "N/A"
                std_text = "N/A"
            
            # Check if they match
            match = "✅" if (comp_id != "N/A" and std_id != "N/A" and comp_id == std_id) else "❌"
            
            print(f"{i:<4} {comp_id:<12} '{comp_text}'<{20} {std_id:<12} '{std_text}'<{20} {match:<6}")
        
        # Display full generated text
        print(f"\n📝 FULL GENERATED TEXT COMPARISON:")
        print(f"-" * 80)
        
        compressed_full = results["compressed_generation"]["full_text"]
        standard_full = results["standard_generation"]["full_text"]
        
        print(f"\n🗜️  COMPRESSED GENERATION:")
        print(f"'{compressed_full}'")
        
        print(f"\n📝 STANDARD GENERATION:")
        print(f"'{standard_full}'")
        
        # Memory usage if available
        if "cache_stats" in results["compressed_generation"]:
            cache_stats = results["compressed_generation"]["cache_stats"]
            print(f"\n💾 COMPRESSION STATISTICS:")
            if "memory_savings" in cache_stats:
                print(f"Memory savings: {cache_stats['memory_savings']:.1%}")
            if "compression_mapping" in results["compressed_generation"]:
                mapping = results["compressed_generation"]["compression_mapping"]
                profile_counts = {}
                for profile in mapping:
                    profile_counts[profile] = profile_counts.get(profile, 0) + 1
                print(f"Compression profiles used: {profile_counts}")

    def run_multiple_generation_tests(self, 
                                    prompts: List[str],
                                    max_new_tokens: int = 15,
                                    temperature: float = 0.7) -> Dict[str, Any]:
        """Run generation tests on multiple prompts"""
        print(f"\n🧪 Running Multiple Token Generation Tests")
        print(f"=" * 70)
        
        all_results = {
            "test_config": {
                "num_prompts": len(prompts),
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "timestamp": datetime.now().isoformat()
            },
            "prompt_results": {},
            "aggregate_stats": {}
        }
        
        for i, prompt in enumerate(prompts):
            print(f"\n🎯 Test {i+1}/{len(prompts)}")
            
            results = self.test_token_generation_comparison(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
            
            all_results["prompt_results"][f"prompt_{i+1}"] = results
            
            # Display the detailed comparison for each prompt
            self.display_token_comparison(results)
            
            print(f"\n" + "="*50)
        
        # Calculate aggregate statistics
        self._calculate_aggregate_stats(all_results)
        
        return all_results

    def _calculate_aggregate_stats(self, all_results: Dict[str, Any]):
        """Calculate aggregate statistics across all tests"""
        prompt_results = all_results["prompt_results"]
        
        successful_tests = [r for r in prompt_results.values() 
                          if r["compressed_generation"]["success"] and r["standard_generation"]["success"]]
        
        if not successful_tests:
            print("❌ No successful tests to aggregate")
            return
        
        # Aggregate metrics
        token_accuracies = [r["comparison"]["token_accuracy"] for r in successful_tests]
        speedups = [r["comparison"]["speedup"] for r in successful_tests]
        text_overlaps = [r["comparison"]["text_word_overlap"] for r in successful_tests]
        
        avg_token_accuracy = sum(token_accuracies) / len(token_accuracies)
        avg_speedup = sum(speedups) / len(speedups)
        avg_text_overlap = sum(text_overlaps) / len(text_overlaps)
        
        all_results["aggregate_stats"] = {
            "successful_tests": len(successful_tests),
            "total_tests": len(prompt_results),
            "success_rate": len(successful_tests) / len(prompt_results),
            "average_token_accuracy": avg_token_accuracy,
            "average_speedup": avg_speedup,
            "average_text_overlap": avg_text_overlap
        }
        
        print(f"\n🎯 AGGREGATE RESULTS ACROSS ALL TESTS")
        print(f"=" * 60)
        print(f"Successful tests: {len(successful_tests)}/{len(prompt_results)}")
        print(f"Average token accuracy: {avg_token_accuracy:.2%}")
        print(f"Average speedup: {avg_speedup:.2f}x")
        print(f"Average text overlap: {avg_text_overlap:.2%}")


def main():
    """Run token generation tests with real data"""
    print(f"🚀 REAL TOKEN GENERATION TEST")
    print(f"Comparing Compressed vs Standard Token Generation")
    print(f"Using REAL LLaMA-3 8B Model")
    
    # Create test suite
    test_suite = RealTokenGenerationTest()
    
    # Define test prompts
    test_prompts = [
        "The future of artificial intelligence",
        "Climate change affects",
        "The transformer architecture revolutionized",
        "Large language models can"
    ]
    
    # Run comprehensive generation tests
    results = test_suite.run_multiple_generation_tests(
        prompts=test_prompts,
        max_new_tokens=12,  # Generate 12 new tokens per prompt
        temperature=0.7
    )
    
    # Final summary
    print(f"\n🏆 FINAL TOKEN GENERATION SUMMARY")
    print(f"=" * 70)
    
    if "aggregate_stats" in results:
        stats = results["aggregate_stats"]
        print(f"✅ Tests completed: {stats['successful_tests']}/{stats['total_tests']}")
        print(f"🎯 Average token accuracy: {stats['average_token_accuracy']:.1%}")
        print(f"⚡ Average speedup: {stats['average_speedup']:.2f}x")
        print(f"📝 Average text overlap: {stats['average_text_overlap']:.1%}")
        
        if stats['average_token_accuracy'] >= 0.8:
            print(f"🎉 EXCELLENT: High token accuracy with compressed generation!")
        elif stats['average_token_accuracy'] >= 0.6:
            print(f"✅ GOOD: Reasonable token accuracy with compression benefits")
        elif stats['average_token_accuracy'] >= 0.4:
            print(f"⚠️ MODERATE: Some token divergence but still usable")
        else:
            print(f"❌ POOR: Significant token divergence - needs optimization")
    
    # Save results
    results_path = f"tests/token_generation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Convert tensors to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    results_json = convert_for_json(results)
    
    import json
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2, default=str)
    
    print(f"💾 Results saved to {results_path}")
    
    return results


if __name__ == "__main__":
    main()
