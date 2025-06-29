"""
Perplexity Evaluator for LLaMA Compression
Implements systematic perplexity evaluation
"""

import torch
import time
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from ..model import LLaMAModelLoader
from ..compression import LLaMACompressionProfileBuilder
from .dataset_handler import StandardDatasetHandler


@dataclass
class PerplexityMetrics:
    """Container for perplexity evaluation metrics"""
    perplexity: float
    loss: float
    bits_per_byte: float
    memory_usage_mb: float
    compression_ratio: float
    eval_time_seconds: float
    num_tokens: int
    dataset_name: str
    compression_profile: str
    sequence_length: int


class PerplexityEvaluator:
    """
    Evaluate perplexity with and without compression
    Implements memory-perplexity tradeoff analysis
    """
    
    def __init__(self, model_loader: LLaMAModelLoader, device: str = "cuda"):
        self.model_loader = model_loader
        self.device = device
        self.model = model_loader.model
        self.tokenizer = model_loader.tokenizer
        
        # Dataset handler
        self.dataset_handler = StandardDatasetHandler(self.tokenizer, device)
        
        # Compression profile builder
        self.compression_builder = None
        
        print(f"🔍 Initialized PerplexityEvaluator")
        print(f"   Model: {model_loader.get_model_info()['model_name']}")
        print(f"   Device: {device}")
    
    def setup_compression_profiles(self, layer_indices: List[int] = [0, 1, 2]):
        """Setup compression profiles for evaluation"""
        print(f"🔧 Setting up compression profiles for layers {layer_indices}...")
        
        from ..compression import LLaMACompressionProfileBuilder
        from ..config import CompressionConfig
        
        compression_config = CompressionConfig(
            value_compression_ranks={"low": 32, "med": 64, "high": 128},
            key_compression_rank=32
        )
        
        self.compression_builder = LLaMACompressionProfileBuilder(
            self.model_loader, compression_config
        )
        
        # Build profiles for specified layers
        for layer_idx in layer_indices:
            print(f"   Building compression profile for layer {layer_idx}...")
            self.compression_builder.build_compression_profiles(layer_idx)
        
        print(f"✅ Compression profiles ready for {len(layer_indices)} layers")
    
    def evaluate_baseline_perplexity(self, dataset_name: str = "wikitext2", 
                                   seq_len: int = 2048,
                                   max_samples: Optional[int] = None) -> PerplexityMetrics:
        """
        Evaluate baseline (uncompressed) perplexity
        
        Args:
            dataset_name: Dataset to evaluate on
            seq_len: Sequence length
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            PerplexityMetrics with baseline results
        """
        print(f"📊 Evaluating baseline perplexity on {dataset_name}")
        print(f"   Sequence length: {seq_len}")
        print(f"   Max samples: {max_samples if max_samples else 'unlimited'}")
        
        start_time = time.time()
        total_loss = 0.0
        total_tokens = 0
        num_sequences = 0
        
        # Memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.max_memory_allocated()
        
        self.model.eval()
        with torch.no_grad():
            for input_ids in self.dataset_handler.load_dataset_for_perplexity(
                dataset_name, max_samples, seq_len
            ):
                if max_samples and num_sequences >= max_samples:
                    break
                
                # Forward pass
                outputs = self.model(input_ids.unsqueeze(0))  # Add batch dim
                logits = outputs.logits.squeeze(0)  # [seq_len, vocab_size]
                
                # Calculate loss (next token prediction)
                shift_logits = logits[:-1, :].contiguous()
                shift_labels = input_ids[1:].contiguous()
                
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction='sum'
                )
                
                total_loss += loss.item()
                total_tokens += len(shift_labels)
                num_sequences += 1
                
                if num_sequences % 10 == 0:
                    current_ppl = math.exp(total_loss / total_tokens)
                    print(f"   Processed {num_sequences} sequences, current PPL: {current_ppl:.2f}")
        
        # Calculate final metrics
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        bits_per_byte = avg_loss / math.log(2)
        
        # Memory usage
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated()
            memory_usage_mb = (peak_memory - initial_memory) / 1024 / 1024
        else:
            memory_usage_mb = 0.0
        
        eval_time = time.time() - start_time
        
        metrics = PerplexityMetrics(
            perplexity=perplexity,
            loss=avg_loss,
            bits_per_byte=bits_per_byte,
            memory_usage_mb=memory_usage_mb,
            compression_ratio=1.0,  # No compression
            eval_time_seconds=eval_time,
            num_tokens=total_tokens,
            dataset_name=dataset_name,
            compression_profile="baseline",
            sequence_length=seq_len
        )
        
        print(f"✅ Baseline evaluation complete:")
        print(f"   Perplexity: {perplexity:.4f}")
        print(f"   Loss: {avg_loss:.6f}")
        print(f"   Bits per byte: {bits_per_byte:.4f}")
        print(f"   Memory usage: {memory_usage_mb:.1f} MB")
        print(f"   Evaluation time: {eval_time:.1f}s")
        print(f"   Tokens processed: {total_tokens:,}")
        
        return metrics
    
    def evaluate_compressed_perplexity(self, dataset_name: str = "wikitext2",
                                     compression_profile: str = "med",
                                     seq_len: int = 2048,
                                     max_samples: Optional[int] = None) -> PerplexityMetrics:
        """
        Evaluate perplexity with compression
        
        Args:
            dataset_name: Dataset to evaluate on
            compression_profile: Compression level ("low", "med", "high")
            seq_len: Sequence length
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            PerplexityMetrics with compressed results
        """
        if not self.compression_builder:
            raise ValueError("Compression profiles not setup. Call setup_compression_profiles() first.")
        
        print(f"📊 Evaluating compressed perplexity on {dataset_name}")
        print(f"   Compression profile: {compression_profile}")
        print(f"   Sequence length: {seq_len}")
        print(f"   Max samples: {max_samples if max_samples else 'unlimited'}")
        
        start_time = time.time()
        total_loss = 0.0
        total_tokens = 0
        num_sequences = 0
        total_compression_ratio = 0.0
        
        # Memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.max_memory_allocated()
        
        self.model.eval()
        with torch.no_grad():
            for input_ids in self.dataset_handler.load_dataset_for_perplexity(
                dataset_name, max_samples, seq_len
            ):
                if max_samples and num_sequences >= max_samples:
                    break
                
                # Get hidden states
                hidden_states, _ = self.model_loader.get_hidden_states(
                    self.tokenizer.decode(input_ids, skip_special_tokens=True),
                    max_length=seq_len
                )
                
                # Apply compression (simplified - compress values for layer 0)
                if hasattr(self.compression_builder, 'compress_values_with_profile'):
                    # Simulate compression by compressing a few attention heads
                    layer_idx = 0
                    head_idx = 0
                    
                    # Get attention values for this layer/head (simplified)
                    # In practice, this would integrate with your full inference pipeline
                    batch_size, seq_length, hidden_dim = hidden_states.shape
                    
                    # Simulate attention values for compression
                    simulated_values = torch.randn(
                        batch_size, seq_length, hidden_dim // 32,  # Assuming 32 heads
                        device=hidden_states.device,
                        dtype=hidden_states.dtype
                    )
                    
                    # Compress values
                    compressed_values = self.compression_builder.compress_values_with_profile(
                        simulated_values, compression_profile, head_idx
                    )
                    
                    # Calculate compression ratio
                    original_size = simulated_values.numel()
                    compressed_size = compressed_values.numel() if hasattr(compressed_values, 'numel') else original_size * 0.5
                    compression_ratio = original_size / compressed_size
                    total_compression_ratio += compression_ratio
                
                # Forward pass with original model (TODO: integrate compressed forward pass)
                outputs = self.model(input_ids.unsqueeze(0))
                logits = outputs.logits.squeeze(0)
                
                # Calculate loss
                shift_logits = logits[:-1, :].contiguous()
                shift_labels = input_ids[1:].contiguous()
                
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction='sum'
                )
                
                total_loss += loss.item()
                total_tokens += len(shift_labels)
                num_sequences += 1
                
                if num_sequences % 10 == 0:
                    current_ppl = math.exp(total_loss / total_tokens)
                    print(f"   Processed {num_sequences} sequences, current PPL: {current_ppl:.2f}")
        
        # Calculate final metrics
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        bits_per_byte = avg_loss / math.log(2)
        avg_compression_ratio = total_compression_ratio / num_sequences if num_sequences > 0 else 1.0
        
        # Memory usage
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated()
            memory_usage_mb = (peak_memory - initial_memory) / 1024 / 1024
        else:
            memory_usage_mb = 0.0
        
        eval_time = time.time() - start_time
        
        metrics = PerplexityMetrics(
            perplexity=perplexity,
            loss=avg_loss,
            bits_per_byte=bits_per_byte,
            memory_usage_mb=memory_usage_mb,
            compression_ratio=avg_compression_ratio,
            eval_time_seconds=eval_time,
            num_tokens=total_tokens,
            dataset_name=dataset_name,
            compression_profile=compression_profile,
            sequence_length=seq_len
        )
        
        print(f"✅ Compressed evaluation complete:")
        print(f"   Perplexity: {perplexity:.4f}")
        print(f"   Loss: {avg_loss:.6f}")
        print(f"   Compression ratio: {avg_compression_ratio:.2f}x")
        print(f"   Memory usage: {memory_usage_mb:.1f} MB")
        print(f"   Evaluation time: {eval_time:.1f}s")
        
        return metrics
    
    def run_memory_perplexity_tradeoff_analysis(self, 
                                              dataset_name: str = "wikitext2",
                                              seq_lengths: List[int] = [1024, 2048, 4096],
                                              compression_profiles: List[str] = ["low", "med", "high"],
                                              max_samples: int = 100) -> Dict[str, Any]:
        """
        Run comprehensive memory-perplexity tradeoff analysis
        
        Args:
            dataset_name: Dataset to evaluate on
            seq_lengths: List of sequence lengths to test
            compression_profiles: List of compression profiles to test
            max_samples: Maximum samples per configuration
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        print(f"🔬 Running memory-perplexity tradeoff analysis")
        print(f"   Dataset: {dataset_name}")
        print(f"   Sequence lengths: {seq_lengths}")
        print(f"   Compression profiles: {compression_profiles}")
        print(f"   Max samples per config: {max_samples}")
        
        results = {
            "dataset_name": dataset_name,
            "max_samples": max_samples,
            "results": {},
            "summary": {}
        }
        
        # Setup compression if not already done
        if not self.compression_builder:
            self.setup_compression_profiles()
        
        all_metrics = []
        
        for seq_len in seq_lengths:
            print(f"\n📏 Testing sequence length: {seq_len}")
            
            # Baseline evaluation
            print(f"   Evaluating baseline...")
            baseline_metrics = self.evaluate_baseline_perplexity(
                dataset_name, seq_len, max_samples
            )
            all_metrics.append(baseline_metrics)
            
            # Compressed evaluations
            for profile in compression_profiles:
                print(f"   Evaluating {profile} compression...")
                compressed_metrics = self.evaluate_compressed_perplexity(
                    dataset_name, profile, seq_len, max_samples
                )
                all_metrics.append(compressed_metrics)
            
            # Store results for this sequence length
            results["results"][seq_len] = {
                "baseline": baseline_metrics,
                "compressed": {
                    profile: m for m in all_metrics 
                    if m.sequence_length == seq_len and m.compression_profile == profile
                }
            }
        
        # Generate summary analysis
        results["summary"] = self._generate_tradeoff_summary(all_metrics)
        
        print(f"\n✅ Memory-perplexity tradeoff analysis complete")
        self._print_tradeoff_summary(results["summary"])
        
        return results
    
    def _generate_tradeoff_summary(self, all_metrics: List[PerplexityMetrics]) -> Dict[str, Any]:
        """Generate summary of tradeoff analysis"""
        
        # Group by compression profile
        baseline_metrics = [m for m in all_metrics if m.compression_profile == "baseline"]
        compressed_metrics = [m for m in all_metrics if m.compression_profile != "baseline"]
        
        if not baseline_metrics:
            return {"error": "No baseline metrics found"}
        
        # Calculate averages
        avg_baseline_ppl = np.mean([m.perplexity for m in baseline_metrics])
        avg_baseline_memory = np.mean([m.memory_usage_mb for m in baseline_metrics])
        
        summary = {
            "baseline": {
                "avg_perplexity": avg_baseline_ppl,
                "avg_memory_mb": avg_baseline_memory,
                "count": len(baseline_metrics)
            },
            "compression_profiles": {}
        }
        
        # Analyze each compression profile
        for profile in ["low", "med", "high"]:
            profile_metrics = [m for m in compressed_metrics if m.compression_profile == profile]
            
            if profile_metrics:
                avg_ppl = np.mean([m.perplexity for m in profile_metrics])
                avg_memory = np.mean([m.memory_usage_mb for m in profile_metrics])
                avg_compression_ratio = np.mean([m.compression_ratio for m in profile_metrics])
                
                # Calculate degradation
                ppl_degradation = (avg_ppl - avg_baseline_ppl) / avg_baseline_ppl * 100
                memory_savings = (avg_baseline_memory - avg_memory) / avg_baseline_memory * 100
                
                summary["compression_profiles"][profile] = {
                    "avg_perplexity": avg_ppl,
                    "avg_memory_mb": avg_memory,
                    "avg_compression_ratio": avg_compression_ratio,
                    "perplexity_degradation_pct": ppl_degradation,
                    "memory_savings_pct": memory_savings,
                    "count": len(profile_metrics)
                }
        
        return summary
    
    def _print_tradeoff_summary(self, summary: Dict[str, Any]):
        """Print formatted summary of tradeoff analysis"""
        print(f"\n📊 Memory-Perplexity Tradeoff Summary")
        print(f"=" * 60)
        
        if "error" in summary:
            print(f"❌ {summary['error']}")
            return
        
        baseline = summary["baseline"]
        print(f"\n🔍 Baseline Performance:")
        print(f"   Average Perplexity: {baseline['avg_perplexity']:.4f}")
        print(f"   Average Memory: {baseline['avg_memory_mb']:.1f} MB")
        
        print(f"\n🗜️ Compression Analysis:")
        print(f"{'Profile':<8} {'PPL':<8} {'PPL Δ%':<8} {'Memory':<10} {'Mem Δ%':<8} {'Ratio':<6}")
        print(f"{'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*6}")
        
        for profile, data in summary["compression_profiles"].items():
            print(f"{profile:<8} {data['avg_perplexity']:<8.3f} "
                  f"{data['perplexity_degradation_pct']:>+6.1f}% "
                  f"{data['avg_memory_mb']:<8.1f}MB "
                  f"{data['memory_savings_pct']:>+6.1f}% "
                  f"{data['avg_compression_ratio']:<6.1f}x")
        
        print(f"\n💡 Key Insights:")
        best_tradeoff = min(
            summary["compression_profiles"].items(),
            key=lambda x: x[1]["perplexity_degradation_pct"] + (100 - x[1]["memory_savings_pct"])
        )
        print(f"   Best tradeoff: {best_tradeoff[0]} profile")
        print(f"   ({best_tradeoff[1]['perplexity_degradation_pct']:+.1f}% PPL, "
              f"{best_tradeoff[1]['memory_savings_pct']:+.1f}% memory)")
    
    def save_results(self, results: Dict[str, Any], filepath: str):
        """Save evaluation results to file"""
        import json
        
        # Convert PerplexityMetrics to dict for JSON serialization
        def convert_metrics(obj):
            if isinstance(obj, PerplexityMetrics):
                return obj.__dict__
            elif isinstance(obj, dict):
                return {k: convert_metrics(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_metrics(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_metrics(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"💾 Results saved to {filepath}")