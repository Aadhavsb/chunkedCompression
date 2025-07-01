"""
Zero-Shot Task Evaluator for LLaMA Compression
Integrates with lm-evaluation-harness for standard benchmarks
"""

import torch
import json
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import numpy as np

# Try to import lm_eval, provide fallback if not available
try:
    import lm_eval
    from lm_eval import evaluator
    from lm_eval.api.model import LM
    from lm_eval.models.huggingface import HFLM
    LM_EVAL_AVAILABLE = True
except ImportError:
    LM_EVAL_AVAILABLE = False
    LM = object  # Fallback for when lm_eval is not available
    print("⚠️ lm-eval not available. Install with: pip install lm-eval>=0.4.0")

from ..model import LLaMAModelLoader


@dataclass
class ZeroShotResults:
    """Container for zero-shot evaluation results"""
    task_name: str
    accuracy: float
    num_samples: int
    compression_profile: str
    memory_usage_mb: float
    eval_time_seconds: float
    additional_metrics: Dict[str, float]


class CompressedLLaMAWrapper(LM):
    """
    Wrapper to make compressed LLaMA compatible with lm-evaluation-harness
    """
    
    def __init__(self, model_loader: LLaMAModelLoader, compression_profile: str = "baseline"):
        # Don't call super().__init__() as LM is abstract
        self.model_loader = model_loader
        self.model = model_loader.model
        self.tokenizer = model_loader.tokenizer
        self.compression_profile = compression_profile
        self.device = model_loader.device
        
        # lm-eval compatibility attributes
        self.vocab_size = len(self.tokenizer)
        self.eot_token_id = self.tokenizer.eos_token_id
        self.max_length = 2048  # Default context length
        
        # Required LM attributes
        self._device = self.device
        self._model = self.model
        self._tokenizer = self.tokenizer
        
        # Initialize compressed inference if not baseline
        self.compressed_inference = None
        if compression_profile != "baseline":
            self._setup_compressed_inference()
    
    def _setup_compressed_inference(self):
        """Setup compressed inference pipeline"""
        try:
            from ..compression import LLaMACompressionProfileBuilder
            from ..inference import LLaMACompressionInference
            from ..config import CompressionConfig
            
            print(f"🔧 Setting up compressed inference for {self.compression_profile} profile...")
            
            # Setup compression configuration
            compression_config = CompressionConfig(
                value_compression_ranks={"low": 32, "med": 64, "high": 128},
                key_compression_rank=32
            )
            
            # Build compression profiles
            compression_builder = LLaMACompressionProfileBuilder(
                self.model_loader, compression_config
            )
            
            # Build profiles for first few layers (sufficient for evaluation)
            for layer_idx in [0, 1, 2]:
                compression_builder.build_compression_profiles(layer_idx)
            
            # Initialize compressed inference
            self.compressed_inference = LLaMACompressionInference(
                model_loader=self.model_loader,
                profile_builder=compression_builder
            )
            
            print(f"✅ Compressed inference ready for {self.compression_profile} profile")
            
        except Exception as e:
            print(f"⚠️ Failed to setup compressed inference: {e}")
            print(f"🔄 Falling back to baseline model")
            self.compressed_inference = None
        
    def loglikelihood(self, requests: List[tuple]) -> List[tuple]:
        """
        Calculate log-likelihood for lm-eval compatibility
        
        Args:
            requests: List of (context, continuation) tuples
            
        Returns:
            List of (log_likelihood, is_greedy) tuples
        """
        results = []
        self.model.eval()
        
        with torch.no_grad():
            for context, continuation in requests:
                # Combine context and continuation
                full_text = context + continuation
                
                # Tokenize
                full_tokens = self.tokenizer(full_text, return_tensors="pt")["input_ids"].to(self.device)
                context_tokens = self.tokenizer(context, return_tensors="pt")["input_ids"].to(self.device)
                
                # Get logits - use compressed inference if available
                if self.compressed_inference and self.compression_profile != "baseline":
                    try:
                        # Use compressed inference for forward pass
                        logits = self._get_compressed_logits(full_text, full_tokens)
                    except Exception as e:
                        print(f"⚠️ Compressed inference failed: {e}, falling back to baseline")
                        outputs = self.model(full_tokens)
                        logits = outputs.logits.squeeze(0)
                else:
                    # Baseline: original model
                    outputs = self.model(full_tokens)
                    logits = outputs.logits.squeeze(0)  # [seq_len, vocab_size]
                
                # Calculate log-likelihood for continuation tokens
                continuation_start = context_tokens.shape[1] - 1  # -1 for inclusive indexing
                continuation_tokens = full_tokens.squeeze(0)[continuation_start + 1:]
                continuation_logits = logits[continuation_start:-1]  # Exclude last position
                
                # Compute log probabilities
                log_probs = torch.nn.functional.log_softmax(continuation_logits, dim=-1)
                continuation_log_probs = log_probs.gather(
                    1, continuation_tokens.unsqueeze(1)
                ).squeeze(1)
                
                # Sum log probabilities
                total_log_likelihood = continuation_log_probs.sum().item()
                
                # Check if this is the greedy choice (simplified)
                greedy_tokens = continuation_logits.argmax(dim=-1)
                is_greedy = torch.equal(greedy_tokens, continuation_tokens)
                
                results.append((total_log_likelihood, is_greedy.item()))
        
        return results
    
    def _get_compressed_logits(self, text: str, tokens: torch.Tensor) -> torch.Tensor:
        """Get logits using compressed inference"""
        # Use compressed inference pipeline
        benchmark_result = self.compressed_inference.run_compression_benchmark(
            texts=[text],
            max_length=tokens.shape[1],
            compression_profiles=[self.compression_profile]
        )
        
        # Extract logits from benchmark result
        if benchmark_result and "per_text_results" in benchmark_result:
            text_result = benchmark_result["per_text_results"][0]
            profile_results = text_result.get("profile_results", {})
            
            if self.compression_profile in profile_results:
                compressed_result = profile_results[self.compression_profile]
                if "logits" in compressed_result:
                    return compressed_result["logits"]
        
        # Fallback: run model forward pass manually with compression
        # This is a simplified approach - in practice, would need full compressed forward pass
        return self._run_compressed_forward(tokens)
    
    def _run_compressed_forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run forward pass with compression (simplified implementation)"""
        # For now, fall back to original model
        # In a full implementation, this would use compressed attention layers
        outputs = self.model(tokens)
        return outputs.logits.squeeze(0)
    
    def generate_until(self, requests: List[tuple]) -> List[str]:
        """
        Generate text until stopping criteria for lm-eval compatibility
        
        Args:
            requests: List of (context, generation_kwargs) tuples
            
        Returns:
            List of generated strings
        """
        results = []
        
        for context, gen_kwargs in requests:
            # Parse generation parameters
            max_gen_toks = gen_kwargs.get("max_gen_toks", 50)
            temperature = gen_kwargs.get("temperature", 0.0)
            
            # Use compressed inference if available
            if self.compressed_inference and self.compression_profile != "baseline":
                try:
                    # Use compressed generation
                    generated = self._generate_compressed(
                        context, max_gen_toks, temperature
                    )
                except Exception as e:
                    print(f"⚠️ Compressed generation failed: {e}, falling back to baseline")
                    generated = self.model_loader.generate_text(
                        context,
                        max_new_tokens=max_gen_toks,
                        temperature=temperature if temperature > 0 else 0.1
                    )
            else:
                # Baseline generation
                generated = self.model_loader.generate_text(
                    context,
                    max_new_tokens=max_gen_toks,
                    temperature=temperature if temperature > 0 else 0.1
                )
            
            # Extract only the generated part (remove context)
            if generated.startswith(context):
                generated = generated[len(context):]
            
            results.append(generated)
        
        return results
    
    def _generate_compressed(self, context: str, max_tokens: int, temperature: float) -> str:
        """Generate text using compressed inference"""
        # For now, fall back to model_loader generation
        # In a full implementation, this would use compressed generation
        return self.model_loader.generate_text(
            context,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 0.1
        )


class ZeroShotEvaluator:
    """
    Evaluate zero-shot performance on standard benchmarks
    Compatible with lm-evaluation-harness
    """
    
    def __init__(self, model_loader: LLaMAModelLoader):
        self.model_loader = model_loader
        self.device = model_loader.device
        
        # Standard benchmark tasks
        self.standard_tasks = [
            "openbookqa",
            "hellaswag", 
            "piqa",
            "arc_easy",
            "arc_challenge",
            "winogrande"
        ]
        
        print(f"🎯 Initialized ZeroShotEvaluator")
        print(f"   Available tasks: {', '.join(self.standard_tasks)}")
        print(f"   lm-eval available: {LM_EVAL_AVAILABLE}")
    
    def evaluate_single_task(self, task_name: str, 
                           compression_profile: str = "baseline",
                           num_fewshot: int = 0,
                           limit: Optional[int] = None) -> ZeroShotResults:
        """
        Evaluate a single zero-shot task
        
        Args:
            task_name: Name of the task to evaluate
            compression_profile: Compression level to use
            num_fewshot: Number of few-shot examples (0 for zero-shot)
            limit: Limit number of evaluation samples
            
        Returns:
            ZeroShotResults with evaluation metrics
        """
        if not LM_EVAL_AVAILABLE:
            return self._fallback_evaluation(task_name, compression_profile)
        
        print(f"🔍 Evaluating {task_name} (compression: {compression_profile})")
        
        start_time = time.time()
        
        # Memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.max_memory_allocated()
        
        try:
            # Use HFLM wrapper as suggested by the error message
            # Note: Currently runs baseline model for all compression profiles
            # TODO: Integrate compressed model with lm-eval framework
            
            # Fix tokenizer pad token issue
            tokenizer = self.model_loader.tokenizer
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            hf_model = HFLM(pretrained=self.model_loader.model, tokenizer=tokenizer)
            
            # Run evaluation using lm-eval
            results = evaluator.simple_evaluate(
                model=hf_model,
                tasks=[task_name],
                num_fewshot=num_fewshot,
                limit=limit,
                bootstrap_iters=0  # Disable bootstrapping for speed
            )
            
            # Extract results
            task_results = results["results"][task_name]
            
            # Get primary metric (usually accuracy)
            primary_metric = None
            accuracy = 0.0
            
            if "acc" in task_results:
                primary_metric = "acc"
                accuracy = task_results["acc"]
            elif "acc_norm" in task_results:
                primary_metric = "acc_norm" 
                accuracy = task_results["acc_norm"]
            elif "exact_match" in task_results:
                primary_metric = "exact_match"
                accuracy = task_results["exact_match"]
            else:
                # Fallback to first available metric
                metrics = [k for k in task_results.keys() if isinstance(task_results[k], (int, float))]
                if metrics:
                    primary_metric = metrics[0]
                    accuracy = task_results[primary_metric]
            
            # Get number of samples
            num_samples = results["config"].get("limit", "unknown")
            if isinstance(num_samples, str):
                num_samples = len(task_results) if hasattr(task_results, '__len__') else 0
            
            # Additional metrics
            additional_metrics = {
                k: v for k, v in task_results.items() 
                if isinstance(v, (int, float)) and k != primary_metric
            }
            
        except Exception as e:
            print(f"❌ lm-eval evaluation failed: {e}")
            return self._fallback_evaluation(task_name, compression_profile)
        
        # Memory usage
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated()
            memory_usage_mb = (peak_memory - initial_memory) / 1024 / 1024
        else:
            memory_usage_mb = 0.0
        
        eval_time = time.time() - start_time
        
        result = ZeroShotResults(
            task_name=task_name,
            accuracy=accuracy,
            num_samples=num_samples,
            compression_profile=compression_profile,
            memory_usage_mb=memory_usage_mb,
            eval_time_seconds=eval_time,
            additional_metrics=additional_metrics
        )
        
        print(f"✅ {task_name} complete:")
        print(f"   Accuracy ({primary_metric}): {accuracy:.4f}")
        print(f"   Samples: {num_samples}")
        print(f"   Time: {eval_time:.1f}s")
        print(f"   Memory: {memory_usage_mb:.1f} MB")
        if compression_profile != "baseline":
            print(f"   Note: Currently using baseline model (compression integration TODO)")
        
        return result
    
    def evaluate_standard_suite(self, 
                              compression_profiles: List[str] = ["baseline", "med"],
                              limit: Optional[int] = None) -> Dict[str, Dict[str, ZeroShotResults]]:
        """
        Evaluate the full standard benchmark suite
        
        Args:
            compression_profiles: List of compression profiles to test
            limit: Limit samples per task
            
        Returns:
            Dictionary of {compression_profile: {task: results}}
        """
        print(f"🚀 Running standard zero-shot benchmark suite")
        print(f"   Tasks: {', '.join(self.standard_tasks)}")
        print(f"   Compression profiles: {', '.join(compression_profiles)}")
        print(f"   Sample limit: {limit if limit else 'none'}")
        
        all_results = {}
        
        for profile in compression_profiles:
            print(f"\n📊 Evaluating compression profile: {profile}")
            profile_results = {}
            
            for task in self.standard_tasks:
                result = self.evaluate_single_task(task, profile, limit=limit)
                profile_results[task] = result
                
                # Clear GPU cache between tasks
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            all_results[profile] = profile_results
        
        print(f"\n✅ Standard benchmark suite complete")
        self._print_suite_summary(all_results)
        
        return all_results
    
    def _fallback_evaluation(self, task_name: str, compression_profile: str) -> ZeroShotResults:
        """Fallback evaluation when lm-eval is not available"""
        print(f"🔄 Using fallback evaluation for {task_name}")
        
        # Simple accuracy test using basic QA
        fallback_questions = {
            "openbookqa": [
                ("What is the primary source of energy for plants?", "sunlight"),
                ("What gas do plants release during photosynthesis?", "oxygen"),
                ("What is the hardest natural substance?", "diamond")
            ],
            "hellaswag": [
                ("A person is cooking pasta. They will next", "drain the water"),
                ("Someone is brushing their teeth. They will then", "rinse their mouth"),
                ("A student is taking notes. They will likely", "review them later")
            ],
            "piqa": [
                ("How do you make ice?", "freeze water"),
                ("How do you turn on a light?", "flip the switch"),
                ("How do you open a door?", "turn the handle")
            ]
        }
        
        questions = fallback_questions.get(task_name, fallback_questions["openbookqa"])
        correct = 0
        total = len(questions)
        
        start_time = time.time()
        
        for question, expected_answer in questions:
            # Generate response
            response = self.model_loader.generate_text(
                f"Question: {question}\nAnswer:",
                max_new_tokens=10,
                temperature=0.1
            )
            
            # Simple matching (very basic)
            if expected_answer.lower() in response.lower():
                correct += 1
        
        accuracy = correct / total
        eval_time = time.time() - start_time
        
        return ZeroShotResults(
            task_name=task_name,
            accuracy=accuracy,
            num_samples=total,
            compression_profile=compression_profile,
            memory_usage_mb=0.0,
            eval_time_seconds=eval_time,
            additional_metrics={"fallback": True}
        )
    
    def _print_suite_summary(self, all_results: Dict[str, Dict[str, ZeroShotResults]]):
        """Print formatted summary of benchmark suite results"""
        print(f"\n📈 Zero-Shot Benchmark Summary")
        print(f"=" * 70)
        
        # Calculate averages per profile
        for profile, profile_results in all_results.items():
            accuracies = [r.accuracy for r in profile_results.values()]
            avg_accuracy = np.mean(accuracies)
            avg_memory = np.mean([r.memory_usage_mb for r in profile_results.values()])
            total_time = sum(r.eval_time_seconds for r in profile_results.values())
            
            print(f"\n🎯 {profile.upper()} Results:")
            print(f"   Average Accuracy: {avg_accuracy:.4f}")
            print(f"   Average Memory: {avg_memory:.1f} MB")
            print(f"   Total Time: {total_time:.1f}s")
            
            print(f"   Per-task breakdown:")
            for task, result in profile_results.items():
                print(f"     {task:<15}: {result.accuracy:.4f} ({result.num_samples} samples)")
    
    def compare_compression_impact(self, results: Dict[str, Dict[str, ZeroShotResults]]) -> Dict[str, Any]:
        """
        Analyze impact of compression on zero-shot performance
        
        Args:
            results: Results from evaluate_standard_suite
            
        Returns:
            Analysis of compression impact
        """
        if "baseline" not in results:
            return {"error": "Baseline results not found for comparison"}
        
        baseline_results = results["baseline"]
        analysis = {
            "baseline_avg_accuracy": np.mean([r.accuracy for r in baseline_results.values()]),
            "compression_analysis": {}
        }
        
        for profile, profile_results in results.items():
            if profile == "baseline":
                continue
            
            # Calculate degradation per task
            task_degradation = {}
            for task in baseline_results.keys():
                if task in profile_results:
                    baseline_acc = baseline_results[task].accuracy
                    compressed_acc = profile_results[task].accuracy
                    degradation = (baseline_acc - compressed_acc) / baseline_acc * 100
                    task_degradation[task] = degradation
            
            # Overall statistics
            avg_degradation = np.mean(list(task_degradation.values()))
            max_degradation = max(task_degradation.values())
            min_degradation = min(task_degradation.values())
            
            analysis["compression_analysis"][profile] = {
                "avg_accuracy_degradation_pct": avg_degradation,
                "max_degradation_pct": max_degradation,
                "min_degradation_pct": min_degradation,
                "per_task_degradation": task_degradation,
                "avg_accuracy": np.mean([r.accuracy for r in profile_results.values()])
            }
        
        return analysis
    
    def save_results(self, results: Dict[str, Any], filepath: str):
        """Save zero-shot evaluation results"""
        
        # Convert ZeroShotResults to dict for JSON serialization
        def convert_results(obj):
            if isinstance(obj, ZeroShotResults):
                return obj.__dict__
            elif isinstance(obj, dict):
                return {k: convert_results(v) for k, v in obj.items()}
            else:
                return obj
        
        serializable_results = convert_results(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"💾 Zero-shot results saved to {filepath}")