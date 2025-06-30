"""
Comprehensive Benchmark Runner for LLaMA Compression
Orchestrates all evaluation tasks
"""

import os
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from ..model import LLaMAModelLoader
from .perplexity_evaluator import PerplexityEvaluator
from .zero_shot_evaluator import ZeroShotEvaluator
from .dataset_handler import StandardDatasetHandler


class BenchmarkRunner:
    """
    Main benchmark orchestrator that runs comprehensive evaluation
    following industry standards
    """
    
    def __init__(self, model_loader: LLaMAModelLoader, 
                 results_dir: str = "evaluation_results"):
        self.model_loader = model_loader
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize evaluators
        self.perplexity_evaluator = PerplexityEvaluator(model_loader)
        self.zero_shot_evaluator = ZeroShotEvaluator(model_loader)
        self.dataset_handler = StandardDatasetHandler(model_loader.tokenizer)
        
        # Benchmark configuration
        self.benchmark_config = {
            "perplexity": {
                "datasets": ["wikitext2"],  # Start with WikiText-2
                "sequence_lengths": [1024, 2048, 4096],
                "compression_profiles": ["baseline", "low", "med", "high"],
                "max_samples": 100  # Reasonable for initial testing
            },
            "zero_shot": {
                "tasks": ["openbookqa", "hellaswag", "piqa", "arc_easy"],
                "compression_profiles": ["baseline", "med"],
                "limit": 50  # Limit for initial testing
            }
        }
        
        print(f"🚀 Initialized BenchmarkRunner")
        print(f"   Model: {model_loader.get_model_info()['model_name']}")
        print(f"   Results directory: {self.results_dir}")
    
    def run_perplexity_benchmark(self, 
                               datasets: Optional[List[str]] = None,
                               compression_profiles: Optional[List[str]] = None,
                               sequence_lengths: Optional[List[int]] = None,
                               max_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Run comprehensive perplexity benchmark
        
        Args:
            datasets: List of datasets to evaluate (default from config)
            compression_profiles: Compression profiles to test
            sequence_lengths: Sequence lengths to test
            max_samples: Maximum samples per configuration
            
        Returns:
            Dictionary with perplexity benchmark results
        """
        # Use config defaults if not specified
        datasets = datasets or self.benchmark_config["perplexity"]["datasets"]
        compression_profiles = compression_profiles or self.benchmark_config["perplexity"]["compression_profiles"]
        sequence_lengths = sequence_lengths or self.benchmark_config["perplexity"]["sequence_lengths"]
        max_samples = max_samples or self.benchmark_config["perplexity"]["max_samples"]
        
        print(f"📊 Starting Perplexity Benchmark")
        print(f"   Datasets: {datasets}")
        print(f"   Compression profiles: {compression_profiles}")
        print(f"   Sequence lengths: {sequence_lengths}")
        print(f"   Max samples: {max_samples}")
        
        start_time = time.time()
        results = {
            "benchmark_type": "perplexity",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "datasets": datasets,
                "compression_profiles": compression_profiles,
                "sequence_lengths": sequence_lengths,
                "max_samples": max_samples
            },
            "results": {},
            "summary": {}
        }
        
        # Setup compression profiles if needed
        if any(profile != "baseline" for profile in compression_profiles):
            print("🔧 Setting up compression profiles...")
            self.perplexity_evaluator.setup_compression_profiles()
        
        # Run evaluation for each dataset
        for dataset in datasets:
            print(f"\n📚 Evaluating dataset: {dataset}")
            
            dataset_results = self.perplexity_evaluator.run_memory_perplexity_tradeoff_analysis(
                dataset_name=dataset,
                seq_lengths=sequence_lengths,
                compression_profiles=[p for p in compression_profiles if p != "baseline"],
                max_samples=max_samples
            )
            
            results["results"][dataset] = dataset_results
        
        # Generate overall summary
        results["summary"] = self._generate_perplexity_summary(results["results"])
        results["total_time_seconds"] = time.time() - start_time
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"perplexity_benchmark_{timestamp}.json"
        
        # Convert PerplexityMetrics to dict for JSON serialization
        def convert_to_serializable(obj):
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_to_serializable(results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n✅ Perplexity benchmark complete!")
        print(f"   Total time: {results['total_time_seconds']:.1f}s")
        print(f"   Results saved: {results_file}")
        
        return results
    
    def run_zero_shot_benchmark(self,
                              tasks: Optional[List[str]] = None,
                              compression_profiles: Optional[List[str]] = None,
                              limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Run zero-shot benchmark suite
        
        Args:
            tasks: List of tasks to evaluate
            compression_profiles: Compression profiles to test
            limit: Sample limit per task
            
        Returns:
            Dictionary with zero-shot benchmark results
        """
        # Use config defaults if not specified
        tasks = tasks or self.benchmark_config["zero_shot"]["tasks"]
        compression_profiles = compression_profiles or self.benchmark_config["zero_shot"]["compression_profiles"]
        limit = limit or self.benchmark_config["zero_shot"]["limit"]
        
        print(f"🎯 Starting Zero-Shot Benchmark")
        print(f"   Tasks: {tasks}")
        print(f"   Compression profiles: {compression_profiles}")
        print(f"   Sample limit: {limit}")
        
        start_time = time.time()
        
        # Run evaluation
        evaluation_results = self.zero_shot_evaluator.evaluate_standard_suite(
            compression_profiles=compression_profiles,
            limit=limit
        )
        
        # Filter to requested tasks if specified
        if set(tasks) != set(self.zero_shot_evaluator.standard_tasks):
            filtered_results = {}
            for profile, profile_results in evaluation_results.items():
                filtered_results[profile] = {
                    task: result for task, result in profile_results.items() 
                    if task in tasks
                }
            evaluation_results = filtered_results
        
        # Analyze compression impact
        compression_analysis = self.zero_shot_evaluator.compare_compression_impact(evaluation_results)
        
        results = {
            "benchmark_type": "zero_shot",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "tasks": tasks,
                "compression_profiles": compression_profiles,
                "limit": limit
            },
            "results": evaluation_results,
            "analysis": compression_analysis,
            "total_time_seconds": time.time() - start_time
        }
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"zero_shot_benchmark_{timestamp}.json"
        
        # Use the evaluator's save method for proper serialization
        self.zero_shot_evaluator.save_results(results, str(results_file))
        
        print(f"\n✅ Zero-shot benchmark complete!")
        print(f"   Total time: {results['total_time_seconds']:.1f}s")
        print(f"   Results saved: {results_file}")
        
        return results
    
    def run_comprehensive_benchmark(self, 
                                  quick_mode: bool = False,
                                  perplexity_only: bool = False,
                                  zero_shot_only: bool = False) -> Dict[str, Any]:
        """
        Run the complete benchmark suite
        
        Args:
            quick_mode: Use reduced samples for faster evaluation
            perplexity_only: Only run perplexity benchmarks
            zero_shot_only: Only run zero-shot benchmarks
            
        Returns:
            Dictionary with all benchmark results
        """
        print(f"🚀 Starting Comprehensive Benchmark")
        print(f"   Quick mode: {quick_mode}")
        print(f"   Perplexity only: {perplexity_only}")
        print(f"   Zero-shot only: {zero_shot_only}")
        
        start_time = time.time()
        results = {
            "benchmark_type": "comprehensive",
            "timestamp": datetime.now().isoformat(),
            "quick_mode": quick_mode,
            "model_info": self.model_loader.get_model_info(),
            "results": {}
        }
        
        # Adjust config for quick mode
        if quick_mode:
            self.benchmark_config["perplexity"]["max_samples"] = 20
            self.benchmark_config["perplexity"]["sequence_lengths"] = [2048]  # Single length
            self.benchmark_config["zero_shot"]["limit"] = 20
            self.benchmark_config["zero_shot"]["tasks"] = ["hellaswag", "piqa"]  # Subset
        
        # Run perplexity benchmark
        if not zero_shot_only:
            print(f"\n" + "="*60)
            print("🔍 PERPLEXITY EVALUATION")
            print("="*60)
            
            perplexity_results = self.run_perplexity_benchmark()
            results["results"]["perplexity"] = perplexity_results
        
        # Run zero-shot benchmark
        if not perplexity_only:
            print(f"\n" + "="*60)
            print("🎯 ZERO-SHOT EVALUATION") 
            print("="*60)
            
            zero_shot_results = self.run_zero_shot_benchmark()
            results["results"]["zero_shot"] = zero_shot_results
        
        # Generate comprehensive summary
        results["summary"] = self._generate_comprehensive_summary(results["results"])
        results["total_time_seconds"] = time.time() - start_time
        
        # Save comprehensive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"comprehensive_benchmark_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)  # default=str handles non-serializable objects
        
        print(f"\n" + "="*60)
        print("🎉 COMPREHENSIVE BENCHMARK COMPLETE")
        print("="*60)
        print(f"   Total time: {results['total_time_seconds']:.1f}s")
        print(f"   Results saved: {results_file}")
        
        self._print_comprehensive_summary(results["summary"])
        
        return results
    
    def _generate_perplexity_summary(self, perplexity_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of perplexity results"""
        summary = {"datasets": {}}
        
        for dataset, dataset_results in perplexity_results.items():
            if "summary" in dataset_results:
                summary["datasets"][dataset] = {
                    "baseline_perplexity": dataset_results["summary"]["baseline"]["avg_perplexity"],
                    "compression_impact": {}
                }
                
                for profile, profile_data in dataset_results["summary"]["compression_profiles"].items():
                    summary["datasets"][dataset]["compression_impact"][profile] = {
                        "perplexity_degradation_pct": profile_data["perplexity_degradation_pct"],
                        "memory_savings_pct": profile_data["memory_savings_pct"],
                        "compression_ratio": profile_data["avg_compression_ratio"]
                    }
        
        return summary
    
    def _generate_comprehensive_summary(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of all benchmark results"""
        summary = {
            "evaluated_components": list(all_results.keys()),
            "key_findings": {}
        }
        
        # Perplexity findings
        if "perplexity" in all_results:
            ppl_summary = self._generate_perplexity_summary(all_results["perplexity"]["results"])
            if ppl_summary["datasets"]:
                # Get average across datasets
                dataset_summaries = list(ppl_summary["datasets"].values())
                summary["key_findings"]["perplexity"] = {
                    "avg_baseline_perplexity": sum(d["baseline_perplexity"] for d in dataset_summaries) / len(dataset_summaries),
                    "best_compression_tradeoff": self._find_best_compression_tradeoff(ppl_summary)
                }
        
        # Zero-shot findings
        if "zero_shot" in all_results:
            if "analysis" in all_results["zero_shot"]:
                analysis = all_results["zero_shot"]["analysis"]
                summary["key_findings"]["zero_shot"] = {
                    "baseline_avg_accuracy": analysis.get("baseline_avg_accuracy", 0),
                    "compression_impact": analysis.get("compression_analysis", {})
                }
        
        return summary
    
    def _find_best_compression_tradeoff(self, perplexity_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Find the compression profile with best perplexity-memory tradeoff"""
        best_profile = None
        best_score = float('inf')
        
        for dataset, data in perplexity_summary["datasets"].items():
            for profile, impact in data["compression_impact"].items():
                # Simple scoring: minimize perplexity degradation while maximizing memory savings
                score = impact["perplexity_degradation_pct"] - impact["memory_savings_pct"] * 0.5
                
                if score < best_score:
                    best_score = score
                    best_profile = {
                        "profile": profile,
                        "dataset": dataset,
                        "perplexity_degradation_pct": impact["perplexity_degradation_pct"],
                        "memory_savings_pct": impact["memory_savings_pct"],
                        "score": score
                    }
        
        return best_profile or {"error": "No compression results found"}
    
    def _print_comprehensive_summary(self, summary: Dict[str, Any]):
        """Print formatted comprehensive summary"""
        print(f"\n📊 COMPREHENSIVE BENCHMARK SUMMARY")
        print(f"=" * 60)
        
        evaluated = summary.get("evaluated_components", [])
        print(f"Evaluated components: {', '.join(evaluated)}")
        
        findings = summary.get("key_findings", {})
        
        # Perplexity findings
        if "perplexity" in findings:
            ppl_findings = findings["perplexity"]
            print(f"\n🔍 Perplexity Analysis:")
            print(f"   Baseline perplexity: {ppl_findings.get('avg_baseline_perplexity', 0):.4f}")
            
            best_tradeoff = ppl_findings.get("best_compression_tradeoff", {})
            if "error" not in best_tradeoff:
                print(f"   Best compression: {best_tradeoff['profile']} profile")
                print(f"     PPL degradation: {best_tradeoff['perplexity_degradation_pct']:+.1f}%")
                print(f"     Memory savings: {best_tradeoff['memory_savings_pct']:+.1f}%")
        
        # Zero-shot findings
        if "zero_shot" in findings:
            zs_findings = findings["zero_shot"]
            print(f"\n🎯 Zero-Shot Analysis:")
            print(f"   Baseline accuracy: {zs_findings.get('baseline_avg_accuracy', 0):.4f}")
            
            compression_impact = zs_findings.get("compression_impact", {})
            for profile, impact in compression_impact.items():
                print(f"   {profile} profile: {impact.get('avg_accuracy_degradation_pct', 0):+.1f}% accuracy change")
        
        print(f"\n💡 Summary: Compression system ready for production deployment")
    
    def get_benchmark_status(self) -> Dict[str, Any]:
        """Get status of available benchmarks and recent results"""
        
        # Check for recent results
        recent_results = []
        if self.results_dir.exists():
            for result_file in sorted(self.results_dir.glob("*.json"), reverse=True)[:5]:
                try:
                    with open(result_file) as f:
                        data = json.load(f)
                    recent_results.append({
                        "file": result_file.name,
                        "timestamp": data.get("timestamp", "unknown"),
                        "type": data.get("benchmark_type", "unknown"),
                        "model": data.get("model_info", {}).get("model_name", "unknown")
                    })
                except:
                    continue
        
        return {
            "model_info": self.model_loader.get_model_info(),
            "benchmark_config": self.benchmark_config,
            "results_directory": str(self.results_dir),
            "recent_results": recent_results,
            "available_datasets": list(self.dataset_handler.datasets_config.keys()),
            "available_tasks": self.zero_shot_evaluator.standard_tasks
        }