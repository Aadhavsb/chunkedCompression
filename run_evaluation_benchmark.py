#!/usr/bin/env python3
"""
Comprehensive LLaMA Compression Evaluation Script
Runs industry-standard benchmarks

Usage:
    python run_evaluation_benchmark.py --mode [quick|full|perplexity|zero-shot]
    python run_evaluation_benchmark.py --help
"""

import argparse
import sys
import os
from pathlib import Path

# Add current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive LLaMA compression evaluation benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick evaluation (reduced samples)
    python run_evaluation_benchmark.py --mode quick
    
    # Full comprehensive benchmark
    python run_evaluation_benchmark.py --mode full
    
    # Only perplexity evaluation
    python run_evaluation_benchmark.py --mode perplexity
    
    # Only zero-shot evaluation
    python run_evaluation_benchmark.py --mode zero-shot
    
    # Custom configuration
    python run_evaluation_benchmark.py --mode custom \\
        --datasets wikitext2 --compression low med \\
        --seq-lengths 2048 --max-samples 50
        """
    )
    
    # Main arguments
    parser.add_argument(
        "--mode", 
        choices=["quick", "full", "perplexity", "zero-shot", "custom"],
        default="quick",
        help="Evaluation mode (default: quick)"
    )
    
    parser.add_argument(
        "--results-dir",
        default="evaluation_results",
        help="Directory to save results (default: evaluation_results)"
    )
    
    # Custom configuration options
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["wikitext2", "c4", "ptb"],
        default=["wikitext2"],
        help="Datasets for perplexity evaluation"
    )
    
    parser.add_argument(
        "--compression",
        nargs="+",
        choices=["baseline", "low", "med", "high"],
        default=["baseline", "med"],
        help="Compression profiles to evaluate"
    )
    
    parser.add_argument(
        "--seq-lengths",
        nargs="+",
        type=int,
        default=[2048],
        help="Sequence lengths for evaluation"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum samples per configuration"
    )
    
    parser.add_argument(
        "--zero-shot-tasks",
        nargs="+",
        choices=["openbookqa", "hellaswag", "piqa", "arc_easy", "arc_challenge", "winogrande"],
        default=["hellaswag", "piqa"],
        help="Zero-shot tasks to evaluate"
    )
    
    parser.add_argument(
        "--zero-shot-limit",
        type=int,
        default=50,
        help="Sample limit per zero-shot task"
    )
    
    # System options
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use (default: cuda)"
    )
    
    parser.add_argument(
        "--model-path",
        help="Override model path (default: from environment/config)"
    )
    
    parser.add_argument(
        "--no-compression-setup",
        action="store_true",
        help="Skip compression profile setup (baseline only)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("🚀 LLaMA Compression Evaluation Benchmark")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Results directory: {args.results_dir}")
    print(f"Device: {args.device}")
    
    if args.debug:
        print(f"Debug mode enabled")
        print(f"Arguments: {vars(args)}")
    
    try:
        # Import after path setup
        from core.model import LLaMAModelLoader
        from core.config import ModelConfig
        from core.evaluation import BenchmarkRunner
        
        print("\n📦 Initializing model and benchmark runner...")
        
        # Initialize model configuration
        if args.model_path:
            model_config = ModelConfig(
                model_path=args.model_path,
                device=args.device
            )
        else:
            model_config = ModelConfig.from_env()
            if args.device != "cuda":
                model_config.device = args.device
        
        # Initialize model loader
        print(f"🔧 Loading model from: {model_config.model_path}")
        model_loader = LLaMAModelLoader(model_config)
        model_loader.load_model()
        
        print(f"✅ Model loaded: {model_loader.get_model_info()['model_name']}")
        
        # Initialize benchmark runner
        benchmark_runner = BenchmarkRunner(model_loader, args.results_dir)
        
        # Run evaluation based on mode
        if args.mode == "quick":
            print(f"\n🏃 Running quick evaluation...")
            results = benchmark_runner.run_comprehensive_benchmark(quick_mode=True)
            
        elif args.mode == "full":
            print(f"\n🔍 Running full comprehensive benchmark...")
            results = benchmark_runner.run_comprehensive_benchmark(quick_mode=False)
            
        elif args.mode == "perplexity":
            print(f"\n📊 Running perplexity evaluation...")
            results = benchmark_runner.run_perplexity_benchmark(
                datasets=args.datasets,
                compression_profiles=args.compression,
                sequence_lengths=args.seq_lengths,
                max_samples=args.max_samples
            )
            
        elif args.mode == "zero-shot":
            print(f"\n🎯 Running zero-shot evaluation...")
            results = benchmark_runner.run_zero_shot_benchmark(
                tasks=args.zero_shot_tasks,
                compression_profiles=args.compression,
                limit=args.zero_shot_limit
            )
            
        elif args.mode == "custom":
            print(f"\n⚙️ Running custom evaluation...")
            
            # Run perplexity if datasets specified
            perplexity_results = None
            if args.datasets:
                perplexity_results = benchmark_runner.run_perplexity_benchmark(
                    datasets=args.datasets,
                    compression_profiles=args.compression,
                    sequence_lengths=args.seq_lengths,
                    max_samples=args.max_samples
                )
            
            # Run zero-shot if tasks specified
            zero_shot_results = None
            if args.zero_shot_tasks:
                zero_shot_results = benchmark_runner.run_zero_shot_benchmark(
                    tasks=args.zero_shot_tasks,
                    compression_profiles=args.compression,
                    limit=args.zero_shot_limit
                )
            
            results = {
                "mode": "custom",
                "perplexity": perplexity_results,
                "zero_shot": zero_shot_results
            }
        
        print(f"\n🎉 Evaluation complete!")
        print(f"📊 Results summary available in: {args.results_dir}")
        
        # Show status
        status = benchmark_runner.get_benchmark_status()
        print(f"\n📋 Recent results files:")
        for result in status["recent_results"][:3]:
            print(f"   - {result['file']} ({result['type']})")
        
        return 0
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print(f"💡 Make sure to install dependencies: pip install -r requirements.txt")
        return 1
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())