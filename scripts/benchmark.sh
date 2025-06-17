#!/bin/bash

# 📊 Comprehensive Benchmarking Script
#
# Runs comprehensive performance benchmarks for the LLaMA-3 8B Compression System
# with different configurations, profiles, and metrics collection.
#
# Prerequisites:
# - GPU environment setup
# - LLaMA-3 8B model available
# - Development environment configured
#
# Usage:
#   ./scripts/benchmark.sh [OPTIONS]
#
# Options:
#   --profiles PROFILES   Comma-separated compression profiles (default: low,med,high)
#   --layers LAYERS       Comma-separated layer indices (default: 0,15,31)
#   --iterations N        Number of benchmark iterations (default: 3)
#   --output DIR          Output directory for results (default: benchmark_results)
#   --monitor             Enable resource monitoring during benchmark
#   --compare             Compare with baseline (uncompressed)
#   --export FORMAT       Export format: json,csv,html (default: json)
#   --help                Show this help

set -e

# Default values
PROFILES="low,med,high"
LAYERS="0,15,31"
ITERATIONS=3
OUTPUT_DIR="benchmark_results"
ENABLE_MONITORING=False
COMPARE_BASELINE=False
EXPORT_FORMAT="json"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --profiles)
            PROFILES="$2"
            shift 2
            ;;
        --layers)
            LAYERS="$2"
            shift 2
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --monitor)
            ENABLE_MONITORING=true
            shift
            ;;
        --compare)
            COMPARE_BASELINE=true
            shift
            ;;
        --export)
            EXPORT_FORMAT="$2"
            shift 2
            ;;
        --help)
            echo "📊 Comprehensive Benchmarking Script for LLaMA-3 8B Compression System"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --profiles PROFILES   Comma-separated compression profiles (default: low,med,high)"
            echo "  --layers LAYERS       Comma-separated layer indices (default: 0,15,31)"
            echo "  --iterations N        Number of benchmark iterations (default: 3)"
            echo "  --output DIR          Output directory for results (default: benchmark_results)"
            echo "  --monitor             Enable resource monitoring during benchmark"
            echo "  --compare             Compare with baseline (uncompressed)"
            echo "  --export FORMAT       Export format: json,csv,html (default: json)"
            echo "  --help                Show this help"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Basic benchmark"
            echo "  $0 --profiles low,med --iterations 5 # Custom profiles and iterations"
            echo "  $0 --monitor --compare --export csv  # Full benchmark with monitoring"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate iterations
if ! [[ "$ITERATIONS" =~ ^[0-9]+$ ]] || [ "$ITERATIONS" -lt 1 ]; then
    echo "❌ Error: Iterations must be a positive integer"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_FILE="$OUTPUT_DIR/benchmark_$TIMESTAMP.$EXPORT_FORMAT"

echo "📊 LLaMA-3 8B Compression System Benchmark"
echo "==========================================="
echo "📍 Host: $(hostname)"
echo "📅 Start: $(date)"
echo "⚙️  Profiles: $PROFILES"
echo "🔢 Layers: $LAYERS"
echo "🔄 Iterations: $ITERATIONS"
echo "📁 Output: $RESULT_FILE"
echo ""

# Check prerequisites
echo "🔍 Checking prerequisites..."
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "❌ Error: CUDA not available"
    exit 1
fi

if ! python3 -c "from core.model import LLaMAModelLoader" 2>/dev/null; then
    echo "❌ Error: Core modules not available. Run: pip install -e ."
    exit 1
fi

model_path="/mnt/vstor/CSE_ECSE_GXD234/Meta-Llama-3-8B-Instruct"
if [ ! -d "$model_path" ]; then
    echo "❌ Error: LLaMA-3 8B model not found at $model_path"
    exit 1
fi
echo "✓ Prerequisites checked"

# Start resource monitoring if enabled
MONITOR_PID=""
if [ "$ENABLE_MONITORING" = true ]; then
    echo ""
    echo "📊 Starting resource monitoring..."
    ./scripts/monitor_resources.sh --log "$OUTPUT_DIR/monitor_$TIMESTAMP.log" --quiet &
    MONITOR_PID=$!
    echo "✓ Monitoring started (PID: $MONITOR_PID)"
fi

# Function to stop monitoring
stop_monitoring() {
    if [ ! -z "$MONITOR_PID" ]; then
        kill $MONITOR_PID 2>/dev/null || true
        echo "✓ Resource monitoring stopped"
    fi
}
trap stop_monitoring EXIT

# Run benchmark
echo ""
echo "🚀 Starting benchmark execution..."

# Create Python benchmark script
cat > "$OUTPUT_DIR/run_benchmark_$TIMESTAMP.py" << EOF
#!/usr/bin/env python3
"""
Generated benchmark script for LLaMA-3 8B Compression System
Generated at: $(date)
"""
import json
import time
import torch
from typing import Dict, List, Any
from core.model import LLaMAModelLoader
from core.config import ModelConfig, CompressionConfig
from core.compression import LLaMACompressionProfileBuilder
from core.inference import LLaMACompressionInference

def run_benchmark():
    """Run comprehensive benchmark"""
    print("🔧 Initializing benchmark...")
    
    # Configuration
    profiles = "$PROFILES".split(',')
    layers = [int(x) for x in "$LAYERS".split(',')]
    iterations = $ITERATIONS
    compare_baseline = $COMPARE_BASELINE
    
    # Results storage
    results = {
        'timestamp': '$(date -Iseconds)',
        'host': '$(hostname)',
        'profiles': profiles,
        'layers': layers,
        'iterations': iterations,
        'gpu_info': {},
        'benchmark_results': {}
    }
    
    # Get GPU info
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        results['gpu_info'] = {
            'name': props.name,
            'total_memory_gb': props.total_memory / (1024**3),
            'cuda_version': torch.version.cuda
        }
    
    # Initialize model
    print("📥 Loading LLaMA-3 8B model...")
    model_config = ModelConfig.from_env()
    model_loader = LLaMAModelLoader(model_config)
    model_loader.load_model()
    
    # Run benchmarks for each profile and layer
    for profile in profiles:
        print(f"\\n🧪 Testing profile: {profile}")
        results['benchmark_results'][profile] = {}
        
        for layer_idx in layers:
            print(f"  📊 Layer {layer_idx}...")
            layer_results = []
            
            for iteration in range(iterations):
                print(f"    🔄 Iteration {iteration + 1}/{iterations}")
                
                # Setup compression
                compression_config = CompressionConfig()
                profile_builder = LLaMACompressionProfileBuilder(
                    model_loader, compression_config
                )
                profile_builder.build_compression_profiles(layer_idx=layer_idx)
                
                # Run inference
                inference = LLaMACompressionInference(model_loader, profile_builder)
                
                start_time = time.time()
                benchmark_results = inference.run_compression_benchmark()
                end_time = time.time()
                
                # Extract metrics
                metrics = benchmark_results['aggregate_metrics']
                iteration_result = {
                    'iteration': iteration + 1,
                    'total_time': end_time - start_time,
                    'memory_savings': metrics['avg_memory_savings'],
                    'cosine_similarity': metrics['avg_cosine_similarity'],
                    'mse': metrics['avg_mse'],
                    'compression_ratio': metrics.get('avg_compression_ratio', 0),
                }
                
                layer_results.append(iteration_result)
                
                # Clear caches
                inference.clear_all_caches()
                torch.cuda.empty_cache()
            
            # Calculate averages
            avg_results = {}
            for key in ['total_time', 'memory_savings', 'cosine_similarity', 'mse', 'compression_ratio']:
                values = [r[key] for r in layer_results]
                avg_results[f'avg_{key}'] = sum(values) / len(values)
                avg_results[f'std_{key}'] = (sum((x - avg_results[f'avg_{key}'])**2 for x in values) / len(values))**0.5
            
            results['benchmark_results'][profile][f'layer_{layer_idx}'] = {
                'iterations': layer_results,
                'averages': avg_results
            }
    
    # Save results
    output_file = "$RESULT_FILE"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\\n💾 Results saved to: {output_file}")
    
    # Print summary
    print("\\n📊 Benchmark Summary:")
    print("=====================")
    for profile in profiles:
        print(f"\\n🔸 Profile: {profile}")
        for layer_idx in layers:
            layer_key = f'layer_{layer_idx}'
            if layer_key in results['benchmark_results'][profile]:
                avg = results['benchmark_results'][profile][layer_key]['averages']
                print(f"  Layer {layer_idx}: {avg['avg_memory_savings']:.2%} memory savings, {avg['avg_cosine_similarity']:.4f} similarity")

if __name__ == "__main__":
    run_benchmark()
EOF

# Make script executable and run it
chmod +x "$OUTPUT_DIR/run_benchmark_$TIMESTAMP.py"
python3 "$OUTPUT_DIR/run_benchmark_$TIMESTAMP.py"

# Stop monitoring
stop_monitoring

# Generate report
echo ""
echo "📋 Generating benchmark report..."

# Create summary report
cat > "$OUTPUT_DIR/summary_$TIMESTAMP.md" << EOF
# LLaMA-3 8B Compression Benchmark Report

**Generated:** $(date)  
**Host:** $(hostname)  
**Profiles:** $PROFILES  
**Layers:** $LAYERS  
**Iterations:** $ITERATIONS  

## Configuration
- GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
- CUDA: $(python3 -c "import torch; print(torch.version.cuda)")
- PyTorch: $(python3 -c "import torch; print(torch.__version__)")

## Results
Results saved to: \`$RESULT_FILE\`

EOF

if [ "$ENABLE_MONITORING" = true ]; then
    echo "Monitor log: \`$OUTPUT_DIR/monitor_$TIMESTAMP.log\`" >> "$OUTPUT_DIR/summary_$TIMESTAMP.md"
fi

echo "✓ Summary report: $OUTPUT_DIR/summary_$TIMESTAMP.md"

# Export to different formats if requested
if [ "$EXPORT_FORMAT" = "csv" ]; then
    echo "📊 Converting to CSV format..."
    python3 -c "
import json
import csv

with open('$RESULT_FILE', 'r') as f:
    data = json.load(f)

# Extract flat data for CSV
rows = []
for profile, profile_data in data['benchmark_results'].items():
    for layer, layer_data in profile_data.items():
        avg = layer_data['averages']
        rows.append({
            'profile': profile,
            'layer': layer,
            'avg_memory_savings': avg['avg_memory_savings'],
            'avg_cosine_similarity': avg['avg_cosine_similarity'],
            'avg_mse': avg['avg_mse'],
            'avg_total_time': avg['avg_total_time']
        })

csv_file = '$OUTPUT_DIR/benchmark_$TIMESTAMP.csv'
with open(csv_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f'CSV exported to: {csv_file}')
"
fi

echo ""
echo "🎉 Benchmark Complete!"
echo "====================="
echo "📁 Output directory: $OUTPUT_DIR"
echo "📊 Results file: $RESULT_FILE"
echo "📋 Summary: $OUTPUT_DIR/summary_$TIMESTAMP.md"
if [ "$ENABLE_MONITORING" = true ]; then
    echo "📈 Monitor log: $OUTPUT_DIR/monitor_$TIMESTAMP.log"
fi
echo ""
echo "📋 Next steps:"
echo "  • Review results: cat $RESULT_FILE"
echo "  • Analyze performance: python -c \"import json; print(json.load(open('$RESULT_FILE')))\""
echo "  • Compare configurations: diff between benchmark files"