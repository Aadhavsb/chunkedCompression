# LLaMA Compression Evaluation Guide

This guide covers the new industry-standard evaluation capabilities added to the chunkedCompression project.

- 🚀 **Real WikiText-2, C4, Penn Treebank datasets**
- 🚀 **lm-evaluation-harness integration** for zero-shot tasks
- 🚀 **Memory-perplexity tradeoff analysis**
- 🚀 **Multiple sequence lengths** (1024, 2048, 4096 tokens)
- 🚀 **Systematic perplexity evaluation**
- 🚀 **Standard benchmark tasks** (OpenBookQA, HellaSwag, PIQA, ARC, WinoGrande)

## 🛠️ Setup

### Install Evaluation Dependencies

```bash
# Option 1: Install evaluation dependencies only
pip install -e ".[evaluation]"

# Option 2: Install everything
pip install -e ".[all]"

# Option 3: Manual installation
pip install lm-eval[all]>=0.4.0 evaluate>=0.4.0
```

### Verify Installation

```bash
# Check if lm-eval is available
python -c "import lm_eval; print('lm-eval available')"

# Check datasets access
python -c "from datasets import load_dataset; print('datasets available')"
```

## 🚀 Quick Start

### Option 1: Command Line Interface

```bash
# Quick evaluation (reduced samples for testing)
python run_evaluation_benchmark.py --mode quick

# Full comprehensive benchmark
python run_evaluation_benchmark.py --mode full

# Only perplexity evaluation
python run_evaluation_benchmark.py --mode perplexity

# Only zero-shot evaluation  
python run_evaluation_benchmark.py --mode zero-shot

# Custom configuration
python run_evaluation_benchmark.py --mode custom \
    --datasets wikitext2 --compression baseline med \
    --seq-lengths 2048 --max-samples 50
```

### Option 2: Python API

```python
from core.model import LLaMAModelLoader
from core.config import ModelConfig
from core.evaluation import BenchmarkRunner

# Initialize model
model_config = ModelConfig.from_env()
model_loader = LLaMAModelLoader(model_config)
model_loader.load_model()

# Initialize benchmark runner
benchmark_runner = BenchmarkRunner(model_loader)

# Run comprehensive benchmark
results = benchmark_runner.run_comprehensive_benchmark(quick_mode=True)

# Or run specific evaluations
perplexity_results = benchmark_runner.run_perplexity_benchmark(
    datasets=["wikitext2"],
    compression_profiles=["baseline", "med"],
    sequence_lengths=[2048],
    max_samples=100
)

zero_shot_results = benchmark_runner.run_zero_shot_benchmark(
    tasks=["hellaswag", "piqa"],
    compression_profiles=["baseline", "med"],
    limit=50
)
```

## 📊 Evaluation Types

### 1. Perplexity Evaluation

**What it measures:** Language modeling quality on standard datasets

**Datasets:**
- **WikiText-2**: Primary benchmark (default)
- **C4**: Alternative large-scale dataset
- **Penn Treebank**: Classic language modeling benchmark

**Metrics:**
- Perplexity (lower is better)
- Cross-entropy loss
- Bits per byte
- Memory usage
- Compression ratio

**Configuration:**
```python
perplexity_results = benchmark_runner.run_perplexity_benchmark(
    datasets=["wikitext2", "c4"],           # Datasets to evaluate
    compression_profiles=["baseline", "low", "med", "high"],
    sequence_lengths=[1024, 2048, 4096],    # Test different context lengths  
    max_samples=100                         # Limit for faster evaluation
)
```

### 2. Zero-Shot Task Evaluation

**What it measures:** Performance on downstream tasks without fine-tuning

**Tasks (following lm-evaluation-harness):**
- **OpenBookQA**: Science question answering
- **HellaSwag**: Commonsense reasoning
- **PIQA**: Physical interaction QA
- **ARC Easy/Challenge**: AI2 Reasoning Challenge
- **WinoGrande**: Winograd schema challenge

**Metrics:**
- Accuracy (primary metric)
- Exact match (for some tasks)
- Task-specific metrics

**Configuration:**
```python
zero_shot_results = benchmark_runner.run_zero_shot_benchmark(
    tasks=["hellaswag", "piqa", "arc_easy"],
    compression_profiles=["baseline", "med"],
    limit=100  # Samples per task
)
```

### 3. Memory-Perplexity Tradeoff Analysis

**What it measures:** Compression efficiency vs quality tradeoff

**Analysis includes:**
- Perplexity degradation vs baseline
- Memory savings percentage
- Compression ratios
- Best tradeoff identification

**Example output:**
```
Memory-Perplexity Tradeoff Summary
Profile  PPL      PPL Δ%   Memory    Mem Δ%   Ratio
low      12.34   +2.1%    45.2MB    -65.3%   4.2x
med      12.56   +4.2%    32.1MB    -78.1%   8.1x  
high     13.21   +8.7%    18.9MB    -87.4%   15.3x
```

## 📁 Results Structure

Results are saved to `evaluation_results/` directory:

```
evaluation_results/
├── perplexity_benchmark_20241201_143022.json
├── zero_shot_benchmark_20241201_144531.json  
└── comprehensive_benchmark_20241201_150145.json
```

### Result File Structure

```json
{
  "benchmark_type": "comprehensive",
  "timestamp": "2024-12-01T15:01:45",
  "model_info": {
    "model_name": "Meta-Llama-3-8B-Instruct",
    "device": "cuda"
  },
  "results": {
    "perplexity": {
      "summary": {
        "baseline": {"avg_perplexity": 12.1},
        "compression_profiles": {
          "med": {
            "avg_perplexity": 12.5,
            "perplexity_degradation_pct": 3.3,
            "memory_savings_pct": 78.1
          }
        }
      }
    },
    "zero_shot": {
      "baseline": {
        "hellaswag": {"accuracy": 0.764, "num_samples": 100},
        "piqa": {"accuracy": 0.812, "num_samples": 100}
      },
      "med": {
        "hellaswag": {"accuracy": 0.751, "num_samples": 100},
        "piqa": {"accuracy": 0.798, "num_samples": 100}
      }
    }
  },
  "summary": {
    "key_findings": {
      "perplexity": {
        "best_compression_tradeoff": {
          "profile": "med",
          "perplexity_degradation_pct": 3.3,
          "memory_savings_pct": 78.1
        }
      }
    }
  }
}
```

## 🔧 Advanced Usage

### Custom Dataset Evaluation

```python
from core.evaluation import StandardDatasetHandler

# Initialize dataset handler
dataset_handler = StandardDatasetHandler(tokenizer)

# Load calibration data for compression
calibration_data = dataset_handler.load_calibration_data(
    dataset_name="wikitext2",
    num_samples=256,
    seq_len=2048
)

# Evaluate perplexity on custom data
for input_ids in dataset_handler.load_dataset_for_perplexity("c4", max_samples=50):
    # Your evaluation logic here
    pass
```

### Adding Custom Compression Profiles

```python
# Update compression configuration
from core.config import CompressionConfig

compression_config = CompressionConfig(
    value_compression_ranks={
        "aggressive": 16,    # Very high compression
        "standard": 64,      # Standard compression  
        "conservative": 128  # Light compression
    },
    key_compression_rank=32
)
```

### Integration with Existing Tests

The new evaluation system integrates with your existing test suite:

```python
# Your existing comprehensive test
from tests.integration.run_comprehensive_test import main as run_legacy_tests

# New evaluation benchmark
from run_evaluation_benchmark import main as run_evaluation_benchmark

# Run both
run_legacy_tests()        # Your existing model/compression tests
run_evaluation_benchmark()  # New industry-standard benchmarks
```

## 📈 Interpreting Results

### Perplexity Analysis
- **Lower perplexity = better language modeling**
- **Target: <5% degradation for practical deployment**
- **Memory savings vs quality tradeoff is key**

### Zero-Shot Analysis  
- **Accuracy degradation should be minimal (<2-3%)**
- **Some tasks more sensitive to compression than others**
- **Reasoning tasks (ARC, HellaSwag) typically most sensitive**

### Memory Analysis
- **Memory savings calculated during actual inference**
- **Compression ratios show theoretical improvements**
- **Real memory usage includes overhead**

## 🚨 Troubleshooting

### Common Issues

1. **lm-eval import errors**
   ```bash
   pip install lm-eval[all]>=0.4.0
   ```

2. **Dataset download failures**
   ```bash
   # Fallback samples are automatically used
   # Check internet connection for full datasets
   ```

3. **GPU memory issues**
   ```bash
   # Use quick mode or reduce samples
   python run_evaluation_benchmark.py --mode quick --max-samples 20
   ```

4. **Model path issues**
   ```bash
   # Override model path
   python run_evaluation_benchmark.py --model-path /your/model/path
   ```

### Debug Mode

```bash
python run_evaluation_benchmark.py --debug --mode quick
```

## 🎉 Summary

Your LLaMA compression project now has industry-standard evaluation capabilities:

✅ **Real dataset evaluation** (vs hardcoded samples)  
✅ **Standard benchmark integration** (lm-evaluation-harness)  
✅ **Memory-perplexity tradeoff analysis**  
✅ **Multiple evaluation metrics** (perplexity, accuracy, memory)  
✅ **Systematic sequence length testing**  
✅ **Comprehensive result tracking**  
