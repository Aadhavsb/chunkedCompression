# CLAUDE.md - LLaMA-3 8B Chunked Compression System

## 🚀 Project Overview

This is a **production-grade transformer attention compression research project** that implements chunked KV cache compression for the LLaMA-3 8B Instruct model. The system uses **real model weights** (no synthetic data) and performs SVD-based compression on attention mechanisms to reduce memory usage while maintaining model performance.

### 🔑 Key Features
- **Real LLaMA-3 8B Integration**: Uses actual Meta-Llama-3-8B-Instruct model weights
- **SVD-based Compression**: Mathematical guarantees with three compression levels (low/med/high)
- **Production KV Caching**: Memory-optimized storage with compressed representations
- **Fused Output Projections**: Direct vocabulary decoding from compressed states
- **Modular Architecture**: Clean, extensible design with proper interfaces
- **Comprehensive Testing**: Real data testing with performance benchmarking

## 🏗️ Modern Architecture

### **Core Module Structure**

```
core/
├── 📁 config/                      # Configuration Management
│   ├── model_config.py             # Model loading configuration
│   ├── compression_config.py       # Compression parameters
│   ├── inference_config.py         # Inference and benchmarking
│   └── __init__.py
├── 📁 interfaces/                  # Abstract Base Classes
│   ├── model_interface.py          # Model loading interfaces
│   ├── compression_interface.py    # Compression interfaces
│   ├── cache_interface.py          # KV cache interfaces
│   ├── inference_interface.py      # Inference interfaces
│   ├── data_interface.py           # Data handling interfaces
│   └── __init__.py
├── 📁 model/                       # Model Loading Components
│   ├── model_loader.py             # Modern LLaMAModelLoader
│   ├── model_config_wrapper.py     # Config abstraction
│   ├── llama_loader.py             # Cluster loader utilities
│   └── __init__.py
├── 📁 compression/                 # Compression Algorithms
│   ├── compression_algorithms.py   # SVD algorithms
│   ├── profile_builder.py          # Profile builder
│   ├── legacy_wrapper.py           # Backward compatibility
│   └── __init__.py
├── 📁 cache/                       # KV Cache Implementations
│   ├── kv_cache_llama.py           # Compressed cache
│   ├── standard_kv_cache.py        # Standard cache baseline
│   └── __init__.py
├── 📁 inference/                   # Inference Pipeline
│   ├── llama_inference.py          # Main inference engine
│   ├── llama_full_forward.py       # Forward pass utilities
│   ├── compressed_autoregressive_decoder.py # Decoder
│   └── __init__.py
├── 📁 data/                        # Data Handling
│   ├── dataset_llama.py            # Dataset utilities
│   └── __init__.py
├── 📁 utils/                       # Utility Functions
│   ├── memory_manager.py           # Memory management
│   └── __init__.py
└── __init__.py                     # Core exports
```

### **Component Details**

#### **1. Model Management** (`core/model/`)
- **LLaMAModelLoader**: Loads real LLaMA-3 8B Instruct from cluster paths
- **ModelConfigWrapper**: Unified interface for model configuration
- **LLaMA3Loader**: Memory-safe cluster loading with bfloat16 support
- **Path**: `/mnt/vstor/CSE_ECSE_GXD234/Meta-Llama-3-8B-Instruct`

#### **2. Compression System** (`core/compression/`)
- **SVDCompressionAlgorithm**: Mathematical SVD compression foundation
- **LLaMACompressionProfileBuilder**: Profile creation from real weights
- **Compression levels**: 32, 64, 128 ranks (aggressive optimization)
- **Fixed key compression**: Rank 32 for memory efficiency

#### **3. Caching Layer** (`core/cache/`)
- **LLaMAKVCache**: Compressed KV storage with metadata tracking
- **StandardKVCache**: Baseline for performance comparison
- **Memory optimization**: Automatic cleanup and monitoring

#### **4. Inference Pipeline** (`core/inference/`)
- **LLaMACompressionInference**: End-to-end compression workflow
- **Attention computation**: Compressed K/V with quality metrics
- **Benchmarking**: Performance analysis and quality validation

#### **5. Configuration Management** (`core/config/`)
- **Environment-based**: Load settings from environment variables
- **Validation**: Automatic validation with sensible defaults
- **Flexibility**: Support multiple deployment scenarios

## 🛠️ Development Setup

### **Prerequisites**
- **GPU Memory**: 16GB+ VRAM for LLaMA-3 8B
- **RAM**: 32GB+ system memory recommended
- **CUDA**: Compatible GPU with CUDA 11.8+
- **Storage**: ~30GB for model weights

### **🐳 Automated Cluster Setup (Recommended)**

The project includes automation scripts for SLURM cluster environments with Singularity:

```bash
# 1. Initial setup (one-time)
./scripts/setup.sh
# - Loads Singularity module
# - Pulls PyTorch container from Docker Hub
# - Builds writable sandbox environment

# 2. Allocate GPU resources
./scripts/run.sh
# - Requests: 2 GPUs, 24 cores, 24GB memory, 2 hours
# - Returns node assignment (e.g., gpu-node-123)

# 3. SSH to allocated node and navigate to project
ssh gpu-node-123
cd chunkedCompression

# 4. Start container with GPU support
./scripts/start_container.sh
# - Loads Singularity module on compute node
# - Starts container with --nv (NVIDIA support)
# - Drops you into interactive shell

# 5. Install dependencies inside container
pip install -r requirements.txt
# OR for modern development setup:
pip install -e ".[dev]"        # Development tools + testing
```

### **🛠️ Development & Utility Scripts**

Additional scripts for development workflow:

```bash
# Development environment setup
./scripts/dev_setup.sh          # Install dev dependencies, pre-commit hooks

# Testing and quality
./scripts/run_tests.sh           # Flexible test runner with options
./scripts/check_gpu.sh           # GPU environment validation

# Performance and monitoring  
./scripts/monitor_resources.sh   # Real-time resource monitoring
./scripts/benchmark.sh           # Comprehensive performance benchmarking

# Maintenance
./scripts/cleanup.sh             # Clean temporary files and artifacts
```

### **⚙️ Manual Installation**
```bash
# Install dependencies
pip install -r requirements.txt

# Ensure LLaMA-3 8B Instruct model is available at:
# /mnt/vstor/CSE_ECSE_GXD234/Meta-Llama-3-8B-Instruct
```

### **📦 Modern Package Management**

The project uses `pyproject.toml` for modern Python packaging and development:

```bash
# Development installation (recommended)
pip install -e ".[dev]"        # Development tools + testing
pip install -e ".[research]"   # Full research environment  
pip install -e ".[all]"        # Everything included

# Traditional approach (container/CI)
pip install -r requirements.txt
```

**Key features:**
- **Optional dependencies**: Different profiles for different use cases
- **CLI commands**: `llama-benchmark`, `llama-compress`
- **Tool configuration**: pytest, black, mypy, coverage
- **Build system**: Modern setuptools with version management

### **🔑 Core Dependencies**
```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.0.0
numpy>=1.21.0
accelerate>=0.20.0
safetensors>=0.3.0
```

## 💻 Development Commands

### **Two Evaluation Systems**

The project now has **two complementary evaluation approaches**:

1. **`llama-benchmark`**: Model validation and integration testing (your existing system)
2. **`llama-evaluate`**: Industry-standard benchmarking following Palu methodology (new)

### **🧪 Model Validation Tests**
```bash
# Comprehensive test suite (model validation)
python tests/integration/run_comprehensive_test.py
# OR
llama-benchmark

# Individual test categories
python -m pytest tests/unit/               # Unit tests
python -m pytest tests/integration/        # Integration tests

# Specific test files
python tests/unit/test_llama_compression.py        # Core compression tests
python tests/unit/test_kv_cache_comparison.py      # Cache performance
python tests/unit/test_real_kv_comparison.py       # Real vs compressed
python tests/unit/test_baseline_vs_compressed.py   # Baseline comparison
python tests/integration/test_refactored_imports.py # Import verification

# Token generation testing
python tests/integration/test_token_generation.py
```

### **📊 Industry-Standard Evaluation**
```bash
# Quick evaluation (reduced samples for testing)
python run_evaluation_benchmark.py --mode quick
# OR
llama-evaluate --mode quick

# Full comprehensive benchmark (like Palu research)
llama-evaluate --mode full

# Perplexity evaluation with real datasets
llama-evaluate --mode perplexity \
    --datasets wikitext2 c4 ptb \
    --compression baseline low med high \
    --seq-lengths 1024 2048 4096 \
    --max-samples 100

# Zero-shot evaluation with standard tasks
llama-evaluate --mode zero-shot \
    --zero-shot-tasks openbookqa hellaswag piqa arc_easy arc_challenge winogrande \
    --compression baseline med \
    --zero-shot-limit 100

# Memory-perplexity tradeoff analysis (like Palu)
llama-evaluate --mode perplexity --compression baseline low med high \
    --seq-lengths 1024 2048 4096 --max-samples 50

# Custom evaluation configuration
llama-evaluate --mode custom \
    --datasets wikitext2 --compression baseline med \
    --seq-lengths 2048 --max-samples 50 \
    --zero-shot-tasks hellaswag piqa --zero-shot-limit 50
```

### **🚀 Usage Examples**

#### **Modern API (Recommended)**
```python
from core.model import LLaMAModelLoader
from core.config import ModelConfig, CompressionConfig
from core.compression import LLaMACompressionProfileBuilder

# Environment-based configuration
model_config = ModelConfig.from_env()
compression_config = CompressionConfig(
    value_compression_ranks={"low": 32, "med": 64, "high": 128},
    key_compression_rank=32
)

# Initialize pipeline
model_loader = LLaMAModelLoader(model_config)
model_loader.load_model()

# Build compression profiles
profile_builder = LLaMACompressionProfileBuilder(
    model_loader, compression_config
)
profile_builder.build_compression_profiles(layer_idx=0)

# Use compression
compressed_values = profile_builder.compress_values_with_profile(
    values, "med", head_idx=0
)
```

#### **Legacy API (Backward Compatible)**
```python
# Existing code continues to work unchanged
from profiles_llama_new import LLaMACompressionProfiles

# Initialize pipeline (loads real LLaMA-3 8B model)
model_loader = None  # Use legacy model loader
profiles = LLaMACompressionProfiles(model_loader)

# Run compression benchmark
compressed_values = profiles.compress_values(values, "med", head_idx=0)
```

#### **Memory Management**
```python
from core.utils import MemoryManager

memory_manager = MemoryManager(cleanup_threshold=0.8)
with memory_manager.managed_computation():
    # Memory-intensive operations with automatic cleanup
    result = expensive_computation()
    
# Get memory statistics
stats = memory_manager.get_memory_usage()
print(memory_manager.get_memory_summary())
```

## 📊 Compression Technology

### **Mathematical Foundation**

**Value Compression:**
- `W_V = U @ S @ V^T` (SVD decomposition)
- `A_V = S_trunc @ V_trunc^T` (compression matrix)
- `W_fused = W_LM_HEAD @ U_trunc` (fused projection)

**Key Compression:**
- `W_K = U_K @ S_K @ V_K^T`
- `A_K = S_K_trunc @ V_K_trunc^T` (compression)
- `B_K = V_K_trunc` (reconstruction)

### **Compression Profiles**
| Profile | Value Rank | Key Rank | Typical Compression Ratio | Memory Savings |
|---------|-----------|----------|---------------------------|----------------|
| **Low** | 32 | 32 | ~15x | 93% |
| **Med** | 64 | 32 | ~8x | 87% |
| **High** | 128 | 32 | ~4x | 75% |

## 🧪 Testing Strategy

The project includes a comprehensive test suite with proper organization:

### **Test Hierarchy**
```
tests/
├── unit/           # Component-level tests
├── integration/    # End-to-end workflow tests
└── results/        # Historical test results (JSON files)
```

### **Test Coverage**
1. **Model Loading**: Verifies real LLaMA-3 8B loading and basic inference
2. **Compression Profiles**: Validates SVD matrices and shape correctness
3. **Hidden States Processing**: Tests real transformer output processing
4. **KV Cache Operations**: Verifies compressed storage and retrieval
5. **End-to-End Pipeline**: Complete compression workflow validation
6. **Memory Management**: Resource cleanup and monitoring tests
7. **Configuration**: Environment-based setup validation

### **Performance Metrics Tracked**
- **Quality**: Output MSE, cosine similarity, perplexity
- **Efficiency**: Memory savings, compression ratios, timing
- **Cache Performance**: Hit rates, reconstruction overhead
- **Model Accuracy**: Ground truth comparison, token prediction

## ⚙️ Configuration

### **Environment Variables**
```bash
# Model configuration
export LLAMA_MODEL_PATH="/custom/path/to/model"
export MODEL_NAME="llama-3-8b-custom"
export DEVICE="cuda"
export DTYPE="bfloat16"

# Compression settings
export COMPRESSION_RANK_LOW=32
export COMPRESSION_RANK_MED=64
export COMPRESSION_RANK_HIGH=128
export KEY_COMPRESSION_RANK=32

# Inference settings
export MAX_LENGTH=50
export BATCH_SIZE=1
export TEMPERATURE=0.7
```

### **Configuration Classes**
```python
# Model configuration
config = ModelConfig(
    model_path="/custom/path",
    device="cuda",
    dtype="bfloat16",
    low_memory_mode=True
)

# Compression configuration
compression_config = CompressionConfig(
    value_compression_ranks={"aggressive": 16, "standard": 64},
    key_compression_rank=32,
    use_memory_efficient_svd=True
)
```

## 📁 Project Structure

### **Organized File Layout**
```
chunkedCompression/
├── 📋 Documentation (4 files)
│   ├── README.md, CLAUDE.md, REFACTORING_SUMMARY.md, PROJECT_STRUCTURE.md
├── 🏗️ core/ (21 files)
│   ├── Modern modular architecture
├── 📜 legacy/ (3 files)
│   ├── Original implementation (preserved)
├── 🧪 tests/ (9 + 14 result files)
│   ├── Organized test hierarchy
├── 🚀 scripts/ (9 files)
│   ├── Deployment automation
└── 🔗 profiles_llama_new.py
    ├── Backward compatibility wrapper
```

## 🔄 Migration Guide

### **For Existing Code**
```python
# Old imports still work
from profiles_llama_new import LLaMACompressionProfiles
# No changes required to existing code
```

### **For New Development**
```python
# Use modern modular imports
from core.model import LLaMAModelLoader
from core.compression import LLaMACompressionProfileBuilder
from core.config import ModelConfig, CompressionConfig
```

## 📈 Recent Development

Based on git history and refactoring, recent work includes:
- **🏗️ Complete modular architecture**: Organized 20+ scattered files into clean hierarchy
- **⚙️ Configuration management**: Environment-based, flexible settings
- **💾 Memory management**: Centralized resource handling with monitoring
- **🧪 Test organization**: Proper unit/integration/results separation
- **📚 Complete documentation**: Comprehensive guides and examples
- **🔄 Backward compatibility**: All existing code continues to work
- **📐 Interface design**: Abstract base classes for extensibility

## 🎯 Important Notes

- **No Synthetic Data**: All compression performed on actual transformer representations
- **Memory Requirements**: Requires significant GPU memory for LLaMA-3 8B
- **Research Project**: Educational/research use only
- **LLaMA License**: Subject to Meta's LLaMA-3 license terms
- **Cluster Environment**: Designed for cluster computing environment
- **Production Ready**: Modular architecture suitable for production deployment

## 📊 Output Files

### **Model Validation Results** 
Test results are saved to `tests/results/` directory with timestamped JSON files containing:
- Compression performance metrics
- Memory usage statistics
- Quality measurements (MSE, cosine similarity)
- Cache performance data

### **Industry Benchmark Results**
Evaluation results are saved to `evaluation_results/` directory:
- `perplexity_benchmark_YYYYMMDD_HHMMSS.json`: Perplexity analysis results
- `zero_shot_benchmark_YYYYMMDD_HHMMSS.json`: Zero-shot task performance
- `comprehensive_benchmark_YYYYMMDD_HHMMSS.json`: Combined evaluation results

**Example benchmark result structure:**
```json
{
  "benchmark_type": "perplexity",
  "results": {
    "wikitext2": {
      "summary": {
        "baseline": {"avg_perplexity": 12.1},
        "compression_profiles": {
          "med": {
            "perplexity_degradation_pct": 3.3,
            "memory_savings_pct": 78.1,
            "avg_compression_ratio": 8.1
          }
        }
      }
    }
  }
}
```

## 🚀 Development Workflow

1. **Environment Setup**: Use automated scripts for cluster deployment
2. **Model Loading**: Initialize LLaMAModelLoader with configuration
3. **Profile Creation**: Generate compression matrices from actual attention weights
4. **Data Processing**: Process real text through LLaMA model
5. **Compression**: Apply SVD-based compression to K/V representations
6. **Caching**: Store compressed representations in optimized cache
7. **Inference**: Perform attention with compressed K/V and fused outputs
8. **Evaluation**: Compare against baseline with comprehensive metrics

This system represents a complete production-grade implementation of transformer attention compression using real LLaMA-3 8B model weights with a modern, maintainable architecture.