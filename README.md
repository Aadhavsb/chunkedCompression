# LLaMA-3 8B Chunked Compression System

**Production-grade transformer attention compression using real LLaMA-3 8B Instruct model weights**

## 🚀 Overview

This system implements **chunked attention compression** for the LLaMA-3 8B Instruct model using:

- **Real SVD-based compression** on actual model weights (no synthetic data)
- **Adaptive value compression** with three compression levels (15x, 8x, 4x ratios)
- **Fixed key compression** with on-the-fly reconstruction 
- **Fused output projections** for direct vocabulary decoding
- **Production-grade KV caching** with memory optimization
- **Modular architecture** with clean separation of concerns

## ⚠️ Python & pip Version Requirements

To use pip install -e . or pip install -e ".[all]" with only a pyproject.toml (no setup.py/setup.cfg), you must have:

- **Python 3.8+** (Python 3.10+ recommended)
- **pip 21.3+** (pip 25+ recommended)
- **setuptools 61+** (setuptools 80+ recommended)

If you see errors about editable installs or missing setup.py, upgrade pip and setuptools:

```bash
python3.10 -m pip install --upgrade pip setuptools
```

Older pip/setuptools do not support editable installs with PEP 660 (pyproject.toml only). On Mac, Homebrew and recent Python distributions usually have new enough versions by default. On Linux/cluster, you may need to upgrade manually.

---

## 🏗️ Architecture

### **Modern Modular Structure**

```
core/
├── config/          # Configuration management
├── model/           # Model loading and management  
├── compression/     # Compression algorithms and profiles
├── cache/           # KV cache implementations
├── inference/       # Inference pipeline and benchmarking
├── data/            # Dataset handling utilities
├── utils/           # Utility functions and memory management
└── interfaces/      # Abstract base classes for extensibility
```

### **Core Components**

#### **🤖 Model Management** (`core/model/`)
- **LLaMAModelLoader**: Advanced model loading with configuration support
- **ModelConfigWrapper**: Unified model configuration interface
- **LLaMA3Loader**: Memory-safe cluster loading utilities

#### **🗜️ Compression System** (`core/compression/`)
- **SVDCompressionAlgorithm**: Mathematical SVD-based compression
- **LLaMACompressionProfileBuilder**: Profile building and management
- **Legacy compatibility**: Backward-compatible wrappers

#### **💾 Caching Layer** (`core/cache/`)
- **LLaMAKVCache**: Compressed KV storage with metadata
- **StandardKVCache**: Baseline cache for comparison
- **Performance metrics**: Hit rates and memory optimization

#### **🚀 Inference Pipeline** (`core/inference/`)
- **LLaMACompressionInference**: End-to-end compression pipeline
- **Compressed decoder**: Autoregressive decoding with compression
- **Benchmarking tools**: Performance and quality metrics

## 📦 Quick Start

### **🔧 Automated Setup (Recommended)**

For cluster environments with Singularity:

```bash
# 1. Initial setup (one-time)
bash scripts/setup.sh        # or ./scripts/setup.sh

# 2. Allocate GPU resources  
bash scripts/run.sh
# Note the assigned node (e.g., gpu-node-123)

# 3. SSH to the allocated node
ssh gpu-node-123

# 4. Start the container
bash scripts/start_container.sh

# 5. Install dependencies inside container
pip install -r requirements.txt
# OR for development setup:
pip install -e ".[dev]"
```

### **⚙️ Manual Setup**

#### **Modern Approach (Recommended)**
```bash
# Install package in development mode
pip install -e .

# Or install with specific profile
pip install -e ".[research]"    # Full research environment
pip install -e ".[dev]"         # Development tools
pip install -e ".[all]"         # Everything
```

#### **Traditional Approach (Container/CI)**
```bash
# Install dependencies only
pip install -r requirements.txt

# Then run from source
python tests/integration/run_comprehensive_test.py
```

#### **Prerequisites**
```bash
# Ensure LLaMA-3 8B Instruct model is available at:
# /mnt/vstor/CSE_ECSE_GXD234/Meta-Llama-3-8B-Instruct
```

## 💻 Usage

### **🆕 Modern API (Recommended)**

```python
from core.model import LLaMAModelLoader
from core.config import ModelConfig, CompressionConfig
from core.compression import LLaMACompressionProfileBuilder
from core.inference import LLaMACompressionInference

# Configure the system
model_config = ModelConfig.from_env()  # Load from environment
compression_config = CompressionConfig(
    value_compression_ranks={"low": 32, "med": 64, "high": 128}
)

# Initialize components (auto-loading enabled)
model_loader = LLaMAModelLoader(model_config)
# model_loader.load_model()  # Optional - auto-called by inference classes

# Build compression profiles
profile_builder = LLaMACompressionProfileBuilder(
    model_loader, compression_config
)
profile_builder.build_compression_profiles(layer_idx=0)

# Run inference
inference = LLaMACompressionInference(
    model_loader, profile_builder
)
results = inference.run_compression_benchmark()

print(f"Memory savings: {results['aggregate_metrics']['avg_memory_savings']:.2%}")
```

### **🔄 Legacy API (Backward Compatible)**

```python
# Existing code continues to work unchanged
from profiles_llama_new import LLaMACompressionProfiles

# Initialize pipeline (loads real LLaMA-3 8B model)
pipeline = LLaMACompressionProfiles(model_loader)

# Run compression  
compressed_values = pipeline.compress_values(values, "med", head_idx=0)
```

### **⚡ Quick Examples**

#### **Memory Management**
```python
from core.utils import MemoryManager

memory_manager = MemoryManager(cleanup_threshold=0.8)
with memory_manager.managed_computation():
    # Memory-intensive operations with automatic cleanup
    result = expensive_computation()
```

#### **Custom Configuration**
```python
from core.config import CompressionConfig

# Custom compression settings
config = CompressionConfig(
    value_compression_ranks={"aggressive": 16, "standard": 64},
    key_compression_rank=32,
    use_memory_efficient_svd=True
)
```

## 🧪 Testing

### **📋 Test Organization**

```
tests/
├── unit/           # Component-level tests
├── integration/    # End-to-end workflow tests  
└── results/        # Historical test results
```

### **🏃 Running Tests**

```bash
# Run comprehensive test suite
python tests/integration/run_comprehensive_test.py

# Run specific test categories
python -m pytest tests/unit/           # Unit tests
python -m pytest tests/integration/    # Integration tests

# Individual test files
python tests/unit/test_llama_compression.py
python tests/unit/test_kv_cache_comparison.py
python tests/integration/test_refactored_imports.py
```

### **📊 Test Coverage**

The test suite verifies:

1. **✅ Model Loading**: Real LLaMA-3 8B loading and inference
2. **✅ Compression Profiles**: SVD matrices and shape validation  
3. **✅ Hidden States**: Real transformer output processing
4. **✅ KV Cache**: Compressed storage and retrieval
5. **✅ End-to-End**: Complete compression pipeline
6. **✅ Memory Management**: Resource cleanup and monitoring
7. **✅ Configuration**: Environment-based setup validation

## 📈 Compression Profiles

| Profile | Value Rank | Key Rank | Compression Ratio | Memory Savings |
|---------|-----------|----------|-------------------|----------------|
| **Low** | 32 | 32 | ~15x | 93% |
| **Med** | 64 | 32 | ~8x | 87% |
| **High** | 128 | 32 | ~4x | 75% |

## 🔧 Key Features

### **🎯 Real Model Integration**
- ✅ **No synthetic data** - all tensors from actual LLaMA-3 8B
- ✅ **Real attention weights** extracted from model layers
- ✅ **Actual hidden states** from transformer forward passes

### **⚙️ Compression Technology**
- ✅ **SVD-based compression** with mathematical guarantees
- ✅ **Adaptive value compression** based on token importance
- ✅ **On-the-fly key reconstruction** for memory efficiency
- ✅ **Fused output projections** for direct vocabulary decoding

### **📊 Performance Monitoring**
- **Quality Metrics**: Output MSE, cosine similarity, perplexity
- **Efficiency Metrics**: Memory savings, compression ratios, timing
- **Cache Performance**: Hit rates, reconstruction overhead
- **Memory Management**: Automatic cleanup and monitoring

### **🏗️ Architecture Benefits**
- **Modular Design**: Clean separation of concerns
- **Configuration Management**: Environment-based settings
- **Backward Compatibility**: Existing code continues to work
- **Extensibility**: Interface-based design for custom implementations
- **Memory Efficiency**: Centralized memory management

## 🖥️ System Requirements

- **GPU Memory**: 16GB+ VRAM for LLaMA-3 8B
- **RAM**: 32GB+ system memory
- **Storage**: ~30GB for model weights
- **CUDA**: Compatible GPU with CUDA 11.8+
- **Python**: 3.8+ with PyTorch 2.0+

## 📚 Documentation

- **📋 [Usage Guide](USAGE_GUIDE.md)**: Complete installation and usage instructions
- **📋 [Project Structure](PROJECT_STRUCTURE.md)**: Detailed file organization
- **📋 [Module Documentation](MODULE_DOCS.md)**: Complete core module documentation
- **📋 [Development Guide](CLAUDE.md)**: Development instructions
- **💡 Individual module docstrings**: See each `core/` module for detailed API docs

## 🔄 Migration Guide

### **From Legacy Code**
```python
# Old imports
from llama_model_loader import LLaMAModelLoader
from profiles_llama import LLaMACompressionProfiles

# New imports (recommended)
from core.model import LLaMAModelLoader  
from core.compression import LLaMACompressionProfileBuilder
```

### **Compatibility Layer**
```python
# This continues to work unchanged
from profiles_llama_new import LLaMACompressionProfiles
# All existing APIs preserved
```

## 🎯 What's New

- **🏗️ Modular Architecture**: Clean, organized codebase structure
- **⚙️ Configuration System**: Flexible, environment-based settings
- **💾 Memory Management**: Centralized resource handling
- **📐 Abstract Interfaces**: Extensible design patterns
- **🧪 Organized Testing**: Proper test hierarchy and structure
- **📚 Complete Documentation**: Comprehensive guides and examples
- **🔄 Backward Compatibility**: All existing code still works

---

**Note**: This system uses REAL LLaMA-3 8B model weights with NO synthetic data. All compression is performed on actual transformer representations using production-grade algorithms.