# 🚀 Usage Guide - LLaMA-3 8B Chunked Compression

## ⚠️ Python, pip, and setuptools Version Requirements

To use `pip install -e .` or `pip install -e ".[all]"` with only a `pyproject.toml` (no `setup.py`/`setup.cfg`), you must have:

- **Python 3.8+** (Python 3.10+ recommended)
- **pip 21.3+** (pip 25+ recommended)
- **setuptools 61+** (setuptools 80+ recommended)

If you see errors about editable installs or missing `setup.py`, upgrade pip and setuptools:

```bash
python3.10 -m pip install --upgrade pip setuptools
```

Older pip/setuptools do not support editable installs with PEP 660 (pyproject.toml only). On Mac, Homebrew and recent Python distributions usually have new enough versions by default. On Linux/cluster, you may need to upgrade manually.

---

## 📦 Installation Methods

### **🎯 Method 1: Package Installation (Recommended)**
```bash
# Clone repository
git clone https://github.com/Aadhavsb/chunkedCompression
cd chunkedCompression

# Install in development mode
pip install -e .

# Or install with specific dependencies
pip install -e ".[research]"    # Full research environment
pip install -e ".[dev]"         # Development + testing tools  
pip install -e ".[all]"         # Everything included
```

### **🐳 Method 2: Container Environment**
```bash
# Use automated scripts
./scripts/setup.sh      # Build container
./scripts/run.sh        # Allocate GPU resources
# SSH to allocated node, then:
./scripts/start_container.sh   # Start container

# Inside container: install dependencies
pip install -r requirements.txt
# OR for development setup:
pip install -e ".[dev]"
```

### **📋 Method 3: Manual Dependencies**
```bash
# Traditional requirements.txt approach
pip install -r requirements.txt

# Then run from source
python -m pytest tests/
```

---

## 🏃 How to Run the Project

### **🧪 Run Comprehensive Tests**
```bash
# Method 1: Using installed package
llama-benchmark  # CLI command from pyproject.toml

# Method 2: Direct execution (LLaMA-3 8B comprehensive test)
python tests/integration/run_comprehensive_test.py

# Method 3: With pytest
pytest tests/integration/ -v
```

### **🚀 Basic Compression Pipeline**
```bash
# Using modern API (auto-loading)
python -c "
from core.model import LLaMAModelLoader
from core.config import ModelConfig
from core.compression import LLaMACompressionProfileBuilder
from core.inference import LLaMACompressionInference

# Configure and run (models auto-load for convenience)
config = ModelConfig.from_env()
loader = LLaMAModelLoader(config)
# loader.load_model()  # Optional - auto-called by inference classes

builder = LLaMACompressionProfileBuilder(loader)
builder.build_compression_profiles(layer_idx=0)

inference = LLaMACompressionInference(loader, builder)
results = inference.run_compression_benchmark()
print(f'Memory savings: {results[\"aggregate_metrics\"][\"avg_memory_savings\"]:.2%}')
"

# Or even simpler - let inference handle everything
python -c "
from core.inference import LLaMACompressionInference

# Simplest usage - everything auto-configured
inference = LLaMACompressionInference()  # Uses default model path
results = inference.run_compression_benchmark()
print(f'Results: {results[\"aggregate_metrics\"]}')
"
```

### **🔬 Individual Test Components**
```bash
# Unit tests
pytest tests/unit/ -v

# Specific test files
python tests/unit/test_llama_compression.py
python tests/unit/test_kv_cache_comparison.py
python tests/integration/test_token_generation.py

# Test with markers
pytest -m "not slow"     # Skip slow tests
pytest -m "gpu"          # Only GPU tests
pytest -m "integration"  # Only integration tests
```

---

## ⚙️ Configuration Options

### **🌍 Environment Variables**
```bash
# Model configuration
export LLAMA_MODEL_PATH="/path/to/Meta-Llama-3-8B-Instruct"
export LLAMA_DEVICE="cuda"
export LLAMA_USE_FLASH_ATTENTION="true"

# Compression settings  
export COMPRESSION_VALUE_RANKS="32,64,128"
export COMPRESSION_KEY_RANK="32"
export COMPRESSION_MEMORY_EFFICIENT="true"

# Run with configuration
python tests/integration/run_comprehensive_test.py
```

### **📝 Programmatic Configuration**
```python
from core.config import ModelConfig, CompressionConfig

# Custom model config
model_config = ModelConfig(
    model_path="/custom/path/to/model",
    device="cuda:1",
    use_flash_attention=True
)

# Custom compression config
compression_config = CompressionConfig(
    value_compression_ranks={"aggressive": 16, "standard": 64},
    key_compression_rank=32,
    use_memory_efficient_svd=True
)
```

---

## 🎯 Common Use Cases

### **📊 Performance Benchmarking**
```bash
# Full benchmark suite
python tests/integration/run_comprehensive_test.py

# Results saved to: tests/results/benchmark_YYYYMMDD_HHMMSS.json
```

### **🧪 Compression Quality Testing**
```python
from core.inference import LLaMACompressionInference

inference = LLaMACompressionInference(model_loader, profile_builder)
results = inference.run_compression_benchmark()

# Check quality metrics
print(f"Cosine similarity: {results['aggregate_metrics']['avg_cosine_similarity']:.4f}")
print(f"MSE: {results['aggregate_metrics']['avg_mse']:.6f}")
print(f"Memory savings: {results['aggregate_metrics']['avg_memory_savings']:.2%}")
```

### **🔧 Custom Compression Profiles**
```python
from core.compression import LLaMACompressionProfileBuilder
from core.config import CompressionConfig

# Create custom compression profile
config = CompressionConfig(
    value_compression_ranks={"ultra_aggressive": 8, "custom": 48},
    key_compression_rank=16
)

builder = LLaMACompressionProfileBuilder(model_loader, config)
builder.build_compression_profiles(layer_idx=0)

# Use custom profile
compressed_values = builder.compress_values(values, "ultra_aggressive", head_idx=0)
```

### **💾 Memory Management**
```python
from core.utils import MemoryManager

memory_manager = MemoryManager(cleanup_threshold=0.8)

with memory_manager.managed_computation():
    # Memory-intensive operations with automatic cleanup
    results = run_heavy_computation()
    # Memory automatically cleaned up when exiting context
```

---

## 🔄 Development Workflow

### **🛠️ Development Setup**
```bash
# Install development dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Run code formatting
black .
isort .

# Type checking
mypy core/

# Run tests with coverage
pytest --cov=core
```

---

## 🚨 Troubleshooting

### **❌ Common Issues**

#### **Import Errors**
```bash
# If you get import errors, try:
pip install -e ".[all]"  # Install all dependencies

# Or check if in correct directory:
cd /path/to/chunkedCompression
python -c "import core; print('Import successful')"
```

#### **CUDA/GPU Issues**
```bash
# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Verify in container
nvidia-smi
```

#### **Model Path Issues**
```bash
# Set explicit model path
export LLAMA_MODEL_PATH="/mnt/vstor/CSE_ECSE_GXD234/Meta-Llama-3-8B-Instruct"

# Or check if model exists
ls -la /mnt/vstor/CSE_ECSE_GXD234/Meta-Llama-3-8B-Instruct
```

#### **Memory Issues**
```python
# Use memory manager for large computations
from core.utils import MemoryManager
memory_manager = MemoryManager(cleanup_threshold=0.7)  # Lower threshold
```

---

## 📋 Quick Reference

### **🎯 Most Common Commands**
```bash
# Install and run comprehensive test
pip install -e ".[research]"
python tests/integration/run_comprehensive_test.py

# Run specific test
python tests/unit/test_llama_compression.py

# Container workflow  
./scripts/setup.sh && ./scripts/run.sh
# (SSH to node) && ./scripts/start_container.sh
# pip install -r requirements.txt
```

### **📊 Key Files**
- **pyproject.toml**: Modern package configuration
- **requirements.txt**: Traditional dependencies (for containers)
- **tests/integration/run_comprehensive_test.py**: Main test runner
- **core/**: Modern modular implementation
- **legacy/**: Original implementation (for reference)

This guide covers all the ways to install, configure, and run the LLaMA-3 8B Chunked Compression system!