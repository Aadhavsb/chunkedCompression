# LLaMA-3 8B Chunked Compression System

**Production-grade transformer attention compression using real LLaMA-3 8B Instruct model weights**

## 🎯 Overview

This system implements **chunked attention compression** for the LLaMA-3 8B Instruct model using:

- **Real SVD-based compression** on actual model weights (no placeholders)
- **Adaptive value compression** with three compression levels (low/med/high)
- **Fixed key compression** with on-the-fly reconstruction 
- **Fused output projections** for direct vocabulary decoding
- **Production-grade KV caching** with memory optimization

## 🏗️ Architecture

### Core Components

1. **LLaMAModelLoader** (`llama_model_loader.py`)
   - Loads real LLaMA-3 8B Instruct from local path
   - Extracts actual attention matrices (W_Q, W_K, W_V, W_O)
   - Provides real hidden states extraction

2. **LLaMACompressionProfiles** (`profiles_llama.py`)
   - Creates SVD-based compression matrices from real weights
   - Three value compression levels: 64, 128, 256 ranks
   - Fixed key compression at rank 128
   - Fused language model head projections

3. **LLaMAKVCache** (`kv_cache_llama.py`)
   - Compressed KV storage with metadata tracking
   - Memory usage optimization
   - Standard cache comparison baseline

4. **LLaMACompressionInference** (`llama_inference.py`)
   - End-to-end compression pipeline
   - Attention computation with compressed K/V
   - Performance benchmarking

5. **LLaMADatasetHandler** (`dataset_llama.py`)
   - Real text processing through LLaMA model
   - Hidden states extraction and analysis
   - Ground truth logits and perplexity calculation

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure LLaMA-3 8B Instruct model is available at:
# /mnt/vstor/CSE_ECSE_GXD234/Meta-Llama-3-8B-Instruct
```

### Basic Usage

```python
from llama_inference import LLaMACompressionInference

# Initialize pipeline (loads real LLaMA-3 8B model)
pipeline = LLaMACompressionInference()

# Run compression benchmark
results = pipeline.run_compression_benchmark()

# View results
print(f"Average memory savings: {results['aggregate_metrics']['avg_memory_savings']:.2%}")
print(f"Average cosine similarity: {results['aggregate_metrics']['avg_cosine_similarity']:.4f}")
```

### Running Tests

```bash
# Run comprehensive test suite
cd tests/
python test_llama_compression.py
```

## 📊 Compression Profiles

| Profile | Value Rank | Key Rank | Typical Compression Ratio |
|---------|-----------|----------|---------------------------|
| **Low** | 64 | 128 | ~15x |
| **Med** | 128 | 128 | ~8x |
| **High** | 256 | 128 | ~4x |

### Mathematical Foundation

**Value Compression:**
```
W_V = U @ S @ V^T  (SVD decomposition)
A_V = S_trunc @ V_trunc^T  (compression matrix)
W_fused = W_LM_HEAD @ U_trunc  (fused projection)
```

**Key Compression:**
```
W_K = U_K @ S_K @ V_K^T
A_K = S_K_trunc @ V_K_trunc^T  (compression)
B_K = V_K_trunc  (reconstruction)
```

## 🔬 Key Features

### Real Model Integration
- ✅ **No synthetic data** - all tensors from actual LLaMA-3 8B
- ✅ **Real attention weights** extracted from model layers
- ✅ **Actual hidden states** from transformer forward passes
- ✅ **True language model head** integration

### Compression Technology
- ✅ **SVD-based compression** with mathematical guarantees
- ✅ **Adaptive value compression** based on token importance
- ✅ **On-the-fly key reconstruction** for memory efficiency
- ✅ **Fused output matrices** for direct vocabulary projection

### Production Features
- ✅ **Memory tracking** and optimization
- ✅ **Performance benchmarking** with detailed metrics
- ✅ **Cache hit/miss statistics**
- ✅ **Comprehensive error handling**

## 📈 Performance Metrics

The system tracks:

- **Quality Metrics**: Output MSE, cosine similarity, perplexity
- **Efficiency Metrics**: Memory savings, compression ratios, timing
- **Cache Performance**: Hit rates, reconstruction overhead
- **Model Accuracy**: Ground truth comparison, token prediction

## 🗂️ Project Structure

```
chunkedCompression/
├── llama_model_loader.py      # Real LLaMA-3 8B model loading
├── profiles_llama.py          # SVD compression profiles
├── dataset_llama.py           # Real data processing
├── kv_cache_llama.py          # Production KV caching
├── llama_inference.py         # End-to-end pipeline
├── compression.py             # Core SVD utilities
├── tests/
│   └── test_llama_compression.py  # Comprehensive test suite
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 🧪 Testing

The test suite verifies:

1. **Model Loading**: Real LLaMA-3 8B loading and inference
2. **Compression Profiles**: SVD matrices and shape validation
3. **Hidden States**: Real transformer output processing
4. **KV Cache**: Compressed storage and retrieval
5. **End-to-End**: Complete compression pipeline

Run tests to verify system integrity:

```bash
python tests/test_llama_compression.py
```

## 📋 System Requirements

- **GPU Memory**: 16GB+ VRAM for LLaMA-3 8B
- **RAM**: 32GB+ system memory
- **Storage**: ~30GB for model weights
- **CUDA**: Compatible GPU with CUDA 11.8+

## 🔧 Configuration

### Model Path
Update model path in components if different location:
```python
model_path = "/your/path/to/Meta-Llama-3-8B-Instruct"
```

### Compression Ranks
Modify compression ranks in `profiles_llama.py`:
```python
self.value_compression_ranks = {
    "low": 64,    # Adjust as needed
    "med": 128,   # Adjust as needed  
    "high": 256   # Adjust as needed
}
```

## 📄 License

Research and educational use only. LLaMA-3 model subject to Meta's license terms.

## 🤝 Contributing

This is a research prototype. Contributions welcome for:
- Additional compression algorithms
- Performance optimizations
- Extended evaluation metrics
- Multi-layer compression

---

**Note**: This system uses REAL LLaMA-3 8B model weights with NO synthetic data or placeholders. All compression is performed on actual transformer representations.
