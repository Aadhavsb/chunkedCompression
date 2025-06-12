# LLaMA-3 8B Chunked Compression System

**Production-grade transformer attention compression using real LLaMA-3 8B Instruct model weights**

## Overview

This system implements **chunked attention compression** for the LLaMA-3 8B Instruct model using:

- **Real SVD-based compression** on actual model weights (no synthetic data)
- **Adaptive value compression** with three compression levels (15x, 8x, 4x ratios)
- **Fixed key compression** with on-the-fly reconstruction 
- **Fused output projections** for direct vocabulary decoding
- **Production-grade KV caching** with memory optimization

## Architecture

### Core Components

1. **LLaMAModelLoader** (`llama_model_loader.py`)
   - Loads real LLaMA-3 8B Instruct from cluster path
   - Extracts actual attention matrices (W_Q, W_K, W_V, W_O)
   - Provides real hidden states extraction

2. **LLaMACompressionProfiles** (`profiles_llama.py`)
   - Creates SVD-based compression matrices from real weights
   - Three value compression levels: 64, 128, 256 ranks
   - Fixed key compression at rank 128
   - Fused language model head projections

3. **LLaMAKVCache** (`kv_cache_llama.py`)
   - Compressed KV storage with metadata tracking
   - Memory usage optimization and cache performance metrics

4. **LLaMACompressionInference** (`llama_inference.py`)
   - End-to-end compression pipeline
   - Performance benchmarking with quality metrics

## Quick Start

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
print(f"Memory savings: {results['aggregate_metrics']['avg_memory_savings']:.2%}")
print(f"Cosine similarity: {results['aggregate_metrics']['avg_cosine_similarity']:.4f}")
```

### Running Tests

```bash
# Run comprehensive 5-stage test suite
python tests/test_llama_compression.py
```

## Compression Profiles

| Profile | Value Rank | Key Rank | Compression Ratio |
|---------|-----------|----------|-------------------|
| **Low** | 64 | 128 | ~15x |
| **Med** | 128 | 128 | ~8x |
| **High** | 256 | 128 | ~4x |

## Key Features

### Real Model Integration
- ✅ **No synthetic data** - all tensors from actual LLaMA-3 8B
- ✅ **Real attention weights** extracted from model layers
- ✅ **Actual hidden states** from transformer forward passes

### Compression Technology
- ✅ **SVD-based compression** with mathematical guarantees
- ✅ **Adaptive value compression** based on token importance
- ✅ **On-the-fly key reconstruction** for memory efficiency

### Performance Metrics
- **Quality**: Output MSE, cosine similarity, perplexity
- **Efficiency**: Memory savings, compression ratios, timing
- **Cache Performance**: Hit rates, reconstruction overhead

## System Requirements

- **GPU Memory**: 16GB+ VRAM for LLaMA-3 8B
- **RAM**: 32GB+ system memory
- **Storage**: ~30GB for model weights
- **CUDA**: Compatible GPU with CUDA 11.8+

## Testing

The comprehensive test suite verifies:

1. **Model Loading**: Real LLaMA-3 8B loading and inference
2. **Compression Profiles**: SVD matrices and shape validation
3. **Hidden States**: Real transformer output processing
4. **KV Cache**: Compressed storage and retrieval
5. **End-to-End**: Complete compression pipeline

---

**Note**: This system uses REAL LLaMA-3 8B model weights with NO synthetic data. All compression is performed on actual transformer representations.
