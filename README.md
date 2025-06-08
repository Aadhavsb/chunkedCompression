# Barebones Chunked-Fused KV Compression

A minimal testbed for validating per-token, per-chunk compression + fused output projection in a transformer-style decoder.

## Overview

This system implements:
- Per-token compression assignment (low/medium/high compression ratios)
- Compressed latent storage in KV cache
- Attention over compressed representations
- Fused output projection per compression group

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the test
python test.py
```

## Files

- `main.py` - Entry point and full pipeline
- `model.py` - Barebones transformer decoder with compression
- `compression.py` - Compression matrix generation (simulated SVD)
- `kv_cache.py` - Compressed latent vector storage
- `profiles.py` - Hardcoded compression profiles
- `dataset.py` - Mock WikiText-2 data loading
- `utils.py` - Token-to-compression mapping utilities
- `test.py` - Test suite and validation

## Output

The system processes ~128 tokens and outputs:
- Compression assignment statistics
- Latent vector shapes per compression profile
- Final projected output tensor [T, d_model]
- Validation that logic doesn't crash

This is a **logic validation prototype** - no training, no model fidelity checks, just testing the compression flow.
