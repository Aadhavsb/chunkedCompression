# 📚 Core Module Documentation

## 🎯 Overview

This document provides detailed documentation for each module in the `core/` directory, explaining their purpose, key classes, and usage patterns.

---

## 📁 `core/config/` - Configuration Management

### Purpose
Centralized configuration management with environment variable support and validation.

### Key Classes

#### `ModelConfig` (`model_config.py`)
```python
from core.config import ModelConfig

# Load from environment variables
config = ModelConfig.from_env()

# Manual configuration
config = ModelConfig(
    model_path="/path/to/model",
    device="cuda",
    use_flash_attention=True
)
```

#### `CompressionConfig` (`compression_config.py`)
```python
from core.config import CompressionConfig

config = CompressionConfig(
    value_compression_ranks={"low": 32, "med": 64, "high": 128},
    key_compression_rank=32,
    use_memory_efficient_svd=True
)
```

#### `InferenceConfig` (`inference_config.py`)
```python
from core.config import InferenceConfig

config = InferenceConfig(
    max_sequence_length=2048,
    batch_size=1,
    temperature=0.7
)
```

---

## 📐 `core/interfaces/` - Abstract Base Classes

### Purpose
Defines contracts and interfaces for all major components to ensure consistency and enable extensibility.

### Key Interfaces

#### `ModelLoaderInterface` (`model_interface.py`)
```python
from core.interfaces import ModelLoaderInterface

class CustomModelLoader(ModelLoaderInterface):
    def load_model(self) -> None:
        # Custom implementation
        pass
    
    def get_model_weights(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        # Custom weight extraction
        pass
```

#### `CompressionInterface` (`compression_interface.py`)
```python
from core.interfaces import CompressionInterface

class CustomCompression(CompressionInterface):
    def compress_values(self, values: torch.Tensor, profile: str) -> torch.Tensor:
        # Custom compression algorithm
        pass
```

---

## 🤖 `core/model/` - Model Loading Components

### Purpose
Handles loading and management of LLaMA-3 8B model with various configurations.

### Key Classes

#### `LLaMAModelLoader` (`model_loader.py`)
```python
from core.model import LLaMAModelLoader
from core.config import ModelConfig

config = ModelConfig.from_env()
loader = LLaMAModelLoader(config)
# loader.load_model()  # Optional - auto-called by inference classes

# Extract attention weights (will auto-load if needed)
weights = loader.get_attention_weights(layer_idx=0)
```

**Auto-Loading Feature**: Inference classes automatically call `load_model()` for convenience. Manual loading is only needed for direct model loader usage.

#### `ModelConfigWrapper` (`model_config_wrapper.py`)
```python
from core.model import ModelConfigWrapper

wrapper = ModelConfigWrapper(model_config)
# Provides unified interface for different model configurations
```

#### `LLaMA3Loader` (`llama_loader.py`)
```python
from core.model import LLaMA3Loader

# Memory-safe cluster loading
loader = LLaMA3Loader()
model_data = loader.load_model_data()
```

---

## 🗜️ `core/compression/` - Compression Algorithms

### Purpose
Implements SVD-based compression algorithms and profile management.

### Key Classes

#### `SVDCompressionAlgorithm` (`compression_algorithms.py`)
```python
from core.compression import SVDCompressionAlgorithm

algorithm = SVDCompressionAlgorithm()
compressed_matrix = algorithm.compress_matrix(
    matrix=attention_weights,
    rank=64,
    use_memory_efficient=True
)
```

#### `LLaMACompressionProfileBuilder` (`profile_builder.py`)
```python
from core.compression import LLaMACompressionProfileBuilder

builder = LLaMACompressionProfileBuilder(model_loader, compression_config)
profiles = builder.build_compression_profiles(layer_idx=0)

# Use profiles for compression
compressed_values = builder.compress_values(values, "med", head_idx=0)
```

#### `LegacyCompressionWrapper` (`legacy_wrapper.py`)
```python
# Provides backward compatibility
from core.compression import LegacyCompressionWrapper

wrapper = LegacyCompressionWrapper(new_implementation)
# Maintains old API while using new backend
```

---

## 💾 `core/cache/` - KV Cache Implementations

### Purpose
Implements compressed and standard KV caching with performance monitoring.

### Key Classes

#### `LLaMAKVCache` (`kv_cache_llama.py`)
```python
from core.cache import LLaMAKVCache

cache = LLaMAKVCache()
cache.store_kv(
    keys=compressed_keys,
    values=compressed_values,
    metadata={"compression_ratio": 8.0}
)

# Retrieve with reconstruction
retrieved_kv = cache.get_kv(token_idx=10)
```

#### `StandardKVCache` (`standard_kv_cache.py`)
```python
from core.cache import StandardKVCache

# Baseline cache for comparison
baseline_cache = StandardKVCache()
baseline_cache.store_kv(keys, values)
```

---

## 🚀 `core/inference/` - Inference Pipeline

### Purpose
End-to-end inference pipeline with compression integration and benchmarking.

### Key Classes

#### `LLaMACompressionInference` (`llama_inference.py`)
```python
from core.inference import LLaMACompressionInference

inference = LLaMACompressionInference(model_loader, profile_builder)
results = inference.run_compression_benchmark()

print(f"Memory savings: {results['aggregate_metrics']['avg_memory_savings']:.2%}")
```

#### `CompressedAutoregressiveDecoder` (`compressed_autoregressive_decoder.py`)
```python
from core.inference import CompressedAutoregressiveDecoder

decoder = CompressedAutoregressiveDecoder(inference_config)
tokens = decoder.generate(
    prompt="Hello world",
    max_length=100,
    use_compression=True
)
```

#### `LLaMAFullForward` (`llama_full_forward.py`)
```python
from core.inference import LLaMAFullForward

forward_pass = LLaMAFullForward(model_loader)
outputs = forward_pass.forward(input_ids, use_compression=True)
```

---

## 📊 `core/data/` - Data Handling

### Purpose
Dataset loading and preprocessing utilities for the compression system.

### Key Classes

#### `LLaMADatasetProcessor` (`dataset_llama.py`)
```python
from core.data import LLaMADatasetProcessor

processor = LLaMADatasetProcessor()
dataset = processor.load_dataset("wikitext")
processed_data = processor.preprocess_for_compression(dataset)
```

---

## 🛠️ `core/utils/` - Utility Functions

### Purpose
Common utilities for memory management, logging, and system operations.

### Key Classes

#### `MemoryManager` (`memory_manager.py`)
```python
from core.utils import MemoryManager

memory_manager = MemoryManager(cleanup_threshold=0.8)

# Automatic memory management
with memory_manager.managed_computation():
    result = expensive_computation()
    # Memory automatically cleaned up

# Manual cleanup
memory_manager.cleanup_if_needed()
```

---

## 🔄 Common Usage Patterns

### **Complete Workflow Example**
```python
from core.config import ModelConfig, CompressionConfig
from core.model import LLaMAModelLoader
from core.compression import LLaMACompressionProfileBuilder
from core.inference import LLaMACompressionInference
from core.utils import MemoryManager

# Configuration
model_config = ModelConfig.from_env()
compression_config = CompressionConfig(
    value_compression_ranks={"med": 64},
    key_compression_rank=32
)

# Memory management
memory_manager = MemoryManager()

with memory_manager.managed_computation():
    # Model loading
    model_loader = LLaMAModelLoader(model_config)
    model_loader.load_model()
    
    # Compression setup
    profile_builder = LLaMACompressionProfileBuilder(
        model_loader, compression_config
    )
    profile_builder.build_compression_profiles(layer_idx=0)
    
    # Inference
    inference = LLaMACompressionInference(model_loader, profile_builder)
    results = inference.run_compression_benchmark()
    
    print(f"Compression successful: {results['aggregate_metrics']}")
```

### **Custom Implementation Example**
```python
from core.interfaces import CompressionInterface
from core.compression import SVDCompressionAlgorithm

class CustomCompressionAlgorithm(CompressionInterface):
    def __init__(self):
        self.svd_algorithm = SVDCompressionAlgorithm()
    
    def compress_values(self, values: torch.Tensor, profile: str) -> torch.Tensor:
        # Custom preprocessing
        preprocessed = self.custom_preprocessing(values)
        
        # Use built-in SVD
        compressed = self.svd_algorithm.compress_matrix(
            preprocessed, rank=self.get_rank_for_profile(profile)
        )
        
        return compressed
```

---

## 📖 Migration Guide

### **From Legacy to Modern API**

```python
# Old way
from profiles_llama import LLaMACompressionProfiles
profiles = LLaMACompressionProfiles(model_loader)

# New way (recommended)
from core.compression import LLaMACompressionProfileBuilder
from core.config import CompressionConfig

config = CompressionConfig()
builder = LLaMACompressionProfileBuilder(model_loader, config)
```

### **Using Compatibility Layer**
```python
# This still works unchanged
from profiles_llama_new import LLaMACompressionProfiles
profiles = LLaMACompressionProfiles(model_loader)
# All existing code continues to function
```

---

## 🎯 Key Benefits

- **🏗️ Modular Design**: Each component has a single responsibility
- **📐 Interface-Based**: Easy to extend and customize
- **⚙️ Configuration-Driven**: Environment-based settings
- **💾 Memory Efficient**: Built-in memory management
- **🔄 Backward Compatible**: Legacy code continues to work
- **🧪 Well Tested**: Comprehensive test coverage
- **📚 Well Documented**: Clear examples and patterns

This architecture enables both simple usage for common cases and extensive customization for advanced use cases while maintaining high performance and memory efficiency.