#!/bin/bash

# 🎮 GPU Environment Checker Script
#
# Comprehensive GPU environment validation for the LLaMA-3 8B Compression System.
# Checks CUDA availability, memory, PyTorch integration, and model requirements.
#
# Prerequisites:
# - NVIDIA GPU with CUDA support
# - PyTorch with CUDA support installed
#
# Usage:
#   ./scripts/check_gpu.sh
#
# What this script does:
# 1. Checks NVIDIA driver and CUDA installation
# 2. Validates PyTorch CUDA integration
# 3. Tests GPU memory and capabilities
# 4. Validates model path accessibility
# 5. Provides recommendations

set -e  # Exit on any error

echo "🎮 GPU Environment Checker for LLaMA-3 8B Compression System..."
echo "============================================================="
echo "📍 Host: $(hostname)"
echo "📅 Date: $(date)"
echo ""

# Check NVIDIA driver
echo "🔍 Checking NVIDIA driver..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ nvidia-smi available"
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free --format=csv,noheader,nounits | while IFS=, read name driver memory_total memory_used memory_free; do
        echo "  GPU: $name"
        echo "  Driver: $driver"
        echo "  Memory: ${memory_used}MB used / ${memory_total}MB total (${memory_free}MB free)"
    done
else
    echo "❌ nvidia-smi not available"
    echo "   Make sure you're on a GPU node and NVIDIA drivers are installed"
    exit 1
fi

# Check CUDA availability
echo ""
echo "🔍 Checking CUDA installation..."
if command -v nvcc &> /dev/null; then
    cuda_version=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
    echo "✓ CUDA version: $cuda_version"
else
    echo "⚠️  nvcc not found (CUDA toolkit may not be installed)"
fi

# Check PyTorch CUDA integration
echo ""
echo "🔍 Checking PyTorch CUDA integration..."
python3 -c "
import sys
try:
    import torch
    print(f'✓ PyTorch version: {torch.__version__}')
    
    if torch.cuda.is_available():
        print(f'✓ CUDA available in PyTorch')
        print(f'  CUDA version: {torch.version.cuda}')
        print(f'  cuDNN version: {torch.backends.cudnn.version()}')
        print(f'  Number of GPUs: {torch.cuda.device_count()}')
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f'  GPU {i}: {props.name} ({memory_gb:.1f} GB)')
            
        # Test GPU memory allocation
        try:
            test_tensor = torch.randn(1000, 1000, device='cuda')
            print(f'✓ GPU memory allocation test: PASSED')
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f'❌ GPU memory allocation test: FAILED ({e})')
            
    else:
        print('❌ CUDA not available in PyTorch')
        print('   This may be due to:')
        print('   - PyTorch CPU-only installation')
        print('   - CUDA version mismatch')
        print('   - Missing CUDA libraries')
        sys.exit(1)
        
except ImportError:
    print('❌ PyTorch not installed')
    print('   Install with: pip install torch --index-url https://download.pytorch.org/whl/cu118')
    sys.exit(1)
"

# Check memory requirements for LLaMA-3 8B
echo ""
echo "🔍 Checking LLaMA-3 8B memory requirements..."
python3 -c "
import torch
if torch.cuda.is_available():
    total_memory = torch.cuda.get_device_properties(0).total_memory
    memory_gb = total_memory / (1024**3)
    
    # LLaMA-3 8B requirements
    required_memory = 16  # GB
    recommended_memory = 24  # GB
    
    print(f'  Available GPU memory: {memory_gb:.1f} GB')
    print(f'  Required for LLaMA-3 8B: {required_memory} GB')
    print(f'  Recommended: {recommended_memory} GB')
    
    if memory_gb >= recommended_memory:
        print('✓ Memory: EXCELLENT (recommended or above)')
    elif memory_gb >= required_memory:
        print('✓ Memory: SUFFICIENT (minimum requirements met)')
    else:
        print(f'⚠️  Memory: INSUFFICIENT (need at least {required_memory} GB)')
        print('   Consider using smaller batch sizes or model sharding')
else:
    print('❌ Cannot check memory requirements (no CUDA)')
"

# Check model path accessibility
echo ""
echo "🔍 Checking LLaMA-3 8B model accessibility..."
model_path="/mnt/vstor/CSE_ECSE_GXD234/Meta-Llama-3-8B-Instruct"
if [ -d "$model_path" ]; then
    echo "✓ Model path accessible: $model_path"
    
    # Check key model files
    if [ -f "$model_path/config.json" ]; then
        echo "✓ config.json found"
    else
        echo "⚠️  config.json not found"
    fi
    
    if ls "$model_path"/*.safetensors >/dev/null 2>&1; then
        safetensor_count=$(ls "$model_path"/*.safetensors | wc -l)
        echo "✓ SafeTensor files found ($safetensor_count files)"
    elif ls "$model_path"/*.bin >/dev/null 2>&1; then
        bin_count=$(ls "$model_path"/*.bin | wc -l)
        echo "✓ Bin files found ($bin_count files)"
    else
        echo "⚠️  No model weight files found (.safetensors or .bin)"
    fi
    
    if [ -f "$model_path/tokenizer.json" ] || [ -f "$model_path/tokenizer.model" ]; then
        echo "✓ Tokenizer files found"
    else
        echo "⚠️  Tokenizer files not found"
    fi
    
else
    echo "❌ Model path not accessible: $model_path"
    echo "   Make sure the model is available and mounted correctly"
fi

# Check transformers library
echo ""
echo "🔍 Checking transformers library..."
python3 -c "
try:
    import transformers
    print(f'✓ Transformers version: {transformers.__version__}')
    
    # Test model loading capability
    from transformers import AutoConfig
    try:
        config = AutoConfig.from_pretrained('$model_path', trust_remote_code=True)
        print(f'✓ Model config loadable: {config.model_type}')
    except Exception as e:
        print(f'⚠️  Model config loading issues: {e}')
        
except ImportError:
    print('❌ Transformers not installed')
    print('   Install with: pip install transformers>=4.30.0')
"

# Performance recommendations
echo ""
echo "💡 Performance Recommendations:"
echo "================================"

# Check if we're in a SLURM environment
if [ ! -z "$SLURM_JOB_ID" ]; then
    echo "✓ Running in SLURM environment (Job ID: $SLURM_JOB_ID)"
    echo "  • Allocated GPUs: ${CUDA_VISIBLE_DEVICES:-'all available'}"
else
    echo "• Consider using SLURM for resource management"
fi

# Check CPU count
cpu_count=$(nproc)
echo "• CPU cores available: $cpu_count"
if [ $cpu_count -ge 16 ]; then
    echo "  ✓ Sufficient CPU cores for data loading"
else
    echo "  ⚠️  Consider requesting more CPU cores for optimal data loading"
fi

# Memory recommendations
echo "• For optimal performance:"
echo "  - Use mixed precision training (fp16)"
echo "  - Enable gradient checkpointing if memory limited"
echo "  - Use DeepSpeed for larger models"

echo ""
echo "🎉 GPU Environment Check Complete!"
echo ""
echo "📋 Next steps:"
echo "  • Run benchmark: python tests/integration/run_comprehensive_test.py"
echo "  • Start development: ./scripts/dev_setup.sh"
echo "  • Monitor GPU usage: watch -n 1 nvidia-smi"