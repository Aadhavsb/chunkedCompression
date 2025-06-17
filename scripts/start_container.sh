#!/bin/bash

# 🐳 Container Startup Script
#
# This script starts the Singularity container with GPU support for the
# LLaMA-3 8B Compression System. Run this AFTER SSH'ing into the allocated GPU node.
#
# Prerequisites:
# - Successfully allocated GPU resources (via ./scripts/run.sh)
# - SSH'd into the assigned GPU node
# - Container setup completed (pytorch_sandbox exists)
#
# Usage:
#   ssh <gpu-node-name>
#   cd chunkedCompression
#   ./scripts/start_container.sh
#
# What this script does:
# 1. Validates GPU node environment
# 2. Loads Singularity module
# 3. Starts container with NVIDIA GPU support
# 4. Drops you into interactive container shell
#
# Inside the container you can:
# - Run Python scripts with GPU access
# - Use the core compression modules
# - Execute tests and benchmarks
# - Access the LLaMA-3 8B model

set -e  # Exit on any error

echo "🐳 Starting LLaMA-3 8B Compression Container..."
echo "=============================================="
echo "📍 Node: $(hostname)"
echo "📅 Time: $(date)"
echo ""

# Validate GPU environment
echo "🔍 Validating GPU environment..."
if ! nvidia-smi &> /dev/null; then
    echo "⚠️  Warning: nvidia-smi not available. Make sure you're on a GPU node."
    echo "   Expected to be on a node like: gpu-node-XXX"
else
    echo "✅ GPU environment detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1 | \
    while IFS=, read name memory; do
        echo "   GPU: $name ($memory MB)"
    done
fi

# Validate container environment
echo ""
echo "🔍 Validating container environment..."
if [ ! -d "pytorch_sandbox" ]; then
    echo "❌ Error: pytorch_sandbox directory not found!"
    echo ""
    echo "📋 Troubleshooting:"
    echo "  • Make sure you're in the chunkedCompression directory"
    echo "  • Run: cd chunkedCompression"
    echo "  • Run setup first: ./scripts/setup.sh"
    echo "  • Check current directory: pwd"
    echo ""
    exit 1
fi
echo "✅ Container sandbox found"

# Load required modules
echo ""
echo "📦 Loading Singularity module..."
module load singularity

# Start container
echo ""
echo "🚀 Starting Singularity container with GPU support..."
echo ""
echo "📋 Inside the container you can:"
echo "  • Install: pip install -e '.[research]'"  
echo "  • Run: python tests/integration/run_comprehensive_test.py"
echo "  • Use: from core.model import LLaMAModelLoader"
echo "  • Execute: llama-benchmark  # CLI command"
echo "  • Access: /mnt/vstor/CSE_ECSE_GXD234/Meta-Llama-3-8B-Instruct"
echo ""
echo "🔗 Documentation: See README.md for usage examples"
echo ""
echo "⚡ Entering container shell..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

singularity shell --nv --writable pytorch_sandbox/