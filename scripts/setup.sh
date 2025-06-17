#!/bin/bash

# 🐳 Singularity Container Setup Script
# 
# This script builds the PyTorch Singularity container for the 
# LLaMA-3 8B Chunked Compression System
#
# Prerequisites:
# - Access to cluster environment with Singularity module
# - Docker Hub access for pulling containers
# - Sufficient storage space (~10GB for container)
#
# Usage:
#   ./scripts/setup.sh
#
# What this script does:
# 1. Loads Singularity module
# 2. Pulls PyTorch container from Docker Hub
# 3. Builds writable sandbox environment
# 4. Validates setup completion

set -e  # Exit on any error

echo "🐳 Setting up Singularity container for LLaMA-3 8B Compression System..."
echo "================================================================="

# Load singularity module
echo "Loading Singularity module..."
module load singularity

# Pull PyTorch container from Docker Hub
echo "Pulling PyTorch container from Docker Hub..."
if [ ! -f "pytorch.sif" ]; then
    singularity pull pytorch.sif docker://pytorch/pytorch:latest
    echo "✓ PyTorch container pulled successfully"
else
    echo "✓ PyTorch container already exists"
fi

# Build sandbox from the container
echo "Building writable sandbox..."
if [ ! -d "pytorch_sandbox" ]; then
    singularity build --sandbox pytorch_sandbox/ pytorch.sif
    echo "✓ Sandbox built successfully"
else
    echo "✓ Sandbox already exists"
fi

echo ""
echo "🎉 Setup complete! Container environment is ready."
echo ""
echo "📋 Next steps:"
echo "  1. Run: ./scripts/run.sh (to allocate GPU resources)"
echo "  2. Note the assigned node name (e.g., gpu-node-123)"
echo "  3. SSH: ssh <node-name>"
echo "  4. Run: ./scripts/start_container.sh (to start working)"
echo ""
echo "📍 Requirements:"
echo "  • LLaMA-3 8B model available at: /mnt/vstor/CSE_ECSE_GXD234/Meta-Llama-3-8B-Instruct"
echo "  • GPU resources: 2 GPUs, 24GB memory recommended"
echo ""
echo "🔗 For more information, see: README.md"