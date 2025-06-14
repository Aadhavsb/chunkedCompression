#!/bin/bash

# Container startup script - run this AFTER ssh'ing into the allocated node
# Usage: ./start_container.sh

set -e  # Exit on any error

echo "Starting Singularity container on $(hostname)..."

# Check if we're on a GPU node
if ! nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not available. Make sure you're on a GPU node."
fi

# Check if sandbox exists
if [ ! -d "pytorch_sandbox" ]; then
    echo "Error: pytorch_sandbox directory not found!"
    echo "Make sure you're in the correct directory and ran setup.sh first."
    exit 1
fi

# Load singularity module
echo "Loading Singularity module..."
module load singularity

# Start the container with GPU support
echo "Starting Singularity container with GPU and NVIDIA support..."
echo "You'll be dropped into the container shell where you can run your Python scripts."
echo ""

singularity shell --nv --writable pytorch_sandbox/