#!/bin/bash

# Setup script for Singularity container
# This script builds the PyTorch Singularity container for the chunked compression project

set -e  # Exit on any error

echo "Setting up Singularity container for LLaMA compression project..."

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

echo "Setup complete! You can now use run.sh to start working with the container."
echo "Note: Make sure the LLaMA-3 8B model is available at /mnt/vstor/CSE_ECSE_GXD234/Meta-Llama-3-8B-Instruct"