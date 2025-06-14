#!/bin/bash

# Run script for LLaMA compression project
# This script allocates GPU resources and provides instructions for SSH

set -e  # Exit on any error

echo "Starting LLaMA compression project environment..."

# Check if sandbox exists
if [ ! -d "pytorch_sandbox" ]; then
    echo "Error: pytorch_sandbox directory not found!"
    echo "Please run setup.sh first to create the container."
    exit 1
fi

# Allocate GPU resources
echo "Allocating GPU resources..."
echo "Requesting: 2 GPUs, 24 cores, 24GB memory for 2 hours"
echo ""
echo "After allocation, you'll need to:"
echo "1. Note the node name given (e.g., gpu-node-123)"
echo "2. SSH into that node: ssh gpu-node-123"
echo "3. Run start_container.sh on that node"
echo ""

# Use salloc to get interactive allocation
salloc -p gpu -c 24 --gres=gpu:2 --mem=24gb -t 2:00:00