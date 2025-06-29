#!/bin/bash

# 🚀 GPU Resource Allocation Script
#
# This script allocates GPU resources for the LLaMA-3 8B Compression System
# and provides clear instructions for the next steps.
#
# Prerequisites:
# - Container setup completed (run ./scripts/setup.sh first)
# - Access to SLURM cluster with GPU partition
# - Valid cluster account with GPU allocation permissions
#
# Usage:
#   ./scripts/run.sh
#
# What this script does:
# 1. Validates container environment exists
# 2. Requests GPU allocation from SLURM scheduler
# 3. Provides instructions for SSH and container startup
#
# Resource Requirements:
# - 2 GPUs (for LLaMA-3 8B model)
# - 24 CPU cores
# - 24GB memory  
# - 2 hour time limit

set -e  # Exit on any error

echo "🚀 Starting LLaMA-3 8B Compression System GPU Allocation..."
echo "========================================================"

# Validate prerequisites
echo "🔍 Validating environment..."
if [ ! -d "pytorch_sandbox" ]; then
    echo "❌ Error: pytorch_sandbox directory not found!"
    echo ""
    echo "📋 Please run setup first:"
    echo "   ./scripts/setup.sh"
    echo ""
    exit 1
fi
echo "✅ Container environment validated"

# Display resource request details
echo ""
echo "📊 Requesting GPU resources from SLURM..."
echo "  • GPUs: 1 (for LLaMA-3 8B model)"
echo "  • CPU cores: 24"
echo "  • Memory: 24GB"
echo "  • Time limit: 2 hours"
echo "  • Partition: gpu"
echo ""

echo "⏳ Waiting for allocation... (this may take a few minutes)"
echo ""
echo "📋 After allocation, follow these steps:"
echo "  1. 📝 Note the assigned node name (e.g., gpu-node-123)"
echo "  2. 🔗 SSH into that node: ssh <node-name>"
echo "  3. 🐳 Start container: ./scripts/start_container.sh"
echo "  4. 🚀 Begin development with full GPU access"
echo ""

# Request GPU allocation through SLURM
echo "🎯 Executing: salloc -p gpu -c 24 --gres=gpu:1 --mem=24gb -t 2:00:00"
echo ""
salloc -p gpu -c 24 --gres=gpu:1 --mem=24gb -t 2:00:00