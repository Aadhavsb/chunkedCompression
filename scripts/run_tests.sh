#!/bin/bash

# 🧪 Comprehensive Test Runner Script
#
# Runs various test suites for the LLaMA-3 8B Compression System
# with different configurations and reporting options.
#
# Prerequisites:
# - Development environment setup (run ./scripts/dev_setup.sh)
# - GPU access (for GPU-specific tests)
# - LLaMA-3 8B model available
#
# Usage:
#   ./scripts/run_tests.sh [OPTIONS]
#
# Options:
#   --unit          Run only unit tests
#   --integration   Run only integration tests
#   --fast          Skip slow tests
#   --gpu           Run only GPU tests
#   --coverage      Generate coverage report
#   --benchmark     Run performance benchmarks
#   --all           Run all tests (default)
#   --help          Show this help

set -e  # Exit on any error

# Default options
RUN_UNIT=false
RUN_INTEGRATION=false
RUN_FAST=false
RUN_GPU=false
RUN_COVERAGE=false
RUN_BENCHMARK=false
RUN_ALL=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --unit)
            RUN_UNIT=true
            RUN_ALL=false
            shift
            ;;
        --integration)
            RUN_INTEGRATION=true
            RUN_ALL=false
            shift
            ;;
        --fast)
            RUN_FAST=true
            shift
            ;;
        --gpu)
            RUN_GPU=true
            RUN_ALL=false
            shift
            ;;
        --coverage)
            RUN_COVERAGE=true
            shift
            ;;
        --benchmark)
            RUN_BENCHMARK=true
            RUN_ALL=false
            shift
            ;;
        --all)
            RUN_ALL=true
            shift
            ;;
        --help)
            echo "🧪 Comprehensive Test Runner for LLaMA-3 8B Compression System"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --unit          Run only unit tests"
            echo "  --integration   Run only integration tests"
            echo "  --fast          Skip slow tests"
            echo "  --gpu           Run only GPU tests"
            echo "  --coverage      Generate coverage report"
            echo "  --benchmark     Run performance benchmarks"
            echo "  --all           Run all tests (default)"
            echo "  --help          Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "🧪 Running LLaMA-3 8B Compression System Tests..."
echo "=============================================="
echo "📍 Working directory: $(pwd)"
echo "📅 Timestamp: $(date)"
echo ""

# Check prerequisites
echo "🔍 Checking prerequisites..."
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found. Are you in the project root?"
    exit 1
fi

if ! python3 -c "import pytest" 2>/dev/null; then
    echo "❌ Error: pytest not installed. Run: pip install -e '.[dev]'"
    exit 1
fi
echo "✓ Prerequisites checked"

# Create results directory
mkdir -p test_results/
timestamp=$(date +"%Y%m%d_%H%M%S")

# Build pytest command
PYTEST_CMD="python3 -m pytest"
PYTEST_ARGS="-v --tb=short"

if [ "$RUN_COVERAGE" = true ]; then
    PYTEST_ARGS="$PYTEST_ARGS --cov=core --cov-report=html --cov-report=term-missing --cov-report=xml"
fi

if [ "$RUN_FAST" = true ]; then
    PYTEST_ARGS="$PYTEST_ARGS -m 'not slow'"
fi

# Run specific test suites
if [ "$RUN_UNIT" = true ]; then
    echo ""
    echo "🔬 Running Unit Tests..."
    echo "========================"
    $PYTEST_CMD tests/unit/ $PYTEST_ARGS --junit-xml=test_results/unit_tests_$timestamp.xml
    
elif [ "$RUN_INTEGRATION" = true ]; then
    echo ""
    echo "🔗 Running Integration Tests..."
    echo "==============================="
    $PYTEST_CMD tests/integration/ $PYTEST_ARGS --junit-xml=test_results/integration_tests_$timestamp.xml
    
elif [ "$RUN_GPU" = true ]; then
    echo ""
    echo "🎮 Running GPU Tests..."
    echo "======================"
    # Check GPU availability
    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "✓ GPU available"
        $PYTEST_CMD -m gpu $PYTEST_ARGS --junit-xml=test_results/gpu_tests_$timestamp.xml
    else
        echo "⚠️  No GPU available, skipping GPU tests"
    fi
    
elif [ "$RUN_BENCHMARK" = true ]; then
    echo ""
    echo "📊 Running Performance Benchmarks..."
    echo "===================================="
    python3 tests/integration/run_comprehensive_test.py
    echo "✓ Benchmark results saved to tests/results/"
    
elif [ "$RUN_ALL" = true ]; then
    echo ""
    echo "🎯 Running All Test Suites..."
    echo "============================="
    
    # Unit tests
    echo ""
    echo "1️⃣ Unit Tests:"
    $PYTEST_CMD tests/unit/ $PYTEST_ARGS --junit-xml=test_results/unit_tests_$timestamp.xml
    
    # Integration tests
    echo ""
    echo "2️⃣ Integration Tests:"
    $PYTEST_CMD tests/integration/ $PYTEST_ARGS --junit-xml=test_results/integration_tests_$timestamp.xml
    
    # GPU tests (if available)
    echo ""
    echo "3️⃣ GPU Tests:"
    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "✓ GPU available"
        $PYTEST_CMD -m gpu $PYTEST_ARGS --junit-xml=test_results/gpu_tests_$timestamp.xml
    else
        echo "⚠️  No GPU available, skipping GPU tests"
    fi
    
    # Comprehensive benchmark
    echo ""
    echo "4️⃣ Performance Benchmark:"
    python3 tests/integration/run_comprehensive_test.py
fi

# Coverage report
if [ "$RUN_COVERAGE" = true ]; then
    echo ""
    echo "📊 Coverage Report Generated:"
    echo "  HTML: htmlcov/index.html"
    echo "  XML: coverage.xml"
    echo "  Terminal: (displayed above)"
fi

# Summary
echo ""
echo "🎉 Test Execution Complete!"
echo "=========================="
echo "📁 Results saved to: test_results/"
echo "📅 Timestamp: $timestamp"
echo ""
echo "📋 Next steps:"
echo "  • Review test results in test_results/"
if [ "$RUN_COVERAGE" = true ]; then
    echo "  • Open coverage report: open htmlcov/index.html"
fi
echo "  • Check CI/CD integration with generated XML reports"
echo "  • Run specific tests: pytest tests/unit/test_specific.py"