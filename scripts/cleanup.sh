#!/bin/bash

# 🧹 Cleanup Script
#
# Cleans up temporary files, logs, test results, and other artifacts
# from the LLaMA-3 8B Compression System development and testing.
#
# Prerequisites:
# - Write access to project directory
#
# Usage:
#   ./scripts/cleanup.sh [OPTIONS]
#
# Options:
#   --all               Clean everything (cache, logs, results, build)
#   --cache             Clean Python cache files (__pycache__, .pyc)
#   --logs              Clean log files and monitoring data
#   --results           Clean test and benchmark results
#   --build             Clean build artifacts and distributions
#   --containers        Clean container artifacts (pytorch_sandbox, *.sif)
#   --coverage          Clean coverage reports
#   --interactive       Interactive mode (ask before each deletion)
#   --dry-run           Show what would be deleted without deleting
#   --help              Show this help

set -e

# Default values
CLEAN_ALL=false
CLEAN_CACHE=false
CLEAN_LOGS=false
CLEAN_RESULTS=false
CLEAN_BUILD=false
CLEAN_CONTAINERS=false
CLEAN_COVERAGE=false
INTERACTIVE=false
DRY_RUN=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            CLEAN_ALL=true
            shift
            ;;
        --cache)
            CLEAN_CACHE=true
            shift
            ;;
        --logs)
            CLEAN_LOGS=true
            shift
            ;;
        --results)
            CLEAN_RESULTS=true
            shift
            ;;
        --build)
            CLEAN_BUILD=true
            shift
            ;;
        --containers)
            CLEAN_CONTAINERS=true
            shift
            ;;
        --coverage)
            CLEAN_COVERAGE=true
            shift
            ;;
        --interactive)
            INTERACTIVE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            echo "🧹 Cleanup Script for LLaMA-3 8B Compression System"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --all               Clean everything (cache, logs, results, build)"
            echo "  --cache             Clean Python cache files (__pycache__, .pyc)"
            echo "  --logs              Clean log files and monitoring data"
            echo "  --results           Clean test and benchmark results"
            echo "  --build             Clean build artifacts and distributions"
            echo "  --containers        Clean container artifacts (pytorch_sandbox, *.sif)"
            echo "  --coverage          Clean coverage reports"
            echo "  --interactive       Interactive mode (ask before each deletion)"
            echo "  --dry-run           Show what would be deleted without deleting"
            echo "  --help              Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 --cache                    # Clean only Python cache"
            echo "  $0 --results --logs           # Clean results and logs"
            echo "  $0 --all --dry-run            # Preview all cleanup operations"
            echo "  $0 --all --interactive        # Interactive cleanup"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# If --all is specified, enable all cleanup types
if [ "$CLEAN_ALL" = true ]; then
    CLEAN_CACHE=true
    CLEAN_LOGS=true
    CLEAN_RESULTS=true
    CLEAN_BUILD=true
    CLEAN_CONTAINERS=true
    CLEAN_COVERAGE=true
fi

# If no specific cleanup type specified, default to cache
if [ "$CLEAN_CACHE" = false ] && [ "$CLEAN_LOGS" = false ] && [ "$CLEAN_RESULTS" = false ] && [ "$CLEAN_BUILD" = false ] && [ "$CLEAN_CONTAINERS" = false ] && [ "$CLEAN_COVERAGE" = false ]; then
    CLEAN_CACHE=true
fi

echo "🧹 LLaMA-3 8B Compression System Cleanup"
echo "========================================"
echo "📍 Directory: $(pwd)"
echo "📅 Time: $(date)"

if [ "$DRY_RUN" = true ]; then
    echo "🔍 DRY RUN MODE - No files will be deleted"
fi

if [ "$INTERACTIVE" = true ]; then
    echo "🤝 INTERACTIVE MODE - Will ask before each deletion"
fi

echo ""

# Function to confirm deletion in interactive mode
confirm_deletion() {
    local item="$1"
    if [ "$INTERACTIVE" = true ]; then
        read -p "Delete $item? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return 1
        fi
    fi
    return 0
}

# Function to execute deletion command
execute_cleanup() {
    local description="$1"
    local command="$2"
    
    echo "🗑️  $description..."
    
    if [ "$DRY_RUN" = true ]; then
        echo "   Would execute: $command"
        # Show what would be found
        eval "find . -name '*' ${command#*find*}" 2>/dev/null | head -5 | while read item; do
            echo "   Would delete: $item"
        done | head -5
        local count=$(eval "find . -name '*' ${command#*find*}" 2>/dev/null | wc -l)
        if [ $count -gt 5 ]; then
            echo "   ... and $((count - 5)) more files"
        fi
    else
        if confirm_deletion "$description"; then
            eval "$command" 2>/dev/null || true
            echo "   ✓ Completed"
        else
            echo "   ⏭️  Skipped"
        fi
    fi
}

# Clean Python cache files
if [ "$CLEAN_CACHE" = true ]; then
    echo "🐍 Cleaning Python cache files..."
    execute_cleanup "Python cache directories" "find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true"
    execute_cleanup "Compiled Python files" "find . -name '*.pyc' -delete 2>/dev/null || true"
    execute_cleanup "Python optimization files" "find . -name '*.pyo' -delete 2>/dev/null || true"
    execute_cleanup "Python cache files" "find . -name '*.pycache' -delete 2>/dev/null || true"
    echo ""
fi

# Clean log files
if [ "$CLEAN_LOGS" = true ]; then
    echo "📋 Cleaning log files..."
    execute_cleanup "Log directory" "rm -rf logs/"
    execute_cleanup "Monitor log files" "find . -name 'monitor_*.log' -delete 2>/dev/null || true"
    execute_cleanup "Debug log files" "find . -name '*.log' -delete 2>/dev/null || true"
    execute_cleanup "Temporary log files" "find . -name '*.tmp' -delete 2>/dev/null || true"
    echo ""
fi

# Clean test and benchmark results
if [ "$CLEAN_RESULTS" = true ]; then
    echo "📊 Cleaning test and benchmark results..."
    execute_cleanup "Test results directory" "rm -rf test_results/"
    execute_cleanup "Benchmark results directory" "rm -rf benchmark_results/"
    execute_cleanup "Test result JSON files" "find tests/results/ -name '*.json' -delete 2>/dev/null || true"
    execute_cleanup "JUnit XML files" "find . -name '*.xml' -path '*/test*' -delete 2>/dev/null || true"
    execute_cleanup "Temporary benchmark scripts" "find . -name 'run_benchmark_*.py' -delete 2>/dev/null || true"
    echo ""
fi

# Clean build artifacts
if [ "$CLEAN_BUILD" = true ]; then
    echo "🔨 Cleaning build artifacts..."
    execute_cleanup "Build directory" "rm -rf build/"
    execute_cleanup "Distribution directory" "rm -rf dist/"
    execute_cleanup "Egg info directories" "find . -name '*.egg-info' -type d -exec rm -rf {} + 2>/dev/null || true"
    execute_cleanup "Wheel cache" "rm -rf .cache/"
    execute_cleanup "Setuptools cache" "rm -rf .eggs/"
    echo ""
fi

# Clean container artifacts
if [ "$CLEAN_CONTAINERS" = true ]; then
    echo "🐳 Cleaning container artifacts..."
    execute_cleanup "PyTorch sandbox directory" "rm -rf pytorch_sandbox/"
    execute_cleanup "Singularity image files" "find . -name '*.sif' -delete 2>/dev/null || true"
    execute_cleanup "Container temporary files" "find . -name '.singularity*' -delete 2>/dev/null || true"
    echo ""
fi

# Clean coverage reports
if [ "$CLEAN_COVERAGE" = true ]; then
    echo "📈 Cleaning coverage reports..."
    execute_cleanup "HTML coverage report" "rm -rf htmlcov/"
    execute_cleanup "Coverage data files" "find . -name '.coverage*' -delete 2>/dev/null || true"
    execute_cleanup "Coverage XML reports" "find . -name 'coverage.xml' -delete 2>/dev/null || true"
    execute_cleanup "Coverage data directory" "rm -rf .coverage_data/"
    echo ""
fi

# Additional cleanup for common development artifacts
if [ "$CLEAN_ALL" = true ]; then
    echo "🧽 Additional cleanup..."
    execute_cleanup "Jupyter checkpoint directories" "find . -name '.ipynb_checkpoints' -type d -exec rm -rf {} + 2>/dev/null || true"
    execute_cleanup "MyPy cache" "rm -rf .mypy_cache/"
    execute_cleanup "Pytest cache" "rm -rf .pytest_cache/"
    execute_cleanup "Tox environments" "rm -rf .tox/"
    execute_cleanup "Virtual environment artifacts" "find . -name 'pyvenv.cfg' -delete 2>/dev/null || true"
    execute_cleanup "Editor backup files" "find . -name '*~' -delete 2>/dev/null || true"
    execute_cleanup "Editor swap files" "find . -name '*.swp' -delete 2>/dev/null || true"
    execute_cleanup "MacOS metadata files" "find . -name '.DS_Store' -delete 2>/dev/null || true"
    echo ""
fi

# Show disk space summary
if [ "$DRY_RUN" = false ]; then
    echo "💾 Disk space in project directory:"
    du -sh . 2>/dev/null || echo "   Unable to calculate disk usage"
fi

echo ""
echo "🎉 Cleanup Complete!"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "📋 To actually perform cleanup, run without --dry-run"
else
    echo "📋 Cleanup summary:"
    echo "  • Python cache files: $([ "$CLEAN_CACHE" = true ] && echo "cleaned" || echo "skipped")"
    echo "  • Log files: $([ "$CLEAN_LOGS" = true ] && echo "cleaned" || echo "skipped")"
    echo "  • Test results: $([ "$CLEAN_RESULTS" = true ] && echo "cleaned" || echo "skipped")"
    echo "  • Build artifacts: $([ "$CLEAN_BUILD" = true ] && echo "cleaned" || echo "skipped")"
    echo "  • Container files: $([ "$CLEAN_CONTAINERS" = true ] && echo "cleaned" || echo "skipped")"
    echo "  • Coverage reports: $([ "$CLEAN_COVERAGE" = true ] && echo "cleaned" || echo "skipped")"
fi

echo ""
echo "💡 Tip: Use --dry-run to preview cleanup operations before executing"