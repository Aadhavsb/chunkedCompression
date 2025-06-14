#!/bin/bash

# 🛠️ Development Environment Setup Script
#
# Sets up a complete development environment for the LLaMA-3 8B Compression System
# including code quality tools, pre-commit hooks, and testing infrastructure.
#
# Prerequisites:
# - Python 3.8+ installed
# - Git repository initialized
# - Write access to project directory
#
# Usage:
#   ./scripts/dev_setup.sh
#
# What this script does:
# 1. Installs development dependencies
# 2. Sets up pre-commit hooks
# 3. Configures code quality tools
# 4. Initializes testing environment
# 5. Validates setup

set -e  # Exit on any error

echo "🛠️ Setting up development environment for LLaMA-3 8B Compression System..."
echo "================================================================"

# Check Python version
echo "🐍 Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "✓ Python version: $python_version"

# Install development dependencies
echo ""
echo "📦 Installing development dependencies..."
pip install -e ".[dev]"
echo "✓ Development dependencies installed"

# Set up pre-commit hooks
echo ""
echo "🪝 Setting up pre-commit hooks..."
if command -v pre-commit &> /dev/null; then
    pre-commit install
    echo "✓ Pre-commit hooks installed"
else
    echo "⚠️  pre-commit not found, installing..."
    pip install pre-commit
    pre-commit install
    echo "✓ Pre-commit hooks installed"
fi

# Create necessary directories
echo ""
echo "📁 Creating development directories..."
mkdir -p logs/
mkdir -p test_results/
mkdir -p .coverage_data/
echo "✓ Development directories created"

# Initialize git hooks if not exists
echo ""
echo "🔧 Configuring git hooks..."
if [ ! -f .git/hooks/pre-commit ]; then
    echo "#!/bin/bash" > .git/hooks/pre-commit
    echo "exec pre-commit run --hook-stage pre-commit" >> .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit
    echo "✓ Git pre-commit hook configured"
fi

# Validate code quality tools
echo ""
echo "🧪 Validating code quality tools..."

# Test black
if black --check --diff core/ > /dev/null 2>&1; then
    echo "✓ Black formatting: PASSED"
else
    echo "⚠️  Black formatting issues detected, run: black ."
fi

# Test isort
if isort --check-only --diff core/ > /dev/null 2>&1; then
    echo "✓ Import sorting: PASSED"
else
    echo "⚠️  Import sorting issues detected, run: isort ."
fi

# Test mypy
if mypy core/ > /dev/null 2>&1; then
    echo "✓ Type checking: PASSED"
else
    echo "⚠️  Type checking issues detected, run: mypy core/"
fi

# Test basic imports
echo ""
echo "🔍 Testing basic imports..."
python3 -c "
try:
    import core
    from core.config import ModelConfig
    print('✓ Core imports: PASSED')
except ImportError as e:
    print(f'⚠️  Import issues: {e}')
"

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "  1. Run tests: pytest tests/"
echo "  2. Format code: black . && isort ."
echo "  3. Type check: mypy core/"
echo "  4. Run pre-commit: pre-commit run --all-files"
echo ""
echo "🔗 Documentation: See USAGE_GUIDE.md for detailed instructions"