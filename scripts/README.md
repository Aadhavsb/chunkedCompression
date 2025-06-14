# 🚀 Scripts Directory

This directory contains utility scripts for automating common tasks in the LLaMA-3 8B Chunked Compression System.

## 📁 Available Scripts

### **🐳 Container Management**
| Script | Purpose | Usage |
|--------|---------|-------|
| `setup.sh` | Build Singularity container | `./scripts/setup.sh` |
| `run.sh` | Allocate GPU resources via SLURM | `./scripts/run.sh` |
| `start_container.sh` | Start container with GPU support | `./scripts/start_container.sh` |

### **🛠️ Development Tools**
| Script | Purpose | Usage |
|--------|---------|-------|
| `dev_setup.sh` | Setup development environment | `./scripts/dev_setup.sh` |
| `run_tests.sh` | Comprehensive test runner | `./scripts/run_tests.sh [OPTIONS]` |
| `cleanup.sh` | Clean temporary files and artifacts | `./scripts/cleanup.sh [OPTIONS]` |

### **📊 Performance & Monitoring**
| Script | Purpose | Usage |
|--------|---------|-------|
| `check_gpu.sh` | Validate GPU environment | `./scripts/check_gpu.sh` |
| `monitor_resources.sh` | Real-time resource monitoring | `./scripts/monitor_resources.sh [OPTIONS]` |
| `benchmark.sh` | Comprehensive performance benchmarking | `./scripts/benchmark.sh [OPTIONS]` |

## 🎯 Quick Start Workflows

### **🔧 Development Setup**
```bash
# First-time setup
bash scripts/dev_setup.sh        # or ./scripts/dev_setup.sh if executable
bash scripts/check_gpu.sh

# Run tests
bash scripts/run_tests.sh --all --coverage

# Clean up
bash scripts/cleanup.sh --cache
```

### **🐳 Container Workflow**
```bash
# Setup (one-time)
bash scripts/setup.sh

# For each session
bash scripts/run.sh          # Note the assigned node
ssh <gpu-node-name>          # SSH to allocated node
bash scripts/start_container.sh
```

### **📊 Performance Analysis**
```bash
# Monitor resources during development
bash scripts/monitor_resources.sh --log performance.log &

# Run comprehensive benchmark
bash scripts/benchmark.sh --profiles low,med,high --monitor

# Check GPU environment
bash scripts/check_gpu.sh
```

## 🖥️ HPC Usage Notes

### **Running on HPC Systems**
HPC systems often have different permission models. Use these approaches:

#### **✅ Method 1: Use `bash` explicitly (Always works)**
```bash
bash scripts/setup.sh
bash scripts/run.sh  
bash scripts/check_gpu.sh --help
```

#### **✅ Method 2: Set permissions after clone**
```bash
# After git clone or file transfer
chmod +x scripts/*.sh
./scripts/setup.sh
```

#### **✅ Method 3: Git preserves permissions**
If using `git clone`, permissions are preserved automatically.

### **HPC-Specific Considerations**
- **Module loading**: Scripts use `module load singularity`
- **SLURM integration**: `scripts/run.sh` uses SLURM commands
- **GPU nodes**: Scripts detect GPU availability with `nvidia-smi`
- **File systems**: Scripts create directories as needed

### **Typical HPC Workflow**
```bash
# 1. Clone repository
git clone <repo-url>
cd chunkedCompression

# 2. Check environment  
bash scripts/check_gpu.sh

# 3. Setup container (if using Singularity)
bash scripts/setup.sh

# 4. Get GPU allocation
bash scripts/run.sh
# Note: gpu-node-123

# 5. SSH to allocated node
ssh gpu-node-123

# 6. Start development
bash scripts/start_container.sh
```

## 📋 Script Details

### **`dev_setup.sh`** - Development Environment Setup
Sets up complete development environment with code quality tools.

**Features:**
- Installs development dependencies
- Configures pre-commit hooks
- Sets up code quality tools (black, isort, mypy)
- Validates environment

**Usage:**
```bash
./scripts/dev_setup.sh
```

### **`run_tests.sh`** - Comprehensive Test Runner
Flexible test runner with multiple options and configurations.

**Options:**
- `--unit` - Run only unit tests
- `--integration` - Run only integration tests  
- `--fast` - Skip slow tests
- `--gpu` - Run only GPU tests
- `--coverage` - Generate coverage report
- `--benchmark` - Run performance benchmarks

**Examples:**
```bash
./scripts/run_tests.sh --unit --fast
./scripts/run_tests.sh --integration --coverage
./scripts/run_tests.sh --gpu
```

### **`check_gpu.sh`** - GPU Environment Validator
Comprehensive GPU environment validation and diagnostics.

**Checks:**
- NVIDIA driver and CUDA installation
- PyTorch CUDA integration
- GPU memory and capabilities
- Model path accessibility
- Performance recommendations

**Usage:**
```bash
./scripts/check_gpu.sh
```

### **`monitor_resources.sh`** - Resource Monitoring
Real-time monitoring with logging and alerting capabilities.

**Options:**
- `--interval SECONDS` - Monitoring interval (default: 2)
- `--duration MINUTES` - Total monitoring time
- `--log FILE` - Log output to file
- `--alert-memory PCT` - Alert when GPU memory > PCT%
- `--alert-temp TEMP` - Alert when GPU temp > TEMP°C
- `--quiet` - Minimal output mode

**Examples:**
```bash
./scripts/monitor_resources.sh
./scripts/monitor_resources.sh --interval 5 --duration 30
./scripts/monitor_resources.sh --log monitor.log --quiet
```

### **`benchmark.sh`** - Performance Benchmarking
Comprehensive benchmarking with multiple configurations.

**Options:**
- `--profiles PROFILES` - Compression profiles (default: low,med,high)
- `--layers LAYERS` - Layer indices (default: 0,15,31)
- `--iterations N` - Number of iterations (default: 3)
- `--output DIR` - Output directory
- `--monitor` - Enable resource monitoring
- `--compare` - Compare with baseline
- `--export FORMAT` - Export format (json,csv,html)

**Examples:**
```bash
./scripts/benchmark.sh
./scripts/benchmark.sh --profiles low,med --iterations 5
./scripts/benchmark.sh --monitor --compare --export csv
```

### **`cleanup.sh`** - Cleanup Utility
Cleans temporary files, logs, and build artifacts.

**Options:**
- `--all` - Clean everything
- `--cache` - Clean Python cache files
- `--logs` - Clean log files
- `--results` - Clean test/benchmark results
- `--build` - Clean build artifacts
- `--containers` - Clean container artifacts
- `--coverage` - Clean coverage reports
- `--interactive` - Ask before deletion
- `--dry-run` - Preview without deleting

**Examples:**
```bash
./scripts/cleanup.sh --cache
./scripts/cleanup.sh --all --dry-run
./scripts/cleanup.sh --results --logs
```

## 🔧 Script Configuration

### **Environment Variables**
Scripts respect these environment variables:

```bash
# Model configuration
export LLAMA_MODEL_PATH="/path/to/model"
export LLAMA_DEVICE="cuda"

# SLURM configuration  
export SLURM_PARTITION="gpu"
export SLURM_MEMORY="24gb"
export SLURM_GPUS="2"

# Monitoring configuration
export MONITOR_INTERVAL="2"
export ALERT_MEMORY_THRESHOLD="90"
export ALERT_TEMP_THRESHOLD="85"
```

### **Customization**
Scripts can be customized by:

1. **Editing default values** in script headers
2. **Setting environment variables** 
3. **Creating wrapper scripts** for common configurations
4. **Modifying pyproject.toml** for tool configurations

## 🎯 Integration with Project

### **CI/CD Integration**
Scripts generate machine-readable outputs:

- **JUnit XML** for test results
- **Coverage XML** for coverage reports  
- **JSON/CSV** for benchmark data
- **Structured logs** for monitoring

### **Development Workflow**
Recommended workflow:

1. **Setup**: `./scripts/dev_setup.sh`
2. **Development**: Use `./scripts/monitor_resources.sh` during heavy tasks
3. **Testing**: `./scripts/run_tests.sh --fast` for quick feedback
4. **Benchmarking**: `./scripts/benchmark.sh` for performance validation
5. **Cleanup**: `./scripts/cleanup.sh --cache` regularly

### **Production Deployment**
For cluster deployment:

1. **Container setup**: `./scripts/setup.sh`
2. **Resource allocation**: `./scripts/run.sh`
3. **Monitoring**: `./scripts/monitor_resources.sh --log production.log`
4. **Benchmarking**: `./scripts/benchmark.sh --export csv`

## 💡 Tips & Best Practices

### **Performance**
- Use `--fast` flag during development to skip slow tests
- Monitor resources during long-running tasks
- Use `--dry-run` before cleanup operations

### **Debugging**
- Check GPU environment first: `./scripts/check_gpu.sh`
- Use monitoring for resource bottlenecks
- Enable verbose output in test scripts

### **Maintenance**
- Regular cleanup: `./scripts/cleanup.sh --cache`
- Update dependencies: Re-run `./scripts/dev_setup.sh`
- Archive benchmark results before cleanup

## 🔗 Related Documentation

- **[Usage Guide](../USAGE_GUIDE.md)**: Complete installation and usage instructions
- **[Development Guide](../CLAUDE.md)**: Development workflow and best practices
- **[Project Structure](../PROJECT_STRUCTURE.md)**: File organization details

All scripts are designed to work together as part of a comprehensive development and deployment toolkit for the LLaMA-3 8B Chunked Compression System.