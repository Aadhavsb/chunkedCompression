#!/bin/bash

# 📊 Resource Monitoring Script
#
# Real-time monitoring of GPU, CPU, and memory usage during LLaMA-3 8B compression tasks.
# Provides continuous monitoring with logging and alerting capabilities.
#
# Prerequisites:
# - NVIDIA GPU with nvidia-smi
# - System monitoring tools (top, free)
#
# Usage:
#   ./scripts/monitor_resources.sh [OPTIONS]
#
# Options:
#   --interval SECONDS    Monitoring interval (default: 2)
#   --duration MINUTES    Total monitoring time (default: unlimited)
#   --log FILE           Log output to file
#   --alert-memory PCT   Alert when GPU memory > PCT% (default: 90)
#   --alert-temp TEMP    Alert when GPU temp > TEMP°C (default: 85)
#   --quiet              Minimal output mode
#   --help               Show this help

set -e

# Default values
INTERVAL=2
DURATION=""
LOG_FILE=""
ALERT_MEMORY=90
ALERT_TEMP=85
QUIET=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --interval)
            INTERVAL="$2"
            shift 2
            ;;
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --log)
            LOG_FILE="$2"
            shift 2
            ;;
        --alert-memory)
            ALERT_MEMORY="$2"
            shift 2
            ;;
        --alert-temp)
            ALERT_TEMP="$2"
            shift 2
            ;;
        --quiet)
            QUIET=true
            shift
            ;;
        --help)
            echo "📊 Resource Monitoring Script for LLaMA-3 8B Compression System"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --interval SECONDS    Monitoring interval (default: 2)"
            echo "  --duration MINUTES    Total monitoring time (default: unlimited)"
            echo "  --log FILE           Log output to file"
            echo "  --alert-memory PCT   Alert when GPU memory > PCT% (default: 90)"
            echo "  --alert-temp TEMP    Alert when GPU temp > TEMP°C (default: 85)"
            echo "  --quiet              Minimal output mode"
            echo "  --help               Show this help"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Basic monitoring"
            echo "  $0 --interval 5 --duration 30        # 5s interval for 30 minutes"
            echo "  $0 --log monitor.log --quiet          # Log to file, minimal output"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate interval
if ! [[ "$INTERVAL" =~ ^[0-9]+$ ]] || [ "$INTERVAL" -lt 1 ]; then
    echo "❌ Error: Interval must be a positive integer"
    exit 1
fi

# Setup logging
if [ ! -z "$LOG_FILE" ]; then
    mkdir -p "$(dirname "$LOG_FILE")"
    exec > >(tee -a "$LOG_FILE")
    exec 2>&1
fi

# Check prerequisites
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ Error: nvidia-smi not found"
    exit 1
fi

# Calculate end time if duration specified
if [ ! -z "$DURATION" ]; then
    if ! [[ "$DURATION" =~ ^[0-9]+$ ]]; then
        echo "❌ Error: Duration must be a positive integer (minutes)"
        exit 1
    fi
    END_TIME=$(($(date +%s) + $DURATION * 60))
fi

# Print header
if [ "$QUIET" = false ]; then
    echo "📊 Resource Monitoring - LLaMA-3 8B Compression System"
    echo "======================================================"
    echo "📍 Host: $(hostname)"
    echo "📅 Start: $(date)"
    echo "⏱️  Interval: ${INTERVAL}s"
    if [ ! -z "$DURATION" ]; then
        echo "⏰ Duration: ${DURATION}m"
    fi
    if [ ! -z "$LOG_FILE" ]; then
        echo "📝 Logging to: $LOG_FILE"
    fi
    echo ""
fi

# Function to format bytes
format_bytes() {
    local bytes=$1
    if [ $bytes -gt 1073741824 ]; then
        echo "$(($bytes / 1073741824))GB"
    elif [ $bytes -gt 1048576 ]; then
        echo "$(($bytes / 1048576))MB"
    elif [ $bytes -gt 1024 ]; then
        echo "$(($bytes / 1024))KB"
    else
        echo "${bytes}B"
    fi
}

# Function to get system memory info
get_system_memory() {
    local mem_info=$(free -b | grep "Mem:")
    local total=$(echo $mem_info | awk '{print $2}')
    local used=$(echo $mem_info | awk '{print $3}')
    local available=$(echo $mem_info | awk '{print $7}')
    local used_pct=$((used * 100 / total))
    
    echo "RAM: $(format_bytes $used)/$(format_bytes $total) (${used_pct}%)"
}

# Function to get CPU info
get_cpu_info() {
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    local load_avg=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | tr -d ',')
    
    echo "CPU: ${cpu_usage}% | Load: ${load_avg}"
}

# Function to get GPU info
get_gpu_info() {
    nvidia-smi --query-gpu=index,name,temperature.gpu,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits | while IFS=, read idx name temp mem_used mem_total gpu_util power; do
        local mem_pct=$((mem_used * 100 / mem_total))
        
        # Check alerts
        local alerts=""
        if [ $mem_pct -gt $ALERT_MEMORY ]; then
            alerts="${alerts}⚠️ HIGH-MEM "
        fi
        if [ $temp -gt $ALERT_TEMP ]; then
            alerts="${alerts}⚠️ HIGH-TEMP "
        fi
        
        echo "GPU$idx: ${mem_used}MB/${mem_total}MB (${mem_pct}%) | ${gpu_util}% | ${temp}°C | ${power}W ${alerts}"
    done
}

# Function to get process info
get_process_info() {
    local python_processes=$(pgrep -f python | wc -l)
    local pytorch_processes=$(pgrep -f "python.*torch" | wc -l)
    
    if [ $python_processes -gt 0 ]; then
        echo "Processes: ${python_processes} Python, ${pytorch_processes} PyTorch"
    else
        echo "Processes: No Python processes detected"
    fi
}

# Trap for cleanup
cleanup() {
    if [ "$QUIET" = false ]; then
        echo ""
        echo "📊 Monitoring stopped at $(date)"
        echo "🎉 Resource monitoring complete!"
    fi
    exit 0
}
trap cleanup SIGINT SIGTERM

# Main monitoring loop
COUNTER=0
while true; do
    # Check if duration limit reached
    if [ ! -z "$END_TIME" ] && [ $(date +%s) -gt $END_TIME ]; then
        break
    fi
    
    COUNTER=$((COUNTER + 1))
    TIMESTAMP=$(date +"%H:%M:%S")
    
    if [ "$QUIET" = false ]; then
        echo "[$TIMESTAMP] Monitor #$COUNTER"
        echo "$(get_system_memory)"
        echo "$(get_cpu_info)"
        get_gpu_info
        echo "$(get_process_info)"
        echo "----------------------------------------"
    else
        # Quiet mode: only show critical alerts
        gpu_info=$(get_gpu_info)
        if echo "$gpu_info" | grep -q "⚠️"; then
            echo "[$TIMESTAMP] $gpu_info"
        fi
    fi
    
    sleep $INTERVAL
done

cleanup