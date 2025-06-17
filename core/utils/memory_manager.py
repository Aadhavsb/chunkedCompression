"""
Memory management utilities for efficient GPU and CPU memory usage.
"""
import gc
import time
from contextlib import contextmanager
from typing import Dict, Optional, Any

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class MemoryManager:
    """
    Centralized memory management for the chunked compression system.
    Provides context managers and utilities for efficient memory usage.
    """
    
    def __init__(self, cleanup_threshold: float = 0.8, auto_cleanup: bool = True):
        """
        Initialize memory manager.
        
        Args:
            cleanup_threshold: GPU memory usage threshold (0-1) to trigger cleanup
            auto_cleanup: Whether to automatically clean up when threshold is reached
        """
        self.cleanup_threshold = cleanup_threshold
        self.auto_cleanup = auto_cleanup
        self.peak_memory_usage = 0.0
        self.cleanup_count = 0
        
    @contextmanager
    def managed_computation(self):
        """
        Context manager for automatic memory cleanup after computation.
        
        Usage:
            with memory_manager.managed_computation():
                # Perform memory-intensive operations
                result = expensive_computation()
        """
        # Record memory state before computation
        initial_memory = self.get_memory_usage()
        
        try:
            yield
        finally:
            # Clean up after computation
            if self.auto_cleanup:
                self.cleanup_if_needed()
                
            # Update peak memory tracking
            final_memory = self.get_memory_usage()
            if final_memory['gpu_allocated_gb'] > self.peak_memory_usage:
                self.peak_memory_usage = final_memory['gpu_allocated_gb']
    
    def cleanup_if_needed(self) -> bool:
        """
        Clean up memory if usage exceeds threshold.
        
        Returns:
            True if cleanup was performed, False otherwise
        """
        if not torch.cuda.is_available():
            return False
        
        memory_usage = self.get_gpu_memory_usage_ratio()
        
        if memory_usage > self.cleanup_threshold:
            self.force_cleanup()
            self.cleanup_count += 1
            return True
        
        return False
    
    def force_cleanup(self) -> None:
        """Force immediate memory cleanup."""
        # Clear Python garbage
        gc.collect()
        
        # Clear CUDA cache if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def get_gpu_memory_usage_ratio(self) -> float:
        """
        Get GPU memory usage as a ratio (0-1).
        
        Returns:
            Memory usage ratio, or 0.0 if CUDA not available
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return 0.0
        
        allocated = torch.cuda.memory_allocated()
        cached = torch.cuda.memory_reserved()
        
        # Use the larger of allocated or cached memory
        used_memory = max(allocated, cached)
        total_memory = torch.cuda.get_device_properties(0).total_memory
        
        return used_memory / total_memory if total_memory > 0 else 0.0
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get comprehensive memory usage statistics.
        
        Returns:
            Dictionary containing memory usage metrics
        """
        # CPU memory stats (if psutil available)
        if PSUTIL_AVAILABLE:
            vm = psutil.virtual_memory()
            stats = {
                'cpu_percent': vm.percent,
                'cpu_available_gb': vm.available / (1024**3),
                'cpu_used_gb': vm.used / (1024**3),
                'cpu_total_gb': vm.total / (1024**3),
            }
        else:
            stats = {
                'cpu_percent': 0.0,
                'cpu_available_gb': 0.0,
                'cpu_used_gb': 0.0,
                'cpu_total_gb': 0.0,
            }
        
        # GPU memory stats (if torch and CUDA available)
        if TORCH_AVAILABLE and torch.cuda.is_available():
            allocated_bytes = torch.cuda.memory_allocated()
            cached_bytes = torch.cuda.memory_reserved()
            total_bytes = torch.cuda.get_device_properties(0).total_memory
            
            stats.update({
                'gpu_allocated_gb': allocated_bytes / (1024**3),
                'gpu_cached_gb': cached_bytes / (1024**3),
                'gpu_total_gb': total_bytes / (1024**3),
                'gpu_usage_ratio': self.get_gpu_memory_usage_ratio(),
                'gpu_available_gb': (total_bytes - cached_bytes) / (1024**3),
            })
        else:
            stats.update({
                'gpu_allocated_gb': 0.0,
                'gpu_cached_gb': 0.0,
                'gpu_total_gb': 0.0,
                'gpu_usage_ratio': 0.0,
                'gpu_available_gb': 0.0,
            })
        
        return stats
    
    def get_memory_summary(self) -> str:
        """
        Get a human-readable memory usage summary.
        
        Returns:
            Formatted string with memory usage information
        """
        stats = self.get_memory_usage()
        
        summary = f"Memory Usage Summary:\n"
        summary += f"  CPU: {stats['cpu_used_gb']:.1f}GB / {stats['cpu_total_gb']:.1f}GB ({stats['cpu_percent']:.1f}%)\n"
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            summary += f"  GPU: {stats['gpu_allocated_gb']:.1f}GB / {stats['gpu_total_gb']:.1f}GB ({stats['gpu_usage_ratio']*100:.1f}%)\n"
            summary += f"  GPU Cached: {stats['gpu_cached_gb']:.1f}GB\n"
        else:
            summary += f"  GPU: Not available\n"
        
        summary += f"  Peak GPU Usage: {self.peak_memory_usage:.1f}GB\n"
        summary += f"  Cleanup Count: {self.cleanup_count}"
        
        return summary
    
    @contextmanager
    def temporary_cleanup_threshold(self, threshold: float):
        """
        Temporarily change the cleanup threshold.
        
        Args:
            threshold: Temporary threshold value (0-1)
        """
        original_threshold = self.cleanup_threshold
        self.cleanup_threshold = threshold
        
        try:
            yield
        finally:
            self.cleanup_threshold = original_threshold
    
    def monitor_memory_usage(self, interval: float = 1.0, duration: float = 10.0) -> Dict[str, Any]:
        """
        Monitor memory usage over time.
        
        Args:
            interval: Monitoring interval in seconds
            duration: Total monitoring duration in seconds
            
        Returns:
            Dictionary with monitoring results
        """
        measurements = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            measurement = {
                'timestamp': time.time() - start_time,
                **self.get_memory_usage()
            }
            measurements.append(measurement)
            time.sleep(interval)
        
        # Calculate statistics
        if measurements:
            gpu_usage = [m['gpu_usage_ratio'] for m in measurements]
            cpu_usage = [m['cpu_percent'] for m in measurements]
            
            return {
                'measurements': measurements,
                'statistics': {
                    'avg_gpu_usage': sum(gpu_usage) / len(gpu_usage),
                    'max_gpu_usage': max(gpu_usage),
                    'min_gpu_usage': min(gpu_usage),
                    'avg_cpu_usage': sum(cpu_usage) / len(cpu_usage),
                    'max_cpu_usage': max(cpu_usage),
                    'min_cpu_usage': min(cpu_usage),
                },
                'duration': duration,
                'interval': interval,
                'total_measurements': len(measurements),
            }
        
        return {'measurements': [], 'statistics': {}, 'duration': duration}
    
    def reset_statistics(self) -> None:
        """Reset memory usage statistics."""
        self.peak_memory_usage = 0.0
        self.cleanup_count = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get memory manager statistics.
        
        Returns:
            Dictionary containing memory manager statistics
        """
        return {
            'cleanup_threshold': self.cleanup_threshold,
            'auto_cleanup': self.auto_cleanup,
            'peak_memory_usage_gb': self.peak_memory_usage,
            'cleanup_count': self.cleanup_count,
            'current_memory_usage': self.get_memory_usage(),
        }