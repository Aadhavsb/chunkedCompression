"""
Inference and benchmarking configuration management.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class InferenceConfig:
    """Configuration for inference operations."""
    
    # Generation parameters
    max_length: int = 50
    max_new_tokens: Optional[int] = None
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    do_sample: bool = False
    
    # Batch processing
    batch_size: int = 1
    max_batch_size: int = 8
    
    # Memory management
    use_cache: bool = True
    cache_implementation: str = "compressed"  # "compressed", "standard", "hybrid"
    gradient_checkpointing: bool = False
    
    # Attention parameters
    attention_implementation: str = "compressed"  # "compressed", "standard", "flash"
    use_scaled_dot_product_attention: bool = False
    
    # Performance monitoring
    track_memory: bool = True
    track_timing: bool = True
    profile_attention: bool = False
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")
        
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        
        valid_cache_implementations = ["compressed", "standard", "hybrid"]
        if self.cache_implementation not in valid_cache_implementations:
            raise ValueError(f"Invalid cache implementation: {self.cache_implementation}. "
                           f"Must be one of {valid_cache_implementations}")
        
        valid_attention_implementations = ["compressed", "standard", "flash"]
        if self.attention_implementation not in valid_attention_implementations:
            raise ValueError(f"Invalid attention implementation: {self.attention_implementation}. "
                           f"Must be one of {valid_attention_implementations}")
    
    @classmethod
    def from_env(cls) -> 'InferenceConfig':
        """Create configuration from environment variables."""
        return cls(
            max_length=int(os.getenv('MAX_LENGTH', cls.max_length)),
            temperature=float(os.getenv('TEMPERATURE', cls.temperature)),
            top_p=float(os.getenv('TOP_P', cls.top_p)),
            top_k=int(os.getenv('TOP_K', cls.top_k)),
            do_sample=os.getenv('DO_SAMPLE', 'false').lower() == 'true',
            batch_size=int(os.getenv('BATCH_SIZE', cls.batch_size)),
            use_cache=os.getenv('USE_CACHE', 'true').lower() == 'true',
            cache_implementation=os.getenv('CACHE_IMPLEMENTATION', cls.cache_implementation),
            attention_implementation=os.getenv('ATTENTION_IMPLEMENTATION', cls.attention_implementation),
            track_memory=os.getenv('TRACK_MEMORY', 'true').lower() == 'true',
            track_timing=os.getenv('TRACK_TIMING', 'true').lower() == 'true',
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'max_length': self.max_length,
            'max_new_tokens': self.max_new_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'top_k': self.top_k,
            'do_sample': self.do_sample,
            'batch_size': self.batch_size,
            'max_batch_size': self.max_batch_size,
            'use_cache': self.use_cache,
            'cache_implementation': self.cache_implementation,
            'gradient_checkpointing': self.gradient_checkpointing,
            'attention_implementation': self.attention_implementation,
            'use_scaled_dot_product_attention': self.use_scaled_dot_product_attention,
            'track_memory': self.track_memory,
            'track_timing': self.track_timing,
            'profile_attention': self.profile_attention,
        }


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark operations."""
    
    # Test parameters
    num_samples: int = 10
    num_iterations: int = 5
    warmup_iterations: int = 2
    
    # Test data
    test_texts: Optional[List[str]] = None
    test_dataset: str = "openwebtext"  # Dataset name for testing
    
    # Metrics to collect
    collect_quality_metrics: bool = True
    collect_performance_metrics: bool = True
    collect_memory_metrics: bool = True
    
    # Quality metrics
    calculate_mse: bool = True
    calculate_cosine_similarity: bool = True
    calculate_perplexity: bool = False  # Computationally expensive
    
    # Comparison baselines
    compare_with_standard: bool = True
    compare_with_uncompressed: bool = True
    
    # Output configuration
    save_results: bool = True
    results_dir: str = "tests/results"
    include_timestamp: bool = True
    
    # Visualization
    generate_plots: bool = False
    plot_compression_ratios: bool = True
    plot_quality_metrics: bool = True
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {self.num_samples}")
        
        if self.num_iterations <= 0:
            raise ValueError(f"num_iterations must be positive, got {self.num_iterations}")
        
        if self.warmup_iterations < 0:
            raise ValueError(f"warmup_iterations must be non-negative, got {self.warmup_iterations}")
        
        if self.warmup_iterations >= self.num_iterations:
            raise ValueError(f"warmup_iterations ({self.warmup_iterations}) must be less than "
                           f"num_iterations ({self.num_iterations})")
    
    @classmethod
    def from_env(cls) -> 'BenchmarkConfig':
        """Create configuration from environment variables."""
        return cls(
            num_samples=int(os.getenv('BENCHMARK_SAMPLES', cls.num_samples)),
            num_iterations=int(os.getenv('BENCHMARK_ITERATIONS', cls.num_iterations)),
            warmup_iterations=int(os.getenv('BENCHMARK_WARMUP', cls.warmup_iterations)),
            test_dataset=os.getenv('TEST_DATASET', cls.test_dataset),
            collect_quality_metrics=os.getenv('COLLECT_QUALITY', 'true').lower() == 'true',
            collect_performance_metrics=os.getenv('COLLECT_PERFORMANCE', 'true').lower() == 'true',
            collect_memory_metrics=os.getenv('COLLECT_MEMORY', 'true').lower() == 'true',
            calculate_perplexity=os.getenv('CALCULATE_PERPLEXITY', 'false').lower() == 'true',
            save_results=os.getenv('SAVE_RESULTS', 'true').lower() == 'true',
            results_dir=os.getenv('RESULTS_DIR', cls.results_dir),
            generate_plots=os.getenv('GENERATE_PLOTS', 'false').lower() == 'true',
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'num_samples': self.num_samples,
            'num_iterations': self.num_iterations,
            'warmup_iterations': self.warmup_iterations,
            'test_texts': self.test_texts,
            'test_dataset': self.test_dataset,
            'collect_quality_metrics': self.collect_quality_metrics,
            'collect_performance_metrics': self.collect_performance_metrics,
            'collect_memory_metrics': self.collect_memory_metrics,
            'calculate_mse': self.calculate_mse,
            'calculate_cosine_similarity': self.calculate_cosine_similarity,
            'calculate_perplexity': self.calculate_perplexity,
            'compare_with_standard': self.compare_with_standard,
            'compare_with_uncompressed': self.compare_with_uncompressed,
            'save_results': self.save_results,
            'results_dir': self.results_dir,
            'include_timestamp': self.include_timestamp,
            'generate_plots': self.generate_plots,
            'plot_compression_ratios': self.plot_compression_ratios,
            'plot_quality_metrics': self.plot_quality_metrics,
        }