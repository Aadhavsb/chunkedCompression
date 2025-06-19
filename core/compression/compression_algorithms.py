"""
Core compression algorithms and SVD utilities.
"""
from typing import Tuple, Optional, Dict, Any
from ..interfaces.compression_interface import CompressionAlgorithmInterface

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy torch for type hints
    torch = None


class SVDCompressionAlgorithm(CompressionAlgorithmInterface):
    """
    SVD-based compression algorithm for transformer attention weights.
    Implements the core mathematical operations for matrix decomposition and compression.
    """
    
    def __init__(self, solver: str = "auto", tolerance: float = 1e-6):
        """
        Initialize the SVD compression algorithm.
        
        Args:
            solver: SVD solver type ("auto", "full", "arpack", "randomized")
            tolerance: Numerical tolerance for SVD computation
        """
        self.solver = solver
        self.tolerance = tolerance
        self._validate_solver()
    
    def _validate_solver(self) -> None:
        """Validate SVD solver configuration."""
        valid_solvers = ["auto", "full", "arpack", "randomized"]
        if self.solver not in valid_solvers:
            raise ValueError(f"Invalid SVD solver: {self.solver}. Must be one of {valid_solvers}")
    
    def perform_svd_compression(self, weight_matrix, rank: int):
        """
        Perform SVD-based compression on weight matrix.
        
        Args:
            weight_matrix: Weight matrix to compress [input_dim, output_dim]
            rank: Target compression rank
            
        Returns:
            Tuple of (compression_matrix, reconstruction_matrix)
            - compression_matrix: [rank, input_dim] for compressing inputs
            - reconstruction_matrix: [input_dim, rank] for reconstructing
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for compression algorithms but is not installed")
        
        if weight_matrix.numel() == 0:
            raise ValueError("Cannot compress empty weight matrix")
        
        if rank <= 0:
            raise ValueError(f"Compression rank must be positive, got {rank}")
        
        # Convert to float32 for SVD if needed (bfloat16 not supported)
        original_dtype = weight_matrix.dtype
        if weight_matrix.dtype in [torch.bfloat16, torch.float16]:
            weight_matrix = weight_matrix.float()
        
        # Perform SVD decomposition
        try:
            U, S, V = torch.svd(weight_matrix, some=True)
        except RuntimeError as e:
            raise RuntimeError(f"SVD decomposition failed: {e}")
        
        # Determine actual rank (may be limited by matrix dimensions)
        actual_rank = min(rank, U.shape[1], S.shape[0], V.shape[1])
        
        if actual_rank < rank:
            print(f"Warning: Requested rank {rank} reduced to {actual_rank} due to matrix dimensions")
        
        # Truncate to actual rank
        U_truncated = U[:, :actual_rank]  # [input_dim, actual_rank]
        S_truncated = S[:actual_rank]     # [actual_rank]
        V_truncated = V[:, :actual_rank]  # [output_dim, actual_rank]
        
        # Create compression and reconstruction matrices
        # For compressing: input -> compressed: compression_matrix @ input
        # For reconstructing: compressed -> output: U @ diag(S) @ compressed
        compression_matrix = (S_truncated.unsqueeze(1) * V_truncated.T)  # [actual_rank, output_dim]
        reconstruction_matrix = U_truncated  # [input_dim, actual_rank]
        
        # Convert back to original dtype
        compression_matrix = compression_matrix.to(original_dtype)
        reconstruction_matrix = reconstruction_matrix.to(original_dtype)
        
        return compression_matrix, reconstruction_matrix
    
    def perform_value_svd_with_fusion(self, 
                                    value_weight: torch.Tensor,
                                    output_weight: torch.Tensor,
                                    lm_head_weight: torch.Tensor,
                                    rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform SVD compression on value weights with fused output projection.
        
        Args:
            value_weight: Value projection weight [head_dim, hidden_size]
            output_weight: Output projection weight [hidden_size, head_dim]
            lm_head_weight: Language model head weight [vocab_size, hidden_size]
            rank: Target compression rank
            
        Returns:
            Tuple of (compression_matrix, fused_projection)
            - compression_matrix: [rank, head_dim] for value compression
            - fused_projection: [vocab_size, rank] for direct vocabulary projection
        """
        # Validate inputs
        if value_weight.shape[0] != output_weight.shape[1]:
            raise ValueError(f"Dimension mismatch: value_weight head_dim {value_weight.shape[0]} "
                           f"!= output_weight head_dim {output_weight.shape[1]}")
        
        if output_weight.shape[0] != lm_head_weight.shape[1]:
            raise ValueError(f"Dimension mismatch: output_weight hidden_size {output_weight.shape[0]} "
                           f"!= lm_head_weight hidden_size {lm_head_weight.shape[1]}")
        
        # Convert to float32 for SVD
        original_dtype = value_weight.dtype
        if value_weight.dtype in [torch.bfloat16, torch.float16]:
            value_weight = value_weight.float()
            output_weight = output_weight.float()
            lm_head_weight = lm_head_weight.float()
        
        # Perform SVD on value weight
        U, S, V = torch.svd(value_weight, some=True)
        
        # Determine actual rank
        actual_rank = min(rank, U.shape[1], S.shape[0])
        
        # Truncate to actual rank
        U_truncated = U[:, :actual_rank]      # [head_dim, actual_rank]
        S_truncated = S[:actual_rank]         # [actual_rank]
        
        # Create compression matrix: transforms values from [head_dim] to [actual_rank]
        compression_matrix = U_truncated.T    # [actual_rank, head_dim]
        
        # Create fused projection: compressed_value [actual_rank] -> logits [vocab_size]
        # Path: compressed_value -> U @ diag(S) -> output_proj -> lm_head
        fused_projection = lm_head_weight @ output_weight @ U_truncated @ torch.diag(S_truncated)
        # Shape: [vocab_size, actual_rank]
        
        # Convert back to original dtype
        compression_matrix = compression_matrix.to(original_dtype)
        fused_projection = fused_projection.to(original_dtype)
        
        return compression_matrix, fused_projection
    
    def compute_compression_ratio(self, 
                                original_shape: Tuple[int, ...], 
                                compressed_shapes: Tuple[Tuple[int, ...], ...]) -> float:
        """
        Compute compression ratio between original and compressed representations.
        
        Args:
            original_shape: Shape of original tensor
            compressed_shapes: Shapes of compressed tensors (can be multiple)
            
        Returns:
            Compression ratio (original_params / compressed_params)
        """
        # Calculate original parameters
        original_params = 1
        for dim in original_shape:
            original_params *= dim
        
        # Calculate compressed parameters
        compressed_params = 0
        for shape in compressed_shapes:
            shape_params = 1
            for dim in shape:
                shape_params *= dim
            compressed_params += shape_params
        
        if compressed_params == 0:
            return float('inf')
        
        return original_params / compressed_params
    
    def estimate_memory_savings(self, 
                               original_shapes: Dict[str, Tuple[int, ...]], 
                               compressed_shapes: Dict[str, Tuple[int, ...]], 
                               dtype_bytes: int = 2) -> Dict[str, float]:
        """
        Estimate memory savings from compression.
        
        Args:
            original_shapes: Dictionary of original tensor shapes
            compressed_shapes: Dictionary of compressed tensor shapes
            dtype_bytes: Bytes per parameter (2 for float16/bfloat16, 4 for float32)
            
        Returns:
            Dictionary with memory usage statistics
        """
        original_memory = 0
        compressed_memory = 0
        
        # Calculate original memory
        for name, shape in original_shapes.items():
            params = 1
            for dim in shape:
                params *= dim
            original_memory += params * dtype_bytes
        
        # Calculate compressed memory
        for name, shape in compressed_shapes.items():
            params = 1
            for dim in shape:
                params *= dim
            compressed_memory += params * dtype_bytes
        
        # Convert to MB
        original_mb = original_memory / (1024 * 1024)
        compressed_mb = compressed_memory / (1024 * 1024)
        
        savings_mb = original_mb - compressed_mb
        savings_percent = (savings_mb / original_mb * 100) if original_mb > 0 else 0
        compression_ratio = original_mb / compressed_mb if compressed_mb > 0 else float('inf')
        
        return {
            'original_memory_mb': original_mb,
            'compressed_memory_mb': compressed_mb,
            'memory_savings_mb': savings_mb,
            'memory_savings_percent': savings_percent,
            'compression_ratio': compression_ratio,
        }
    
    def validate_compression_quality(self, 
                                   original_matrix: torch.Tensor,
                                   compression_matrix: torch.Tensor,
                                   reconstruction_matrix: torch.Tensor) -> Dict[str, float]:
        """
        Validate the quality of compression by comparing reconstruction error.
        
        Args:
            original_matrix: Original weight matrix
            compression_matrix: Compression matrix
            reconstruction_matrix: Reconstruction matrix
            
        Returns:
            Dictionary with quality metrics
        """
        # Reconstruct the matrix
        reconstructed = reconstruction_matrix @ compression_matrix
        
        # Calculate quality metrics
        mse = torch.mean((original_matrix - reconstructed) ** 2).item()
        
        # Frobenius norm relative error
        original_norm = torch.norm(original_matrix, 'fro').item()
        error_norm = torch.norm(original_matrix - reconstructed, 'fro').item()
        relative_error = error_norm / original_norm if original_norm > 0 else float('inf')
        
        # Cosine similarity (flattened)
        original_flat = original_matrix.flatten()
        reconstructed_flat = reconstructed.flatten()
        cosine_sim = torch.nn.functional.cosine_similarity(
            original_flat.unsqueeze(0), 
            reconstructed_flat.unsqueeze(0)
        ).item()
        
        return {
            'mse': mse,
            'relative_error': relative_error,
            'cosine_similarity': cosine_sim,
            'original_norm': original_norm,
            'reconstruction_norm': torch.norm(reconstructed, 'fro').item(),
        }