"""
Abstract interfaces for data handling operations.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Iterator, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class DatasetInterface(ABC):
    """Abstract interface for dataset operations."""
    
    @abstractmethod
    def load_dataset(self, **kwargs) -> None:
        """Load dataset."""
        pass
    
    @abstractmethod
    def get_samples(self, num_samples: int, **kwargs) -> List[str]:
        """
        Get dataset samples.
        
        Args:
            num_samples: Number of samples to retrieve
            **kwargs: Additional sampling parameters
            
        Returns:
            List of text samples
        """
        pass
    
    @abstractmethod
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text sample.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text
        """
        pass
    
    @abstractmethod
    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary containing dataset statistics
        """
        pass


class DataProcessorInterface(ABC):
    """Abstract interface for data processing operations."""
    
    @abstractmethod
    def tokenize_text(self, text: str, **kwargs) -> Dict[str, 'torch.Tensor']:
        """
        Tokenize input text.
        
        Args:
            text: Input text to tokenize
            **kwargs: Tokenization parameters
            
        Returns:
            Dictionary containing token tensors
        """
        pass
    
    @abstractmethod
    def process_batch(self, texts: List[str], **kwargs) -> Dict[str, 'torch.Tensor']:
        """
        Process batch of texts.
        
        Args:
            texts: List of input texts
            **kwargs: Processing parameters
            
        Returns:
            Dictionary containing processed batch tensors
        """
        pass
    
    @abstractmethod
    def extract_features(self, inputs: Dict[str, 'torch.Tensor'], **kwargs) -> 'torch.Tensor':
        """
        Extract features from processed inputs.
        
        Args:
            inputs: Processed input tensors
            **kwargs: Feature extraction parameters
            
        Returns:
            Feature tensor
        """
        pass


class DataValidatorInterface(ABC):
    """Abstract interface for data validation."""
    
    @abstractmethod
    def validate_input_format(self, data: Any) -> bool:
        """
        Validate input data format.
        
        Args:
            data: Input data to validate
            
        Returns:
            True if valid, False otherwise
        """
        pass
    
    @abstractmethod
    def validate_tensor_shapes(self, tensors: Dict[str, 'torch.Tensor'], 
                              expected_shapes: Dict[str, Tuple[int, ...]]) -> bool:
        """
        Validate tensor shapes.
        
        Args:
            tensors: Dictionary of tensors to validate
            expected_shapes: Expected shapes for each tensor
            
        Returns:
            True if all shapes are valid, False otherwise
        """
        pass
    
    @abstractmethod
    def get_validation_errors(self) -> List[str]:
        """
        Get list of validation errors.
        
        Returns:
            List of validation error messages
        """
        pass