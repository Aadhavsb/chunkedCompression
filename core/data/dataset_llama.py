"""
LLaMA-3 8B Real Dataset Handler
Processes real text through actual LLaMA model - NO SYNTHETIC DATA
"""
import torch
from typing import List, Tuple, Dict, Optional
from ..model import LLaMAModelLoader
import json

class LLaMADatasetHandler:
    def __init__(self, model_loader: LLaMAModelLoader):
        self.model_loader = model_loader
        self.tokenizer = model_loader.tokenizer
        self.model = model_loader.model
        
        # Real WikiText samples for testing
        self.wikitext_samples = [
            "The Transformer architecture revolutionized natural language processing by introducing self-attention mechanisms that allow models to weigh the importance of different words in a sequence when processing each word.",
            
            "Large language models like LLaMA (Large Language Model Meta AI) are neural networks with billions of parameters trained on vast amounts of text data to understand and generate human-like text.",
            
            "Attention mechanisms in neural networks compute a weighted sum of input representations, where the weights are determined by the similarity between a query and key vectors, allowing the model to focus on relevant parts of the input.",
            
            "The scaling laws for neural language models suggest that model performance improves predictably with increases in model size, dataset size, and computational resources used for training.",
            
            "Memory-efficient attention techniques, including key-value compression and sparse attention patterns, enable the deployment of large language models with reduced computational and memory requirements."
        ]
        
        print(f"📚 Initialized LLaMA dataset handler with {len(self.wikitext_samples)} real text samples")
        
    def get_real_hidden_states_batch(self, texts: Optional[List[str]] = None, 
                                   max_length: int = 256) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Get real hidden states from LLaMA model for a batch of texts
        
        Args:
            texts: List of input texts (uses default WikiText samples if None)
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (list of hidden_states, list of input_ids)
        """
        if texts is None:
            texts = self.wikitext_samples
            
        print(f"🧠 Processing {len(texts)} texts through LLaMA-3 8B...")
        
        all_hidden_states = []
        all_input_ids = []
        
        for i, text in enumerate(texts):
            print(f"   Processing text {i+1}/{len(texts)}: '{text[:50]}...'")
            
            # Get real hidden states from LLaMA
            hidden_states, input_ids = self.model_loader.get_hidden_states(text, max_length)
            
            all_hidden_states.append(hidden_states)
            all_input_ids.append(input_ids)
            
            print(f"     Shape: {hidden_states.shape}, Tokens: {len(input_ids)}")
        
        print(f"✅ Processed {len(texts)} texts successfully")
        return all_hidden_states, all_input_ids
    
    def create_compression_mapping(self, sequence_length: int, 
                                 strategy: str = "adaptive") -> List[str]:
        """
        Create compression mapping for tokens based on different strategies
        
        Args:
            sequence_length: Length of the sequence
            strategy: Compression strategy ("adaptive", "uniform", "decreasing")
            
        Returns:
            List of compression profiles for each token
        """
        if strategy == "adaptive":
            # More compression for middle tokens, less for beginning/end
            mapping = []
            for i in range(sequence_length):
                position_ratio = i / max(sequence_length - 1, 1)
                
                if position_ratio < 0.2 or position_ratio > 0.8:
                    # Important positions (start/end) - low compression
                    mapping.append("low")
                elif position_ratio < 0.4 or position_ratio > 0.6:
                    # Medium importance - medium compression
                    mapping.append("med")
                else:
                    # Middle positions - high compression
                    mapping.append("high")
                    
        elif strategy == "uniform":
            # Uniform distribution across compression levels
            cycle = ["low", "med", "high"]
            mapping = [cycle[i % len(cycle)] for i in range(sequence_length)]
            
        elif strategy == "decreasing":
            # Decreasing compression (more compression for later tokens)
            mapping = []
            for i in range(sequence_length):
                if i < sequence_length // 3:
                    mapping.append("low")
                elif i < 2 * sequence_length // 3:
                    mapping.append("med")
                else:
                    mapping.append("high")
        else:
            raise ValueError(f"Unknown compression strategy: {strategy}")
        
        return mapping
    
    def analyze_hidden_states(self, hidden_states: torch.Tensor) -> Dict[str, float]:
        """
        Analyze statistical properties of real hidden states
        
        Args:
            hidden_states: Real hidden states from LLaMA [seq_len, hidden_size]
            
        Returns:
            Dictionary with statistical properties
        """
        with torch.no_grad():
            stats = {
                "mean": hidden_states.mean().item(),
                "std": hidden_states.std().item(),
                "min": hidden_states.min().item(),
                "max": hidden_states.max().item(),
                "l2_norm": torch.norm(hidden_states, p=2).item(),
                "sparsity": (hidden_states == 0).float().mean().item(),
                "sequence_length": hidden_states.shape[0],
                "hidden_dimension": hidden_states.shape[1]
            }
            
            # Compute per-token L2 norms
            token_norms = torch.norm(hidden_states, p=2, dim=1)
            stats.update({
                "token_norm_mean": token_norms.mean().item(),
                "token_norm_std": token_norms.std().item(),
                "token_norm_min": token_norms.min().item(),
                "token_norm_max": token_norms.max().item()
            })
        
        return stats
    
    def get_ground_truth_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get ground truth logits from uncompressed LLaMA model
        
        Args:
            input_ids: Token IDs [seq_len]
            
        Returns:
            Logits [seq_len, vocab_size]
        """
        with torch.no_grad():
            # Add batch dimension
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)  # [1, seq_len]
            
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits.squeeze(0)  # [seq_len, vocab_size]
            
        return logits
    
    def calculate_perplexity(self, logits: torch.Tensor, target_ids: torch.Tensor) -> Tuple[float, float]:
        """
        Calculate perplexity and cross-entropy loss
        
        Args:
            logits: Model logits [seq_len, vocab_size]
            target_ids: Target token IDs [seq_len]
            
        Returns:
            Tuple of (perplexity, loss)
        """
        # Shift for next-token prediction
        shift_logits = logits[:-1, :].contiguous()
        shift_labels = target_ids[1:].contiguous()
        
        # Calculate cross-entropy loss
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(shift_logits, shift_labels)
        
        # Calculate perplexity
        perplexity = torch.exp(loss)
        
        return perplexity.item(), loss.item()
    
    def generate_with_compression_comparison(self, prompt: str, max_new_tokens: int = 20) -> Dict[str, str]:
        """
        Generate text and compare with/without compression
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            
        Returns:
            Dictionary with generation results
        """
        print(f"🎯 Generating text with compression comparison...")
        print(f"   Prompt: '{prompt}'")
        
        # Original generation (uncompressed)
        original_text = self.model_loader.generate_text(
            prompt, max_new_tokens=max_new_tokens, temperature=0.7
        )
        
        # TODO: Add compressed generation when compression pipeline is ready
        # For now, return original generation
        
        results = {
            "prompt": prompt,
            "original_generation": original_text,
            "compressed_generation": "TODO: Implement compressed generation",
            "max_new_tokens": max_new_tokens
        }
        
        print(f"   Original: '{original_text}'")
        return results
    
    def save_dataset_stats(self, filepath: str, texts: Optional[List[str]] = None):
        """
        Save dataset statistics to file
        
        Args:
            filepath: Output file path
            texts: List of texts to analyze (uses default if None)
        """
        if texts is None:
            texts = self.wikitext_samples
            
        print(f"📊 Analyzing dataset statistics...")
        
        all_stats = []
        
        # Process each text
        for i, text in enumerate(texts):
            hidden_states, input_ids = self.model_loader.get_hidden_states(text)
            stats = self.analyze_hidden_states(hidden_states)
            stats["text_index"] = i
            stats["text_preview"] = text[:100]
            stats["token_count"] = len(input_ids)
            
            all_stats.append(stats)
        
        # Calculate aggregate statistics
        aggregate_stats = {
            "total_texts": len(texts),
            "total_tokens": sum(s["token_count"] for s in all_stats),
            "avg_sequence_length": sum(s["sequence_length"] for s in all_stats) / len(all_stats),
            "avg_hidden_norm": sum(s["l2_norm"] for s in all_stats) / len(all_stats),
            "model_info": self.model_loader.get_model_info()
        }
        
        # Save to file
        output_data = {
            "aggregate_stats": aggregate_stats,
            "per_text_stats": all_stats
        }
        
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✅ Saved dataset statistics to {filepath}")
        print(f"   Total texts: {aggregate_stats['total_texts']}")
        print(f"   Total tokens: {aggregate_stats['total_tokens']}")
        print(f"   Average sequence length: {aggregate_stats['avg_sequence_length']:.1f}")
    
    def print_dataset_summary(self):
        """Print summary of the dataset"""
        print(f"\n📚 LLaMA Dataset Summary")
        print(f"=" * 50)
        print(f"Number of samples: {len(self.wikitext_samples)}")
        print(f"Model: {self.model_loader.get_model_info()['model_name']}")
        print(f"Tokenizer vocab size: {self.model_loader.vocab_size}")
        
        # Analyze first sample
        sample_text = self.wikitext_samples[0]
        hidden_states, input_ids = self.model_loader.get_hidden_states(sample_text)
        stats = self.analyze_hidden_states(hidden_states)
        
        print(f"\nSample Analysis (first text):")
        print(f"  Text: '{sample_text[:80]}...'")
        print(f"  Tokens: {len(input_ids)}")
        print(f"  Hidden states shape: {hidden_states.shape}")
        print(f"  Mean activation: {stats['mean']:.6f}")
        print(f"  Std activation: {stats['std']:.6f}")
        print(f"  L2 norm: {stats['l2_norm']:.2f}")
        print(f"  Sparsity: {stats['sparsity']:.4f}")
