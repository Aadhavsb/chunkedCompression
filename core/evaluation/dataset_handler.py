"""
Standard Dataset Handler for LLaMA Compression Evaluation
Implements WikiText-2, C4, and PTB evaluation
"""

import torch
from datasets import load_dataset
from typing import List, Dict, Iterator, Optional, Tuple
import numpy as np
from transformers import AutoTokenizer


class StandardDatasetHandler:
    """Handle standard evaluation datasets with proper preprocessing"""
    
    def __init__(self, tokenizer: AutoTokenizer, device: str = "cuda"):
        self.tokenizer = tokenizer
        self.device = device
        
        # Dataset configurations
        self.datasets_config = {
            "wikitext2": {
                "name": "wikitext",
                "config": "wikitext-2-raw-v1", 
                "split": "test",
                "seq_len": 2048,
                "stride": 512
            },
            "c4": {
                "name": "c4",
                "config": "en",
                "split": "validation",
                "seq_len": 2048, 
                "stride": 512
            },
            "ptb": {
                "name": "ptb_text_only",
                "config": None,
                "split": "test",
                "seq_len": 2048,
                "stride": 512
            }
        }
        
    def load_dataset_for_perplexity(self, dataset_name: str, 
                                  max_samples: Optional[int] = None,
                                  seq_len: Optional[int] = None) -> Iterator[torch.Tensor]:
        """
        Load dataset for perplexity evaluation
        
        Args:
            dataset_name: One of 'wikitext2', 'c4', 'ptb'
            max_samples: Maximum number of samples to load
            seq_len: Sequence length override
            
        Yields:
            input_ids: Tokenized sequences [seq_len]
        """
        if dataset_name not in self.datasets_config:
            raise ValueError(f"Dataset {dataset_name} not supported. Available: {list(self.datasets_config.keys())}")
        
        config = self.datasets_config[dataset_name]
        effective_seq_len = seq_len or config["seq_len"]
        
        print(f"📚 Loading {dataset_name} dataset for perplexity evaluation...")
        print(f"   Sequence length: {effective_seq_len}")
        print(f"   Max samples: {max_samples if max_samples else 'unlimited'}")
        
        try:
            # Load dataset
            if config["config"]:
                dataset = load_dataset(config["name"], config["config"], split=config["split"])
            else:
                dataset = load_dataset(config["name"], split=config["split"])
            
            # Preprocess text field
            if dataset_name == "wikitext2":
                text_field = "text" 
            elif dataset_name == "c4":
                text_field = "text"
            elif dataset_name == "ptb":
                text_field = "sentence"
            else:
                text_field = "text"
            
            # Concatenate all text (don't limit by max_samples here)
            all_text = ""
            sample_count = 0
            
            for item in dataset:
                text = item[text_field].strip()
                if text:  # Skip empty lines
                    all_text += text + " "
                    sample_count += 1
                    
                # Stop after reasonable amount of text to avoid memory issues
                if len(all_text) > 10_000_000:  # 10MB of text should be plenty
                    break
            
            print(f"   Loaded {sample_count} text samples")
            print(f"   Total characters: {len(all_text):,}")
            
            # Tokenize entire text
            encoded = self.tokenizer(
                all_text,
                return_tensors="pt",
                truncation=False,
                padding=False
            )
            
            input_ids = encoded["input_ids"].squeeze(0)  # [total_tokens]
            print(f"   Total tokens: {len(input_ids):,}")
            
            # Generate sliding windows
            stride = config["stride"]
            num_sequences = 0
            
            for i in range(0, len(input_ids) - effective_seq_len + 1, stride):
                if max_samples and num_sequences >= max_samples:
                    break
                    
                sequence = input_ids[i:i + effective_seq_len]
                if len(sequence) == effective_seq_len:
                    yield sequence.to(self.device)
                    num_sequences += 1
            
            print(f"   Generated {num_sequences} sequences of length {effective_seq_len}")
            
        except Exception as e:
            print(f"❌ Error loading {dataset_name}: {e}")
            print("🔄 Falling back to sample data...")
            
            # Fallback to sample data
            fallback_texts = self._get_fallback_samples(dataset_name)
            for text in fallback_texts[:max_samples] if max_samples else fallback_texts:
                encoded = self.tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=effective_seq_len,
                    truncation=True,
                    padding=False
                )
                yield encoded["input_ids"].squeeze(0).to(self.device)
    
    def load_calibration_data(self, dataset_name: str = "wikitext2", 
                            num_samples: int = 256,
                            seq_len: int = 2048) -> List[torch.Tensor]:
        """
        Load calibration data for compression profile building
        
        Args:
            dataset_name: Dataset to use for calibration
            num_samples: Number of calibration samples
            seq_len: Sequence length for calibration
            
        Returns:
            List of tokenized sequences for calibration
        """
        print(f"🎯 Loading calibration data from {dataset_name}...")
        print(f"   Samples: {num_samples}, Sequence length: {seq_len}")
        
        calibration_data = []
        sample_count = 0
        
        for input_ids in self.load_dataset_for_perplexity(dataset_name, seq_len=seq_len):
            if sample_count >= num_samples:
                break
            calibration_data.append(input_ids)
            sample_count += 1
        
        print(f"✅ Loaded {len(calibration_data)} calibration samples")
        return calibration_data
    
    def _get_fallback_samples(self, dataset_name: str) -> List[str]:
        """Get fallback samples when dataset loading fails"""
        
        fallback_data = {
            "wikitext2": [
                "The Tower of London is a historic castle located on the north bank of the River Thames in central London. It was founded towards the end of 1066 as part of the Norman Conquest of England. The White Tower, which gives the entire castle its name, was built by William the Conqueror in 1078 and was a resented symbol of oppression, inflicted upon London by the new ruling elite.",
                
                "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of intelligent agents: any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.",
                
                "The Internet is a global system of interconnected computer networks that uses the Internet protocol suite (TCP/IP) to communicate between networks and devices. It is a network of networks that consists of private, public, academic, business, and government networks of local to global scope.",
                
                "Machine learning (ML) is a type of artificial intelligence (AI) that allows software applications to become more accurate at predicting outcomes without being explicitly programmed to do so. Machine learning algorithms use historical data as input to predict new output values.",
                
                "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data."
            ],
            
            "c4": [
                "Climate change refers to long-term shifts in global or regional climate patterns. Since the mid-20th century, scientists have observed that the primary driver has been the increase in greenhouse gases produced by human activities, particularly the burning of fossil fuels.",
                
                "Renewable energy comes from natural sources or processes that are constantly replenished. For example, sunlight or wind keep shining and blowing, even if their availability depends on time and weather conditions.",
                
                "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.",
                
                "Quantum computing is a type of computation that harnesses the collective properties of quantum states, such as superposition, interference, and entanglement, to perform calculations.",
                
                "Blockchain technology is a decentralized, distributed and public digital ledger that is used to record transactions across many computers so that the record cannot be altered retroactively without the alteration of all subsequent blocks."
            ],
            
            "ptb": [
                "The company reported strong quarterly earnings despite economic headwinds affecting the broader market.",
                "Researchers at the university published findings that could revolutionize treatment approaches for the disease.",
                "Government officials announced new policy measures aimed at addressing infrastructure challenges.",
                "Technology stocks rallied following positive developments in the semiconductor industry.",
                "The central bank's decision to adjust interest rates reflects ongoing economic uncertainties."
            ]
        }
        
        return fallback_data.get(dataset_name, fallback_data["wikitext2"])
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, any]:
        """Get information about a dataset"""
        if dataset_name not in self.datasets_config:
            return {"error": f"Dataset {dataset_name} not supported"}
        
        config = self.datasets_config[dataset_name]
        
        try:
            # Try to load a small sample to get info
            if config["config"]:
                dataset = load_dataset(config["name"], config["config"], split=config["split"])
            else:
                dataset = load_dataset(config["name"], split=config["split"])
            
            return {
                "dataset_name": dataset_name,
                "hf_name": config["name"],
                "config": config["config"],
                "split": config["split"],
                "num_examples": len(dataset),
                "features": list(dataset.features.keys()),
                "default_seq_len": config["seq_len"],
                "stride": config["stride"]
            }
        
        except Exception as e:
            return {
                "dataset_name": dataset_name,
                "error": str(e),
                "fallback_available": True
            }
    
    def print_dataset_summary(self):
        """Print summary of all available datasets"""
        print("\n📊 Available Evaluation Datasets")
        print("=" * 50)
        
        for dataset_name in self.datasets_config.keys():
            info = self.get_dataset_info(dataset_name)
            print(f"\n{dataset_name.upper()}:")
            
            if "error" in info:
                print(f"  ❌ {info['error']}")
                if info.get("fallback_available"):
                    print(f"  🔄 Fallback samples available")
            else:
                print(f"  📚 HuggingFace: {info['hf_name']}")
                if info['config']:
                    print(f"  🔧 Config: {info['config']}")
                print(f"  📄 Split: {info['split']} ({info['num_examples']:,} examples)")
                print(f"  🔤 Sequence length: {info['default_seq_len']}")
                print(f"  📐 Features: {', '.join(info['features'])}")