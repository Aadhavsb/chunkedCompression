"""
KV Cache for storing compressed latent vectors
"""
import torch
from typing import Dict, List, Tuple
from collections import defaultdict

class KVCache:
    def __init__(self):
        # Storage: (group_id, option) -> List[latent_vectors]
        self.cache: Dict[Tuple[int, str], List[torch.Tensor]] = defaultdict(list)
        
    def append(self, token_idx: int, group_id: int, h_t: torch.Tensor, option: str):
        """
        Store a compressed latent vector
        
        Args:
            token_idx: Token position index
            group_id: Head group identifier
            h_t: Latent vector [r_opt]
            option: Compression option ("low", "med", "high")
        """
        key = (group_id, option)
        self.cache[key].append(h_t)
    
    def retrieve(self, group_id: int, option: str) -> torch.Tensor:
        """
        Retrieve all latent vectors for a group/option combination
        
        Args:
            group_id: Head group identifier
            option: Compression option
            
        Returns:
            Tensor of shape [T, r_opt] where T is number of tokens for this group/option
        """
        key = (group_id, option)
        if key not in self.cache or len(self.cache[key]) == 0:
            return torch.empty(0, 0)
        
        # Stack all latent vectors for this group/option
        return torch.stack(self.cache[key], dim=0)
    
    def get_all_keys(self) -> List[Tuple[int, str]]:
        """Get all (group_id, option) keys in the cache"""
        return list(self.cache.keys())
    
    def clear(self):
        """Clear the cache"""
        self.cache.clear()
    
    def size(self) -> int:
        """Total number of cached entries"""
        return sum(len(vectors) for vectors in self.cache.values())
