"""
KV Cache for storing compressed latent vectors (both keys and values)
"""
import torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

class KVCache:
    def __init__(self):
        # Storage for compressed values: (group_id, option) -> List[latent_vectors]
        self.v_cache: Dict[Tuple[int, str], List[torch.Tensor]] = defaultdict(list)
        
        # Storage for compressed keys: (group_id, option) -> List[compressed_keys]
        self.k_cache: Dict[Tuple[int, str], List[torch.Tensor]] = defaultdict(list)
        
        # Storage for token metadata
        self.token_metadata: Dict[Tuple[int, str], List[int]] = defaultdict(list)  # token indices
        
    def append(self, token_idx: int, group_id: int, h_v: torch.Tensor, h_k: torch.Tensor, option: str):
        """
        Store compressed latent vectors for both keys and values
        
        Args:
            token_idx: Token position index
            group_id: Head group identifier
            h_v: Compressed value vector [r_v]
            h_k: Compressed key vector [r_k]
            option: Compression option ("low", "med", "high")
        """
        key = (group_id, option)
        self.v_cache[key].append(h_v)
        self.k_cache[key].append(h_k)
        self.token_metadata[key].append(token_idx)
    
    def retrieve_values(self, group_id: int, option: str) -> torch.Tensor:
        """
        Retrieve all compressed value vectors for a group/option combination
        
        Args:
            group_id: Head group identifier
            option: Compression option
            
        Returns:
            Tensor of shape [T, r_v] where T is number of tokens for this group/option
        """
        key = (group_id, option)
        if key not in self.v_cache or len(self.v_cache[key]) == 0:
            return torch.empty(0, 0)
        
        # Stack all value latent vectors for this group/option
        return torch.stack(self.v_cache[key], dim=0)
    
    def retrieve_keys(self, group_id: int, option: str) -> torch.Tensor:
        """
        Retrieve all compressed key vectors for a group/option combination
        
        Args:
            group_id: Head group identifier
            option: Compression option
            
        Returns:
            Tensor of shape [T, r_k] where T is number of tokens for this group/option
        """
        key = (group_id, option)
        if key not in self.k_cache or len(self.k_cache[key]) == 0:
            return torch.empty(0, 0)
        
        # Stack all key latent vectors for this group/option
        return torch.stack(self.k_cache[key], dim=0)
    
    def get_token_indices(self, group_id: int, option: str) -> List[int]:
        """Get token indices for a group/option combination"""
        key = (group_id, option)
        return self.token_metadata.get(key, [])
    
    def get_all_keys(self) -> List[Tuple[int, str]]:
        """Get all (group_id, option) keys in the cache"""
        return list(self.v_cache.keys())
    
    def clear(self):
        """Clear the cache"""
        self.v_cache.clear()
        self.k_cache.clear()
        self.token_metadata.clear()
    
    def size(self) -> int:
        """Total number of cached entries"""
        return sum(len(vectors) for vectors in self.v_cache.values())
    
    # Backward compatibility methods
    def retrieve(self, group_id: int, option: str) -> torch.Tensor:
        """Backward compatibility: retrieve values"""
        return self.retrieve_values(group_id, option)
