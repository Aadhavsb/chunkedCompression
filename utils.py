"""
Utility functions for token to compression option mapping
"""
from typing import List

def get_compression_map(tokens: List[int]) -> List[str]:
    """
    Map each token to a compression option for testing
    
    Args:
        tokens: List of token IDs
        
    Returns:
        List of compression options ("low", "med", "high") per token
    """
    T = len(tokens)
    compression_map = []
    
    # Simple rule: first 40% -> low, next 40% -> med, rest -> high
    for i in range(T):
        if i < T * 0.4:
            compression_map.append("low")
        elif i < T * 0.8:
            compression_map.append("med")
        else:
            compression_map.append("high")
    
    return compression_map

def print_compression_stats(compression_map: List[str], tokens: List[int]):
    """Print statistics about compression assignment"""
    from collections import Counter
    
    stats = Counter(compression_map)
    print("\n📊 Compression Assignment Stats:")
    print(f"Total tokens: {len(compression_map)}")
    for option, count in stats.items():
        print(f"  {option}: {count} tokens ({count/len(compression_map)*100:.1f}%)")
    
    print("\n🔍 First 10 token assignments:")
    for i in range(min(10, len(tokens))):
        print(f"  Token {i} (ID: {tokens[i]}): {compression_map[i]}")
