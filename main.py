"""
Main entry point for the chunked compression system
"""
import torch
from dataset import WikiTextDataset
from model import BareDecoder
from utils import get_compression_map, print_compression_stats
from profiles import profiles

def main():
    """Run the full compression pipeline"""
    print("🚀 Starting Barebones Chunked-Fused KV Compression Test")
    print("=" * 60)
    
    # Load data
    print("\n📚 Loading WikiText data...")
    dataset = WikiTextDataset(max_tokens=128)
    tokens, X = dataset.get_tokenized_batch()
    
    # Get compression mapping
    print("\n🗺️  Generating compression mapping...")
    compression_map = get_compression_map(tokens)
    print_compression_stats(compression_map, tokens)
    
    # Initialize model
    print("\n🤖 Initializing BareDecoder...")
    model = BareDecoder()
    
    # Print profile information
    print("\n📋 Compression Profiles:")
    for option, profile in profiles.items():
        A_shape = profile["A"].shape
        W_shape = profile["W_fused"].shape
        print(f"  {option}: A{A_shape} -> W_fused{W_shape} (rank {profile['r']})")
    
    # Forward pass
    print("\n🎬 Running forward pass...")
    with torch.no_grad():
        outputs = model.forward(X, compression_map)
    
    # Print results
    print("\n📊 Results:")
    print(f"  Input shape: {X.shape}")
    print(f"  Output shape: {outputs.shape}")
    print(f"  Output mean: {outputs.mean().item():.6f}")
    print(f"  Output std: {outputs.std().item():.6f}")
    print(f"  Output range: [{outputs.min().item():.6f}, {outputs.max().item():.6f}]")
    
    # Validate shapes
    assert outputs.shape == X.shape, f"Shape mismatch: expected {X.shape}, got {outputs.shape}"
    print("✅ Shape validation passed!")
    
    print("\n🎉 Test completed successfully!")
    return outputs, compression_map, tokens

if __name__ == "__main__":
    outputs, compression_map, tokens = main()
