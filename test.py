"""
Test runner for the chunked compression system
"""
import torch
from main import main
from compression import validate_compression_matrices
from profiles import profiles

def run_sanity_checks():
    """Run sanity checks on the system"""
    print("🧪 Running sanity checks...")
    
    # Check compression matrix shapes
    print("\n🔍 Validating compression matrices...")
    all_valid = True
    for option, profile in profiles.items():
        A = profile["A"]
        W_fused = profile["W_fused"]
        rank = profile["r"]
        
        valid = validate_compression_matrices(A, W_fused, 64, rank)  # d_head=64
        print(f"  {option} profile: {'✅' if valid else '❌'}")
        all_valid = all_valid and valid
    
    if not all_valid:
        print("❌ Some compression matrices have invalid shapes!")
        return False
    
    print("✅ All compression matrices have valid shapes")
    return True

def test_full_pipeline():
    """Test the full pipeline"""
    print("\n🏃‍♂️ Testing full pipeline...")
    
    try:
        outputs, compression_map, tokens = main()
        
        # Additional checks
        T = len(tokens)
        d_model = 512
        
        # Verify output properties
        assert outputs.shape == (T, d_model), f"Wrong output shape: {outputs.shape}"
        assert not torch.isnan(outputs).any(), "Output contains NaN values"
        assert not torch.isinf(outputs).any(), "Output contains infinite values"
        assert len(compression_map) == T, f"Compression map length mismatch: {len(compression_map)} vs {T}"
        
        print("✅ Full pipeline test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False

def print_detailed_results(outputs, compression_map, tokens):
    """Print detailed analysis of results"""
    print("\n📈 Detailed Results Analysis:")
    
    # Per-compression-option statistics
    from collections import defaultdict
    import numpy as np
    
    option_outputs = defaultdict(list)
    for i, option in enumerate(compression_map):
        option_outputs[option].append(outputs[i])
    
    print("\n📊 Per-compression-option output statistics:")
    for option in ["low", "med", "high"]:
        if option in option_outputs:
            option_tensors = torch.stack(option_outputs[option])
            mean_val = option_tensors.mean().item()
            std_val = option_tensors.std().item()
            count = len(option_outputs[option])
            print(f"  {option}: {count} tokens, mean={mean_val:.6f}, std={std_val:.6f}")
    
    # Print sample outputs
    print(f"\n🔍 First 5 token outputs:")
    for i in range(min(5, len(tokens))):
        output_norm = torch.norm(outputs[i]).item()
        print(f"  Token {i} ({compression_map[i]}): ||output||={output_norm:.6f}")

def main_test():
    """Main test function"""
    print("🧪 Barebones Chunked-Fused KV Compression Test Suite")
    print("=" * 65)
    
    # Run sanity checks
    if not run_sanity_checks():
        print("❌ Sanity checks failed!")
        return
    
    # Test full pipeline
    if not test_full_pipeline():
        print("❌ Full pipeline test failed!")
        return
    
    print("\n🎉 All tests passed! System is ready for logic validation.")

if __name__ == "__main__":
    main_test()
