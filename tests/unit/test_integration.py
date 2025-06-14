#!/usr/bin/env python3
"""
Integration test to validate the complete compression system
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from core.compression import SVDCompressionAlgorithm
from legacy.compression import decompose_and_fuse, compress_keys, compress_key_states, reconstruct_keys

def test_basic_svd_compression():
    """Test basic SVD compression functionality"""
    print("🧪 Testing Basic SVD Compression...")
    
    # Set up test dimensions
    d_model = 768
    d_head = 64
    rank = 32
    
    # Create mock matrices
    torch.manual_seed(42)
    W_v = torch.randn(d_model, d_head) * 0.02
    W_o = torch.randn(d_model, d_head) * 0.02
    W_k = torch.randn(d_model, d_head) * 0.02
    
    print(f"   Input matrices: W_v{W_v.shape}, W_o{W_o.shape}, W_k{W_k.shape}")
    
    # Test value compression
    A_v, W_fused = decompose_and_fuse(W_v, W_o, rank)
    print(f"   Value compression: A_v{A_v.shape}, W_fused{W_fused.shape}")
    
    # Test key compression
    A_k, B_k = compress_keys(W_k, rank)
    print(f"   Key compression: A_k{A_k.shape}, B_k{B_k.shape}")
    
    # Test key compression/reconstruction pipeline
    seq_len = 16
    key_states = torch.randn(seq_len, d_head) * 0.1
    
    # Compress keys
    compressed_keys = compress_key_states(key_states, A_k)
    print(f"   Compressed keys: {key_states.shape} -> {compressed_keys.shape}")
    
    # Reconstruct keys
    reconstructed_keys = reconstruct_keys(compressed_keys, B_k)
    print(f"   Reconstructed keys: {compressed_keys.shape} -> {reconstructed_keys.shape}")
    
    # Calculate reconstruction error
    reconstruction_error = F.mse_loss(reconstructed_keys, key_states)
    print(f"   Reconstruction MSE: {reconstruction_error:.6f}")
    
    # Validate shapes
    assert A_v.shape == (rank, d_head), f"A_v shape wrong: {A_v.shape}"
    assert A_k.shape == (rank, d_head), f"A_k shape wrong: {A_k.shape}"
    assert B_k.shape == (d_head, rank), f"B_k shape wrong: {B_k.shape}"
    assert compressed_keys.shape == (seq_len, rank), f"Compressed keys shape wrong: {compressed_keys.shape}"
    assert reconstructed_keys.shape == key_states.shape, f"Reconstructed keys shape wrong: {reconstructed_keys.shape}"
    
    print("   ✅ All SVD compression tests passed!")
    return True

def test_kv_cache():
    """Test KV cache functionality"""
    print("\n🧪 Testing KV Cache...")
    
    from kv_cache import KVCache
    
    cache = KVCache()
    
    # Test data
    seq_len = 8
    rank_v = 32
    rank_k = 16
    
    compressed_values = torch.randn(seq_len, rank_v)
    compressed_keys = torch.randn(seq_len, rank_k)
    
    # Store compressed KV pairs
    for t in range(seq_len):
        cache.append(
            token_idx=t,
            group_id=0,
            h_v=compressed_values[t],
            h_k=compressed_keys[t],
            option="med"
        )
    
    # Retrieve and validate
    cached_values = cache.retrieve_values(0, "med")
    cached_keys = cache.retrieve_keys(0, "med")
    
    print(f"   Stored: values{compressed_values.shape}, keys{compressed_keys.shape}")
    print(f"   Retrieved: values{cached_values.shape}, keys{cached_keys.shape}")
    
    # Validate
    assert torch.allclose(cached_values, compressed_values), "Values don't match"
    assert torch.allclose(cached_keys, compressed_keys), "Keys don't match"
    assert cache.size() == seq_len, f"Cache size wrong: {cache.size()}"
    
    print("   ✅ KV cache tests passed!")
    return True

def test_profiles():
    """Test compression profiles"""
    print("\n🧪 Testing Compression Profiles...")
    
    from profiles import profiles
    
    print(f"   Available profiles: {list(profiles.keys())}")
    
    for option, profile in profiles.items():
        # Check required keys
        required_keys = ["A", "W_fused", "r", "A_K", "B_K", "r_k"]
        for key in required_keys:
            assert key in profile, f"Missing key '{key}' in {option} profile"
        
        A_v = profile["A"]
        W_fused = profile["W_fused"]
        A_k = profile["A_K"]
        B_k = profile["B_K"]
        value_rank = profile["r"]
        key_rank = profile["r_k"]
        
        # Validate shapes
        assert A_v.shape == (value_rank, 64), f"{option}: A_v shape wrong: {A_v.shape}"
        assert A_k.shape == (key_rank, 64), f"{option}: A_k shape wrong: {A_k.shape}"
        assert B_k.shape == (64, key_rank), f"{option}: B_k shape wrong: {B_k.shape}"
        
        print(f"   {option}: ✅ value_rank={value_rank}, key_rank={key_rank}")
        print(f"     A_v{A_v.shape}, A_k{A_k.shape}, B_k{B_k.shape}")
    
    print("   ✅ All profile tests passed!")
    return True

def test_model_integration():
    """Test basic model integration"""
    print("\n🧪 Testing Model Integration...")
    
    from model import BareDecoder
    from utils import get_compression_map
    
    # Create model
    model = BareDecoder(d_model=512, d_head=64)
    
    # Create test data
    seq_len = 16
    d_model = 512
    X = torch.randn(seq_len, d_model) * 0.1
    
    # Create compression map
    tokens = list(range(seq_len))  # Mock token IDs
    compression_map = get_compression_map(tokens)
    
    print(f"   Input: {X.shape}")
    print(f"   Compression map: {compression_map}")
    
    # Run forward pass
    with torch.no_grad():
        outputs = model.forward(X, compression_map)
    
    print(f"   Output: {outputs.shape}")
    
    # Validate
    assert outputs.shape == X.shape, f"Output shape mismatch: {outputs.shape} vs {X.shape}"
    assert not torch.isnan(outputs).any(), "Output contains NaN"
    assert not torch.isinf(outputs).any(), "Output contains Inf"
    
    print("   ✅ Model integration test passed!")
    return True

def main():
    """Run all integration tests"""
    print("🚀 STARTING INTEGRATION TESTS")
    print("="*50)
    
    tests = [
        test_basic_svd_compression,
        test_kv_cache,
        test_profiles,
        test_model_integration
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"   ❌ Test failed with error: {e}")
            failed += 1
    
    print("\n" + "="*50)
    print(f"📊 INTEGRATION TEST RESULTS")
    print(f"   Passed: {passed}/{len(tests)}")
    print(f"   Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("✅ ALL INTEGRATION TESTS PASSED!")
        print("🎯 System is ready for real LLM testing!")
    else:
        print("❌ Some tests failed. Please fix before proceeding.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
