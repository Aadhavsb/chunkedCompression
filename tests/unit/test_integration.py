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

def test_kv_cache():
    """Test KV cache functionality"""
    print("\n🧪 Testing KV Cache...")
    
    from core.cache.standard_kv_cache import StandardKVCache as KVCache
    
    cache = KVCache()
    
    # Test data
    seq_len = 8
    rank_v = 32
    rank_k = 16
    
    compressed_values = torch.randn(seq_len, rank_v)
    compressed_keys = torch.randn(seq_len, rank_k)
    
    # Store KV pairs (using layer_idx=0, head_idx=0 for test)
    layer_idx, head_idx = 0, 0
    for t in range(seq_len):
        cache.append(
            layer_idx=layer_idx,
            head_idx=head_idx,
            token_idx=t,
            k_vector=compressed_keys[t],
            v_vector=compressed_values[t]
        )
    
    # Retrieve and validate
    cached_keys, cached_values = cache.retrieve_kv(layer_idx, head_idx)
    
    print(f"   Stored: values{compressed_values.shape}, keys{compressed_keys.shape}")
    print(f"   Retrieved: values{cached_values.shape}, keys{cached_keys.shape}")
    
    # Validate
    assert torch.allclose(cached_values, compressed_values), "Values don't match"
    assert torch.allclose(cached_keys, compressed_keys), "Keys don't match"
    assert len(cache.k_cache[(layer_idx, head_idx)]) == seq_len, f"Cache size wrong: {len(cache.k_cache[(layer_idx, head_idx)])}"
    
    print("   ✅ KV cache tests passed!")


def main():
    """Run all integration tests"""
    print("🚀 STARTING INTEGRATION TESTS")
    print("="*50)
    
    tests = [
        test_basic_svd_compression,
        test_kv_cache
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
