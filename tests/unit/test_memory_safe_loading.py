#!/usr/bin/env python3
"""
Test script for memory-safe LLaMA-3 loading
Run this to verify the cluster loader works without OOM errors
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, '/home/asb269/chunkedCompression')

from llama_loader import LLaMA3Loader
import torch

def test_memory_safe_loading():
    """Test the memory-safe cluster loader"""
    print("🚀 Testing Memory-Safe LLaMA-3 Cluster Loading")
    print("=" * 60)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ No GPU available. This test requires CUDA.")
        return False
    
    print(f"✅ GPU detected: {torch.cuda.get_device_name()}")
    print(f"💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    model_path = "/mnt/vstor/CSE_ECSE_GXD234/Meta-Llama-3-8B-Instruct"
    
    try:
        # Test the cluster loader
        print(f"\n🦙 Initializing LLaMA3Loader...")
        loader = LLaMA3Loader(model_path, dtype="bfloat16")
        
        print(f"🔄 Loading model (this may take a few minutes)...")
        model, tokenizer = loader.load_model()
        
        print(f"✅ Model loaded successfully!")
        
        # Get model info
        model_info = loader.get_model_info()
        print(f"\n📊 Model Information:")
        for key, value in model_info.items():
            print(f"   {key}: {value}")
        
        # Test tokenization
        print(f"\n🔤 Testing tokenization...")
        test_text = "The capital of France is"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"   Text: '{test_text}'")
        print(f"   Tokens: {tokens['input_ids'].shape}")
        
        # Test inference (small)
        print(f"\n🧠 Testing inference...")
        with torch.no_grad():
            outputs = model.generate(
                tokens['input_ids'].to(model.device),
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   Generated: '{generated_text}'")
        
        # Check GPU memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1e9
            memory_cached = torch.cuda.memory_reserved() / 1e9
            print(f"\n💾 GPU Memory Usage:")
            print(f"   Allocated: {memory_used:.2f} GB")
            print(f"   Cached: {memory_cached:.2f} GB")
        
        # Cleanup
        print(f"\n🧹 Cleaning up...")
        loader.cleanup()
        
        print(f"\n🎉 Memory-safe loading test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Memory-safe loading test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_memory_safe_loading()
    exit(0 if success else 1)
