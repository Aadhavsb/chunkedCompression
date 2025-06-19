#!/usr/bin/env python3
"""
Debug script for PDR: Investigate B_K shape mismatch and meta tensor issues
Part of Project Design Report implementation
"""

import sys
import torch
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def debug_compression_shapes():
    """Phase 1: Investigate B_K shape issue"""
    print("🔍 PDR Phase 1: Debugging B_K Tensor Shapes")
    print("=" * 60)
    
    try:
        from core.model import LLaMAModelLoader
        from core.compression import LLaMACompressionProfileBuilder
        from core.config import ModelConfig, CompressionConfig
        
        print("✅ Successfully imported core modules")
        
        # Initialize configs
        model_config = ModelConfig()
        compression_config = CompressionConfig()
        
        print(f"📊 Configuration loaded:")
        print(f"   Key compression rank: {compression_config.key_compression_rank}")
        print(f"   Value compression ranks: {compression_config.value_compression_ranks}")
        
        # Initialize model loader (lightweight check first)
        print("\n🏗️  Initializing model loader...")
        model_loader = LLaMAModelLoader(model_config)
        
        # Check if model loading will cause meta tensor issues
        print("🔍 Checking model device placement...")
        try:
            model_loader.load_model()
            print(f"   Model device: {next(model_loader.model.parameters()).device}")
            print(f"   Model dtype: {next(model_loader.model.parameters()).dtype}")
            print(f"   Is meta tensor: {next(model_loader.model.parameters()).is_meta}")
        except Exception as e:
            print(f"⚠️  Model loading issue detected: {e}")
            print("   This may be the source of meta tensor errors")
        
        # Initialize compression profile builder
        print("\n🗜️  Building compression profiles...")
        profile_builder = LLaMACompressionProfileBuilder(model_loader, compression_config)
        
        # Build profiles and capture shapes
        profile_builder.build_compression_profiles(layer_idx=-1)
        
        print("\n📏 Analyzing tensor shapes:")
        print("-" * 40)
        
        for profile_name in compression_config.get_compression_profile_names():
            if profile_name in profile_builder.profiles:
                profile = profile_builder.profiles[profile_name]
                
                print(f"\n{profile_name.upper()} Profile:")
                print(f"   A_V shape: {profile['A_V'].shape}")
                print(f"   W_fused shape: {profile['W_fused'].shape}")
                print(f"   A_K shape: {profile['A_K'].shape}")
                print(f"   B_K shape: {profile['B_K'].shape}")
                print(f"   Expected B_K: (8, 128, 32)")
                print(f"   Actual B_K: {profile['B_K'].shape}")
                
                # Check for shape mismatch
                expected_B_K_shape = (8, 128, 32)  # (num_kv_heads, head_dim, key_rank)
                actual_B_K_shape = profile['B_K'].shape
                
                if actual_B_K_shape != expected_B_K_shape:
                    print(f"   🚨 SHAPE MISMATCH DETECTED!")
                    print(f"      Expected: {expected_B_K_shape}")
                    print(f"      Actual: {actual_B_K_shape}")
                    print(f"      Suggested fix: Apply .transpose(-1, -2)")
                else:
                    print(f"   ✅ Shape is correct")
        
        # Check key compression matrices directly
        print(f"\n🔑 Key compression matrices:")
        print(f"   Number of KV heads: {len(profile_builder.key_compression_matrices)}")
        for i, (A_K, B_K) in enumerate(zip(profile_builder.key_compression_matrices, 
                                          profile_builder.key_reconstruction_matrices)):
            print(f"   Head {i}: A_K {A_K.shape}, B_K {B_K.shape}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error during shape debugging: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return False

def debug_meta_tensors():
    """Phase 2: Investigate meta tensor issues"""
    print("\n🔍 PDR Phase 2: Debugging Meta Tensor Issues")
    print("=" * 60)
    
    try:
        from core.model import LLaMAModelLoader
        from core.config import ModelConfig
        
        model_config = ModelConfig()
        model_loader = LLaMAModelLoader(model_config)
        
        print("🏗️  Loading model with device debugging...")
        model_loader.load_model()
        
        # Check all parameters for meta device
        meta_params = []
        regular_params = []
        
        for name, param in model_loader.model.named_parameters():
            if param.is_meta:
                meta_params.append(name)
            else:
                regular_params.append(name)
        
        print(f"📊 Parameter analysis:")
        print(f"   Total parameters: {len(list(model_loader.model.parameters()))}")
        print(f"   Meta tensors: {len(meta_params)}")
        print(f"   Regular tensors: {len(regular_params)}")
        
        if meta_params:
            print(f"🚨 Meta tensors detected (first 5):")
            for name in meta_params[:5]:
                print(f"     {name}")
            print("   This explains the 'Cannot copy out of meta tensor' error")
            
        # Test data copy operations
        print("\n🧪 Testing data copy operations...")
        try:
            sample_param = next(model_loader.model.parameters())
            if sample_param.is_meta:
                print("   ⚠️  Cannot test copy - parameter is meta tensor")
                print("   Recommended fix: Force tensor materialization")
            else:
                _ = sample_param.cpu().clone()
                print("   ✅ Data copy successful")
        except Exception as copy_error:
            print(f"   ❌ Copy error: {copy_error}")
            
        return len(meta_params) == 0
        
    except Exception as e:
        print(f"❌ Error during meta tensor debugging: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return False

def main():
    """Execute PDR debugging phases"""
    print("🚀 Starting PDR Implementation: LLaMA-3 8B Compression Debug")
    print("=" * 70)
    
    # Phase 1: Shape debugging
    shapes_ok = debug_compression_shapes()
    
    # Phase 2: Meta tensor debugging  
    meta_ok = debug_meta_tensors()
    
    print("\n📋 PDR Debug Summary:")
    print("=" * 30)
    print(f"✅ Shape debugging: {'PASSED' if shapes_ok else 'FAILED'}")
    print(f"✅ Meta tensor check: {'PASSED' if meta_ok else 'FAILED'}")
    
    if shapes_ok and meta_ok:
        print("\n🎉 All checks passed - system ready for full testing")
    else:
        print("\n🔧 Issues identified - fixes needed as per PDR plan")
        
    return shapes_ok and meta_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
