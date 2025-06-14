#!/usr/bin/env python3
"""
Test script to verify the refactored imports work correctly.
This script tests the new modular structure without requiring heavy dependencies.
"""

def test_core_imports():
    """Test that core module imports work correctly."""
    print("Testing core module imports...")
    
    try:
        # Test config imports
        from core.config import ModelConfig, CompressionConfig, InferenceConfig, BenchmarkConfig
        print("✅ Config imports successful")
        
        # Test interface imports
        from core.interfaces import ModelLoaderInterface, CompressionProfileInterface
        print("✅ Interface imports successful")
        
        # Test utility imports
        from core.utils import MemoryManager
        print("✅ Utility imports successful")
        
        # Test core module import
        import core
        print(f"✅ Core module import successful, heavy components available: {core.HEAVY_COMPONENTS_AVAILABLE}")
        
        # Test that we can create config objects
        model_config = ModelConfig()
        compression_config = CompressionConfig()
        inference_config = InferenceConfig()
        benchmark_config = BenchmarkConfig()
        print("✅ Config object creation successful")
        
        # Test memory manager
        memory_manager = MemoryManager()
        memory_stats = memory_manager.get_memory_usage()
        print(f"✅ Memory manager works, found {len(memory_stats)} metrics")
        
        return True
        
    except Exception as e:
        print(f"❌ Core imports failed: {e}")
        return False

def test_legacy_compatibility():
    """Test that legacy compatibility wrapper can be imported."""
    print("\nTesting legacy compatibility...")
    
    try:
        # Test that the new profiles module can be imported
        import profiles_llama_new
        print(f"✅ Profiles module import successful, components available: {profiles_llama_new.COMPONENTS_AVAILABLE}")
        
        if profiles_llama_new.COMPONENTS_AVAILABLE:
            # Test that components can be imported if available
            from profiles_llama_new import LLaMACompressionProfiles, ModelConfig
            print("✅ Component imports from compatibility module successful")
        else:
            print("ℹ️  Components not available (expected in environment without torch)")
        
        return True
        
    except Exception as e:
        print(f"❌ Legacy compatibility failed: {e}")
        return False

def test_config_functionality():
    """Test configuration functionality."""
    print("\nTesting configuration functionality...")
    
    try:
        from core.config import ModelConfig, CompressionConfig
        
        # Test config creation and validation
        config = CompressionConfig(
            value_compression_ranks={"low": 32, "med": 64, "high": 128},
            key_compression_rank=64
        )
        
        # Test config methods
        profiles = config.get_compression_profile_names()
        assert "low" in profiles and "med" in profiles and "high" in profiles
        
        value_rank = config.get_value_rank("med")
        assert value_rank == 64
        
        # Test config validation
        compatibility = config.validate_profile_compatibility(4096)  # LLaMA hidden size
        assert isinstance(compatibility, dict)
        
        print("✅ Configuration functionality tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration functionality failed: {e}")
        return False

def test_file_structure():
    """Test that the expected files exist."""
    print("\nTesting file structure...")
    
    import os
    
    expected_files = [
        "core/__init__.py",
        "core/config/__init__.py",
        "core/config/model_config.py",
        "core/config/compression_config.py",
        "core/config/inference_config.py",
        "core/interfaces/__init__.py",
        "core/interfaces/model_interface.py",
        "core/interfaces/compression_interface.py",
        "core/interfaces/cache_interface.py",
        "core/interfaces/inference_interface.py",
        "core/interfaces/data_interface.py",
        "core/model/__init__.py",
        "core/model/model_loader.py",
        "core/model/model_config_wrapper.py",
        "core/compression/__init__.py",
        "core/compression/compression_algorithms.py",
        "core/compression/profile_builder.py",
        "core/compression/legacy_wrapper.py",
        "core/utils/__init__.py",
        "core/utils/memory_manager.py",
        "profiles_llama_new.py"
    ]
    
    missing_files = []
    for file_path in expected_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print(f"✅ All {len(expected_files)} expected files found")
        return True

def main():
    """Run all tests."""
    print("🧪 Testing Refactored Chunked Compression System")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_core_imports,
        test_config_functionality,
        test_legacy_compatibility,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print(f"\n📊 Test Results:")
    print(f"   Passed: {passed}")
    print(f"   Failed: {failed}")
    print(f"   Total:  {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! Refactoring successful.")
        return True
    else:
        print(f"\n⚠️  {failed} tests failed. Review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)