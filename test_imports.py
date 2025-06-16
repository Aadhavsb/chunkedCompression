#!/usr/bin/env python3
"""
Quick import test to debug the issue
"""
import sys
import os

# Add paths like the comprehensive test does
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    print("Testing imports...")
    from tests.unit.test_llama_compression import LLaMACompressionTestSuite
    print("✅ LLaMACompressionTestSuite import successful")
    
    from core.inference import LLaMACompressionInference  
    print("✅ LLaMACompressionInference import successful")
    
    # Test instantiation
    print("Testing instantiation...")
    # Don't actually instantiate since we don't have the model
    print("✅ Import test completed successfully")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("📍 Current working directory:", os.getcwd())
    print("📍 Python path:", sys.path[:3])
except Exception as e:
    print(f"❌ Other error: {e}")