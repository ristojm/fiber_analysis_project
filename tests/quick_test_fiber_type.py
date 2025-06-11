#!/usr/bin/env python3
"""
Quick test to check what detect_fiber_type actually returns
"""
import sys
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "modules"))

print("🔍 Testing detect_fiber_type function interface...")

try:
    from modules.fiber_type_detection import detect_fiber_type
    print("✅ detect_fiber_type imported successfully")
    
    # Create test image
    test_image = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    print("✅ Test image created")
    
    # Test the function
    print("🧪 Calling detect_fiber_type...")
    result = detect_fiber_type(test_image)
    
    print(f"📊 Result type: {type(result)}")
    if hasattr(result, '__len__'):
        print(f"📊 Result length: {len(result)}")
        print(f"📊 Result contents: {result}")
        
        if isinstance(result, tuple):
            for i, item in enumerate(result):
                print(f"   Item {i}: {type(item)} = {item}")
    else:
        print(f"📊 Result: {result}")
        
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()