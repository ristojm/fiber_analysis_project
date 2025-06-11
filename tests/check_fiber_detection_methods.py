#!/usr/bin/env python3
"""
Check what methods are available in fiber type detection
"""
import sys
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "modules"))

print("🔍 Checking fiber detection module methods...")

try:
    from modules.fiber_type_detection import FiberTypeDetector
    import modules.fiber_type_detection as ftd
    
    print("✅ Fiber type detection module imported")
    
    # Check what methods are available
    print("\n📋 Available functions in fiber_type_detection module:")
    for name in dir(ftd):
        if not name.startswith('_'):
            obj = getattr(ftd, name)
            if callable(obj):
                print(f"   📌 {name}: {type(obj)}")
    
    # Test FiberTypeDetector class
    print("\n🧪 Testing FiberTypeDetector class...")
    detector = FiberTypeDetector()
    
    print("\n📋 Available methods in FiberTypeDetector:")
    for name in dir(detector):
        if not name.startswith('_') and callable(getattr(detector, name)):
            print(f"   📌 {name}")
    
    # Test the classify_fiber_type method
    test_image = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    
    print("\n🧪 Testing detector.classify_fiber_type...")
    result = detector.classify_fiber_type(test_image)
    
    print(f"📊 classify_fiber_type result type: {type(result)}")
    if hasattr(result, '__len__'):
        print(f"📊 Result length: {len(result)}")
        if isinstance(result, tuple):
            for i, item in enumerate(result):
                print(f"   Item {i}: {type(item)} = {item if not isinstance(item, dict) else f'dict with {len(item)} keys'}")
        
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()