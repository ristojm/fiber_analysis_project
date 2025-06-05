#!/usr/bin/env python3
"""
Quick test to check if scale detection module loads properly
"""

def test_module_import():
    """Test if the scale detection module imports without errors."""
    
    print("🧪 TESTING MODULE IMPORT")
    print("=" * 30)
    
    try:
        # Test import
        print("1. Importing scale detection module...")
        import sys
        from pathlib import Path
        
        # Add modules to path
        project_root = Path(__file__).parent.parent
        if (project_root / "modules").exists():
            sys.path.insert(0, str(project_root / "modules"))
        else:
            sys.path.insert(0, str(project_root))
        
        from scale_detection import ScaleBarDetector, detect_scale_bar
        print("   ✅ Module imported successfully")
        
        # Test class creation
        print("2. Creating detector...")
        detector = ScaleBarDetector()
        print("   ✅ Detector created successfully")
        
        # Test with dummy parameters
        print("3. Testing with compatibility parameters...")
        detector2 = ScaleBarDetector(use_enhanced_detection=True)
        print("   ✅ Compatibility parameters work")
        
        # Test convenience function
        print("4. Testing convenience function...")
        import numpy as np
        dummy_image = np.ones((100, 100), dtype=np.uint8) * 128
        
        result = detect_scale_bar(dummy_image, use_enhanced=True)
        print("   ✅ Convenience function works")
        print(f"   Result: {result['scale_detected']} (expected: False)")
        
        print("\n🎉 ALL TESTS PASSED - Module is ready!")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_module_import()
    if success:
        print("\n✅ Your scale detection module is working!")
        print("You can now run the batch test.")
    else:
        print("\n❌ Module has issues. Fix the errors above first.")