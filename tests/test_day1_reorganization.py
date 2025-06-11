#!/usr/bin/env python3
"""
Test Day 1 Implementation - Debug System and Enhanced Preprocessing
Tests the new debug configuration system and enhanced preprocessing function.
"""

import sys
import numpy as np
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent if Path(__file__).parent.name != "modules" else Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "modules"))

print(f"🔧 Testing Day 1 Implementation")
print(f"   Project root: {project_root}")

def test_debug_system():
    """Test the new unified debug system."""
    print(f"\n📋 Testing Debug System...")
    
    try:
        # Test import
        from modules.debug_config import DEBUG_CONFIG, enable_global_debug, disable_global_debug, is_debug_enabled
        print(f"✅ Debug system imports successful")
        
        # Test initial state
        assert not is_debug_enabled(), "Debug should be disabled initially"
        print(f"✅ Initial debug state correct (disabled)")
        
        # Test enabling debug
        enable_global_debug(save_images=True, show_plots=False)
        assert is_debug_enabled(), "Debug should be enabled after calling enable"
        assert DEBUG_CONFIG.save_images == True, "Save images should be enabled"
        assert DEBUG_CONFIG.show_plots == False, "Show plots should be disabled"
        print(f"✅ Debug enable functionality working")
        
        # Test disabling debug
        disable_global_debug()
        assert not is_debug_enabled(), "Debug should be disabled after calling disable"
        print(f"✅ Debug disable functionality working")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_modules_init():
    """Test the enhanced modules __init__.py"""
    print(f"\n📋 Testing Enhanced Modules Init...")
    
    try:
        # Test import
        from modules import (DEBUG_CONFIG, enable_global_debug, disable_global_debug, 
                           HAS_ENHANCED_PREPROCESSING, HAS_ENHANCED_FIBER_DETECTION,
                           HAS_ENHANCED_SCALE_DETECTION, HAS_ENHANCED_CRUMBLY_DETECTION)
        print(f"✅ Enhanced modules import successful")
        
        # Check availability flags
        print(f"   Enhanced preprocessing available: {HAS_ENHANCED_PREPROCESSING}")
        print(f"   Enhanced fiber detection available: {HAS_ENHANCED_FIBER_DETECTION}")
        print(f"   Enhanced scale detection available: {HAS_ENHANCED_SCALE_DETECTION}")
        print(f"   Enhanced crumbly detection available: {HAS_ENHANCED_CRUMBLY_DETECTION}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced modules init test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_preprocessing():
    """Test the new enhanced preprocessing function."""
    print(f"\n📋 Testing Enhanced Preprocessing Function...")
    
    try:
        # Try to import the new function
        try:
            from modules.image_preprocessing import preprocess_for_analysis
            print(f"✅ Enhanced preprocessing function imported")
        except ImportError as e:
            print(f"❌ Enhanced preprocessing function not available: {e}")
            print(f"   You need to add the function to modules/image_preprocessing.py")
            return False
        
        # Create a test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print(f"✅ Created test image: {test_image.shape}")
        
        # Test without debug
        result = preprocess_for_analysis(test_image, remove_scale_bar=True, debug=False)
        
        # Validate result structure
        required_keys = ['processed_image', 'original_image', 'processing_steps', 'scale_bar_removed']
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
        print(f"✅ Result structure correct: {list(result.keys())}")
        
        # Validate processing
        assert result['scale_bar_removed'] == True, "Scale bar should be removed"
        assert len(result['processing_steps']) > 0, "Should have processing steps"
        
        # Check image dimensions (should be cropped)
        original_height = test_image.shape[0]
        processed_height = result['processed_image'].shape[0]
        assert processed_height < original_height, "Processed image should be shorter (scale bar removed)"
        print(f"✅ Scale bar removal working: {original_height} → {processed_height}")
        
        # Test with debug enabled
        from modules import enable_global_debug, disable_global_debug
        enable_global_debug(save_images=False, show_plots=False)
        
        print(f"\n   Testing with debug enabled...")
        result_debug = preprocess_for_analysis(test_image, remove_scale_bar=True, debug=True)
        
        disable_global_debug()
        
        # Should have same structure
        assert all(key in result_debug for key in required_keys), "Debug result should have same structure"
        print(f"✅ Debug mode working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced preprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Day 1 tests."""
    print(f"🧪 Day 1 Implementation Tests")
    print(f"=" * 50)
    
    tests = [
        ("Debug System", test_debug_system),
        ("Enhanced Modules Init", test_enhanced_modules_init),
        ("Enhanced Preprocessing", test_enhanced_preprocessing)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        success = test_func()
        results.append((test_name, success))
        
        if success:
            print(f"✅ {test_name}: PASSED")
        else:
            print(f"❌ {test_name}: FAILED")
    
    # Summary
    print(f"\n📊 Test Summary:")
    print(f"=" * 50)
    
    passed = sum(1 for name, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print(f"🎉 All Day 1 tests passed! Ready for Day 2 implementation.")
        return True
    else:
        print(f"⚠️ Some tests failed. Please fix issues before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)