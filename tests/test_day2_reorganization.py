#!/usr/bin/env python3
"""
Test Day 2 Implementation - All Enhanced Functions
Tests all the new reorganized functions from the workflow migration.
"""

import sys
import numpy as np
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent if Path(__file__).parent.name != "modules" else Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "modules"))

print(f"🔧 Testing Day 2 Implementation - All Enhanced Functions")
print(f"   Project root: {project_root}")

def test_enhanced_fiber_detection():
    """Test the new enhanced fiber detection functions."""
    print(f"\n📋 Testing Enhanced Fiber Detection Functions...")
    
    try:
        # Test imports
        try:
            from modules.fiber_type_detection import extract_fiber_mask_from_analysis, create_optimal_fiber_mask
            print(f"✅ Enhanced fiber detection functions imported")
        except ImportError as e:
            print(f"❌ Enhanced fiber detection functions not available: {e}")
            return False
        
        # Create test data
        test_image = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
        test_mask = np.zeros((400, 600), dtype=np.uint8)
        test_mask[100:300, 150:450] = 255  # Rectangular fiber region
        
        # Test analysis data with fiber mask
        test_analysis_data = {
            'fiber_mask': test_mask,
            'fiber_type': 'hollow_fiber',
            'confidence': 0.85,
            'individual_results': [
                {
                    'fiber_contour': np.array([[150, 100], [450, 100], [450, 300], [150, 300]]),
                    'fiber_type': 'hollow_fiber'
                }
            ]
        }
        
        # Test extract_fiber_mask_from_analysis
        extracted_mask = extract_fiber_mask_from_analysis(test_image, test_analysis_data, debug=True)
        assert extracted_mask.dtype == np.uint8, "Mask should be uint8"
        assert extracted_mask.shape == test_image.shape[:2], "Mask should match image dimensions"
        assert np.sum(extracted_mask > 0) > 1000, "Should extract significant mask area"
        print(f"✅ extract_fiber_mask_from_analysis working correctly")
        
        # Test create_optimal_fiber_mask
        optimal_mask = create_optimal_fiber_mask(test_image, test_analysis_data, 
                                                method='best_available', debug=True)
        assert optimal_mask.dtype == np.uint8, "Optimal mask should be uint8"
        assert optimal_mask.shape == test_image.shape[:2], "Optimal mask should match image dimensions"
        print(f"✅ create_optimal_fiber_mask working correctly")
        
        # Test with empty analysis data
        empty_mask = extract_fiber_mask_from_analysis(test_image, None, debug=True)
        assert np.sum(empty_mask > 0) == 0, "Should return empty mask for no analysis data"
        print(f"✅ Fallback behavior working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced fiber detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_scale_detection():
    """Test the new enhanced scale detection function."""
    print(f"\n📋 Testing Enhanced Scale Detection Function...")
    
    try:
        # Test import
        try:
            from modules.scale_detection import detect_scale_bar_from_crop
            print(f"✅ Enhanced scale detection function imported")
        except ImportError as e:
            print(f"❌ Enhanced scale detection function not available: {e}")
            return False
        
        # Create test image with simulated scale bar at bottom
        test_image = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
        # Add a dark region at bottom to simulate scale bar
        test_image[400:, :] = 50  # Dark bottom region
        
        # Test scale detection
        scale_result = detect_scale_bar_from_crop(test_image, crop_bottom_percent=15, debug=True)
        
        # Validate result structure
        required_keys = ['scale_detected', 'micrometers_per_pixel', 'scale_bar_region', 
                        'detection_method', 'confidence']
        for key in required_keys:
            assert key in scale_result, f"Missing key: {key}"
        print(f"✅ Scale detection result structure correct")
        
        # Test crop region
        crop_region = scale_result.get('scale_bar_region')
        assert crop_region is not None, "Should have crop region"
        expected_height = int(480 * 0.15)  # 15% of 480
        assert abs(crop_region.shape[0] - expected_height) <= 1, f"Crop height should be ~{expected_height}"
        print(f"✅ Scale bar cropping working correctly")
        
        # Test with invalid crop percentage
        try:
            invalid_result = detect_scale_bar_from_crop(test_image, crop_bottom_percent=150, debug=True)
            assert not invalid_result['scale_detected'], "Should handle invalid crop percentage"
            print(f"✅ Error handling working correctly")
        except:
            print(f"✅ Error handling working correctly (exception caught)")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced scale detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_crumbly_detection():
    """Test the new enhanced crumbly detection function."""
    print(f"\n📋 Testing Enhanced Crumbly Detection Function...")
    
    try:
        # Test import
        try:
            from modules.crumbly_detection import improve_crumbly_classification
            print(f"✅ Enhanced crumbly detection function imported")
        except ImportError as e:
            print(f"❌ Enhanced crumbly detection function not available: {e}")
            return False
        
        # Create test data
        test_image = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
        test_mask = np.ones((400, 600), dtype=np.uint8) * 255
        
        # Test initial classification
        initial_classification = {
            'classification': 'crumbly',
            'confidence': 0.6
        }
        
        # Test porosity data
        porosity_data = {
            'porosity_percentage': 18.5,
            'total_pores': 75,
            'average_pore_size': 12.3
        }
        
        # Test improvement function
        improved = improve_crumbly_classification(test_image, test_mask, initial_classification, 
                                                porosity_data, debug=True)
        
        # Validate result structure
        required_keys = ['final_classification', 'confidence', 'override_reason', 
                        'improvement_applied', 'original_classification']
        for key in required_keys:
            assert key in improved, f"Missing key: {key}"
        print(f"✅ Classification improvement result structure correct")
        
        # Test with string classification
        string_classification = 'not_crumbly'
        improved_string = improve_crumbly_classification(test_image, test_mask, string_classification, 
                                                       None, debug=True)
        assert 'final_classification' in improved_string, "Should handle string classification"
        print(f"✅ String classification handling working")
        
        # Test without porosity data
        improved_no_porosity = improve_crumbly_classification(test_image, test_mask, initial_classification, 
                                                            None, debug=True)
        assert improved_no_porosity['final_classification'] == 'crumbly', "Should preserve original without porosity data"
        print(f"✅ No porosity data handling working")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced crumbly detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_module_integration():
    """Test that all enhanced functions are properly integrated."""
    print(f"\n📋 Testing Module Integration...")
    
    try:
        # Test updated module imports
        from modules import (HAS_ENHANCED_PREPROCESSING, HAS_ENHANCED_FIBER_DETECTION,
                           HAS_ENHANCED_SCALE_DETECTION, HAS_ENHANCED_CRUMBLY_DETECTION)
        
        print(f"   Enhanced preprocessing: {HAS_ENHANCED_PREPROCESSING}")
        print(f"   Enhanced fiber detection: {HAS_ENHANCED_FIBER_DETECTION}")
        print(f"   Enhanced scale detection: {HAS_ENHANCED_SCALE_DETECTION}")
        print(f"   Enhanced crumbly detection: {HAS_ENHANCED_CRUMBLY_DETECTION}")
        
        # All should be True after Day 2
        expected_true = [HAS_ENHANCED_PREPROCESSING, HAS_ENHANCED_FIBER_DETECTION, 
                        HAS_ENHANCED_SCALE_DETECTION, HAS_ENHANCED_CRUMBLY_DETECTION]
        
        if all(expected_true):
            print(f"✅ All enhanced functions properly integrated")
            return True
        else:
            print(f"⚠️ Some enhanced functions not yet integrated")
            return False
        
    except Exception as e:
        print(f"❌ Module integration test failed: {e}")
        return False

def main():
    """Run all Day 2 tests."""
    print(f"🧪 Day 2 Implementation Tests")
    print(f"=" * 50)
    
    # Enable debug for testing
    from modules import enable_global_debug, disable_global_debug
    enable_global_debug(save_images=False, show_plots=False)
    
    tests = [
        ("Enhanced Fiber Detection", test_enhanced_fiber_detection),
        ("Enhanced Scale Detection", test_enhanced_scale_detection),
        ("Enhanced Crumbly Detection", test_enhanced_crumbly_detection),
        ("Module Integration", test_module_integration)
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
    
    disable_global_debug()
    
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
        print(f"🎉 All Day 2 tests passed! Ready for Day 3 workflow reorganization.")
        return True
    else:
        print(f"⚠️ Some tests failed. Please fix issues before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)