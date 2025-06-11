#!/usr/bin/env python3
"""
Test Scale Detection - Compare Original vs Cropped Approach
"""

import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "modules"))

def test_scale_detection_approaches():
    """Test different scale detection approaches."""
    print(f"🔍 Testing Scale Detection Approaches")
    
    # Use the test image
    image_path = project_root / "sample_images" / "14a_001.jpg"
    
    if not image_path.exists():
        print(f"❌ Test image not found: {image_path}")
        return
    
    try:
        from modules.image_preprocessing import load_image
        from modules.scale_detection import ScaleBarDetector, detect_scale_bar_from_crop
        
        # Load image
        image = load_image(str(image_path))
        print(f"✅ Loaded image: {image.shape}")
        
        # Test 1: Original full image approach
        print(f"\n📋 Test 1: Original Full Image Approach")
        scale_detector = ScaleBarDetector(use_enhanced_detection=True)
        result1 = scale_detector.detect_scale_bar(image, debug=True)
        
        print(f"   Result: {result1.get('scale_detected', False)}")
        if result1.get('scale_detected'):
            print(f"   Scale factor: {result1.get('micrometers_per_pixel', 0):.4f}")
        
        # Test 2: 15% bottom crop
        print(f"\n📋 Test 2: 15% Bottom Crop")
        result2 = detect_scale_bar_from_crop(image, crop_bottom_percent=15, debug=True)
        
        print(f"   Result: {result2.get('scale_detected', False)}")
        if result2.get('scale_detected'):
            print(f"   Scale factor: {result2.get('micrometers_per_pixel', 0):.4f}")
        
        # Test 3: 20% bottom crop
        print(f"\n📋 Test 3: 20% Bottom Crop")
        result3 = detect_scale_bar_from_crop(image, crop_bottom_percent=20, debug=True)
        
        print(f"   Result: {result3.get('scale_detected', False)}")
        if result3.get('scale_detected'):
            print(f"   Scale factor: {result3.get('micrometers_per_pixel', 0):.4f}")
        
        # Test 4: 25% bottom crop
        print(f"\n📋 Test 4: 25% Bottom Crop")
        result4 = detect_scale_bar_from_crop(image, crop_bottom_percent=25, debug=True)
        
        print(f"   Result: {result4.get('scale_detected', False)}")
        if result4.get('scale_detected'):
            print(f"   Scale factor: {result4.get('micrometers_per_pixel', 0):.4f}")
        
        # Summary
        print(f"\n📊 Comparison Summary:")
        tests = [
            ("Full Image", result1),
            ("15% Crop", result2), 
            ("20% Crop", result3),
            ("25% Crop", result4)
        ]
        
        for test_name, result in tests:
            detected = result.get('scale_detected', False)
            factor = result.get('micrometers_per_pixel', 0)
            status = "✅ DETECTED" if detected else "❌ NOT DETECTED"
            print(f"   {test_name:12s}: {status}")
            if detected:
                print(f"                  Scale: {factor:.4f} μm/pixel")
        
        # Recommendation
        successful_tests = [name for name, result in tests if result.get('scale_detected', False)]
        if successful_tests:
            print(f"\n🎯 Recommendation: Use {successful_tests[0]} approach")
        else:
            print(f"\n⚠️ No approach successfully detected the scale bar")
            print(f"   The image may need manual scale calibration")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_scale_detection_approaches()