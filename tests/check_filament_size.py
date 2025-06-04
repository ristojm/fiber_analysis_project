#!/usr/bin/env python3
"""
Check the size of the solid filament to set appropriate threshold
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Setup paths
current_dir = Path(__file__).parent
if current_dir.name == 'tests':
    project_root = current_dir.parent
else:
    project_root = current_dir

modules_dir = project_root / "modules"
sys.path.insert(0, str(modules_dir))

def check_filament_size():
    """Check what size the solid filament actually is"""
    
    print("="*60)
    print("CHECKING SOLID FILAMENT SIZE")
    print("="*60)
    
    from fiber_type_detection import FiberTypeDetector
    from image_preprocessing import load_image
    
    # Load solid filament image
    img = load_image(str(project_root / "sample_images" / "solid_filament_sample.jpg"))
    print(f"Image shape: {img.shape}")
    
    # Create detector with LOWER threshold to see all fibers
    detector_low = FiberTypeDetector(min_fiber_area=1000)  # Very low threshold
    
    # Preprocess and segment
    preprocessed = detector_low.preprocess_for_detection(img)
    fiber_mask, fiber_properties = detector_low.segment_fibers(preprocessed)
    
    print(f"Found {len(fiber_properties)} fibers with min_area=1000:")
    
    if fiber_properties:
        # Sort by area (largest first)
        sorted_fibers = sorted(fiber_properties, key=lambda x: x['area'], reverse=True)
        
        for i, fiber in enumerate(sorted_fibers[:5]):  # Show top 5
            area = fiber['area']
            circularity = fiber['circularity']
            print(f"  Fiber {i+1}: {area:.0f} pixels, circularity: {circularity:.3f}")
        
        largest_area = sorted_fibers[0]['area']
        print(f"\nLargest fiber: {largest_area:.0f} pixels")
        
        # Test different thresholds
        thresholds = [1000, 10000, 20000, 30000, 40000, 50000]
        print(f"\nFiber count with different thresholds:")
        
        for threshold in thresholds:
            count = sum(1 for f in fiber_properties if f['area'] >= threshold)
            print(f"  min_area={threshold:5d}: {count} fibers")
        
        # Recommend threshold
        if largest_area < 50000:
            recommended = max(10000, largest_area // 2)
            print(f"\n💡 RECOMMENDATION:")
            print(f"  Largest filament: {largest_area:.0f} pixels")
            print(f"  Current threshold: 50,000 (too high!)")
            print(f"  Recommended threshold: {recommended:.0f}")
            
            return recommended
        else:
            print(f"\n✅ Current threshold of 50,000 should work")
            return 50000
    else:
        print("❌ No fibers found even with low threshold")
        return None

def test_with_recommended_threshold():
    """Test with the recommended threshold"""
    
    recommended = check_filament_size()
    
    if recommended:
        print(f"\n" + "="*60)
        print(f"TESTING WITH RECOMMENDED THRESHOLD: {recommended}")
        print("="*60)
        
        from fiber_type_detection import FiberTypeDetector
        from image_preprocessing import load_image
        
        # Test both images with recommended threshold
        test_images = [
            ("hollow_fiber_sample.jpg", "hollow_fiber"),
            ("solid_filament_sample.jpg", "filament")
        ]
        
        for img_name, expected in test_images:
            print(f"\n📷 Testing {img_name}:")
            
            img = load_image(str(project_root / "sample_images" / img_name))
            
            # Create detector with recommended threshold
            detector = FiberTypeDetector(min_fiber_area=recommended)
            
            fiber_type, confidence, analysis_data = detector.classify_fiber_type(img)
            
            print(f"  Type: {fiber_type}")
            print(f"  Confidence: {confidence:.3f}")
            
            if 'total_fibers' in analysis_data:
                print(f"  Total fibers: {analysis_data['total_fibers']}")
                print(f"  Hollow: {analysis_data['hollow_fibers']}")
                print(f"  Filaments: {analysis_data['filaments']}")
            
            # Check if correct
            if fiber_type == expected:
                print(f"  ✅ CORRECT!")
            else:
                print(f"  ❌ Expected: {expected}")

if __name__ == "__main__":
    test_with_recommended_threshold()