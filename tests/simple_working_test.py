#!/usr/bin/env python3
"""
Simple test using the WORKING module (we know it works from debug!)
"""

import sys
from pathlib import Path

# Setup paths
current_dir = Path(__file__).parent
if current_dir.name == 'tests':
    project_root = current_dir.parent
else:
    project_root = current_dir

modules_dir = project_root / "modules"
sys.path.insert(0, str(modules_dir))

def test_working_algorithm():
    """Test with the algorithm we KNOW works"""
    
    print("="*60)
    print("TESTING WITH WORKING UPDATED MODULE")
    print("="*60)
    
    from fiber_type_detection import FiberTypeDetector
    from image_preprocessing import load_image
    
    # Test both images
    test_images = [
        ("hollow_fiber_sample.jpg", "Should be hollow_fiber"),
        ("solid_filament_sample.jpg", "Should be filament")
    ]
    
    for img_name, expected in test_images:
        print(f"\n📷 Testing: {img_name}")
        print(f"Expected: {expected}")
        print("-" * 40)
        
        # Load image
        img_path = project_root / "sample_images" / img_name
        img = load_image(str(img_path))
        
        # Create detector with DEFAULT parameters (we know these work!)
        detector = FiberTypeDetector()  # Uses defaults: min_fiber_area=50000, etc.
        
        print(f"✓ Detector parameters:")
        print(f"  min_fiber_area: {detector.min_fiber_area}")
        print(f"  lumen_area_threshold: {detector.lumen_area_threshold}")
        
        # Run classification
        fiber_type, confidence, analysis_data = detector.classify_fiber_type(img)
        
        print(f"\n🎯 RESULTS:")
        print(f"  Type: {fiber_type}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Total fibers: {analysis_data['total_fibers']}")
        print(f"  Hollow: {analysis_data['hollow_fibers']}")
        print(f"  Filaments: {analysis_data['filaments']}")
        
        # Check if it matches expectation
        if "hollow" in expected.lower() and fiber_type == "hollow_fiber":
            print(f"  ✅ CORRECT!")
        elif "filament" in expected.lower() and fiber_type == "filament":
            print(f"  ✅ CORRECT!")
        else:
            print(f"  ❌ WRONG - Expected {expected}")
        
        # Show individual fiber details
        print(f"\n📊 Individual Fibers:")
        for i, result in enumerate(analysis_data['individual_results']):
            props = result['fiber_properties']
            has_lumen = result['has_lumen']
            area = props['area']
            confidence = result['confidence']
            
            print(f"  Fiber {i+1}: {area:.0f} pixels, lumen={has_lumen}, conf={confidence:.3f}")
            
            if has_lumen and 'lumen_properties' in result:
                lumen_ratio = result['lumen_properties'].get('area_ratio', 0)
                print(f"    Lumen ratio: {lumen_ratio:.3f}")

if __name__ == "__main__":
    test_working_algorithm()