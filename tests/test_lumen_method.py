#!/usr/bin/env python3
"""
Test if the lumen detection method itself is updated
"""

import sys
from pathlib import Path
import inspect

# Setup paths
current_dir = Path(__file__).parent
if current_dir.name == 'tests':
    project_root = current_dir.parent
else:
    project_root = current_dir

modules_dir = project_root / "modules"
sys.path.insert(0, str(modules_dir))

def check_lumen_method():
    """Check what's in the detect_lumen method"""
    
    print("="*60)
    print("CHECKING LUMEN DETECTION METHOD")
    print("="*60)
    
    from fiber_type_detection import FiberTypeDetector
    from image_preprocessing import load_image
    
    # Check the detect_lumen method source
    lumen_method_source = inspect.getsource(FiberTypeDetector.detect_lumen)
    
    print("🔍 DETECT_LUMEN METHOD:")
    print("First 20 lines:")
    lines = lumen_method_source.split('\n')[:20]
    for i, line in enumerate(lines, 1):
        print(f"  {i:2d}: {line}")
    
    # Look for key indicators
    if "threshold = 50" in lumen_method_source:
        print("\n✅ Found fixed threshold = 50")
    elif "percentile" in lumen_method_source:
        print("\n❌ Still using percentile method (OLD)")
    else:
        print("\n❓ Unknown threshold method")
    
    # Test actual lumen detection
    print(f"\n🧪 TESTING LUMEN DETECTION:")
    
    detector = FiberTypeDetector()
    img = load_image(str(project_root / "sample_images" / "hollow_fiber_sample.jpg"))
    
    # Get the fiber
    preprocessed = detector.preprocess_for_detection(img)
    fiber_mask, fiber_properties = detector.segment_fibers(preprocessed)
    
    if fiber_properties:
        main_fiber = fiber_properties[0]  # The large one
        contour = main_fiber['contour']
        
        print(f"  Fiber area: {main_fiber['area']:.0f} pixels")
        
        # Test lumen detection
        has_lumen, lumen_props = detector.detect_lumen(img, contour)
        
        print(f"  Lumen detected: {has_lumen}")
        if has_lumen:
            print(f"  Lumen area ratio: {lumen_props.get('area_ratio', 0):.3f}")
            print(f"  Threshold used: {lumen_props.get('threshold_used', 'unknown')}")
        else:
            print(f"  No lumen properties returned")
    
    print(f"\n💡 CONCLUSION:")
    if "threshold = 50" in lumen_method_source:
        print("  ✅ Method is updated but may need debugging")
    else:
        print("  ❌ Method needs to be replaced with fixed version")

if __name__ == "__main__":
    check_lumen_method()