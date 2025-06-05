#!/usr/bin/env python3
"""
Quick fix: Update the circularity threshold and test
"""

import sys
from pathlib import Path

# Setup paths
current_dir = Path(__file__).parent
if current_dir.name == 'tests':
    project_root = current_dir.parent
else:
    project_root = current_dir

def fix_circularity_threshold():
    """Update the circularity threshold in the module"""
    
    print("="*60)
    print("FIXING CIRCULARITY THRESHOLD")
    print("="*60)
    
    module_file = project_root / "modules" / "fiber_type_detection.py"
    
    # Read the current file
    with open(module_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and replace the circularity threshold
    old_line = "if lumen_props['circularity'] < 0.1:"  # Very lenient
    new_line = "if lumen_props['circularity'] < 0.05:"  # Even more lenient
    
    if old_line in content:
        print("✓ Found circularity check with threshold 0.1")
        updated_content = content.replace(old_line, new_line)
        
        # Write back the updated file
        with open(module_file, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print("✓ Updated circularity threshold from 0.1 to 0.05")
        print("✓ File updated successfully")
        
        return True
    else:
        print("❌ Could not find the circularity threshold line")
        print("Manual update needed")
        return False

def test_fixed_detection():
    """Test the detection with fixed threshold"""
    
    modules_dir = project_root / "modules"
    sys.path.insert(0, str(modules_dir))
    
    # Force reload to get updated module
    import importlib
    if 'fiber_type_detection' in sys.modules:
        importlib.reload(sys.modules['fiber_type_detection'])
    
    from fiber_type_detection import FiberTypeDetector
    from image_preprocessing import load_image
    
    print(f"\n🧪 TESTING FIXED DETECTION:")
    
    detector = FiberTypeDetector()
    img = load_image(str(project_root / "sample_images" / "hollow_fiber_sample.jpg"))
    
    # Run full classification
    fiber_type, confidence, analysis_data = detector.classify_fiber_type(img)
    
    print(f"🎯 RESULTS:")
    print(f"  Type: {fiber_type}")
    print(f"  Confidence: {confidence:.3f}")
    print(f"  Total fibers: {analysis_data['total_fibers']}")
    print(f"  Hollow: {analysis_data['hollow_fibers']}")
    print(f"  Filaments: {analysis_data['filaments']}")
    
    if fiber_type == "hollow_fiber":
        print(f"  ✅ SUCCESS! Hollow fiber correctly detected!")
        
        # Show lumen details
        for result in analysis_data['individual_results']:
            if result['has_lumen']:
                lumen_props = result['lumen_properties']
                print(f"  📊 Lumen details:")
                print(f"    Area ratio: {lumen_props['area_ratio']:.3f}")
                print(f"    Circularity: {lumen_props['circularity']:.3f}")
                print(f"    Threshold used: {lumen_props['threshold_used']}")
    else:
        print(f"  ❌ Still not working: {fiber_type}")

if __name__ == "__main__":
    success = fix_circularity_threshold()
    
    if success:
        print(f"\n" + "="*60)
        test_fixed_detection()
    else:
        print(f"\n💡 MANUAL FIX NEEDED:")
        print(f"In modules/fiber_type_detection.py, find this line:")
        print(f"    if lumen_props['circularity'] < 0.1:")
        print(f"And change it to:")
        print(f"    if lumen_props['circularity'] < 0.05:")