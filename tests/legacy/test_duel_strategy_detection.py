#!/usr/bin/env python3
"""
Test the dual strategy detection directly
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Setup paths
project_root = Path(__file__).parent.parent
if (project_root / "modules").exists():
    sys.path.insert(0, str(project_root / "modules"))
else:
    sys.path.insert(0, str(project_root))

def test_dual_strategy():
    """Test the dual strategy detection on a known failing image."""
    
    print("🧪 TESTING DUAL STRATEGY DETECTION")
    print("=" * 50)
    
    # Import the updated module
    from scale_detection import detect_scale_bar
    
    # Test with first failing image
    sample_dir = project_root / "sample_images"
    if not sample_dir.exists():
        sample_dir = project_root / "../sample_images"
    
    test_image = sample_dir / "hollow_fiber_sample.jpg"  # This one was failing
    
    if not test_image.exists():
        print(f"❌ Test image not found: {test_image}")
        return
    
    print(f"📸 Testing with: {test_image.name}")
    print("   (This image has '400.Ojm' text but was failing before)")
    
    # Test the main detection function
    print("\n🔍 Running detect_scale_bar()...")
    result = detect_scale_bar(str(test_image))
    
    print(f"\n📊 RESULTS:")
    print(f"   Scale detected: {result['scale_detected']}")
    print(f"   Method used: {result['method_used']}")
    print(f"   Error: {result.get('error', 'None')}")
    
    if result['scale_detected']:
        info = result.get('scale_info', {})
        print(f"   ✅ SUCCESS!")
        print(f"   Text: '{info.get('text', 'N/A')}'")
        print(f"   Parsed: {info.get('value', 0)} {info.get('unit', '')}")
        print(f"   Scale: {result['micrometers_per_pixel']:.4f} μm/pixel")
        print(f"   Bar length: {result.get('bar_length_pixels', 0)} pixels")
        
        if 'method' in result:
            print(f"   Detection method: {result['method']}")
    else:
        print(f"   ❌ STILL FAILED: {result.get('error', 'Unknown error')}")
        
        # Additional debugging
        print(f"\n🔍 DEBUG INFO:")
        if 'scale_region' in result and result['scale_region'] is not None:
            print(f"   Corner region size: {result['scale_region'].shape}")
        else:
            print(f"   No corner region found")

def test_all_failing_images():
    """Test all the failing images."""
    
    print("\n🧪 TESTING ALL FAILING IMAGES")
    print("=" * 50)
    
    from scale_detection import detect_scale_bar
    
    sample_dir = project_root / "sample_images"
    if not sample_dir.exists():
        sample_dir = project_root / "../sample_images"
    
    failing_images = [
        "hollow_fiber_sample.jpg",      # Has "400.Ojm"
        "hollow_fiber_sample2.jpg",     # Has "400 Opm"  
        "solid_filament_sample.jpg"     # Has "500.Opm"
    ]
    
    successful = 0
    
    for image_name in failing_images:
        image_path = sample_dir / image_name
        if not image_path.exists():
            print(f"⚠️ {image_name} not found")
            continue
            
        print(f"\n📸 Testing: {image_name}")
        result = detect_scale_bar(str(image_path))
        
        if result['scale_detected']:
            info = result.get('scale_info', {})
            print(f"   ✅ SUCCESS: {info.get('value', 0)} {info.get('unit', '')} = {result['micrometers_per_pixel']:.4f} μm/pixel")
            successful += 1
        else:
            print(f"   ❌ FAILED: {result.get('error', 'Unknown error')}")
    
    print(f"\n📊 SUMMARY: {successful}/{len(failing_images)} images now working")
    
    if successful > 0:
        print("🎉 Dual strategy is working!")
    else:
        print("😞 Still having issues - need more debugging")

def main():
    """Main test function."""
    
    # Test single image with detailed output
    test_dual_strategy()
    
    # Test all failing images
    test_all_failing_images()

if __name__ == "__main__":
    main()