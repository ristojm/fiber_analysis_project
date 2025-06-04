#!/usr/bin/env python3
"""
Test the updated hollow fiber detection with enhanced parameters.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Setup paths
project_root = Path(__file__).parent.parent
modules_dir = project_root / "modules"
sys.path.insert(0, str(modules_dir))

def test_enhanced_detection():
    """Test hollow fiber detection with updated parameters."""
    
    print("="*60)
    print("TESTING ENHANCED HOLLOW FIBER DETECTION")
    print("="*60)
    
    # Load image
    sample_dir = project_root / "sample_images"
    image_files = list(sample_dir.glob("*.jpg"))
    
    if not image_files:
        print("❌ No images found")
        return
    
    # Test both images
    for img_file in image_files:
        print(f"\n📷 Testing: {img_file.name}")
        print("-" * 40)
        
        try:
            from image_preprocessing import load_image
            img = load_image(str(img_file))
            print(f"✓ Image loaded: {img.shape}")
            
            # Test with OLD parameters (current)
            print("\n🔴 OLD Parameters:")
            from fiber_type_detection import FiberTypeDetector
            
            old_detector = FiberTypeDetector(
                min_fiber_area=1000,
                lumen_area_threshold=0.05,  # OLD: 5%
                circularity_threshold=0.3,   # OLD: 0.3
                confidence_threshold=0.7     # OLD: 0.7
            )
            
            old_type, old_conf, old_data = old_detector.classify_fiber_type(img)
            print(f"  Type: {old_type}")
            print(f"  Confidence: {old_conf:.3f}")
            print(f"  Total fibers: {old_data['total_fibers']}")
            print(f"  Hollow: {old_data['hollow_fibers']}, Solid: {old_data['filaments']}")
            
            # Test with NEW parameters (enhanced)
            print("\n🟢 NEW Enhanced Parameters:")
            
            new_detector = FiberTypeDetector(
                min_fiber_area=5000,
                lumen_area_threshold=0.02,  # NEW: 2% (more sensitive)
                circularity_threshold=0.2,   # NEW: 0.2 (more lenient)
                confidence_threshold=0.6     # NEW: 0.6 (lower threshold)
            )
            
            new_type, new_conf, new_data = new_detector.classify_fiber_type(img)
            print(f"  Type: {new_type}")
            print(f"  Confidence: {new_conf:.3f}")
            print(f"  Total fibers: {new_data['total_fibers']}")
            print(f"  Hollow: {new_data['hollow_fibers']}, Solid: {new_data['filaments']}")
            
            # Analysis of individual fibers
            print(f"\n🔍 Individual Fiber Analysis:")
            for i, result in enumerate(new_data['individual_results']):
                fiber_props = result['fiber_properties']
                has_lumen = result['has_lumen']
                confidence = result['confidence']
                
                print(f"  Fiber {i+1}:")
                print(f"    Area: {fiber_props['area']:.0f} pixels")
                print(f"    Circularity: {fiber_props['circularity']:.3f}")
                print(f"    Has lumen: {has_lumen}")
                print(f"    Confidence: {confidence:.3f}")
                
                if has_lumen and 'lumen_properties' in result:
                    lumen_props = result['lumen_properties']
                    print(f"    Lumen area ratio: {lumen_props.get('area_ratio', 0):.3f}")
                    print(f"    Lumen circularity: {lumen_props.get('circularity', 0):.3f}")
            
            # Comparison
            print(f"\n📊 Comparison:")
            if old_type != new_type:
                print(f"  🔄 Classification CHANGED: {old_type} → {new_type}")
            else:
                print(f"  ➡️ Classification unchanged: {old_type}")
            
            conf_change = new_conf - old_conf
            print(f"  📈 Confidence change: {conf_change:+.3f}")
            
        except Exception as e:
            print(f"❌ Error testing {img_file.name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_detection()