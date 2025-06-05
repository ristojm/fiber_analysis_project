#!/usr/bin/env python3
"""
Simple test to isolate where the UnicodeDecodeError occurs.
"""

import sys
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent
modules_dir = project_root / "modules"
sys.path.insert(0, str(modules_dir))

def test_step_by_step():
    """Test each step individually to find where the error occurs."""
    
    print("="*60)
    print("STEP-BY-STEP ANALYSIS TEST")
    print("="*60)
    
    # Step 1: Test image loading only
    print("STEP 1: Testing image preprocessing...")
    try:
        from image_preprocessing import load_image
        
        sample_dir = project_root / "sample_images"
        image_files = list(sample_dir.glob("*.jpg"))
        
        if image_files:
            test_image = image_files[0]
            print(f"Loading: {test_image.name}")
            
            img = load_image(str(test_image))
            print(f"✓ Image loaded: {img.shape}")
        else:
            print("❌ No JPG files found")
            return
            
    except Exception as e:
        print(f"❌ Image loading failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Test fiber detection only (no OCR)
    print("\nSTEP 2: Testing fiber type detection...")
    try:
        from fiber_type_detection import FiberTypeDetector
        
        detector = FiberTypeDetector(
            min_fiber_area=1000,
            lumen_area_threshold=0.02,
            circularity_threshold=0.2,
            confidence_threshold=0.6
        )
        
        fiber_type, confidence, analysis_data = detector.classify_fiber_type(img)
        print(f"✓ Fiber type detection: {fiber_type} (confidence: {confidence:.3f})")
        print(f"✓ Fibers detected: {analysis_data['total_fibers']}")
        
    except Exception as e:
        print(f"❌ Fiber detection failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Test scale detection WITHOUT OCR
    print("\nSTEP 3: Testing scale detection (without OCR)...")
    try:
        from scale_detection import ScaleBarDetector
        
        # Create detector but skip OCR for now
        scale_detector = ScaleBarDetector()
        
        # Test just the scale bar line detection part
        scale_region, y_offset = scale_detector.extract_scale_region(img)
        bar_candidates = scale_detector.detect_scale_bar_line(scale_region)
        
        print(f"✓ Scale region extracted: {scale_region.shape}")
        print(f"✓ Scale bar candidates found: {len(bar_candidates)}")
        
        if bar_candidates:
            best_bar = bar_candidates[0]
            print(f"✓ Best bar length: {best_bar['length']} pixels")
        
    except Exception as e:
        print(f"❌ Scale detection failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Test OCR separately (this might be where the error occurs)
    print("\nSTEP 4: Testing OCR text extraction...")
    try:
        # Skip OCR for now to see if that's the issue
        print("⏳ Skipping OCR test for now...")
        print("✓ Manual scale calibration available as fallback")
        
    except Exception as e:
        print(f"❌ OCR failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*60)
    print("✅ STEP-BY-STEP TEST COMPLETE!")
    print("="*60)
    print(f"Results summary:")
    print(f"- Image: {test_image.name}")
    print(f"- Type: {fiber_type}")
    print(f"- Confidence: {confidence:.3f}")
    print(f"- Scale bar candidates: {len(bar_candidates)}")

if __name__ == "__main__":
    test_step_by_step()