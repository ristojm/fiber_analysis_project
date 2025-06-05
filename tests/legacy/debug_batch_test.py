#!/usr/bin/env python3
"""
Debug Batch Test - See exactly what's happening with scale detection
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "modules"))

def debug_single_image(image_path):
    """Debug scale detection on a single image with detailed output."""
    
    print(f"\n🔍 DEBUGGING: {image_path.name}")
    print("=" * 50)
    
    try:
        from modules.image_preprocessing import load_image
        from modules.scale_detection import ScaleBarDetector
        
        # Load image
        image = load_image(str(image_path))
        print(f"✅ Image loaded: {image.shape}")
        
        # Test both methods side by side
        detector = ScaleBarDetector()
        
        # Method 1: Enhanced detection (what batch test uses)
        print(f"\n🧪 METHOD 1: Enhanced Detection (Batch Test Method)")
        result_enhanced = detector.detect_scale_bar(image)
        
        print(f"   Scale detected: {result_enhanced.get('scale_detected', False)}")
        print(f"   Method used: {result_enhanced.get('method_used', 'unknown')}")
        print(f"   Error: {result_enhanced.get('error', 'None')}")
        
        if result_enhanced.get('scale_detected', False):
            scale_info = result_enhanced.get('scale_info', {})
            print(f"   Scale text: {scale_info.get('original_text', 'N/A')}")
            print(f"   Scale value: {scale_info.get('value', 0)} {scale_info.get('unit', '')}")
            print(f"   Scale factor: {result_enhanced.get('micrometers_per_pixel', 0):.4f} μm/pixel")
        
        # Method 2: Simple detection (what worked individually)
        print(f"\n🎯 METHOD 2: Simple Detection (Individual Test Method)")
        
        # Extract scale region
        scale_region, y_offset = detector.extract_scale_region(image)
        print(f"   Scale region: {scale_region.shape}")
        
        # Try EasyOCR directly
        try:
            import easyocr
            reader = easyocr.Reader(['en'])
            results = reader.readtext(scale_region, detail=1)
            
            print(f"   EasyOCR found {len(results)} text elements:")
            
            for (bbox, text, confidence) in results:
                print(f"     '{text}' (confidence: {confidence:.3f})")
                
                # Try simple parsing
                scale_info = simple_smart_parse(text)
                if scale_info:
                    print(f"       ✅ Parsed: {scale_info['value']} {scale_info['unit']} = {scale_info['micrometers']} μm")
                else:
                    print(f"       ❌ Could not parse")
        
        except Exception as e:
            print(f"   EasyOCR failed: {e}")
        
        # Method 3: Force enhanced=False (original method)
        print(f"\n🔧 METHOD 3: Original Detection (Enhanced=False)")
        result_original = detector.detect_scale_bar(image)
        result_original['method_used'] = 'original'
        
        print(f"   Scale detected: {result_original.get('scale_detected', False)}")
        print(f"   Error: {result_original.get('error', 'None')}")
        
        if result_original.get('scale_detected', False):
            scale_info = result_original.get('scale_info', {})
            print(f"   Scale text: {scale_info.get('original_text', 'N/A')}")
            print(f"   Scale value: {scale_info.get('value', 0)} {scale_info.get('unit', '')}")
            print(f"   Scale factor: {result_original.get('micrometers_per_pixel', 0):.4f} μm/pixel")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def simple_smart_parse(text: str) -> dict:
    """
    Simple OCR error-tolerant parsing (same as working individual test).
    """
    import re
    
    # Extract number
    number_match = re.search(r'(\d+\.?\d*)', text)
    if not number_match:
        return None
    
    number = float(number_match.group(1))
    
    # Simple unit fixing with common OCR errors
    text_lower = text.lower()
    
    # Look for micrometer patterns (most common for SEM)
    if any(pattern in text_lower for pattern in [
        'μm', 'µm', 'um', 'jm', 'jim', 'qum', 'opm', 'ojm', 'om', 'pm'
    ]):
        unit = 'μm'
        micrometers = number
    
    # Look for millimeter patterns
    elif 'mm' in text_lower:
        unit = 'mm'
        micrometers = number * 1000
    
    # Look for nanometer patterns
    elif 'nm' in text_lower:
        unit = 'nm' 
        micrometers = number * 0.001
    
    else:
        return None
    
    # Filter reasonable SEM scale values
    if 10 <= micrometers <= 3000:  # 10μm to 3mm range
        return {
            'value': number,
            'unit': unit,
            'micrometers': micrometers,
            'original_text': text
        }
    
    return None

def debug_batch():
    """Debug all images in batch."""
    
    print("🧪 DEBUG BATCH SCALE DETECTION")
    print("=" * 60)
    
    # Find images
    sample_dir = project_root / "sample_images"
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(sample_dir.glob(f'*{ext}'))
        image_files.extend(sample_dir.glob(f'*{ext.upper()}'))
    
    image_files = sorted(set(image_files))  # Remove duplicates
    
    print(f"Found {len(image_files)} images:")
    for img in image_files:
        print(f"  - {img.name}")
    
    # Debug each image
    for image_path in image_files:
        debug_single_image(image_path)
        print()  # Add spacing

def main():
    """Main debug function."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug scale detection')
    parser.add_argument('--single', '-s', help='Debug single image file')
    parser.add_argument('--all', '-a', action='store_true', help='Debug all images')
    
    args = parser.parse_args()
    
    if args.single:
        image_path = Path(args.single)
        if image_path.exists():
            debug_single_image(image_path)
        else:
            print(f"❌ Image not found: {image_path}")
    
    elif args.all:
        debug_batch()
    
    else:
        # Default: debug the working image
        working_image = project_root / "sample_images" / "hollow_fiber_sample.jpg"
        if working_image.exists():
            print("🔍 Debugging the known working image first...")
            debug_single_image(working_image)
        else:
            print("❌ Default image not found, use --all or --single")

if __name__ == "__main__":
    main()