#!/usr/bin/env python3
"""
Simple Scale Detection Fix
Just fix the OCR parsing issue without overcomplicating things
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import re

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "modules"))

def simple_smart_parse(text: str) -> dict:
    """
    Simple OCR error-tolerant parsing. Just handles the common OCR mistakes.
    """
    
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

def simple_scale_detection():
    """
    Simple scale detection - just get it working without overcomplicating.
    """
    
    print("🔧 SIMPLE SCALE DETECTION")
    print("=" * 40)
    
    from image_preprocessing import load_image
    from scale_detection import ScaleBarDetector
    
    # Load image
    img_path = project_root / "sample_images" / "hollow_fiber_sample3.jpg"
    image = load_image(str(img_path))
    
    # Extract scale region
    detector = ScaleBarDetector()
    scale_region, y_offset = detector.extract_scale_region(image)
    
    print(f"Scale region: {scale_region.shape}")
    
    # Get OCR results
    try:
        import easyocr
        reader = easyocr.Reader(['en'])
        results = reader.readtext(scale_region, detail=1)
        
        print(f"\nOCR Results:")
        
        scale_candidates = []
        
        for (bbox, text, confidence) in results:
            print(f"  '{text}' (conf: {confidence:.3f})")
            
            # Simple parsing
            scale_info = simple_smart_parse(text)
            
            if scale_info:
                # Simple scoring: OCR confidence + scale reasonableness
                if 100 <= scale_info['micrometers'] <= 1000:  # Common range
                    scale_score = 1.0
                else:
                    scale_score = 0.5
                
                combined_score = confidence * 0.7 + scale_score * 0.3
                
                bbox_array = np.array(bbox)
                center_x = int(np.mean(bbox_array[:, 0]))
                center_y = int(np.mean(bbox_array[:, 1]))
                
                scale_candidates.append({
                    'text': text,
                    'value': scale_info['value'],
                    'unit': scale_info['unit'],
                    'micrometers': scale_info['micrometers'],
                    'center_x': center_x,
                    'center_y': center_y,
                    'bbox': bbox_array,
                    'score': combined_score
                })
                
                print(f"    ✅ {scale_info['value']} {scale_info['unit']} = {scale_info['micrometers']} μm (score: {combined_score:.3f})")
        
        if not scale_candidates:
            print("❌ No scale found")
            return
        
        # Use best candidate
        best = max(scale_candidates, key=lambda x: x['score'])
        print(f"\n🎯 Best: {best['value']} {best['unit']} = {best['micrometers']} μm")
        
        # Now find the scale bar lines (use existing working code)
        from complete_scale_bar_detection import find_complete_scale_bar_span
        
        best_span, all_segments = find_complete_scale_bar_span(
            scale_region, 
            best['center_x'], 
            best['center_y'], 
            best['bbox']
        )
        
        if best_span:
            span_pixels = best_span['total_span']
            calculated_scale = best['micrometers'] / span_pixels
            
            print(f"\n📏 RESULTS:")
            print(f"  Text: {best['value']} {best['unit']}")
            print(f"  Span: {span_pixels:.1f} pixels")
            print(f"  Scale: {calculated_scale:.4f} μm/pixel")
            
            # Compare with expected
            expected = 0.444
            error = abs(calculated_scale - expected) / expected * 100
            print(f"  Expected: {expected:.4f} μm/pixel")
            print(f"  Error: {error:.1f}%")
            
            if error < 20:
                print("  ✅ GOOD RESULT!")
            else:
                print("  ⚠️ Check result")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    simple_scale_detection()
    
    print(f"\n💡 SIMPLE APPROACH:")
    print("- Just handle common OCR errors: jm→μm, Qum→μm, Opm→μm")
    print("- Use basic scoring: OCR confidence + reasonable scale values") 
    print("- Keep existing line detection code that works")
    print("- Total: ~100 lines instead of 5000+")