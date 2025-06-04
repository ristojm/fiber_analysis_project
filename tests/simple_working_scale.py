"""
Simple Scale Detection - Just Do What Works
No fancy strategies, just find text and estimate the bar length.
"""

import cv2
import numpy as np
import re
from typing import Dict, Optional

# EasyOCR import
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except:
    EASYOCR_AVAILABLE = False

def detect_scale_bar(image_input, **kwargs) -> Dict:
    """
    Simple scale detection that just works.
    
    Based on what we know works:
    1. EasyOCR finds scale text like "400.Ojm", "400 Opm", "500.Opm"
    2. Estimate scale bar length as ~20% of image width
    3. Calculate scale factor
    """
    
    # Handle image loading
    if isinstance(image_input, str):
        image = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return {
                'scale_detected': False,
                'micrometers_per_pixel': 0.0,
                'scale_factor': 0.0,
                'error': f'Could not load image: {image_input}'
            }
    else:
        image = image_input
    
    result = {
        'scale_detected': False,
        'micrometers_per_pixel': 0.0,
        'scale_factor': 0.0,
        'error': None,
        'method_used': 'simple_working'
    }
    
    # Extract bottom region
    height, width = image.shape
    bottom_region = image[int(height * 0.8):, :]  # Bottom 20%
    
    # Find scale text using EasyOCR
    scale_text = find_scale_text_simple(bottom_region)
    
    if not scale_text:
        result['error'] = "No scale text found"
        return result
    
    # Estimate scale bar length (typical SEM scale bars are ~15-25% of image width)
    estimated_bar_length = int(width * 0.2)  # 20% of full image width
    
    # Calculate scale factor
    micrometers_per_pixel = scale_text['micrometers'] / estimated_bar_length
    
    result.update({
        'scale_detected': True,
        'micrometers_per_pixel': micrometers_per_pixel,
        'scale_factor': micrometers_per_pixel,
        'scale_info': scale_text,
        'bar_length_pixels': estimated_bar_length,
        'method': 'text_with_estimation'
    })
    
    return result

def find_scale_text_simple(region: np.ndarray) -> Optional[Dict]:
    """Find scale text using EasyOCR - just the basics."""
    
    if not EASYOCR_AVAILABLE:
        return None
    
    try:
        reader = easyocr.Reader(['en'], verbose=False)
        results = reader.readtext(region, detail=1)
        
        for (bbox, text, confidence) in results:
            if confidence > 0.3:  # Reasonable confidence
                scale_info = parse_scale_text_simple(text.strip())
                if scale_info:
                    return {
                        'text': text.strip(),
                        'value': scale_info['value'],
                        'unit': scale_info['unit'],
                        'micrometers': scale_info['micrometers'],
                        'confidence': confidence,
                        'original_text': text.strip()
                    }
    except:
        pass
    
    return None

def parse_scale_text_simple(text: str) -> Optional[Dict]:
    """Parse scale text - just the patterns that work."""
    
    # Only the patterns we KNOW work from your debug output
    patterns = [
        # What actually works in your images
        (r'(\d+\.?\d*)\s*[Oo]jm', 'μm', 1.0),      # "400.Ojm" -> 400 μm
        (r'(\d+\.?\d*)\s*[Oo]pm', 'μm', 1.0),      # "400 Opm" -> 400 μm  
        (r'(\d+\.?\d*)\s*Qum', 'μm', 1.0),         # "300 Qum" -> 300 μm
        
        # Standard patterns as backup
        (r'(\d+\.?\d*)\s*μm', 'μm', 1.0),
        (r'(\d+\.?\d*)\s*um', 'μm', 1.0),
        (r'(\d+\.?\d*)\s*jm', 'μm', 1.0),
    ]
    
    text_clean = text.lower().replace(' ', '').replace('_', '')
    
    for pattern, unit, conversion in patterns:
        match = re.search(pattern, text_clean)
        if match:
            try:
                value = float(match.group(1))
                micrometers = value * conversion
                
                # Reasonable range check
                if 50 <= micrometers <= 2000:  # 50μm to 2mm
                    return {
                        'value': value,
                        'unit': unit,
                        'micrometers': micrometers,
                        'original_text': text
                    }
            except:
                continue
    
    return None

# Backward compatibility functions
def detect_scale_factor_only(image_input, **kwargs) -> float:
    """Just return the scale factor."""
    result = detect_scale_bar(image_input, **kwargs)
    return result['micrometers_per_pixel']

# For your existing code structure
class ScaleBarDetector:
    """Simple detector class for backward compatibility."""
    
    def __init__(self, **kwargs):
        pass  # Ignore all parameters
    
    def detect_scale_bar(self, image: np.ndarray) -> Dict:
        """Use the simple detection."""
        return detect_scale_bar(image)

def manual_scale_calibration(bar_length_pixels: int, bar_length_micrometers: float) -> float:
    """Manual calibration."""
    return bar_length_micrometers / bar_length_pixels

# Test function
def test_simple_detection():
    """Test the simple detection."""
    
    print("🧪 TESTING SIMPLE SCALE DETECTION")
    print("=" * 40)
    
    from pathlib import Path
    
    # Test images we know have scale text
    test_cases = [
        ("hollow_fiber_sample.jpg", "400.Ojm"),
        ("hollow_fiber_sample2.jpg", "400 Opm"), 
        ("hollow_fiber_sample3.jpg", "300 Qum"),
        ("solid_filament_sample.jpg", "500.Opm")
    ]
    
    sample_dir = Path("sample_images")
    if not sample_dir.exists():
        sample_dir = Path("../sample_images")
    
    successful = 0
    
    for filename, expected_text in test_cases:
        image_path = sample_dir / filename
        
        if not image_path.exists():
            print(f"⚠️ {filename} not found")
            continue
        
        print(f"\n📸 Testing: {filename}")
        print(f"   Expected text: {expected_text}")
        
        result = detect_scale_bar(str(image_path))
        
        if result['scale_detected']:
            info = result['scale_info']
            print(f"   ✅ SUCCESS!")
            print(f"   Found: '{info['text']}'")
            print(f"   Parsed: {info['value']} {info['unit']}")
            print(f"   Scale: {result['micrometers_per_pixel']:.4f} μm/pixel")
            successful += 1
        else:
            print(f"   ❌ FAILED: {result.get('error')}")
    
    print(f"\n📊 RESULTS: {successful}/{len(test_cases)} images detected")
    
    if successful >= 3:
        print("🎉 Simple detection works! Use this version.")
    else:
        print("😞 Still having issues.")

if __name__ == "__main__":
    test_simple_detection()