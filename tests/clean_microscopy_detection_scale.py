"""
Advanced Scale Detection for SEM Images
Based on your comprehensive example + EasyOCR + your working patterns
"""

import cv2
import numpy as np
import re
from typing import Dict, Optional, List, Tuple
import matplotlib.pyplot as plt

# OCR imports with fallbacks
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except:
    EASYOCR_AVAILABLE = False

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except:
    PYTESSERACT_AVAILABLE = False

def detect_scale_bar(image_input, debug=False, **kwargs) -> Dict:
    """
    Advanced detection of scale bar in SEM images.
    
    Args:
        image_input: Path to image or numpy array
        debug: If True, show detailed output
        
    Returns:
        Dict with scale detection results in your format
    """
    
    # Handle image loading
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            try:
                from PIL import Image
                pil_img = Image.open(image_input)
                img = np.array(pil_img)
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            except:
                return {
                    'scale_detected': False,
                    'micrometers_per_pixel': 0.0,
                    'scale_factor': 0.0,
                    'error': f'Could not load image: {image_input}'
                }
    else:
        img = image_input if len(image_input.shape) == 3 else cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
    
    if debug:
        print(f"🔍 Analyzing image: {img.shape}")
    
    # Preprocess the image
    enhanced = preprocess_for_scale_detection(img)
    
    # Find scale bar candidates
    scale_bar_candidates = find_scale_bar_candidates(enhanced, debug)
    
    result = {
        'scale_detected': False,
        'micrometers_per_pixel': 0.0,
        'scale_factor': 0.0,
        'error': None,
        'method_used': 'advanced_multi_threshold'
    }
    
    if not scale_bar_candidates:
        result['error'] = "No scale bar candidates found"
        return result
    
    # Select best candidate
    best_candidate = select_best_candidate(scale_bar_candidates, debug)
    
    if debug:
        print(f"✅ Selected best candidate: {best_candidate['width']} pixels wide")
    
    # Extract scale text around the scale bar
    scale_text_info = extract_scale_text_advanced(img, best_candidate, debug)
    
    if not scale_text_info:
        if debug:
            print("❌ No scale text found")
        result['error'] = "Scale bar found but no readable text"
        return result
    
    if debug:
        print(f"✅ Found scale text: '{scale_text_info['text']}' = {scale_text_info['micrometers']} μm")
    
    # Calculate scale factor
    micrometers_per_pixel = scale_text_info['micrometers'] / best_candidate['width']
    
    result.update({
        'scale_detected': True,
        'micrometers_per_pixel': micrometers_per_pixel,
        'scale_factor': micrometers_per_pixel,
        'scale_info': scale_text_info,
        'bar_length_pixels': best_candidate['width'],
        'bar_coordinates': (best_candidate['x'], best_candidate['y'], 
                           best_candidate['width'], best_candidate['height'])
    })
    
    if debug:
        print(f"✅ SUCCESS: {micrometers_per_pixel:.4f} μm/pixel")
    
    return result

def preprocess_for_scale_detection(img: np.ndarray) -> np.ndarray:
    """Preprocess the image to better detect the scale bar."""
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    return enhanced

def find_scale_bar_candidates(enhanced: np.ndarray, debug=False) -> List[Dict]:
    """Find scale bar candidates using multiple thresholds."""
    
    height, width = enhanced.shape
    
    # Focus on the bottom strip (bottom 15% of image)
    bottom_strip_height = int(height * 0.15)
    bottom_strip = enhanced[height - bottom_strip_height:, :]
    
    if debug:
        print(f"   Bottom strip: {bottom_strip.shape}")
    
    # Create multiple binary versions with different thresholds
    binary_versions = []
    threshold_values = [180, 200, 220, 240]
    
    for thresh_val in threshold_values:
        _, binary = cv2.threshold(bottom_strip, thresh_val, 255, cv2.THRESH_BINARY)
        binary_versions.append((binary, f"thresh_{thresh_val}"))
    
    # Also try adaptive thresholding
    adaptive = cv2.adaptiveThreshold(bottom_strip, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    binary_versions.append((adaptive, "adaptive"))
    
    scale_bar_candidates = []
    
    for idx, (binary, method_name) in enumerate(binary_versions):
        # Focus on the right side where scale bar typically is
        scale_region_width = int(width * 0.5)  # Right 50% of the image
        scale_region = binary[:, width - scale_region_width:]
        
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        horizontal_lines = cv2.morphologyEx(scale_region, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Remove noise
        denoised = cv2.morphologyEx(horizontal_lines, cv2.MORPH_CLOSE, 
                                   cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        
        # Find contours
        contours, _ = cv2.findContours(denoised, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours to find scale bar candidates
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Scale bar criteria:
            # - Should be horizontal (width > height)
            # - Should have reasonable width (not too small or too large)
            # - Should be thin (small height)
            if w > 30 and w < scale_region_width * 0.8 and h < 10 and w > h * 10:
                # Adjust coordinates to full image
                bar_x = x + (width - scale_region_width)
                bar_y = y + (height - bottom_strip_height)
                
                scale_bar_candidates.append({
                    'x': bar_x,
                    'y': bar_y,
                    'width': w,
                    'height': h,
                    'threshold_idx': idx,
                    'method': method_name,
                    'aspect_ratio': w / h if h > 0 else 0
                })
    
    if debug:
        print(f"   Found {len(scale_bar_candidates)} scale bar candidates")
        for i, candidate in enumerate(scale_bar_candidates[:3]):  # Show first 3
            print(f"     {i+1}. {candidate['width']}x{candidate['height']} pixels ({candidate['method']})")
    
    return scale_bar_candidates

def select_best_candidate(candidates: List[Dict], debug=False) -> Optional[Dict]:
    """Select the best scale bar candidate."""
    
    if not candidates:
        return None
    
    # Sort by multiple criteria:
    # 1. Width (longer bars are typically better)
    # 2. Aspect ratio (more horizontal is better)
    # 3. Position (rightmost is often better for SEM images)
    candidates.sort(key=lambda c: (c['width'], c['aspect_ratio'], c['x']), reverse=True)
    
    return candidates[0]

def extract_scale_text_advanced(img: np.ndarray, scale_bar: Dict, debug=False) -> Optional[Dict]:
    """Extract scale text around the detected scale bar using multiple OCR methods."""
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    height, width = gray.shape
    
    # Define text search region (area around and above the scale bar)
    x, y, w, h = scale_bar['x'], scale_bar['y'], scale_bar['width'], scale_bar['height']
    
    text_padding = 100
    text_y1 = max(0, y - text_padding)
    text_y2 = min(height, y + h + text_padding)
    text_x1 = max(0, x - text_padding)
    text_x2 = min(width, x + w + text_padding)
    
    text_region = gray[text_y1:text_y2, text_x1:text_x2]
    
    if debug:
        print(f"   Text search region: {text_region.shape}")
    
    # Try EasyOCR first (works best for your images)
    if EASYOCR_AVAILABLE:
        scale_info = extract_with_easyocr_advanced(text_region, debug)
        if scale_info:
            return scale_info
    
    # Try Tesseract with multiple preprocessing methods
    if PYTESSERACT_AVAILABLE:
        scale_info = extract_with_tesseract_advanced(text_region, debug)
        if scale_info:
            return scale_info
    
    if debug:
        print("   ❌ All OCR methods failed")
    
    return None

def extract_with_easyocr_advanced(text_region: np.ndarray, debug=False) -> Optional[Dict]:
    """Extract scale text using EasyOCR with your proven patterns."""
    
    try:
        reader = easyocr.Reader(['en'], verbose=False)
        results = reader.readtext(text_region, detail=1)
        
        if debug:
            print(f"   EasyOCR found {len(results)} text elements")
        
        for (bbox, text, confidence) in results:
            if confidence > 0.3:
                if debug:
                    print(f"     '{text}' (confidence: {confidence:.3f})")
                
                scale_info = parse_scale_text_comprehensive(text.strip())
                if scale_info:
                    if debug:
                        print(f"       ✅ Parsed: {scale_info['value']} {scale_info['unit']} = {scale_info['micrometers']} μm")
                    
                    return {
                        'text': text.strip(),
                        'value': scale_info['value'],
                        'unit': scale_info['unit'],
                        'micrometers': scale_info['micrometers'],
                        'confidence': confidence,
                        'method': 'EasyOCR',
                        'original_text': text.strip()
                    }
                elif debug:
                    print(f"       ❌ Could not parse")
    
    except Exception as e:
        if debug:
            print(f"   ⚠️ EasyOCR failed: {e}")
    
    return None

def extract_with_tesseract_advanced(text_region: np.ndarray, debug=False) -> Optional[Dict]:
    """Extract scale text using Tesseract with multiple preprocessing methods."""
    
    # Try multiple OCR preprocessing methods
    ocr_images = []
    
    # Method 1: Direct thresholding
    _, binary1 = cv2.threshold(text_region, 200, 255, cv2.THRESH_BINARY)
    ocr_images.append((binary1, "thresh_200"))
    
    # Method 2: Inverted thresholding
    _, binary2 = cv2.threshold(text_region, 200, 255, cv2.THRESH_BINARY_INV)
    ocr_images.append((binary2, "thresh_inv"))
    
    # Method 3: Enlarged and enhanced
    enlarged = cv2.resize(text_region, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, binary3 = cv2.threshold(enlarged, 180, 255, cv2.THRESH_BINARY)
    ocr_images.append((binary3, "enlarged"))
    
    # Try OCR on each preprocessed image
    for ocr_img, method_name in ocr_images:
        try:
            # Try different OCR configurations
            configs = ['--psm 7', '--psm 8', '--psm 11', '--psm 13']
            
            for config in configs:
                text = pytesseract.image_to_string(ocr_img, config=config)
                text_clean = text.strip()
                
                if text_clean:
                    if debug:
                        print(f"   Tesseract ({method_name}, {config}): '{text_clean}'")
                    
                    scale_info = parse_scale_text_comprehensive(text_clean)
                    if scale_info:
                        if debug:
                            print(f"     ✅ Parsed: {scale_info['value']} {scale_info['unit']}")
                        
                        return {
                            'text': text_clean,
                            'value': scale_info['value'],
                            'unit': scale_info['unit'],
                            'micrometers': scale_info['micrometers'],
                            'confidence': 0.5,  # Default for Tesseract
                            'method': f'Tesseract_{method_name}_{config}',
                            'original_text': text_clean
                        }
        
        except Exception as e:
            if debug:
                print(f"   ⚠️ Tesseract {method_name} failed: {e}")
            continue
    
    return None

def parse_scale_text_comprehensive(text: str) -> Optional[Dict]:
    """
    Comprehensive scale text parsing including your proven OCR error patterns.
    """
    
    # All patterns that work from your debug output + standard patterns
    patterns = [
        # Your specific OCR error patterns that work
        (r'(\d+\.?\d*)\s*[Oo]jm', 'μm', 1.0),      # "400.Ojm" -> 400 μm
        (r'(\d+\.?\d*)\s*[Oo]pm', 'μm', 1.0),      # "400 Opm" -> 400 μm
        (r'(\d+\.?\d*)\s*Qum', 'μm', 1.0),         # "300 Qum" -> 300 μm
        
        # Standard micrometer patterns
        (r'(\d+\.?\d*)\s*[μµ]m', 'μm', 1.0),
        (r'(\d+\.?\d*)\s*um', 'μm', 1.0),
        (r'(\d+\.?\d*)\s*jm', 'μm', 1.0),
        
        # Other units
        (r'(\d+\.?\d*)\s*nm', 'nm', 0.001),
        (r'(\d+\.?\d*)\s*mm', 'mm', 1000.0),
        
        # Decimal patterns with various separators
        (r'(\d+)[,.](\d+)\s*[Oo]jm', 'μm', 1.0),
        (r'(\d+)[,.](\d+)\s*[Oo]pm', 'μm', 1.0),
        (r'(\d+)[,.](\d+)\s*[μµ]m', 'μm', 1.0),
        (r'(\d+)[,.](\d+)\s*um', 'μm', 1.0),
        
        # Patterns without space
        (r'(\d+\.?\d*)[μµ]m', 'μm', 1.0),
        (r'(\d+\.?\d*)um', 'μm', 1.0),
        (r'(\d+\.?\d*)nm', 'nm', 0.001),
        (r'(\d+\.?\d*)mm', 'mm', 1000.0),
    ]
    
    text_clean = text.lower().replace(' ', '').replace('_', '').replace('\n', '').replace('\t', '')
    
    for pattern, unit, conversion in patterns:
        match = re.search(pattern, text_clean)
        if match:
            try:
                if len(match.groups()) > 1:
                    # Decimal pattern
                    value = float(f"{match.group(1)}.{match.group(2)}")
                else:
                    value = float(match.group(1))
                
                micrometers = value * conversion
                
                # Reasonable range for SEM scales (10nm to 3mm)
                if 0.01 <= micrometers <= 3000:
                    return {
                        'value': value,
                        'unit': unit,
                        'micrometers': micrometers
                    }
            except (ValueError, IndexError):
                continue
    
    return None

# Compatibility functions for your existing codebase
def detect_scale_factor_only(image_input, **kwargs) -> float:
    """Return just the scale factor."""
    result = detect_scale_bar(image_input, **kwargs)
    return result['micrometers_per_pixel']

class ScaleBarDetector:
    """Compatibility class for your existing code."""
    def __init__(self, **kwargs):
        self.debug = kwargs.get('debug', False)
    
    def detect_scale_bar(self, image: np.ndarray) -> Dict:
        return detect_scale_bar(image, debug=self.debug)

def manual_scale_calibration(bar_length_pixels: int, bar_length_micrometers: float) -> float:
    """Manual calibration when automatic detection fails."""
    return bar_length_micrometers / bar_length_pixels

def visualize_scale_detection(image_path: str, scale_info: Dict, output_path: str = None):
    """Visualize the detected scale bar (adapted from your example)."""
    
    img = cv2.imread(image_path)
    if img is None:
        try:
            from PIL import Image
            pil_img = Image.open(image_path)
            img = np.array(pil_img)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        except:
            return None
    
    if scale_info['scale_detected'] and 'bar_coordinates' in scale_info:
        x, y, w, h = scale_info['bar_coordinates']
        
        # Draw rectangle around scale bar in green
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # Draw endpoints
        cv2.circle(img, (x, y + h//2), 5, (0, 255, 0), -1)
        cv2.circle(img, (x + w, y + h//2), 5, (0, 255, 0), -1)
        
        # Add text with scale information
        if 'scale_info' in scale_info:
            info = scale_info['scale_info']
            text = f"{info['value']}{info['unit']} = {w}px"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            
            # Add background for better text visibility
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(img, (x, y - text_height - 20), 
                         (x + text_width + 10, y - 5), (0, 0, 0), -1)
            cv2.putText(img, text, (x + 5, y - 10), font, 
                       font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
            
            # Add pixels per unit info
            ppu_text = f"Scale: {scale_info['micrometers_per_pixel']:.4f} μm/pixel"
            cv2.putText(img, ppu_text, (x + 5, y + h + 30), font, 
                       0.7, (0, 255, 0), 2, cv2.LINE_AA)
    
    if output_path:
        cv2.imwrite(output_path, img)
    
    return img

# Test function
def test_advanced_detection():
    """Test the advanced detection approach."""
    
    print("🧪 TESTING ADVANCED SCALE DETECTION")
    print("=" * 50)
    
    from pathlib import Path
    
    test_images = [
        "hollow_fiber_sample.jpg",      # "400.Ojm"
        "hollow_fiber_sample2.jpg",     # "400 Opm"
        "hollow_fiber_sample3.jpg",     # "300 Qum" (working)
        "solid_filament_sample.jpg"     # "500.Opm"
    ]
    
    sample_dir = Path("sample_images")
    if not sample_dir.exists():
        sample_dir = Path("../sample_images")
    
    successful = 0
    
    for filename in test_images:
        image_path = sample_dir / filename
        
        if not image_path.exists():
            print(f"⚠️ {filename} not found")
            continue
        
        print(f"\n📸 Testing: {filename}")
        
        result = detect_scale_bar(str(image_path), debug=True)
        
        if result['scale_detected']:
            print(f"🎉 SUCCESS: {result['micrometers_per_pixel']:.4f} μm/pixel")
            successful += 1
        else:
            print(f"❌ FAILED: {result.get('error')}")
    
    print(f"\n📊 FINAL RESULTS: {successful}/{len(test_images)} images detected")
    
    if successful >= 3:
        print("🎉 Advanced detection works! This should be your final version.")
    else:
        print("😞 Still need adjustments")

if __name__ == "__main__":
    test_advanced_detection()