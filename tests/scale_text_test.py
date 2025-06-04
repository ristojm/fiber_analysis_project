"""
Enhanced Scale Detection - Split Bar Method
Handles scale bars that are split by the text overlay

Key insight: Scale bar = left_segment + text_width + right_segment
The line should be at text middle height and aligned horizontally.
"""

import cv2
import numpy as np
import re
from typing import Dict, Optional, List, Tuple
from collections import Counter

# EasyOCR import
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except:
    EASYOCR_AVAILABLE = False

def detect_scale_bar_split_aware(image_input, debug=True, **kwargs) -> Dict:
    """
    Enhanced scale detection that properly handles split scale bars.
    
    Method:
    1. Find scale text with precise bounding box
    2. Look for line segments on left and right of text at text's vertical middle
    3. Calculate total span: left_line + text_width + right_line
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
    
    if debug:
        print(f"🔍 Analyzing image: {image.shape}")
    
    result = {
        'scale_detected': False,
        'micrometers_per_pixel': 0.0,
        'scale_factor': 0.0,
        'error': None,
        'method_used': 'split_bar_aware'
    }
    
    # Step 1: Find scale text with precise location
    if debug:
        print("📝 Step 1: Finding scale text with precise bounds...")
    
    text_info = find_scale_text_precise(image, debug)
    
    if not text_info:
        result['error'] = "No scale text found"
        return result
    
    if debug:
        print(f"✅ Found text: '{text_info['text']}' = {text_info['micrometers']} μm")
        print(f"   Text bbox: {text_info['bbox']}")
        print(f"   Text center: ({text_info['center_x']}, {text_info['center_y']})")
        print(f"   Text width: {text_info['width']} pixels")
    
    # Step 2: Find split scale bar segments
    if debug:
        print("📏 Step 2: Finding split scale bar segments...")
    
    total_span = find_split_scale_bar(image, text_info, debug)
    
    if total_span <= 0:
        result['error'] = "Could not detect complete scale bar span"
        return result
    
    # Step 3: Calculate scale factor
    micrometers_per_pixel = text_info['micrometers'] / total_span
    
    result.update({
        'scale_detected': True,
        'micrometers_per_pixel': micrometers_per_pixel,
        'scale_factor': micrometers_per_pixel,
        'scale_info': text_info,
        'bar_length_pixels': total_span,
        'detection_method': 'split_bar_reconstruction'
    })
    
    if debug:
        print(f"✅ SUCCESS: {micrometers_per_pixel:.4f} μm/pixel")
        print(f"   Total span: {total_span} pixels")
    
    return result

def find_scale_text_precise(image: np.ndarray, debug=False) -> Optional[Dict]:
    """
    Find scale text with precise bounding box information.
    """
    
    if not EASYOCR_AVAILABLE:
        if debug:
            print("❌ EasyOCR not available")
        return None
    
    try:
        reader = easyocr.Reader(['en'], verbose=False)
        
        # Search bottom 30% of image
        height, width = image.shape
        search_height = int(height * 0.3)
        bottom_region = image[height - search_height:, :]
        y_offset = height - search_height
        
        if debug:
            print(f"   Searching bottom region: {bottom_region.shape}")
        
        # Get all text with detailed bounding boxes
        results = reader.readtext(bottom_region, detail=1)
        
        if debug:
            print(f"   EasyOCR found {len(results)} text elements")
        
        # Process each text detection
        for (bbox, text, confidence) in results:
            if confidence > 0.3:
                if debug:
                    print(f"     '{text}' (confidence: {confidence:.3f})")
                
                scale_info = parse_scale_text_flexible(text.strip())
                if scale_info:
                    # Calculate precise bounding box in full image coordinates
                    bbox_array = np.array(bbox)
                    
                    # Adjust bbox to full image coordinates
                    bbox_adjusted = bbox_array.copy()
                    bbox_adjusted[:, 1] += y_offset
                    
                    # Calculate precise dimensions
                    min_x = int(np.min(bbox_adjusted[:, 0]))
                    max_x = int(np.max(bbox_adjusted[:, 0]))
                    min_y = int(np.min(bbox_adjusted[:, 1]))
                    max_y = int(np.max(bbox_adjusted[:, 1]))
                    
                    center_x = (min_x + max_x) // 2
                    center_y = (min_y + max_y) // 2
                    text_width = max_x - min_x
                    text_height = max_y - min_y
                    
                    if debug:
                        print(f"       ✅ Parsed: {scale_info['value']} {scale_info['unit']} = {scale_info['micrometers']} μm")
                        print(f"       Text dimensions: {text_width}x{text_height} pixels")
                    
                    return {
                        'text': text.strip(),
                        'value': scale_info['value'],
                        'unit': scale_info['unit'],
                        'micrometers': scale_info['micrometers'],
                        'confidence': confidence,
                        'center_x': center_x,
                        'center_y': center_y,
                        'bbox': bbox_adjusted,
                        'min_x': min_x,
                        'max_x': max_x,
                        'min_y': min_y,
                        'max_y': max_y,
                        'width': text_width,
                        'height': text_height,
                        'original_text': text.strip()
                    }
                elif debug:
                    print(f"       ❌ Could not parse")
        
    except Exception as e:
        if debug:
            print(f"❌ EasyOCR failed: {e}")
    
    return None

def find_split_scale_bar(image: np.ndarray, text_info: Dict, debug=False) -> int:
    """
    Find the complete scale bar that is split by the text overlay.
    
    Strategy:
    1. Look for horizontal lines at the text's vertical middle
    2. Find segments to the left and right of the text
    3. Calculate total span: left_segment + text_width + right_segment
    
    Key insight: The scale bar line runs through the middle height of the text
    """
    
    # Text position info
    text_center_x = text_info['center_x']
    text_center_y = text_info['center_y']
    text_min_x = text_info['min_x']
    text_max_x = text_info['max_x']
    text_width = text_info['width']
    text_height = text_info['height']
    
    # The scale bar line should be at the vertical middle of the text
    line_y_target = text_center_y
    
    if debug:
        print(f"   Text spans from x={text_min_x} to x={text_max_x} (width={text_width})")
        print(f"   Text height: {text_height}, center_y: {text_center_y}")
        print(f"   Looking for scale line at y≈{line_y_target}")
    
    # Create search regions to left and right of text
    search_margin = 300  # Look further from text
    y_tolerance = 8      # Look within 8 pixels of text center vertically (tight)
    
    height, width = image.shape
    
    # Left search region
    left_x1 = max(0, text_min_x - search_margin)
    left_x2 = text_min_x - 5  # Small gap to avoid text artifacts
    search_y1 = max(0, line_y_target - y_tolerance)
    search_y2 = min(height, line_y_target + y_tolerance)
    
    left_region = image[search_y1:search_y2, left_x1:left_x2]
    
    # Right search region  
    right_x1 = text_max_x + 5  # Small gap to avoid text artifacts
    right_x2 = min(width, text_max_x + search_margin)
    
    right_region = image[search_y1:search_y2, right_x1:right_x2]
    
    if debug:
        print(f"   Left search region: {left_region.shape} at x=[{left_x1}:{left_x2}], y=[{search_y1}:{search_y2}]")
        print(f"   Right search region: {right_region.shape} at x=[{right_x1}:{right_x2}], y=[{search_y1}:{search_y2}]")
    
    # Find horizontal lines in each region
    left_lines = find_horizontal_lines_in_region(left_region, debug=debug)
    right_lines = find_horizontal_lines_in_region(right_region, debug=debug)
    
    if debug:
        print(f"   Found {len(left_lines)} left segments, {len(right_lines)} right segments")
    
    # Strategy 1: Try to match left and right segments that align horizontally
    best_span = 0
    best_details = None
    
    for left_line in left_lines:
        for right_line in right_lines:
            # Check if they're at similar Y positions (aligned)
            left_y_global = search_y1 + left_line['center_y']
            right_y_global = search_y1 + right_line['center_y']
            
            if abs(left_y_global - right_y_global) <= 3:  # Within 3 pixels vertically
                # Calculate total span from leftmost to rightmost point
                left_start_global = left_x1 + left_line['start_x']
                right_end_global = right_x1 + right_line['end_x']
                
                total_span = right_end_global - left_start_global
                
                # Score this combination
                score = score_line_pair(left_line, right_line, text_info)
                
                if debug:
                    print(f"     Candidate pair: left={left_line['length']}px, right={right_line['length']}px")
                    print(f"       Y alignment: left={left_y_global}, right={right_y_global}")
                    print(f"       Total span: {total_span}px, score: {score:.3f}")
                
                if score > 0.4 and total_span > best_span and 150 <= total_span <= 800:
                    best_span = total_span
                    best_details = {
                        'left_line': left_line,
                        'right_line': right_line,
                        'left_length': left_line['length'],
                        'right_length': right_line['length'],
                        'text_width': text_width,
                        'total_span': total_span,
                        'score': score,
                        'method': 'paired_segments'
                    }
    
    # Strategy 2: If no good pair found, try individual segments plus text width
    if best_span == 0:
        if debug:
            print("   No aligned pairs found, trying individual segments...")
        
        # Find the best individual lines
        best_left = max(left_lines, key=lambda x: x['score']) if left_lines else None
        best_right = max(right_lines, key=lambda x: x['score']) if right_lines else None
        
        if best_left and best_right:
            # Calculate span: left_segment + text_width + right_segment
            # This assumes the segments connect directly to the text
            estimated_span = best_left['length'] + text_width + best_right['length']
            
            if debug:
                print(f"   Reconstructed span: {best_left['length']} + {text_width} + {best_right['length']} = {estimated_span}")
            
            if 150 <= estimated_span <= 800:
                best_span = estimated_span
                best_details = {
                    'left_length': best_left['length'],
                    'right_length': best_right['length'],
                    'text_width': text_width,
                    'total_span': estimated_span,
                    'method': 'reconstructed'
                }
        
        elif best_left or best_right:
            # Use proportional estimation if we only have one side
            single_line = best_left or best_right
            # Assume the visible line is roughly 35-45% of total span
            estimated_total = int(single_line['length'] / 0.4)
            
            if debug:
                print(f"   Single-side estimation: {single_line['length']} / 0.4 = {estimated_total}")
            
            if 150 <= estimated_total <= 800:
                best_span = estimated_total
                best_details = {
                    'visible_length': single_line['length'],
                    'total_span': estimated_total,
                    'method': 'single_side_proportion'
                }
    
    # Strategy 3: Fallback to scan for any horizontal structures near text
    if best_span == 0:
        if debug:
            print("   Trying fallback: scan for horizontal structures...")
        
        fallback_span = scan_for_horizontal_structures(image, text_info, debug)
        if fallback_span > 0:
            best_span = fallback_span
            best_details = {
                'total_span': fallback_span,
                'method': 'fallback_scan'
            }
    
    if best_span > 0 and debug:
        print(f"✅ Best span found: {best_span} pixels")
        if best_details:
            print(f"   Method: {best_details.get('method', 'unknown')}")
            if 'left_length' in best_details and 'right_length' in best_details:
                print(f"   Components: left={best_details['left_length']}px + text={best_details['text_width']}px + right={best_details['right_length']}px")
    
    return best_span

def scan_for_horizontal_structures(image: np.ndarray, text_info: Dict, debug=False) -> int:
    """
    Fallback method: scan the entire scale region for horizontal structures.
    """
    
    # Look in the bottom 20% of the image
    height, width = image.shape
    scale_region = image[int(height * 0.8):, :]
    
    # Find all horizontal structures
    structures = []
    
    for threshold in [250, 230, 200, 180]:
        _, binary = cv2.threshold(scale_region, threshold, 255, cv2.THRESH_BINARY)
        
        # Enhance horizontal structures
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Connect nearby segments
        connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))
        connected = cv2.morphologyEx(horizontal, cv2.MORPH_CLOSE, connect_kernel)
        
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= 100 and w / h >= 8:  # Reasonable horizontal structure
                structures.append(w)
    
    if structures:
        # Use the most common reasonable structure length
        reasonable = [s for s in structures if 150 <= s <= 600]
        if reasonable:
            from collections import Counter
            structure_counts = Counter(reasonable)
            best_structure = structure_counts.most_common(1)[0][0]
            
            if debug:
                print(f"   Fallback found: {best_structure} pixels")
            
            return best_structure
    
    return 0

def find_horizontal_lines_in_region(region: np.ndarray, debug=False) -> List[Dict]:
    """
    Find horizontal line segments in a region.
    """
    
    if region.size == 0:
        return []
    
    lines = []
    
    # Try multiple thresholds to catch different line intensities
    for threshold in [250, 240, 220, 200, 180]:
        _, binary = cv2.threshold(region, threshold, 255, cv2.THRESH_BINARY)
        
        # Use horizontal morphological opening to enhance horizontal structures
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter for horizontal line-like shapes
            if w >= 15 and h <= 8 and w / h >= 3:
                area = cv2.contourArea(contour)
                fill_ratio = area / (w * h) if w * h > 0 else 0
                
                # Score the line quality
                length_score = min(1.0, w / 100)  # Longer is better
                aspect_score = min(1.0, (w / h) / 10)  # More horizontal is better
                fill_score = fill_ratio  # More filled is better
                
                score = (length_score + aspect_score + fill_score) / 3
                
                lines.append({
                    'start_x': x,
                    'end_x': x + w,
                    'center_x': x + w // 2,
                    'center_y': y + h // 2,
                    'length': w,
                    'thickness': h,
                    'area': area,
                    'fill_ratio': fill_ratio,
                    'score': score,
                    'threshold': threshold
                })
    
    # Remove duplicates (same line detected at different thresholds)
    unique_lines = []
    for line in lines:
        is_duplicate = False
        for existing in unique_lines:
            # Check if lines overlap significantly
            x_overlap = max(0, min(line['end_x'], existing['end_x']) - max(line['start_x'], existing['start_x']))
            if x_overlap > 0.8 * min(line['length'], existing['length']):
                # Keep the better scoring line
                if line['score'] > existing['score']:
                    unique_lines.remove(existing)
                    unique_lines.append(line)
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_lines.append(line)
    
    # Sort by score
    unique_lines.sort(key=lambda x: x['score'], reverse=True)
    
    if debug and unique_lines:
        print(f"     Found {len(unique_lines)} horizontal lines:")
        for i, line in enumerate(unique_lines[:3]):  # Show top 3
            print(f"       {i+1}. Length: {line['length']}px, score: {line['score']:.3f}")
    
    return unique_lines

def score_line_pair(left_line: Dict, right_line: Dict, text_info: Dict) -> float:
    """
    Score how well a pair of lines matches as a split scale bar.
    """
    
    score = 0.0
    
    # Length similarity (both sides should be roughly similar)
    length_diff = abs(left_line['length'] - right_line['length'])
    max_length = max(left_line['length'], right_line['length'])
    length_similarity = 1.0 - (length_diff / max_length) if max_length > 0 else 0
    score += 0.3 * length_similarity
    
    # Y alignment (should be at same height)
    y_diff = abs(left_line['center_y'] - right_line['center_y'])
    y_alignment = max(0, 1.0 - y_diff / 10)  # Perfect if within 10 pixels
    score += 0.3 * y_alignment
    
    # Individual line quality
    line_quality = (left_line['score'] + right_line['score']) / 2
    score += 0.4 * line_quality
    
    return score

def parse_scale_text_flexible(text: str) -> Optional[Dict]:
    """
    Enhanced flexible parser that handles all OCR variations and decimal formats.
    """
    
    # Clean text but preserve important characters
    text_clean = text.strip()
    
    if not text_clean:
        return None
    
    # Try multiple number extraction patterns
    number_patterns = [
        r'(\d+\.\d+)',           # Decimal: 300.0
        r'(\d+,\d+)',            # European decimal: 300,0  
        r'(\d+\.?\d*)',          # Regular: 300 or 300.5
        r'(\d+)',                # Integer only: 300
    ]
    
    value = None
    for pattern in number_patterns:
        number_match = re.search(pattern, text_clean)
        if number_match:
            try:
                # Handle both . and , as decimal separators
                number_str = number_match.group(1).replace(',', '.')
                value = float(number_str)
                break
            except:
                continue
    
    if value is None:
        return None
    
    # Enhanced unit detection with more OCR error patterns
    after_number = text_clean[number_match.end():].strip()
    
    # Create both original and lowercase versions for comparison
    after_lower = after_number.lower()
    after_original = after_number
    
    # Debug the unit detection
    unit_found = None
    micrometers = None
    
    # Micrometers - check for various OCR interpretations
    micrometer_patterns = [
        # Standard patterns
        'μm', 'µm', 'um', 
        # Common OCR errors
        'jm', 'pm', 'qm', 'om', 'ūm', 'ûm', 'ũm',
        # With dots/periods
        'μm.', 'um.', 'jm.',
        # Spaced versions
        'μ m', 'u m', 'j m',
        # Character recognition errors
        'μπι', 'μm', 'μη', 'μμ',
        # Single characters that might be 'm' after a number
        'm', 'n', 'π', 'η'
    ]
    
    # Check for micrometer patterns
    for pattern in micrometer_patterns:
        if pattern in after_lower or pattern in after_original:
            unit_found = 'μm'
            micrometers = value
            break
    
    # If no micrometer pattern found, check other units
    if unit_found is None:
        if any(nm_pattern in after_lower for nm_pattern in ['nm', 'ηm', 'ηπι']):
            unit_found = 'nm'
            micrometers = value * 0.001
        elif any(mm_pattern in after_lower for mm_pattern in ['mm', 'ππι', 'mm.']):
            unit_found = 'mm'  
            micrometers = value * 1000
        elif any(cm_pattern in after_lower for cm_pattern in ['cm', 'cη']):
            unit_found = 'cm'
            micrometers = value * 10000
    
    # Smart default: if we have a reasonable SEM scale value but no clear unit
    if unit_found is None:
        if 50 <= value <= 2000:  # Typical SEM micrometer range
            unit_found = 'μm'
            micrometers = value
        elif 50000 <= value <= 2000000:  # Might be nanometers interpreted as number
            unit_found = 'nm'
            micrometers = value * 0.001
        elif 0.05 <= value <= 2:  # Might be millimeters
            unit_found = 'mm'
            micrometers = value * 1000
    
    # Final validation
    if unit_found and micrometers is not None:
        # Expanded range for different types of SEM images
        if 1 <= micrometers <= 10000:  # 1μm to 10mm range (very generous)
            return {
                'value': value,
                'unit': unit_found,
                'micrometers': micrometers,
                'original_text': text_clean,
                'after_number': after_number  # For debugging
            }
    
    return None

# Compatibility functions
def detect_scale_bar(image_input, debug=True, **kwargs) -> Dict:
    """Main detection function with split-bar awareness."""
    return detect_scale_bar_split_aware(image_input, debug=debug, **kwargs)

def detect_scale_factor_only(image_input, **kwargs) -> float:
    """Return just the scale factor."""
    result = detect_scale_bar_split_aware(image_input, **kwargs)
    return result['micrometers_per_pixel']

class ScaleBarDetector:
    """Compatibility class."""
    def __init__(self, **kwargs):
        self.debug = kwargs.get('debug', True)
    
    def detect_scale_bar(self, image: np.ndarray) -> Dict:
        return detect_scale_bar_split_aware(image, debug=self.debug)

def manual_scale_calibration(bar_length_pixels: int, bar_length_micrometers: float) -> float:
    """Manual calibration."""
    return bar_length_micrometers / bar_length_pixels

def debug_ocr_for_image(image_input, debug=True):
    """
    Debug OCR specifically for problematic images.
    """
    
    # Handle image loading
    if isinstance(image_input, str):
        image = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"❌ Could not load image: {image_input}")
            return
    else:
        image = image_input
    
    if not EASYOCR_AVAILABLE:
        print("❌ EasyOCR not available")
        return
    
    print(f"🔍 DEBUG OCR for image: {image.shape}")
    
    try:
        reader = easyocr.Reader(['en'], verbose=False)
        
        # Try different regions
        regions_to_test = [
            ("Full image", image),
            ("Bottom 30%", image[int(image.shape[0] * 0.7):, :]),
            ("Bottom 20%", image[int(image.shape[0] * 0.8):, :]),
            ("Bottom 15%", image[int(image.shape[0] * 0.85):, :]),
        ]
        
        for region_name, region in regions_to_test:
            print(f"\n📋 Testing {region_name}: {region.shape}")
            
            results = reader.readtext(region, detail=1)
            print(f"   Found {len(results)} text elements:")
            
            for i, (bbox, text, confidence) in enumerate(results):
                print(f"     {i+1}. '{text}' (confidence: {confidence:.3f})")
                
                # Try to parse each text
                scale_info = parse_scale_text_flexible(text.strip())
                if scale_info:
                    print(f"        ✅ PARSED: {scale_info['value']} {scale_info['unit']} = {scale_info['micrometers']} μm")
                    print(f"        Debug info: after_number='{scale_info.get('after_number', 'N/A')}'")
                else:
                    print(f"        ❌ Could not parse")
                    
                    # Show detailed parsing attempt
                    print(f"        Debug: text='{text.strip()}'")
                    number_match = re.search(r'(\d+\.?\d*)', text.strip())
                    if number_match:
                        after_number = text.strip()[number_match.end():].strip()
                        print(f"        Debug: number='{number_match.group(1)}', after='{after_number}'")
                    else:
                        print(f"        Debug: No number found in text")
        
        print(f"\n🎯 Full OCR analysis complete")
        
    except Exception as e:
        print(f"❌ OCR debug failed: {e}")

# Add this to the test function
def test_single_problematic_image(image_path):
    """Test a single problematic image with detailed debugging."""
    
    print(f"🧪 DETAILED DEBUG for: {image_path}")
    print("=" * 60)
    
    # First, run OCR debug
    debug_ocr_for_image(image_path)
    
    # Then run the full detection
    print(f"\n📏 Running full scale detection...")
    result = detect_scale_bar_split_aware(image_path, debug=True)
    
    if result['scale_detected']:
        print(f"🎉 SUCCESS: {result['micrometers_per_pixel']:.4f} μm/pixel")
    else:
        print(f"❌ FAILED: {result.get('error')}")
    
    return result
def test_split_bar_detection():
    """Test the split-bar aware detection on all images in the folder."""
    
    print("🧪 TESTING SPLIT-BAR AWARE DETECTION - ALL IMAGES")
    print("=" * 60)
    
    from pathlib import Path
    
    # Find test images in multiple possible locations
    possible_dirs = [
        Path("sample_images"),
        Path("../sample_images"),
        Path("./"),  # Current directory
        Path("images"),  # Alternative name
        Path("test_images")  # Alternative name
    ]
    
    sample_dir = None
    for dir_path in possible_dirs:
        if dir_path.exists():
            # Check if it contains image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
            has_images = any(
                list(dir_path.glob(f'*{ext}')) + list(dir_path.glob(f'*{ext.upper()}'))
                for ext in image_extensions
            )
            if has_images:
                sample_dir = dir_path
                break
    
    if sample_dir is None:
        print("❌ No directory with image files found")
        print("   Searched in:", [str(d) for d in possible_dirs])
        return False
    
    print(f"📁 Found images in: {sample_dir.absolute()}")
    
    # Get ALL image files
    image_files = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif']
    
    for ext in image_extensions:
        image_files.extend(sample_dir.glob(f'*{ext}'))
        image_files.extend(sample_dir.glob(f'*{ext.upper()}'))
    
    # Remove duplicates and sort
    image_files = sorted(set(image_files))
    
    if not image_files:
        print("❌ No image files found")
        print(f"   Looked for extensions: {image_extensions}")
        return False
    
    print(f"🔍 Found {len(image_files)} image files:")
    for i, img_file in enumerate(image_files, 1):
        print(f"   {i:2d}. {img_file.name}")
    
    print("\n" + "=" * 60)
    print("PROCESSING ALL IMAGES")
    print("=" * 60)
    
    successful = 0
    results_summary = []
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] 📸 Processing: {image_path.name}")
        print("-" * 50)
        
        try:
            result = detect_scale_bar_split_aware(str(image_path), debug=True)
            
            if result['scale_detected']:
                scale_factor = result['micrometers_per_pixel']
                total_span = result['bar_length_pixels']
                scale_value = result['scale_info']['value']
                scale_unit = result['scale_info']['unit']
                
                print(f"🎉 SUCCESS: {scale_factor:.4f} μm/pixel")
                print(f"   Total span: {total_span} pixels")
                print(f"   Scale value: {scale_value} {scale_unit}")
                
                results_summary.append({
                    'filename': image_path.name,
                    'success': True,
                    'scale_factor': scale_factor,
                    'total_span': total_span,
                    'scale_value': scale_value,
                    'scale_unit': scale_unit,
                    'method': result.get('detection_method', 'unknown')
                })
                
                successful += 1
            else:
                error_msg = result.get('error', 'Unknown error')
                print(f"❌ FAILED: {error_msg}")
                
                results_summary.append({
                    'filename': image_path.name,
                    'success': False,
                    'error': error_msg
                })
        
        except Exception as e:
            print(f"💥 ERROR: {str(e)}")
            results_summary.append({
                'filename': image_path.name,
                'success': False,
                'error': f'Exception: {str(e)}'
            })
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"📊 Overall Success Rate: {successful}/{len(image_files)} ({successful/len(image_files)*100:.1f}%)")
    
    if successful > 0:
        successful_results = [r for r in results_summary if r['success']]
        scale_factors = [r['scale_factor'] for r in successful_results]
        
        print(f"\n📏 Scale Factor Statistics:")
        print(f"   Range: {min(scale_factors):.4f} - {max(scale_factors):.4f} μm/pixel")
        print(f"   Mean: {sum(scale_factors)/len(scale_factors):.4f} μm/pixel")
        print(f"   Values: {[f'{sf:.4f}' for sf in sorted(scale_factors)]}")
        
        print(f"\n✅ Successful Images:")
        for result in successful_results:
            print(f"   {result['filename']}: {result['scale_factor']:.4f} μm/px ({result['scale_value']} {result['scale_unit']})")
    
    # Show failures
    failed_results = [r for r in results_summary if not r['success']]
    if failed_results:
        print(f"\n❌ Failed Images ({len(failed_results)}):")
        for result in failed_results:
            print(f"   {result['filename']}: {result['error']}")
    
    # Save results to file
    try:
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = sample_dir / f"scale_detection_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'total_images': len(image_files),
                'successful': successful,
                'success_rate': successful/len(image_files)*100,
                'results': results_summary
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        
    except Exception as e:
        print(f"\n⚠️ Could not save results file: {e}")
    
    print(f"\n🎯 BATCH TEST COMPLETE!")
    
    if successful >= len(image_files) * 0.8:  # 80% success rate
        print("🎉 Excellent! Split-bar detection is working very well!")
        return True
    elif successful >= len(image_files) * 0.6:  # 60% success rate
        print("👍 Good! Split-bar detection is working reasonably well!")
        return True
    else:
        print("😞 Needs improvement - success rate too low")
        return False

if __name__ == "__main__":
    test_split_bar_detection()