"""
Enhanced Scale Detection - Split Bar Method with SEM Parameter Filtering
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
        'method_used': 'split_bar_aware',
        'confidence': 0.0
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
        print(f"   Parse confidence: {text_info.get('parse_confidence', 'unknown')}")
    
    # Step 2: Find split scale bar segments
    if debug:
        print("📏 Step 2: Finding split scale bar segments...")
    
    total_span, bar_method = find_split_scale_bar_with_method(image, text_info, debug)
    
    if total_span <= 0:
        result['error'] = "Could not detect complete scale bar span"
        return result
    
    # Step 3: Calculate scale factor
    micrometers_per_pixel = text_info['micrometers'] / total_span
    
    # Calculate overall confidence
    text_confidence = text_info.get('confidence', 0.5)
    parse_confidence = 1.0 if text_info.get('parse_confidence') == 'high' else 0.7
    validation_score = text_info.get('validation_score', 0.5)
    
    # Method confidence
    method_confidence = {
        'paired_segments': 0.95,
        'reconstructed': 0.85,
        'single_side_proportion': 0.7,
        'fallback_scan': 0.6
    }.get(bar_method, 0.5)
    
    # Combined confidence score
    overall_confidence = (text_confidence * 0.3 + 
                         parse_confidence * 0.2 + 
                         validation_score * 0.2 + 
                         method_confidence * 0.3)
    
    result.update({
        'scale_detected': True,
        'micrometers_per_pixel': micrometers_per_pixel,
        'scale_factor': micrometers_per_pixel,
        'scale_info': text_info,
        'bar_length_pixels': total_span,
        'detection_method': bar_method,
        'confidence': overall_confidence
    })
    
    if debug:
        print(f"✅ SUCCESS: {micrometers_per_pixel:.4f} μm/pixel")
        print(f"   Total span: {total_span} pixels")
        print(f"   Detection method: {bar_method}")
        print(f"   Overall confidence: {overall_confidence:.2%}")
    
    return result

def find_scale_text_precise(image: np.ndarray, debug=False) -> Optional[Dict]:
    """
    Find scale text with precise bounding box information.
    Enhanced to filter out SEM parameter text and focus on actual scale bars.
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
        
        # Try multiple OCR parameter settings for better detection
        all_results = []
        
        # Standard detection
        results = reader.readtext(bottom_region, detail=1)
        all_results.extend(results)
        
        # Also try with lower text threshold for difficult text
        try:
            results_low_thresh = reader.readtext(bottom_region, detail=1, text_threshold=0.5)
            all_results.extend(results_low_thresh)
        except:
            pass
        
        # Also try with preprocessing
        try:
            # Enhance contrast for better OCR
            enhanced = cv2.convertScaleAbs(bottom_region, alpha=1.5, beta=20)
            results_enhanced = reader.readtext(enhanced, detail=1)
            all_results.extend(results_enhanced)
        except:
            pass
        
        # Remove duplicates based on text content
        seen_texts = set()
        unique_results = []
        for result in all_results:
            text = result[1].strip()
            if text not in seen_texts:
                seen_texts.add(text)
                unique_results.append(result)
        
        if debug:
            print(f"   EasyOCR found {len(unique_results)} unique text elements")
            # Show ALL detected text, even low confidence
            for (bbox, text, confidence) in unique_results:
                print(f"     '{text}' (confidence: {confidence:.3f})")
        
        # Collect all scale candidates
        scale_candidates = []
        
        # Process each text detection with a LOWER confidence threshold
        for (bbox, text, confidence) in unique_results:
            if confidence > 0.1:  # Lower threshold to catch more candidates
                if debug and confidence <= 0.3:
                    print(f"     [Low confidence] '{text}' (confidence: {confidence:.3f})")
                
                # FILTER OUT SEM PARAMETERS - this is the key enhancement!
                if is_sem_parameter_text(text):
                    if debug:
                        print(f"       ❌ Filtered out (SEM parameter)")
                    continue
                
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
                    
                    # Additional validation for scale text
                    validation_score = validate_scale_text_context(text, bbox_adjusted, image.shape)
                    
                    if debug:
                        print(f"       ✅ Parsed: {scale_info['value']} {scale_info['unit']} = {scale_info['micrometers']} μm")
                        print(f"       Text dimensions: {text_width}x{text_height} pixels")
                        print(f"       Validation score: {validation_score:.3f}")
                    
                    scale_candidates.append({
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
                        'original_text': text.strip(),
                        'validation_score': validation_score,
                        'parse_confidence': scale_info.get('confidence', 'high')
                    })
                elif debug:
                    print(f"       ❌ Could not parse")
        
        # If no candidates found, try looking specifically in the scale bar area
        if not scale_candidates and debug:
            print("   ⚠️ No scale text found in standard search, trying focused search...")
            # Try searching specifically in the bottom-right area where scale bars typically are
            scale_region = bottom_region[:, int(width * 0.6):]
            try:
                results_focused = reader.readtext(scale_region, detail=1, text_threshold=0.3)
                if results_focused:
                    print(f"   Found {len(results_focused)} texts in focused search:")
                    for (bbox, text, confidence) in results_focused:
                        print(f"     '{text}' (confidence: {confidence:.3f})")
            except:
                pass
        
        # Select the best scale candidate
        if scale_candidates:
            # Sort by validation score and confidence
            scale_candidates.sort(key=lambda x: (x['validation_score'], x['confidence']), reverse=True)
            
            if debug:
                print(f"   Selected best candidate from {len(scale_candidates)} options")
            
            return scale_candidates[0]
    
    except Exception as e:
        if debug:
            print(f"❌ EasyOCR failed: {e}")
    
    return None

def is_sem_parameter_text(text: str) -> bool:
    """
    Filter out SEM imaging parameter text that might contain numbers.
    Uses multiple strategies to avoid false positives.
    """
    
    text_lower = text.lower().strip()
    
    # Strategy 1: Direct keyword matching
    # Common SEM parameter indicators
    sem_keywords = [
        'kv', 'etd', 'wd', 'spot', 'pressure', 'temp', 'mag', 
        'det', 'hv', 'hfov', 'vfov', 'se', 'bse', 'acc', 'mode',
        'scan', 'dwell', 'working distance', 'accelerating',
        'electron', 'vacuum', 'chamber', 'stage', 'tilt',
        'brightness', 'contrast', 'stigmation', 'aperture',
        'emission', 'filament', 'gun', 'lens', 'objective',
        'condenser', 'beam', 'current', 'voltage'
    ]
    
    # Check for keywords
    for keyword in sem_keywords:
        if keyword in text_lower:
            # Special handling for some cases
            if keyword == 'x':
                # Only filter if it's magnification (e.g., "400x")
                if re.search(r'\d+x\s*$', text_lower) or re.search(r'\d+x\s+', text_lower):
                    return True
            elif keyword == 'mm':
                # Only filter if it's working distance (has 'wd' or multiple values)
                if 'wd' in text_lower or len(re.findall(r'\d+', text)) > 1:
                    return True
            else:
                return True
    
    # Strategy 2: Pattern-based detection
    # Check for parameter patterns
    param_patterns = [
        r'\d+\.?\d*\s*kv',          # Voltage
        r'\d+\.?\d*\s*mm\s+wd',     # Working distance with WD
        r'\d+\.?\d*\s*pa',          # Pressure
        r'\d+\.?\d*\s*torr',        # Pressure
        r'\d+x\s*mag',              # Magnification
        r'\d+\.?\d*\s*°',           # Angles
        r'\d+\.?\d*\s*deg',         # Degrees
    ]
    
    for pattern in param_patterns:
        if re.search(pattern, text_lower):
            return True
    
    # Strategy 3: Multi-value detection
    # Parameters often have multiple numbers with different units
    numbers = re.findall(r'\d+\.?\d*', text)
    if len(numbers) >= 3:  # 3 or more numbers likely means parameters
        return True
    
    # Strategy 4: Text length and complexity
    # Scale bars are usually simple and short
    if len(text.strip()) > 30:  # Too long for a scale bar
        return True
    
    # Count different types of characters
    has_letters = bool(re.search(r'[a-zA-Z]', text))
    has_numbers = bool(re.search(r'\d', text))
    has_special = bool(re.search(r'[/:;,|]', text))
    
    # If it has all three and multiple occurrences, likely parameters
    if has_letters and has_numbers and has_special:
        special_count = len(re.findall(r'[/:;,|]', text))
        if special_count >= 2:  # Multiple special chars indicate parameter string
            return True
    
    # Strategy 5: Known facility/manufacturer names
    facility_keywords = ['facility', 'bioimaging', 'wolfson', 'lab', 'laboratory', 
                        'center', 'centre', 'institute', 'university', 'zeiss',
                        'jeol', 'hitachi', 'fei', 'thermo', 'tescan']
    
    for keyword in facility_keywords:
        if keyword in text_lower:
            return True
    
    return False

def validate_scale_text_context(text: str, bbox: np.ndarray, image_shape: Tuple[int, int]) -> float:
    """
    Validate that this text is likely a scale bar based on context.
    Returns a score from 0.0 (unlikely) to 1.0 (very likely).
    """
    
    score = 0.0
    height, width = image_shape
    
    # Position scoring - scale bars are usually in bottom-right or bottom-center
    center_x = np.mean(bbox[:, 0])
    center_y = np.mean(bbox[:, 1])
    
    # Prefer right side of image
    x_ratio = center_x / width
    if x_ratio > 0.7:  # Right side
        score += 0.3
    elif x_ratio > 0.4:  # Center-right area
        score += 0.2
    elif x_ratio > 0.3:  # Center area
        score += 0.1
    
    # Prefer bottom area
    y_ratio = center_y / height
    if y_ratio > 0.85:  # Very bottom
        score += 0.3
    elif y_ratio > 0.7:  # Lower area
        score += 0.2
    
    # Text format scoring
    text_clean = text.strip().lower()
    
    # Prefer simple scale text formats
    # Complete pattern: number + unit
    if re.match(r'^\d+\.?\d*\s*[μµu]?m$', text_clean):  # "300.0μm" format
        score += 0.4
    elif re.match(r'^\d+\.?\d*\s*[μµu]m', text_clean):  # "300μm" with possible extra
        score += 0.3
    elif re.match(r'^\d+\.?\d*\s*[a-z]{1,3}$', text_clean):  # Simple number+unit
        score += 0.2
    
    # Prefer reasonable scale values
    numbers = re.findall(r'\d+\.?\d*', text_clean)
    if numbers:
        try:
            value = float(numbers[0])
            if 50 <= value <= 2000:  # Typical SEM scale range in μm
                score += 0.2
            elif 10 <= value <= 5000:  # Extended range
                score += 0.1
            elif 0.1 <= value <= 10:  # Might be mm
                score += 0.1
        except:
            pass
    
    # Penalize if text is too long (scale bars are usually concise)
    if len(text_clean) > 15:
        score -= 0.1
    
    # Penalize if there are multiple separate numbers
    if len(numbers) > 1:
        score -= 0.2
    
    return max(0.0, min(1.0, score))  # Clamp between 0 and 1

def find_split_scale_bar_with_method(image: np.ndarray, text_info: Dict, debug=False) -> Tuple[int, str]:
    """
    Find the complete scale bar that is split by the text overlay.
    Returns both the span length and the method used for detection.
    """
    
    # Call the original function
    total_span = find_split_scale_bar(image, text_info, debug)
    
    # Determine which method was used based on debug output
    # This is a simplified approach - in production, you'd track this internally
    method = 'unknown'
    if total_span > 0:
        # Based on the span value and text width, infer the method
        text_width = text_info['width']
        if total_span > text_width * 3:
            method = 'paired_segments'
        elif total_span > text_width * 2:
            method = 'reconstructed'
        elif total_span < 500:
            method = 'fallback_scan'
        else:
            method = 'single_side_proportion'
    
    return total_span, method

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
    Intelligent parser that uses fuzzy logic and character similarity to handle OCR errors.
    """
    
    # Clean text but preserve important characters
    text_clean = text.strip()
    
    if not text_clean:
        return None
    
    # Step 1: Extract ALL numbers from the text
    # More robust number extraction that handles decimals, spaces, and OCR errors
    number_patterns = [
        r'(\d+\.\d+)',           # Decimal: 300.0
        r'(\d+,\d+)',            # European decimal: 300,0
        r'(\d+\s+\.\s*\d+)',     # Spaced decimal: 300 . 0
        r'(\d+\.?\d*)',          # Regular: 300 or 300.5
        r'(\d+)',                # Integer only: 300
    ]
    
    value = None
    number_match = None
    
    # Try each pattern
    for pattern in number_patterns:
        match = re.search(pattern, text_clean)
        if match:
            try:
                # Clean up the number string
                number_str = match.group(1).replace(',', '.').replace(' ', '')
                value = float(number_str)
                number_match = match
                break
            except:
                continue
    
    if value is None:
        return None
    
    # Step 2: Smart unit detection using character analysis
    # Get everything after the number
    after_number = text_clean[number_match.end():].strip()
    
    # Also check what's right before the number (sometimes unit comes first)
    before_number = text_clean[:number_match.start()].strip()
    
    # Analyze both parts
    unit_found = None
    micrometers = None
    
    # Function to calculate similarity between strings
    def string_similarity(s1: str, s2: str) -> float:
        """Calculate similarity between two strings (0-1)."""
        s1, s2 = s1.lower(), s2.lower()
        if not s1 or not s2:
            return 0.0
        
        # Simple character overlap ratio
        common = sum(1 for c in s1 if c in s2)
        return common / max(len(s1), len(s2))
    
    # Check for unit patterns with fuzzy matching
    def detect_unit(text_part: str) -> Optional[Tuple[str, float]]:
        """Detect unit type and conversion factor using fuzzy logic."""
        text_lower = text_part.lower()
        
        # Define unit patterns with their conversion factors
        unit_patterns = {
            'μm': {'factor': 1.0, 'variants': ['μm', 'µm', 'um', 'micrometer', 'micron']},
            'nm': {'factor': 0.001, 'variants': ['nm', 'nanometer']},
            'mm': {'factor': 1000.0, 'variants': ['mm', 'millimeter']},
            'cm': {'factor': 10000.0, 'variants': ['cm', 'centimeter']},
        }
        
        # Smart detection logic
        best_match = None
        best_score = 0.0
        
        for unit, info in unit_patterns.items():
            for variant in info['variants']:
                # Direct substring match
                if variant in text_lower:
                    return (unit, info['factor'])
                
                # Fuzzy matching for OCR errors
                # Look for patterns where the text ends with 'm' and has 2-3 chars
                if text_lower.endswith('m') and len(text_lower) <= 4:
                    score = string_similarity(text_lower, variant)
                    if score > best_score and score > 0.5:  # At least 50% similar
                        best_score = score
                        best_match = (unit, info['factor'])
        
        # Special handling for micrometer OCR errors
        # If it's a short string ending in 'm', likely micrometers
        if not best_match and text_lower.endswith('m') and len(text_lower) <= 4:
            # Check if it contains any character that could be a mangled μ
            suspicious_chars = ['u', 'j', 'p', 'q', 'o', '0', 'µ', 'μ']
            if any(c in text_lower for c in suspicious_chars):
                return ('μm', 1.0)
        
        return best_match
    
    # Try to detect unit after the number
    result = detect_unit(after_number)
    if result:
        unit_found, factor = result
        micrometers = value * factor
    else:
        # Try before the number
        result = detect_unit(before_number)
        if result:
            unit_found, factor = result
            micrometers = value * factor
    
    # Step 3: Context-based inference if no unit found
    if unit_found is None:
        # Use the numeric value to infer the most likely unit
        if 0.01 <= value <= 10:  # Likely mm
            unit_found = 'mm'
            micrometers = value * 1000
        elif 10 < value <= 5000:  # Likely μm
            unit_found = 'μm'
            micrometers = value
        elif 5000 < value <= 999999:  # Likely nm
            unit_found = 'nm'
            micrometers = value * 0.001
        else:
            # Default to μm for any other value
            unit_found = 'μm'
            micrometers = value
    
    # Step 4: Validate the result
    # Reasonable range for microscopy: 0.1 μm to 10 mm
    if micrometers is not None and 0.1 <= micrometers <= 10000:
        return {
            'value': value,
            'unit': unit_found,
            'micrometers': micrometers,
            'original_text': text_clean,
            'after_number': after_number,
            'confidence': 'high' if unit_found != 'μm' or after_number else 'inferred'
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
                
                # Check if it's a SEM parameter
                if is_sem_parameter_text(text.strip()):
                    print(f"        ⚠️ SEM parameter detected - would be filtered")
                
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

# Test function
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