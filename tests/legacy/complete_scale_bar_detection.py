#!/usr/bin/env python3
"""
Complete Scale Bar Detection - Find the Full Scale Bar Span
Specifically designed to find both left and right segments of SEM scale bars
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "modules"))

def find_complete_scale_bar_span(scale_region, text_center_x, text_center_y, text_bbox, debug=True):
    """
    Find the complete scale bar span by looking for all white segments at the same Y level as the text.
    Enhanced version with better debugging and more aggressive detection.
    """
    
    if debug:
        print(f"🔍 FINDING COMPLETE SCALE BAR SPAN")
        print(f"  Text center: ({text_center_x}, {text_center_y})")
    
    height, width = scale_region.shape
    
    # Search vertically around the text Y level (much tighter than before)
    y_search_radius = 15  # Very tight vertical search
    search_y_min = max(0, text_center_y - y_search_radius)
    search_y_max = min(height, text_center_y + y_search_radius)
    
    # But search across the full width horizontally
    search_region = scale_region[search_y_min:search_y_max, :]
    
    if debug:
        print(f"  Search region: full width × {search_y_max - search_y_min} pixels")
        print(f"  Y range: {search_y_min} to {search_y_max}")
    
    # Save search region for debugging
    if debug:
        # Show a histogram of the search region intensities
        intensities = search_region.flatten()
        unique, counts = np.unique(intensities, return_counts=True)
        print(f"  Intensity distribution in search region:")
        print(f"    Min: {intensities.min()}, Max: {intensities.max()}, Mean: {intensities.mean():.1f}")
        print(f"    Most common values: {list(zip(unique[-5:], counts[-5:]))}")
    
    # Multiple approaches to find white lines
    white_line_masks = get_white_line_masks(search_region, debug)
    
    all_segments = []
    
    for mask_name, white_mask in white_line_masks:
        segments = extract_horizontal_segments(white_mask, mask_name, search_y_min, debug)
        all_segments.extend(segments)
    
    if debug:
        print(f"  Total segments found: {len(all_segments)}")
        if len(all_segments) == 0:
            print(f"  ⚠️ NO SEGMENTS FOUND - trying fallback detection...")
            # Try fallback detection with very aggressive parameters
            fallback_segments = try_fallback_detection(search_region, search_y_min, debug)
            all_segments.extend(fallback_segments)
    
    if len(all_segments) == 0:
        if debug:
            print(f"  ❌ Still no segments found even with fallback")
            # Save the search region for manual inspection
            try:
                cv2.imwrite(str(project_root / 'analysis_results' / 'debug_search_region.png'), search_region)
                print(f"  💾 Saved search region to debug_search_region.png for inspection")
            except:
                pass
        return None, []
    
    # Group segments by Y level
    y_groups = group_segments_by_y_level(all_segments, text_center_y, debug)
    
    # Find the group that best spans around the text
    best_span = find_best_complete_span(y_groups, text_center_x, text_center_y, debug)
    
    return best_span, all_segments

def try_fallback_detection(search_region, y_offset, debug=True):
    """
    Fallback detection method for very faint or difficult scale bars.
    """
    
    if debug:
        print(f"  🆘 TRYING FALLBACK DETECTION")
    
    fallback_segments = []
    
    # Method 1: Look for any brightness variation
    try:
        # Calculate row-wise mean and look for bright spots
        row_means = np.mean(search_region, axis=1)
        overall_mean = np.mean(row_means)
        
        for y, row_mean in enumerate(row_means):
            if row_mean > overall_mean * 1.1:  # 10% brighter than average
                # Look for continuous bright regions in this row
                row = search_region[y, :]
                bright_threshold = np.mean(row) * 1.2
                bright_mask = row > bright_threshold
                
                # Find continuous regions
                diff = np.diff(np.concatenate(([False], bright_mask, [False])).astype(int))
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]
                
                for start, end in zip(starts, ends):
                    width = end - start
                    if width > 15:  # Minimum width
                        fallback_segments.append({
                            'x': start,
                            'y': y + y_offset,
                            'width': width,
                            'height': 1,
                            'center_x': start + width // 2,
                            'center_y': y + y_offset,
                            'end_x': end,
                            'aspect_ratio': width,
                            'area': width,
                            'approach': 'fallback_brightness'
                        })
        
        if debug and fallback_segments:
            print(f"    Fallback method found {len(fallback_segments)} potential segments")
    
    except Exception as e:
        if debug:
            print(f"    Fallback detection failed: {e}")
    
    return fallback_segments

def get_white_line_masks(search_region, debug=True):
    """
    Generate multiple white line detection masks with more aggressive strategies.
    """
    
    white_masks = []
    
    if debug:
        mean_intensity = np.mean(search_region)
        max_intensity = np.max(search_region)
        print(f"  Search region stats: mean={mean_intensity:.1f}, max={max_intensity:.1f}")
    
    # Strategy 1: High threshold for very bright whites
    _, white_high = cv2.threshold(search_region, 240, 255, cv2.THRESH_BINARY)
    white_masks.append(("high_240", white_high))
    
    # Strategy 2: Medium threshold
    _, white_med = cv2.threshold(search_region, 200, 255, cv2.THRESH_BINARY)
    white_masks.append(("med_200", white_med))
    
    # Strategy 3: Lower threshold for faint lines
    _, white_low = cv2.threshold(search_region, 150, 255, cv2.THRESH_BINARY)
    white_masks.append(("low_150", white_low))
    
    # Strategy 4: Adaptive threshold
    try:
        white_adaptive = cv2.adaptiveThreshold(search_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 21, -10)
        white_masks.append(("adaptive", white_adaptive))
    except:
        pass
    
    # Strategy 5: Otsu threshold
    _, white_otsu = cv2.threshold(search_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_masks.append(("otsu", white_otsu))
    
    # Strategy 6: Percentile-based thresholds
    percentiles = [90, 95, 99]
    for p in percentiles:
        thresh_val = np.percentile(search_region, p)
        if thresh_val > 50:  # Only if reasonable
            _, white_p = cv2.threshold(search_region, int(thresh_val * 0.9), 255, cv2.THRESH_BINARY)
            white_masks.append((f"p{p}", white_p))
    
    # Strategy 7: Edge-based detection (find bright edges)
    try:
        # Enhance contrast first
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(search_region)
        
        # Find edges
        edges = cv2.Canny(enhanced, 30, 100)
        
        # Dilate edges to create lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        white_masks.append(("edges", edges_dilated))
    except:
        pass
    
    # Strategy 8: Very aggressive low threshold
    _, white_very_low = cv2.threshold(search_region, 100, 255, cv2.THRESH_BINARY)
    white_masks.append(("very_low_100", white_very_low))
    
    if debug:
        print(f"  Generated {len(white_masks)} white detection masks")
        for name, mask in white_masks:
            white_pixels = np.sum(mask > 0)
            print(f"    {name}: {white_pixels} white pixels")
    
    return white_masks

def extract_horizontal_segments(white_mask, mask_name, y_offset, debug=True):
    """
    Extract horizontal line segments from a white mask with more aggressive detection.
    """
    
    # Multiple morphological approaches
    segments_all = []
    
    # Approach 1: Horizontal closing only
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    connected1 = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel_h)
    segments1 = find_segments_in_mask(connected1, f"{mask_name}_close", y_offset)
    segments_all.extend(segments1)
    
    # Approach 2: Opening then closing
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    opened = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel_small)
    connected2 = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_h)
    segments2 = find_segments_in_mask(connected2, f"{mask_name}_open_close", y_offset)
    segments_all.extend(segments2)
    
    # Approach 3: Direct contour finding (no morphology)
    segments3 = find_segments_in_mask(white_mask, f"{mask_name}_direct", y_offset)
    segments_all.extend(segments3)
    
    # Remove duplicates (segments that are very similar)
    unique_segments = remove_duplicate_segments(segments_all)
    
    if debug and unique_segments:
        print(f"    {mask_name}: {len(unique_segments)} unique segments")
        for seg in unique_segments[:3]:  # Show first 3
            print(f"      {seg['width']}×{seg['height']} at ({seg['x']}, {seg['y']}) - {seg['approach']}")
    
    return unique_segments

def find_segments_in_mask(mask, approach_name, y_offset):
    """Find line segments in a binary mask."""
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # More lenient criteria for segments
        aspect_ratio = w / h if h > 0 else 0
        
        # Accept anything that's more horizontal than vertical and has some minimum size
        if aspect_ratio > 1.0 and w > 8 and h <= 12:  # Very relaxed criteria
            full_y = y + y_offset
            
            segments.append({
                'x': x,
                'y': full_y,
                'width': w,
                'height': h,
                'center_x': x + w // 2,
                'center_y': full_y + h // 2,
                'end_x': x + w,
                'aspect_ratio': aspect_ratio,
                'area': cv2.contourArea(contour),
                'approach': approach_name
            })
    
    return segments

def remove_duplicate_segments(segments):
    """Remove segments that are too similar."""
    
    if len(segments) <= 1:
        return segments
    
    unique = []
    
    for seg in segments:
        is_duplicate = False
        for existing in unique:
            # Check if segments overlap significantly
            x_overlap = max(0, min(seg['end_x'], existing['end_x']) - max(seg['x'], existing['x']))
            y_overlap = max(0, min(seg['y'] + seg['height'], existing['y'] + existing['height']) - 
                          max(seg['y'], existing['y']))
            
            if x_overlap > 0.7 * min(seg['width'], existing['width']) and y_overlap > 0:
                # Keep the one with better aspect ratio
                if seg['aspect_ratio'] > existing['aspect_ratio']:
                    unique.remove(existing)
                    unique.append(seg)
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique.append(seg)
    
    return unique

def group_segments_by_y_level(all_segments, text_center_y, debug=True, y_tolerance=10):
    """
    Group segments that are at the same Y level (same horizontal line).
    """
    
    if not all_segments:
        return []
    
    # Filter segments to only those close to text Y level
    relevant_segments = []
    for seg in all_segments:
        y_distance = abs(seg['center_y'] - text_center_y)
        if y_distance <= 20:  # Must be close to text level
            relevant_segments.append(seg)
    
    if debug:
        print(f"  Segments near text Y level: {len(relevant_segments)}")
    
    # Sort by Y position
    relevant_segments.sort(key=lambda x: x['center_y'])
    
    groups = []
    i = 0
    
    while i < len(relevant_segments):
        current_y = relevant_segments[i]['center_y']
        group = []
        
        # Collect segments at similar Y level
        for j in range(i, len(relevant_segments)):
            if abs(relevant_segments[j]['center_y'] - current_y) <= y_tolerance:
                group.append(relevant_segments[j])
            else:
                break
        
        if len(group) >= 1:
            groups.append(group)
        
        i += len(group)
    
    if debug:
        print(f"  Y-level groups: {len(groups)}")
        for i, group in enumerate(groups):
            y_avg = sum(seg['center_y'] for seg in group) / len(group)
            print(f"    Group {i+1}: {len(group)} segments at Y≈{y_avg:.1f}")
    
    return groups

def find_best_complete_span(y_groups, text_center_x, text_center_y, debug=True):
    """
    Find the group that gives the best complete span around the text.
    """
    
    if not y_groups:
        return None
    
    span_candidates = []
    
    for group_idx, group in enumerate(y_groups):
        # Sort segments by X position
        group.sort(key=lambda x: x['x'])
        
        # Find the complete span from leftmost to rightmost
        leftmost_x = min(seg['x'] for seg in group)
        rightmost_x = max(seg['end_x'] for seg in group)
        total_span = rightmost_x - leftmost_x
        
        # Calculate text position relative to span
        text_relative_pos = (text_center_x - leftmost_x) / total_span if total_span > 0 else 0.5
        
        # Analyze the gap structure
        gaps = analyze_segment_gaps(group, text_center_x)
        
        # Score this span
        span_score = score_complete_span(group, total_span, text_relative_pos, gaps)
        
        candidate = {
            'group_index': group_idx,
            'segments': group,
            'leftmost_x': leftmost_x,
            'rightmost_x': rightmost_x,
            'total_span': total_span,
            'segment_count': len(group),
            'text_relative_pos': text_relative_pos,
            'gaps': gaps,
            'score': span_score,
            'average_y': sum(seg['center_y'] for seg in group) / len(group)
        }
        
        span_candidates.append(candidate)
        
        if debug:
            print(f"    Span candidate {group_idx + 1}:")
            print(f"      Total span: {total_span:.1f} pixels")
            print(f"      Segments: {len(group)}")
            print(f"      Text position: {text_relative_pos:.3f} (0.5 = centered)")
            print(f"      Score: {span_score:.3f}")
    
    # Sort by score and return best
    span_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    if debug and span_candidates:
        best = span_candidates[0]
        print(f"  🏆 Best span: {best['total_span']:.1f} pixels (score: {best['score']:.3f})")
    
    return span_candidates[0] if span_candidates else None

def analyze_segment_gaps(segments, text_center_x):
    """
    Analyze the gaps between segments to understand the scale bar structure.
    """
    
    if len(segments) < 2:
        return {'has_text_gap': False, 'text_gap_size': 0, 'total_gaps': 0}
    
    segments.sort(key=lambda x: x['x'])
    
    gaps = []
    text_gap_found = False
    text_gap_size = 0
    
    for i in range(len(segments) - 1):
        current_end = segments[i]['end_x']
        next_start = segments[i + 1]['x']
        gap_size = next_start - current_end
        
        if gap_size > 0:
            gap_center = (current_end + next_start) / 2
            gap_distance_to_text = abs(gap_center - text_center_x)
            
            gaps.append({
                'start': current_end,
                'end': next_start,
                'size': gap_size,
                'center': gap_center,
                'distance_to_text': gap_distance_to_text
            })
            
            # Check if this gap likely contains the text
            if gap_distance_to_text < gap_size / 2 and gap_size > 20:
                text_gap_found = True
                text_gap_size = gap_size
    
    return {
        'gaps': gaps,
        'has_text_gap': text_gap_found,
        'text_gap_size': text_gap_size,
        'total_gaps': len(gaps),
        'total_gap_size': sum(g['size'] for g in gaps)
    }

def score_complete_span(segments, total_span, text_relative_pos, gaps):
    """
    Score a complete span based on multiple criteria.
    """
    
    score = 0.0
    
    # 1. Span length score (prefer longer spans, but not unreasonably long)
    if 100 <= total_span <= 600:
        span_score = 1.0
    elif 50 <= total_span < 100 or 600 < total_span <= 800:
        span_score = 0.7
    elif total_span < 50:
        span_score = 0.3
    else:
        span_score = 0.5
    
    score += span_score * 0.3
    
    # 2. Text centrality score
    centrality = 1.0 - 2 * abs(text_relative_pos - 0.5)
    score += max(0, centrality) * 0.25
    
    # 3. Segment structure score
    if len(segments) == 2 and gaps.get('has_text_gap', False):
        # Perfect: two segments with text gap
        structure_score = 1.0
    elif len(segments) == 1:
        # Single long segment (less ideal but possible)
        structure_score = 0.6
    elif 2 <= len(segments) <= 4:
        # Multiple segments (could be broken scale bar)
        structure_score = 0.8
    else:
        # Too many segments (probably noise)
        structure_score = 0.3
    
    score += structure_score * 0.25
    
    # 4. Gap analysis score
    if gaps.get('has_text_gap', False):
        gap_score = 1.0  # Text gap found
    elif gaps.get('total_gaps', 0) == 0:
        gap_score = 0.7  # No gaps (single segment)
    else:
        gap_score = 0.5  # Some gaps but no clear text gap
    
    score += gap_score * 0.2
    
    return score

def test_complete_span_detection():
    """Test the complete span detection on your image."""
    
    print("="*60)
    print("COMPLETE SCALE BAR SPAN DETECTION")
    print("="*60)
    
    from image_preprocessing import load_image
    from scale_detection import ScaleBarDetector
    
    # Load image
    img_path = project_root / "sample_images" / "hollow_fiber_sample3.jpg"
    image = load_image(str(img_path))
    
    # Extract scale region
    detector = ScaleBarDetector()
    scale_region, y_offset = detector.extract_scale_region(image)
    
    print(f"Scale region shape: {scale_region.shape}")
    
    # Use the known text position (400.0μm at approximately the center)
    # We'll find it first, then use it to find the complete span
    text_candidates = find_scale_text_simple(scale_region)
    
    if not text_candidates:
        print("❌ No scale text found")
        return None, None  # FIX: Return tuple instead of None
    
    best_text = text_candidates[0]  # Use the best text candidate
    print(f"✅ Using text: '{best_text['text']}' = {best_text['micrometers']} μm")
    print(f"  Text center: ({best_text['center_x']}, {best_text['center_y']})")
    
    # Find complete scale bar span
    best_span, all_segments = find_complete_scale_bar_span(
        scale_region, 
        best_text['center_x'], 
        best_text['center_y'], 
        best_text['bbox']
    )
    
    if best_span:
        span_pixels = best_span['total_span']
        calculated_scale = best_text['micrometers'] / span_pixels
        manual_scale = 0.444
        error = abs(calculated_scale - manual_scale) / manual_scale * 100
        
        print(f"\n🎯 COMPLETE SPAN RESULTS:")
        print(f"  Text: {best_text['value']} {best_text['unit']}")
        print(f"  Complete span: {span_pixels:.1f} pixels")
        print(f"  Calculated scale: {calculated_scale:.4f} μm/pixel")
        print(f"  Manual scale: {manual_scale:.4f} μm/pixel")
        print(f"  Error: {error:.1f}%")
        print(f"  Segments found: {best_span['segment_count']}")
        print(f"  Text position: {best_span['text_relative_pos']:.3f} (0.5 = centered)")
        
        if error < 10:
            print(f"  🎯 EXCELLENT MATCH!")
        elif error < 20:
            print(f"  ✅ GOOD MATCH!")
        else:
            print(f"  ⚠️ Still needs work")
        
        # Visualize
        visualize_complete_span_detection(scale_region, best_text, best_span, all_segments)
        
        return best_span, calculated_scale
    else:
        print("❌ No complete span found")
        return None, None  # FIX: Return tuple instead of None

def find_scale_text_simple(scale_region):
    """Simplified text finding just to get the scale value with SEM filtering."""
    
    scale_candidates = []
    
    # Try EasyOCR
    try:
        import easyocr
        reader = easyocr.Reader(['en'])
        results = reader.readtext(scale_region, detail=1)
        
        print(f"  EasyOCR found {len(results)} text elements:")
        
        for (bbox, text, confidence) in results:
            print(f"    '{text}' (confidence: {confidence:.3f})")
            scale_info = parse_scale_text_simple(text.strip())
            
            if scale_info and confidence > 0.3:
                bbox_array = np.array(bbox)
                center_x = int(np.mean(bbox_array[:, 0]))
                center_y = int(np.mean(bbox_array[:, 1]))
                
                # Add SEM scale preference scoring
                micrometers = scale_info['micrometers']
                
                # Preference scoring for typical SEM scales
                if 50 <= micrometers <= 1000:  # 50μm to 1mm (most common)
                    preference_score = 1.0
                elif 10 <= micrometers < 50 or 1000 < micrometers <= 3000:  # 10μm-50μm or 1mm-3mm
                    preference_score = 0.8
                elif 1 <= micrometers < 10:  # 1μm-10μm (less common but possible)
                    preference_score = 0.6
                else:  # Very small or large (unusual)
                    preference_score = 0.3
                
                # Combined score: OCR confidence + SEM preference
                combined_score = confidence * 0.7 + preference_score * 0.3
                
                scale_candidates.append({
                    'text': text.strip(),
                    'value': scale_info['value'],
                    'unit': scale_info['unit'],
                    'micrometers': micrometers,
                    'center_x': center_x,
                    'center_y': center_y,
                    'bbox': bbox_array,
                    'confidence': confidence,
                    'preference_score': preference_score,
                    'combined_score': combined_score
                })
                
                print(f"      ✅ Valid SEM scale: {scale_info['value']} {scale_info['unit']} = {micrometers} μm")
                print(f"         Preference score: {preference_score:.2f}, Combined score: {combined_score:.2f}")
            else:
                if scale_info is None:
                    print(f"      ❌ Not a valid scale format")
                else:
                    print(f"      ❌ Low confidence ({confidence:.3f})")
    except Exception as e:
        print(f"  EasyOCR failed: {e}")
    
    # Sort by combined score (OCR confidence + SEM preference)
    scale_candidates.sort(key=lambda x: x['combined_score'], reverse=True)
    
    print(f"  Final candidates after SEM filtering: {len(scale_candidates)}")
    for i, cand in enumerate(scale_candidates):
        print(f"    {i+1}. {cand['value']} {cand['unit']} (score: {cand['combined_score']:.3f})")
    
    return scale_candidates

def parse_scale_text_simple(text):
    """Simple scale text parsing with SEM scale filtering and OCR error handling."""
    
    # Comprehensive patterns including common OCR errors for micrometer symbol
    patterns = [
        # Standard patterns
        (r'(\d+\.?\d*)\s*μm', 'μm', 1.0),
        (r'(\d+\.?\d*)\s*µm', 'μm', 1.0),
        (r'(\d+\.?\d*)\s*um', 'μm', 1.0),
        
        # Common OCR errors for μm
        (r'(\d+\.?\d*)\s*jm', 'μm', 1.0),       # jm instead of μm (very common!)
        (r'(\d+\.?\d*)\s*[|l1]m', 'μm', 1.0),   # |m, lm, 1m instead of μm
        (r'(\d+\.?\d*)\s*[Oo]m', 'μm', 1.0),    # Om instead of μm
        (r'(\d+\.?\d*)\s*[Oo]jm', 'μm', 1.0),   # Ojm instead of μm
        (r'(\d+\.?\d*)\s*[Oo]pm', 'μm', 1.0),   # Opm instead of μm (NEW!)
        (r'(\d+\.?\d*)\s*jim', 'μm', 1.0),      # jim instead of μm
        (r'(\d+\.?\d*)\s*μrn', 'μm', 1.0),      # μrn instead of μm
        (r'(\d+\.?\d*)\s*urn', 'μm', 1.0),      # urn instead of μm
        (r'(\d+\.?\d*)\s*jun', 'μm', 1.0),      # jun instead of μm
        (r'(\d+\.?\d*)\s*μn', 'μm', 1.0),       # μn instead of μm
        (r'(\d+\.?\d*)\s*pm', 'μm', 1.0),       # pm instead of μm (OCR confusion)
        (r'(\d+\.?\d*)\s*0pm', 'μm', 1.0),      # 0pm instead of μm
        (r'(\d+\.?\d*)\s*Qpm', 'μm', 1.0),      # Qpm instead of μm
        (r'(\d+\.?\d*)\s*opm', 'μm', 1.0),      # opm instead of μm (lowercase)
        
        # Other units
        (r'(\d+\.?\d*)\s*nm', 'nm', 0.001),
        (r'(\d+\.?\d*)\s*mm', 'mm', 1000.0),
        (r'(\d+\.?\d*)\s*cm', 'cm', 10000.0),
        
        # Handle decimal separator variations
        (r'(\d+)[,.](\d+)\s*μm', 'μm', 1.0),
        (r'(\d+)[,.](\d+)\s*jm', 'μm', 1.0),    # OCR error version
        (r'(\d+)[,.](\d+)\s*[|l1]m', 'μm', 1.0),
        (r'(\d+)[,.](\d+)\s*[Oo]pm', 'μm', 1.0), # NEW: Decimal Opm
        (r'(\d+)[,.](\d+)\s*mm', 'mm', 1000.0),
    ]
    
    text_clean = text.lower().replace(' ', '').replace('\n', '')
    
    for pattern, unit, conversion in patterns:
        match = re.search(pattern, text_clean)
        if match:
            try:
                if isinstance(match.groups()[0], str) and len(match.groups()) > 1:
                    # Handle decimal patterns like (400, 0) -> 400.0
                    value = float(f"{match.group(1)}.{match.group(2)}")
                else:
                    value = float(match.group(1))
                
                micrometers = value * conversion
                
                # FILTER: Only accept realistic SEM scale bar values
                # SEM scale bars are typically 50nm to 3mm (0.05 to 3000 μm)
                if 0.05 <= micrometers <= 3000:
                    # Show which pattern matched for debugging
                    pattern_desc = pattern.replace('\\d+\\.?\\d*', 'NUMBER').replace('\\s*', '').replace('[', '').replace(']', '').replace('?', '')
                    print(f"    ✅ Matched pattern '{pattern_desc}' → {value} {unit} = {micrometers} μm")
                    return {
                        'value': value,
                        'unit': unit,
                        'micrometers': micrometers,
                        'pattern_matched': pattern_desc,
                        'original_text': text
                    }
                else:
                    print(f"    ❌ Rejected {value} {unit} = {micrometers} μm (outside SEM range 0.05-3000 μm)")
                    
            except (ValueError, IndexError) as e:
                print(f"    ❌ Parse error for pattern {pattern}: {e}")
                continue
    
    print(f"    ❌ No valid scale pattern found in '{text}'")
    return None

def visualize_complete_span_detection(scale_region, best_text, best_span, all_segments):
    """Visualize the complete span detection."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Original with text location
    axes[0, 0].imshow(scale_region, cmap='gray')
    axes[0, 0].plot(best_text['center_x'], best_text['center_y'], 'ro', markersize=8)
    axes[0, 0].text(best_text['center_x'], best_text['center_y'] - 10, 
                   f"{best_text['value']}{best_text['unit']}", 
                   ha='center', color='red', fontsize=10, weight='bold')
    axes[0, 0].set_title('Text Location')
    axes[0, 0].axis('off')
    
    # All segments found
    overlay1 = cv2.cvtColor(scale_region, cv2.COLOR_GRAY2RGB)
    
    colors = [(255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 128, 0), (128, 255, 0)]
    
    for i, seg in enumerate(all_segments):
        color = colors[i % len(colors)]
        cv2.rectangle(overlay1, (seg['x'], seg['y']), 
                     (seg['end_x'], seg['y'] + seg['height']), color, 2)
    
    axes[0, 1].imshow(overlay1)
    axes[0, 1].set_title(f'All Segments ({len(all_segments)} found)')
    axes[0, 1].axis('off')
    
    # Best complete span
    overlay2 = cv2.cvtColor(scale_region, cv2.COLOR_GRAY2RGB)
    
    if best_span:
        # Draw individual segments in green
        for seg in best_span['segments']:
            cv2.rectangle(overlay2, (seg['x'], seg['y']), 
                         (seg['end_x'], seg['y'] + seg['height']), (0, 255, 0), 3)
        
        # Draw complete span in red
        y_line = int(best_span['average_y'])
        cv2.line(overlay2, (int(best_span['leftmost_x']), y_line), 
                (int(best_span['rightmost_x']), y_line), (255, 0, 0), 3)
        
        # Mark text center
        cv2.circle(overlay2, (best_text['center_x'], best_text['center_y']), 5, (255, 0, 0), -1)
        
        # Add span measurement
        mid_x = int((best_span['leftmost_x'] + best_span['rightmost_x']) / 2)
        cv2.putText(overlay2, f"{best_span['total_span']:.0f}px", 
                   (mid_x - 30, y_line - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    axes[1, 0].imshow(overlay2)
    axes[1, 0].set_title('Complete Scale Bar Span')
    axes[1, 0].axis('off')
    
    # Results summary
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Detection Results')
    
    if best_span:
        manual_scale = 0.444
        calculated_scale = best_text['micrometers'] / best_span['total_span']
        error = abs(calculated_scale - manual_scale) / manual_scale * 100
        
        summary = f"""COMPLETE SPAN DETECTION:

Text Detection:
  Value: {best_text['value']} {best_text['unit']}
  Micrometers: {best_text['micrometers']}
  Center: ({best_text['center_x']}, {best_text['center_y']})

Scale Bar Span:
  Total span: {best_span['total_span']:.0f} pixels
  Segments: {best_span['segment_count']}
  Left edge: {best_span['leftmost_x']:.0f}
  Right edge: {best_span['rightmost_x']:.0f}
  Text position: {best_span['text_relative_pos']:.3f}
  Score: {best_span['score']:.3f}

Scale Calculation:
  Calculated: {calculated_scale:.4f} μm/px
  Manual: {manual_scale:.4f} μm/px
  Error: {error:.1f}%

Previous Error: 275.4% (240px span)
New Error: {error:.1f}% ({best_span['total_span']:.0f}px span)

Improvement: {275.4 - error:.1f} percentage points!"""
    else:
        summary = "No complete span detected"
    
    axes[1, 1].text(0.05, 0.95, summary, transform=axes[1, 1].transAxes,
                    verticalalignment='top', fontsize=9, fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(project_root / 'analysis_results' / 'complete_span_detection.png', 
                dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    result, scale = test_complete_span_detection()
    
    print(f"\n" + "="*60)
    print("COMPLETE SPAN DETECTION SUMMARY")
    print("="*60)
    print("This approach finds the COMPLETE scale bar span by:")
    print("1. Detecting the scale text accurately")
    print("2. Searching for ALL white segments at the same Y level")
    print("3. Finding the full span from leftmost to rightmost segment")
    print("4. Accounting for text gaps in the middle")
    print("5. Scoring based on span completeness and text centrality")
    
    if result:
        print(f"\nThis should give a much longer span (~900 pixels instead of 240)")
        print(f"Leading to a scale much closer to your manual measurement!")