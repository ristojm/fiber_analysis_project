#!/usr/bin/env python3
"""
Improved Text-Centered Scale Bar Detection
Better selection criteria for the correct scale bar when multiple candidates exist
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

def find_scale_text_with_confidence(scale_region):
    """
    Find scale text with better confidence scoring and multiple OCR attempts.
    """
    
    print("🔍 FINDING SCALE TEXT WITH IMPROVED CONFIDENCE")
    
    scale_candidates = []
    
    # Strategy 1: Try EasyOCR with different confidence thresholds
    try:
        import easyocr
        reader = easyocr.Reader(['en'])
        
        # Get results with lower confidence threshold for more candidates
        results = reader.readtext(scale_region, detail=1, width_ths=0.4, height_ths=0.4)
        
        for (bbox, text, confidence) in results:
            text_clean = text.strip()
            print(f"  EasyOCR: '{text}' (conf: {confidence:.3f})")
            
            # Parse scale info
            scale_info = parse_comprehensive_scale_text([text_clean])
            
            if scale_info and confidence > 0.2:  # Very low threshold
                bbox_array = np.array(bbox)
                center_x = int(np.mean(bbox_array[:, 0]))
                center_y = int(np.mean(bbox_array[:, 1]))
                
                # Calculate bbox width for quality assessment
                bbox_width = np.max(bbox_array[:, 0]) - np.min(bbox_array[:, 0])
                bbox_height = np.max(bbox_array[:, 1]) - np.min(bbox_array[:, 1])
                
                scale_candidates.append({
                    'text': text_clean,
                    'value': scale_info['value'],
                    'unit': scale_info['unit'],
                    'micrometers': scale_info['micrometers'],
                    'center_x': center_x,
                    'center_y': center_y,
                    'bbox': bbox_array,
                    'bbox_width': bbox_width,
                    'bbox_height': bbox_height,
                    'confidence': confidence,
                    'method': 'EasyOCR',
                    'quality_score': confidence * min(1.0, bbox_width / 50)  # Prefer wider text
                })
                print(f"    ✅ Scale: {scale_info['value']} {scale_info['unit']} = {scale_info['micrometers']} μm")
    
    except Exception as e:
        print(f"  EasyOCR failed: {e}")
    
    # Strategy 2: Enhanced preprocessing for Tesseract
    try:
        import pytesseract
        
        # Try multiple preprocessing approaches
        preprocessed_images = []
        
        # Original
        preprocessed_images.append(("original", scale_region))
        
        # High contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(scale_region)
        preprocessed_images.append(("enhanced", enhanced))
        
        # Threshold to isolate text
        _, thresh = cv2.threshold(scale_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(("thresholded", thresh))
        
        # Inverted threshold
        preprocessed_images.append(("inverted", 255 - thresh))
        
        for name, img in preprocessed_images:
            try:
                # More specific OCR config for numbers and units
                config = '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789.μµumnmkMm '
                
                data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=config)
                
                for i, text in enumerate(data['text']):
                    if text.strip() and int(data['conf'][i]) > 20:
                        scale_info = parse_comprehensive_scale_text([text.strip()])
                        
                        if scale_info:
                            x = data['left'][i]
                            y = data['top'][i]
                            w = data['width'][i]
                            h = data['height'][i]
                            
                            center_x = x + w // 2
                            center_y = y + h // 2
                            confidence = int(data['conf'][i]) / 100.0
                            
                            scale_candidates.append({
                                'text': text.strip(),
                                'value': scale_info['value'],
                                'unit': scale_info['unit'],
                                'micrometers': scale_info['micrometers'],
                                'center_x': center_x,
                                'center_y': center_y,
                                'bbox': np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]]),
                                'bbox_width': w,
                                'bbox_height': h,
                                'confidence': confidence,
                                'method': f'Tesseract-{name}',
                                'quality_score': confidence * min(1.0, w / 30)
                            })
                            print(f"  Tesseract-{name}: '{text}' → {scale_info['value']} {scale_info['unit']}")
            
            except Exception as te:
                continue
    
    except Exception as e:
        print(f"  Tesseract failed: {e}")
    
    # Remove duplicates and sort by quality
    unique_candidates = []
    seen_values = set()
    
    for candidate in scale_candidates:
        key = (candidate['value'], candidate['unit'])
        if key not in seen_values:
            seen_values.add(key)
            unique_candidates.append(candidate)
    
    # Sort by quality score
    unique_candidates.sort(key=lambda x: x['quality_score'], reverse=True)
    
    print(f"  Found {len(unique_candidates)} unique scale candidates")
    for i, cand in enumerate(unique_candidates[:3]):
        print(f"    {i+1}. {cand['value']} {cand['unit']} (quality: {cand['quality_score']:.3f}, method: {cand['method']})")
    
    return unique_candidates

def parse_comprehensive_scale_text(text_lines):
    """
    Enhanced scale text parsing with better error handling.
    """
    
    # More comprehensive patterns including common OCR errors
    scale_patterns = [
        # Standard patterns
        (r'(\d+\.?\d*)\s*μm', 'μm', 1.0),
        (r'(\d+\.?\d*)\s*µm', 'μm', 1.0),
        (r'(\d+\.?\d*)\s*um', 'μm', 1.0),
        (r'(\d+\.?\d*)\s*nm', 'nm', 0.001),
        (r'(\d+\.?\d*)\s*mm', 'mm', 1000.0),
        (r'(\d+\.?\d*)\s*cm', 'cm', 10000.0),
        
        # OCR error patterns for SEM images
        (r'(\d+\.?\d*)\s*[Oo]m', 'μm', 1.0),    # Om instead of μm
        (r'(\d+\.?\d*)\s*[|l1]m', 'μm', 1.0),   # |m or lm or 1m instead of μm
        (r'(\d+\.?\d*)\s*[Oo]jm', 'μm', 1.0),   # Ojm
        (r'(\d+\.?\d*)\s*jim', 'μm', 1.0),      # jim
        (r'(\d+\.?\d*)\s*μrn', 'μm', 1.0),      # μrn instead of μm
        (r'(\d+\.?\d*)\s*urn', 'μm', 1.0),      # urn instead of μm
        
        # With decimal separator variations
        (r'(\d+)[,.](\d+)\s*μm', 'μm', 1.0),
        (r'(\d+)[,.](\d+)\s*mm', 'mm', 1000.0),
    ]
    
    for text in text_lines:
        text_clean = text.lower().replace(' ', '').replace('\n', '')
        
        for pattern, unit_name, conversion in scale_patterns:
            matches = re.findall(pattern, text_clean, re.IGNORECASE)
            for match in matches:
                try:
                    if isinstance(match, tuple):
                        # Handle decimal patterns
                        value = float(f"{match[0]}.{match[1]}")
                    else:
                        value = float(match)
                    
                    micrometers = value * conversion
                    
                    # Filter reasonable scale values
                    if 0.01 <= micrometers <= 50000:  # 10nm to 5cm range
                        return {
                            'value': value,
                            'unit': unit_name,
                            'micrometers': micrometers,
                            'original_text': text,
                            'pattern_used': pattern
                        }
                
                except (ValueError, IndexError):
                    continue
    
    return None

def find_best_scale_bar_match(scale_region, text_candidates, search_radius=120):
    """
    Improved scale bar matching with better selection criteria.
    """
    
    print(f"📏 FINDING BEST SCALE BAR MATCH")
    
    all_results = []
    
    for text_idx, text_candidate in enumerate(text_candidates):
        print(f"\n  Testing text candidate {text_idx + 1}: {text_candidate['value']} {text_candidate['unit']}")
        
        # Find white lines around this text
        line_groups = find_white_lines_improved(
            scale_region, 
            text_candidate['center_x'], 
            text_candidate['center_y'],
            text_candidate['bbox'],
            search_radius
        )
        
        # Score each line group for this text
        for group_idx, group in enumerate(line_groups):
            span = group['total_span']
            
            if span > 30:  # Minimum reasonable span
                # Calculate implied scale
                implied_scale = text_candidate['micrometers'] / span
                
                # Multiple scoring criteria
                scores = calculate_scale_bar_scores(text_candidate, group, implied_scale)
                
                result = {
                    'text_candidate': text_candidate,
                    'line_group': group,
                    'implied_scale': implied_scale,
                    'scores': scores,
                    'total_score': sum(scores.values()),
                    'text_idx': text_idx,
                    'group_idx': group_idx
                }
                
                all_results.append(result)
                
                print(f"    Line group {group_idx + 1}: {span:.1f}px → {implied_scale:.4f} μm/px")
                print(f"      Scores: {', '.join(f'{k}={v:.3f}' for k, v in scores.items())}")
                print(f"      Total: {result['total_score']:.3f}")
    
    if not all_results:
        print("  No valid combinations found")
        return None
    
    # Sort by total score
    all_results.sort(key=lambda x: x['total_score'], reverse=True)
    
    print(f"\n🏆 TOP 3 COMBINATIONS:")
    for i, result in enumerate(all_results[:3]):
        print(f"  {i+1}. Text: {result['text_candidate']['value']} {result['text_candidate']['unit']}")
        print(f"     Span: {result['line_group']['total_span']:.1f}px")
        print(f"     Scale: {result['implied_scale']:.4f} μm/px")
        print(f"     Score: {result['total_score']:.3f}")
    
    return all_results[0]  # Return best match

def calculate_scale_bar_scores(text_candidate, line_group, implied_scale):
    """
    Calculate multiple scoring criteria for scale bar selection.
    """
    
    scores = {}
    
    # 1. Text quality score
    scores['text_quality'] = text_candidate['quality_score'] * 0.3
    
    # 2. Text centrality score (how centered is text between lines)
    centrality = line_group.get('text_centrality_score', 0)
    scores['text_centrality'] = centrality * 0.25
    
    # 3. Scale reasonableness (prefer scales in typical SEM range)
    typical_scales = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]  # μm/pixel
    scale_distances = [abs(implied_scale - ts) / ts for ts in typical_scales]
    min_distance = min(scale_distances)
    scores['scale_reasonableness'] = max(0, 1.0 - min_distance) * 0.2
    
    # 4. Line quality score (prefer continuous, well-defined lines)
    line_count = line_group['count']
    total_span = line_group['total_span']
    
    # Prefer longer spans and reasonable number of segments
    span_score = min(1.0, total_span / 200)  # Normalize to reasonable max
    segment_score = 1.0 if line_count <= 3 else max(0, 1.0 - (line_count - 3) * 0.2)
    scores['line_quality'] = (span_score * segment_score) * 0.15
    
    # 5. Value reasonableness (prefer common scale bar values)
    value = text_candidate['value']
    common_values = [50, 100, 200, 250, 300, 400, 500, 1000, 1500, 2000]
    
    if value in common_values:
        scores['value_reasonableness'] = 0.1
    elif any(abs(value - cv) / cv < 0.1 for cv in common_values):
        scores['value_reasonableness'] = 0.05
    else:
        scores['value_reasonableness'] = 0.0
    
    return scores

def find_white_lines_improved(scale_region, text_center_x, text_center_y, text_bbox, search_radius=120):
    """
    Improved white line detection with better parameter tuning.
    """
    
    height, width = scale_region.shape
    
    # Expand search area but focus vertically around text
    search_y_min = max(0, text_center_y - search_radius // 3)
    search_y_max = min(height, text_center_y + search_radius // 3)
    search_x_min = max(0, text_center_x - search_radius)
    search_x_max = min(width, text_center_x + search_radius)
    
    search_region = scale_region[search_y_min:search_y_max, search_x_min:search_x_max]
    
    # Multiple thresholding approaches
    white_masks = []
    
    # Approach 1: High threshold for bright white
    _, white1 = cv2.threshold(search_region, 220, 255, cv2.THRESH_BINARY)
    white_masks.append(("high_thresh", white1))
    
    # Approach 2: Adaptive threshold
    white2 = cv2.adaptiveThreshold(search_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 21, -5)
    white_masks.append(("adaptive", white2))
    
    # Approach 3: Otsu threshold on inverted image
    inverted = 255 - search_region
    _, white3 = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_masks.append(("otsu_inverted", white3))
    
    all_line_groups = []
    
    for mask_name, white_mask in white_masks:
        # Morphological operations
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        white_connected = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel_h)
        
        # Find contours
        contours, _ = cv2.findContours(white_connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        line_candidates = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Filter for horizontal line-like shapes
            if aspect_ratio > 2 and w > 15:
                # Convert back to full scale region coordinates
                full_x = x + search_x_min
                full_y = y + search_y_min
                
                line_candidates.append({
                    'x': full_x,
                    'y': full_y,
                    'width': w,
                    'height': h,
                    'center_x': full_x + w // 2,
                    'center_y': full_y + h // 2,
                    'aspect_ratio': aspect_ratio,
                    'area': cv2.contourArea(contour),
                    'method': mask_name
                })
        
        # Group lines by Y level
        groups = group_lines_by_y_level(line_candidates, text_center_x, text_center_y)
        all_line_groups.extend(groups)
    
    # Remove duplicate groups and sort by quality
    unique_groups = remove_duplicate_groups(all_line_groups)
    
    return unique_groups

def group_lines_by_y_level(line_candidates, text_center_x, text_center_y, y_tolerance=15):
    """Group lines that are at similar Y levels."""
    
    if not line_candidates:
        return []
    
    line_candidates.sort(key=lambda x: x['center_y'])
    groups = []
    
    i = 0
    while i < len(line_candidates):
        current_y = line_candidates[i]['center_y']
        group = []
        
        # Collect lines at similar Y level
        for j in range(i, len(line_candidates)):
            if abs(line_candidates[j]['center_y'] - current_y) <= y_tolerance:
                group.append(line_candidates[j])
            else:
                break
        
        if len(group) >= 1:
            group.sort(key=lambda x: x['center_x'])
            
            leftmost = min(line['x'] for line in group)
            rightmost = max(line['x'] + line['width'] for line in group)
            total_span = rightmost - leftmost
            
            if total_span >= 20:  # Minimum span
                # Calculate text centrality
                text_relative_pos = (text_center_x - leftmost) / total_span if total_span > 0 else 0.5
                centrality_score = 1.0 - 2 * abs(text_relative_pos - 0.5)
                
                groups.append({
                    'lines': group,
                    'count': len(group),
                    'leftmost': leftmost,
                    'rightmost': rightmost,
                    'total_span': total_span,
                    'average_y': sum(line['center_y'] for line in group) / len(group),
                    'text_relative_pos': text_relative_pos,
                    'text_centrality_score': max(0, centrality_score),
                    'methods': list(set(line['method'] for line in group))
                })
        
        i += len(group)
    
    return groups

def remove_duplicate_groups(groups, span_tolerance=10):
    """Remove groups that are too similar."""
    
    unique_groups = []
    
    for group in groups:
        is_duplicate = False
        for existing in unique_groups:
            span_diff = abs(group['total_span'] - existing['total_span'])
            y_diff = abs(group['average_y'] - existing['average_y'])
            
            if span_diff <= span_tolerance and y_diff <= 10:
                # Keep the one with better centrality
                if group['text_centrality_score'] > existing['text_centrality_score']:
                    unique_groups.remove(existing)
                    unique_groups.append(group)
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_groups.append(group)
    
    return unique_groups

def test_improved_detection():
    """Test the improved text-centered detection."""
    
    print("="*60)
    print("IMPROVED TEXT-CENTERED SCALE BAR DETECTION")
    print("="*60)
    
    from image_preprocessing import load_image
    from scale_detection import ScaleBarDetector
    
    # Load image
    img_path = project_root / "sample_images" / "hollow_fiber_sample.jpg"
    image = load_image(str(img_path))
    
    # Extract scale region
    detector = ScaleBarDetector()
    scale_region, y_offset = detector.extract_scale_region(image)
    
    print(f"Scale region shape: {scale_region.shape}")
    
    # Step 1: Find scale text with improved confidence
    text_candidates = find_scale_text_with_confidence(scale_region)
    
    if not text_candidates:
        print("❌ No scale text found")
        return
    
    # Step 2: Find best scale bar match
    best_result = find_best_scale_bar_match(scale_region, text_candidates)
    
    if best_result:
        best_text = best_result['text_candidate']
        best_group = best_result['line_group']
        best_scale = best_result['implied_scale']
        
        print(f"\n🎯 BEST RESULT:")
        print(f"  Text: '{best_text['text']}' = {best_text['micrometers']} μm")
        print(f"  Method: {best_text['method']}")
        print(f"  Line span: {best_group['total_span']:.1f} pixels")
        print(f"  Calculated scale: {best_scale:.4f} μm/pixel")
        print(f"  Total score: {best_result['total_score']:.3f}")
        
        # Compare with manual
        manual_scale = 0.444
        error = abs(best_scale - manual_scale) / manual_scale * 100
        print(f"  Manual scale: {manual_scale:.4f} μm/pixel")
        print(f"  Error: {error:.1f}%")
        
        if error < 10:
            print(f"  🎯 EXCELLENT MATCH!")
        elif error < 20:
            print(f"  ✅ GOOD MATCH!")
        else:
            print(f"  ⚠️ Needs investigation")
        
        # Visualize
        visualize_improved_detection(scale_region, text_candidates, best_result)
        
        return best_result
    else:
        print("❌ No suitable scale bar found")
        return None

def visualize_improved_detection(scale_region, text_candidates, best_result):
    """Visualize the improved detection results."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original with all text candidates
    axes[0, 0].imshow(scale_region, cmap='gray')
    
    colors = ['red', 'yellow', 'cyan', 'orange', 'purple']
    for i, text in enumerate(text_candidates[:5]):
        color = colors[i % len(colors)]
        bbox = text['bbox']
        
        # Draw bounding box
        bbox_points = bbox.astype(int)
        for j in range(len(bbox_points)):
            start = bbox_points[j]
            end = bbox_points[(j + 1) % len(bbox_points)]
            axes[0, 0].plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=2)
        
        # Label
        axes[0, 0].text(text['center_x'], text['center_y'], 
                       f"{text['value']}{text['unit'][:1]}", 
                       ha='center', va='center', color=color, fontsize=8, weight='bold')
    
    axes[0, 0].set_title(f'Text Candidates ({len(text_candidates)} found)')
    axes[0, 0].axis('off')
    
    # Best text highlighted
    axes[0, 1].imshow(scale_region, cmap='gray')
    
    if best_result:
        best_text = best_result['text_candidate']
        bbox = best_text['bbox'].astype(int)
        
        for j in range(len(bbox)):
            start = bbox[j]
            end = bbox[(j + 1) % len(bbox)]
            axes[0, 1].plot([start[0], end[0]], [start[1], end[1]], color='red', linewidth=3)
        
        axes[0, 1].text(best_text['center_x'], best_text['center_y'], 
                       f"{best_text['value']}{best_text['unit']}", 
                       ha='center', va='center', color='red', fontsize=10, weight='bold')
    
    axes[0, 1].set_title('Selected Text')
    axes[0, 1].axis('off')
    
    # Best scale bar
    overlay = cv2.cvtColor(scale_region, cv2.COLOR_GRAY2RGB)
    
    if best_result:
        best_group = best_result['line_group']
        
        # Draw lines
        for line in best_group['lines']:
            cv2.rectangle(overlay, (line['x'], line['y']), 
                         (line['x'] + line['width'], line['y'] + line['height']), 
                         (0, 255, 0), 2)
        
        # Draw total span
        y_avg = int(best_group['average_y'])
        cv2.line(overlay, (int(best_group['leftmost']), y_avg), 
                (int(best_group['rightmost']), y_avg), (255, 0, 0), 3)
        
        # Text center
        best_text = best_result['text_candidate']
        cv2.circle(overlay, (best_text['center_x'], best_text['center_y']), 5, (255, 0, 0), -1)
    
    axes[0, 2].imshow(overlay)
    axes[0, 2].set_title('Best Scale Bar Match')
    axes[0, 2].axis('off')
    
    # Scores breakdown
    axes[1, 0].axis('off')
    if best_result:
        scores = best_result['scores']
        score_text = "SCORING BREAKDOWN:\n\n"
        for score_name, score_value in scores.items():
            score_text += f"{score_name.replace('_', ' ').title()}:\n  {score_value:.3f}\n"
        score_text += f"\nTotal Score: {best_result['total_score']:.3f}"
        
        axes[1, 0].text(0.05, 0.95, score_text, transform=axes[1, 0].transAxes,
                       verticalalignment='top', fontsize=9, fontfamily='monospace')
    axes[1, 0].set_title('Score Breakdown')
    
    # Results summary
    axes[1, 1].axis('off')
    if best_result:
        best_text = best_result['text_candidate']
        best_group = best_result['line_group']
        best_scale = best_result['implied_scale']
        manual_scale = 0.444
        error = abs(best_scale - manual_scale) / manual_scale * 100
        
        summary = f"""FINAL RESULTS:

Text Detection:
  Value: {best_text['value']} {best_text['unit']}
  Method: {best_text['method']}
  Confidence: {best_text['confidence']:.3f}

Scale Bar:
  Span: {best_group['total_span']:.0f} pixels
  Segments: {best_group['count']}
  Centrality: {best_group['text_centrality_score']:.3f}

Scale Calculation:
  Calculated: {best_scale:.4f} μm/px
  Manual: {manual_scale:.4f} μm/px
  Error: {error:.1f}%

Status: {'🎯 EXCELLENT' if error < 10 else '✅ GOOD' if error < 20 else '⚠️ CHECK'}"""
    else:
        summary = "No results found"
    
    axes[1, 1].text(0.05, 0.95, summary, transform=axes[1, 1].transAxes,
                    verticalalignment='top', fontsize=9, fontfamily='monospace')
    axes[1, 1].set_title('Summary')
    
    # Error analysis
    axes[1, 2].axis('off')
    if best_result and len(text_candidates) > 1:
        comparison_text = "CANDIDATE COMPARISON:\n\n"
        for i, candidate in enumerate(text_candidates[:3]):
            comparison_text += f"{i+1}. {candidate['value']} {candidate['unit']}\n"
            comparison_text += f"   Quality: {candidate['quality_score']:.3f}\n"
            comparison_text += f"   Method: {candidate['method']}\n\n"
        
        axes[1, 2].text(0.05, 0.95, comparison_text, transform=axes[1, 2].transAxes,
                       verticalalignment='top', fontsize=9, fontfamily='monospace')
    axes[1, 2].set_title('Text Candidates')
    
    plt.tight_layout()
    plt.savefig(project_root / 'analysis_results' / 'improved_scale_detection.png', 
                dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    result = test_improved_detection()
    
    print(f"\n" + "="*60)
    print("IMPROVEMENTS MADE:")
    print("="*60)
    print("1. Better text confidence scoring with multiple OCR attempts")
    print("2. Enhanced preprocessing for difficult text")
    print("3. Multi-criteria scoring for scale bar selection:")
    print("   - Text quality and confidence")
    print("   - Text centrality between lines")
    print("   - Scale value reasonableness")
    print("   - Line quality and continuity")
    print("   - Common scale values preference")
    print("4. Duplicate removal and better candidate selection")
    print("5. More robust white line detection with multiple methods")