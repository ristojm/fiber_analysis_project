#!/usr/bin/env python3
"""
Text-Centered Scale Bar Detection (General)
Find ANY scale value text, then look for white lines centered around it
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

def find_scale_text_location(scale_region):
    """
    Find the location of ANY scale text (e.g., "100μm", "2mm", "500nm") in the scale region.
    """
    
    print("🔍 FINDING SCALE TEXT LOCATION")
    
    scale_candidates = []
    
    # Try EasyOCR first (better for bounding boxes)
    try:
        import easyocr
        reader = easyocr.Reader(['en'])
        
        # Get detailed results with bounding boxes
        results = reader.readtext(scale_region, detail=1)
        
        for (bbox, text, confidence) in results:
            text_clean = text.strip()
            print(f"  EasyOCR found: '{text}' (confidence: {confidence:.3f})")
            
            # Look for scale patterns (number + unit)
            scale_info = parse_any_scale_text([text_clean])
            
            if scale_info and confidence > 0.3:
                # Calculate center of bounding box
                bbox_array = np.array(bbox)
                center_x = int(np.mean(bbox_array[:, 0]))
                center_y = int(np.mean(bbox_array[:, 1]))
                
                # Calculate bounding box dimensions
                min_x = int(np.min(bbox_array[:, 0]))
                max_x = int(np.max(bbox_array[:, 0]))
                min_y = int(np.min(bbox_array[:, 1]))
                max_y = int(np.max(bbox_array[:, 1]))
                
                scale_candidates.append({
                    'text': text_clean,
                    'value': scale_info['value'],
                    'unit': scale_info['unit'],
                    'micrometers': scale_info['micrometers'],
                    'center_x': center_x,
                    'center_y': center_y,
                    'bbox': (min_x, min_y, max_x, max_y),
                    'confidence': confidence,
                    'method': 'EasyOCR'
                })
                print(f"    ✅ Valid scale: {scale_info['value']} {scale_info['unit']} = {scale_info['micrometers']} μm")
                print(f"    Center: ({center_x}, {center_y})")
    
    except Exception as e:
        print(f"  EasyOCR failed: {e}")
    
    # Try Tesseract as backup
    try:
        import pytesseract
        
        # Get bounding box info from Tesseract
        data = pytesseract.image_to_data(scale_region, output_type=pytesseract.Output.DICT)
        
        for i, text in enumerate(data['text']):
            if text.strip():
                confidence = int(data['conf'][i])
                
                if confidence > 30:  # Reasonable confidence
                    print(f"  Tesseract found: '{text}' (confidence: {confidence})")
                    
                    scale_info = parse_any_scale_text([text.strip()])
                    
                    if scale_info:
                        # Get bounding box from Tesseract
                        x = data['left'][i]
                        y = data['top'][i]
                        w = data['width'][i]
                        h = data['height'][i]
                        
                        center_x = x + w // 2
                        center_y = y + h // 2
                        
                        scale_candidates.append({
                            'text': text.strip(),
                            'value': scale_info['value'],
                            'unit': scale_info['unit'],
                            'micrometers': scale_info['micrometers'],
                            'center_x': center_x,
                            'center_y': center_y,
                            'bbox': (x, y, x + w, y + h),
                            'confidence': confidence / 100.0,  # Normalize to 0-1
                            'method': 'Tesseract'
                        })
                        print(f"    ✅ Valid scale: {scale_info['value']} {scale_info['unit']} = {scale_info['micrometers']} μm")
                        print(f"    Center: ({center_x}, {center_y})")
    
    except Exception as e:
        print(f"  Tesseract failed: {e}")
    
    # Sort by confidence and return best candidates
    scale_candidates.sort(key=lambda x: x['confidence'], reverse=True)
    
    print(f"  Found {len(scale_candidates)} scale text candidates")
    
    return scale_candidates

def parse_any_scale_text(text_lines):
    """
    Parse any scale text to extract numerical value and unit.
    Handles: 100μm, 2mm, 500nm, 1.5cm, 0.5mm, etc.
    """
    
    # Comprehensive patterns for different units and OCR errors
    scale_patterns = [
        # Standard patterns
        (r'(\d+\.?\d*)\s*μm', 'μm', 1.0),
        (r'(\d+\.?\d*)\s*µm', 'μm', 1.0),      # Different mu character
        (r'(\d+\.?\d*)\s*um', 'μm', 1.0),       # No mu symbol
        (r'(\d+\.?\d*)\s*nm', 'nm', 0.001),
        (r'(\d+\.?\d*)\s*mm', 'mm', 1000.0),
        (r'(\d+\.?\d*)\s*cm', 'cm', 10000.0),
        (r'(\d+\.?\d*)\s*m(?![m])', 'm', 1000000.0),  # Meter but not mm
        
        # OCR error patterns
        (r'(\d+\.?\d*)\s*[Oo]m', 'μm', 1.0),    # Om instead of μm
        (r'(\d+\.?\d*)\s*[Oo]jm', 'μm', 1.0),   # Ojm instead of μm
        (r'(\d+\.?\d*)\s*jim', 'μm', 1.0),      # jim instead of μm
        (r'(\d+\.?\d*)\s*[|l]m', 'μm', 1.0),    # |m or lm instead of μm
        (r'(\d+\.?\d*)\s*[|l]jm', 'μm', 1.0),   # |jm or ljm
    ]
    
    for text in text_lines:
        text_clean = text.lower().replace(' ', '')
        
        for pattern, unit_name, conversion in scale_patterns:
            matches = re.findall(pattern, text_clean, re.IGNORECASE)
            for match in matches:
                try:
                    value = float(match)
                    micrometers = value * conversion
                    
                    # Filter reasonable scale values (0.1μm to 10cm in micrometers)
                    if 0.1 <= micrometers <= 100000:
                        return {
                            'value': value,
                            'unit': unit_name,
                            'micrometers': micrometers,
                            'original_text': text
                        }
                
                except ValueError:
                    continue
    
    return None

def find_white_lines_around_text(scale_region, text_center_x, text_center_y, text_bbox, search_radius=100):
    """
    Find white lines on black background centered around the detected text.
    """
    
    print(f"📏 FINDING WHITE LINES AROUND TEXT")
    print(f"  Text center: ({text_center_x}, {text_center_y})")
    print(f"  Text bbox: {text_bbox}")
    print(f"  Search radius: {search_radius} pixels")
    
    height, width = scale_region.shape
    
    # Create search region around the text
    search_y_min = max(0, text_center_y - search_radius // 4)  # Narrow vertical search
    search_y_max = min(height, text_center_y + search_radius // 4)
    search_x_min = max(0, text_center_x - search_radius)
    search_x_max = min(width, text_center_x + search_radius)
    
    search_region = scale_region[search_y_min:search_y_max, search_x_min:search_x_max]
    
    print(f"  Search region: {search_region.shape} at ({search_x_min}, {search_y_min})")
    
    # Look for white lines on black background
    # Invert the image to make white lines appear as black lines for line detection
    inverted = 255 - search_region
    
    # Use threshold to isolate white areas
    _, white_mask = cv2.threshold(search_region, 200, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to connect broken white lines
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    white_connected = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel_horizontal)
    
    # Find contours of white regions
    contours, _ = cv2.findContours(white_connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    line_candidates = []
    
    for contour in contours:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Check if it looks like a horizontal line
        aspect_ratio = w / h if h > 0 else 0
        
        if aspect_ratio > 3 and w > 20:  # Horizontal and reasonably long
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
                'area': cv2.contourArea(contour)
            })
    
    print(f"  Found {len(line_candidates)} white line candidates")
    
    # Group lines by vertical position (same Y level)
    y_tolerance = 10
    line_groups = []
    
    line_candidates.sort(key=lambda x: x['center_y'])
    
    i = 0
    while i < len(line_candidates):
        current_y = line_candidates[i]['center_y']
        group = []
        
        for j in range(i, len(line_candidates)):
            if abs(line_candidates[j]['center_y'] - current_y) <= y_tolerance:
                group.append(line_candidates[j])
            else:
                break
        
        if len(group) >= 1:
            # Sort by X position
            group.sort(key=lambda x: x['center_x'])
            
            # Calculate total span
            leftmost = min(line['x'] for line in group)
            rightmost = max(line['x'] + line['width'] for line in group)
            total_span = rightmost - leftmost
            
            # Check if the text is roughly centered between the lines
            text_relative_pos = (text_center_x - leftmost) / total_span if total_span > 0 else 0.5
            
            line_groups.append({
                'lines': group,
                'count': len(group),
                'leftmost': leftmost,
                'rightmost': rightmost,
                'total_span': total_span,
                'average_y': sum(line['center_y'] for line in group) / len(group),
                'text_relative_pos': text_relative_pos,
                'text_centrality_score': 1.0 - 2 * abs(text_relative_pos - 0.5)  # Score based on how centered text is
            })
        
        i += len(group)
    
    print(f"  Grouped into {len(line_groups)} line groups:")
    for i, group in enumerate(line_groups):
        print(f"    Group {i+1}: {group['total_span']:.1f}px span, {group['count']} lines")
        print(f"              text position: {group['text_relative_pos']:.2f} (0.5 = centered)")
        print(f"              centrality score: {group['text_centrality_score']:.3f}")
    
    return line_groups, search_region, white_mask

def test_text_centered_detection():
    """Test the text-centered scale bar detection."""
    
    print("="*60)
    print("TEXT-CENTERED SCALE BAR DETECTION")
    print("="*60)
    
    from image_preprocessing import load_image
    from scale_detection import ScaleBarDetector
    
    # Load image and extract scale region
    img_path = project_root / "sample_images" / "hollow_fiber_sample.jpg"
    image = load_image(str(img_path))
    
    # Use a conservative scale region extraction
    detector = ScaleBarDetector()
    scale_region, y_offset = detector.extract_scale_region(image)
    
    print(f"Scale region shape: {scale_region.shape}")
    
    # Step 1: Find scale text
    text_candidates = find_scale_text_location(scale_region)
    
    if not text_candidates:
        print("❌ No scale text found")
        return
    
    # Use the best text candidate
    best_text = text_candidates[0]
    print(f"\n✅ Using text: '{best_text['text']}' = {best_text['micrometers']} μm")
    
    # Step 2: Find white lines around the text
    line_groups, search_region, white_mask = find_white_lines_around_text(
        scale_region, 
        best_text['center_x'], 
        best_text['center_y'],
        best_text['bbox']
    )
    
    # Step 3: Evaluate results
    print(f"\n📊 EVALUATION:")
    
    manual_scale = 0.444
    best_group = None
    best_error = float('inf')
    
    for group in line_groups:
        span = group['total_span']
        if span > 50:  # Reasonable minimum
            implied_scale = best_text['micrometers'] / span
            error = abs(implied_scale - manual_scale) / manual_scale
            centrality = group['text_centrality_score']
            
            # Combined score (scale accuracy + text centrality)
            combined_score = (1 - error) * 0.7 + centrality * 0.3
            
            print(f"  {span:.1f}px span → {implied_scale:.4f} μm/px")
            print(f"    Error: {error*100:.1f}%, Centrality: {centrality:.3f}, Score: {combined_score:.3f}")
            
            if error < best_error:
                best_error = error
                best_group = group
    
    if best_group:
        best_span = best_group['total_span']
        best_scale = best_text['micrometers'] / best_span
        
        print(f"\n🎯 BEST RESULT:")
        print(f"  Text: '{best_text['text']}' = {best_text['micrometers']} μm")
        print(f"  Total span: {best_span:.1f} pixels")
        print(f"  Calculated scale: {best_scale:.4f} μm/pixel")
        print(f"  Manual scale: {manual_scale:.4f} μm/pixel")
        print(f"  Error: {best_error*100:.1f}%")
        print(f"  Text centrality: {best_group['text_centrality_score']:.3f}")
        
        if best_error < 0.1:
            print(f"  🎯 EXCELLENT MATCH!")
        elif best_error < 0.2:
            print(f"  ✅ GOOD MATCH!")
        else:
            print(f"  ⚠️ Needs investigation")
    
    # Visualize
    visualize_text_centered_detection(scale_region, text_candidates, best_text, 
                                     line_groups, search_region, white_mask, best_group)

def visualize_text_centered_detection(scale_region, text_candidates, best_text,
                                     line_groups, search_region, white_mask, best_group):
    """Visualize the text-centered detection."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original scale region with text locations
    axes[0, 0].imshow(scale_region, cmap='gray')
    
    # Mark all text candidates
    for i, text in enumerate(text_candidates):
        color = 'red' if text == best_text else 'yellow'
        bbox = text['bbox']
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], 
                           fill=False, color=color, linewidth=2)
        axes[0, 0].add_patch(rect)
        axes[0, 0].text(text['center_x'], text['center_y'], f"{text['text']}", 
                       ha='center', va='center', color=color, fontsize=8, weight='bold')
    
    axes[0, 0].set_title('Text Detection')
    axes[0, 0].axis('off')
    
    # Search region
    axes[0, 1].imshow(search_region, cmap='gray')
    axes[0, 1].set_title('Search Region Around Text')
    axes[0, 1].axis('off')
    
    # White mask
    axes[0, 2].imshow(white_mask, cmap='gray')
    axes[0, 2].set_title('White Line Detection')
    axes[0, 2].axis('off')
    
    # Line groups on original
    overlay = cv2.cvtColor(scale_region, cv2.COLOR_GRAY2RGB)
    
    colors = [(0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    
    for i, group in enumerate(line_groups):
        color = colors[i % len(colors)]
        
        for line in group['lines']:
            cv2.rectangle(overlay, (line['x'], line['y']), 
                         (line['x'] + line['width'], line['y'] + line['height']), 
                         color, 2)
        
        # Draw total span
        y_avg = int(group['average_y'])
        cv2.line(overlay, (int(group['leftmost']), y_avg), 
                (int(group['rightmost']), y_avg), (255, 0, 0), 2)
    
    # Mark text center
    cv2.circle(overlay, (best_text['center_x'], best_text['center_y']), 5, (255, 0, 0), -1)
    
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title('Line Groups with Text Center')
    axes[1, 0].axis('off')
    
    # Best group highlighted
    if best_group:
        overlay2 = cv2.cvtColor(scale_region, cv2.COLOR_GRAY2RGB)
        
        for line in best_group['lines']:
            cv2.rectangle(overlay2, (line['x'], line['y']), 
                         (line['x'] + line['width'], line['y'] + line['height']), 
                         (0, 255, 0), 3)
        
        # Total span
        y_avg = int(best_group['average_y'])
        cv2.line(overlay2, (int(best_group['leftmost']), y_avg), 
                (int(best_group['rightmost']), y_avg), (255, 0, 0), 3)
        
        # Text center
        cv2.circle(overlay2, (best_text['center_x'], best_text['center_y']), 5, (255, 0, 0), -1)
        
        axes[1, 1].imshow(overlay2)
        axes[1, 1].set_title(f"Best Match: {best_group['total_span']:.0f}px")
    else:
        axes[1, 1].imshow(scale_region, cmap='gray')
        axes[1, 1].set_title('No Best Group Found')
    axes[1, 1].axis('off')
    
    # Results summary
    axes[1, 2].axis('off')
    
    if best_group:
        manual_scale = 0.444
        calc_scale = best_text['micrometers'] / best_group['total_span']
        error = abs(calc_scale - manual_scale) / manual_scale * 100
        
        summary = f"""TEXT-CENTERED RESULTS:

Detected Text: '{best_text['text']}'
Scale Value: {best_text['micrometers']} μm
Method: {best_text['method']}

Best Line Group:
  Total span: {best_group['total_span']:.0f} px
  Line segments: {best_group['count']}
  Text centrality: {best_group['text_centrality_score']:.3f}

Scale Calculation:
  Calculated: {calc_scale:.4f} μm/px
  Manual: {manual_scale:.4f} μm/px
  Error: {error:.1f}%

Status: {'🎯 EXCELLENT' if error < 10 else '✅ GOOD' if error < 20 else '⚠️ CHECK'}"""
    else:
        summary = "No suitable line group found"
    
    axes[1, 2].text(0.05, 0.95, summary, transform=axes[1, 2].transAxes,
                    verticalalignment='top', fontsize=9, fontfamily='monospace')
    axes[1, 2].set_title('Results')
    
    plt.tight_layout()
    plt.savefig(project_root / 'analysis_results' / 'text_centered_detection.png', 
                dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    test_text_centered_detection()
    
    print(f"\n" + "="*60)
    print("This approach should be much more reliable!")
    print("It finds ANY scale value text, then looks for white lines centered around it.")