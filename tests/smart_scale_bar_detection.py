#!/usr/bin/env python3
"""
Smart Scale Bar Detection
Find the actual scale bar strip and focus search on the right side
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "modules"))

def detect_scale_bar_strip(image):
    """
    Detect the actual scale bar strip (dark bottom region) instead of using a fixed fraction.
    """
    
    print("🔍 DETECTING SCALE BAR STRIP")
    
    height, width = image.shape
    
    # Look at bottom portion for horizontal intensity changes
    bottom_fraction = 0.3  # Look at bottom 30%
    bottom_region = image[int(height * (1 - bottom_fraction)):, :]
    
    # Calculate row-wise mean intensity
    row_means = np.mean(bottom_region, axis=1)
    
    # Find the transition to darker region (scale bar strip)
    # Look for a significant drop in intensity
    intensity_threshold = np.mean(row_means) - np.std(row_means)
    
    # Find first row that's significantly darker
    dark_rows = np.where(row_means < intensity_threshold)[0]
    
    if len(dark_rows) > 0:
        # Find the start of the dark strip
        strip_start_relative = dark_rows[0]
        strip_start_absolute = int(height * (1 - bottom_fraction)) + strip_start_relative
        
        # Find the end of the dark strip (bottom of image or return to brightness)
        strip_end_absolute = height  # Default to bottom
        
        # Look for return to brightness
        for i in range(strip_start_relative + 1, len(row_means)):
            if row_means[i] > intensity_threshold:
                strip_end_absolute = int(height * (1 - bottom_fraction)) + i
                break
        
        strip_height = strip_end_absolute - strip_start_absolute
        
        print(f"  Detected strip: rows {strip_start_absolute} to {strip_end_absolute}")
        print(f"  Strip height: {strip_height} pixels")
        
        # Add some erosion for wiggle room
        erosion_pixels = max(5, strip_height // 10)  # 10% erosion or minimum 5 pixels
        
        eroded_start = strip_start_absolute + erosion_pixels
        eroded_end = strip_end_absolute - erosion_pixels
        
        print(f"  After erosion: rows {eroded_start} to {eroded_end}")
        
        return eroded_start, eroded_end
    
    else:
        # Fallback: use bottom 15% with erosion
        print("  No clear strip detected, using fallback")
        fallback_start = int(height * 0.85)
        fallback_end = height
        erosion = 10
        return fallback_start + erosion, fallback_end - erosion

def extract_right_focused_scale_region(image, strip_start, strip_end):
    """
    Extract scale region focused on the right side where scale bars typically appear.
    """
    
    print("🎯 EXTRACTING RIGHT-FOCUSED SCALE REGION")
    
    height, width = image.shape
    
    # Focus on right side (typically right 60% of image)
    right_focus_fraction = 0.6
    left_boundary = int(width * (1 - right_focus_fraction))
    
    # Extract the right-focused scale region
    scale_region = image[strip_start:strip_end, left_boundary:]
    
    print(f"  Full image: {width} × {height}")
    print(f"  Left boundary: {left_boundary} (right {right_focus_fraction*100:.0f}%)")
    print(f"  Scale region: {scale_region.shape[1]} × {scale_region.shape[0]}")
    print(f"  Region covers: x={left_boundary}-{width}, y={strip_start}-{strip_end}")
    
    return scale_region, left_boundary, strip_start

def detect_scale_bar_in_focused_region(scale_region):
    """
    Detect scale bar in the focused region with enhanced parameters.
    """
    
    print("📏 DETECTING SCALE BAR IN FOCUSED REGION")
    
    # Enhanced preprocessing for the focused region
    # Apply gentle smoothing
    smoothed = cv2.GaussianBlur(scale_region, (3, 3), 0)
    
    # Edge detection with lower thresholds for better detection
    edges = cv2.Canny(smoothed, 20, 100, apertureSize=3)
    
    # Morphological operations to connect broken lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))  # Horizontal kernel
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Find line segments with relaxed parameters
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                           threshold=10,      # Lower threshold
                           minLineLength=20,  # Shorter minimum
                           maxLineGap=30)     # Larger gap tolerance
    
    horizontal_segments = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Check if line is horizontal (within 15 degrees)
            if abs(y2 - y1) <= 5 or abs(x2 - x1) == 0:
                continue
                
            angle = np.arctan2(abs(y2 - y1), abs(x2 - x1)) * 180 / np.pi
            if angle <= 15:  # Horizontal lines
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                if length >= 15:  # Minimum segment length
                    horizontal_segments.append({
                        'start_x': min(x1, x2),
                        'end_x': max(x1, x2),
                        'y': (y1 + y2) / 2,
                        'length': length,
                        'center_x': (x1 + x2) / 2,
                        'line': (x1, y1, x2, y2),
                        'angle': angle
                    })
    
    print(f"  Found {len(horizontal_segments)} horizontal segments")
    
    # Group segments by Y level with generous tolerance
    y_tolerance = 20
    groups = []
    
    # Sort by Y position
    horizontal_segments.sort(key=lambda x: x['y'])
    
    i = 0
    while i < len(horizontal_segments):
        current_y = horizontal_segments[i]['y']
        group = []
        
        # Collect segments at similar Y levels
        for j in range(i, len(horizontal_segments)):
            if abs(horizontal_segments[j]['y'] - current_y) <= y_tolerance:
                group.append(horizontal_segments[j])
            else:
                break
        
        if len(group) >= 1:
            # Sort group by X position
            group.sort(key=lambda x: x['start_x'])
            
            # Calculate total span
            total_start = min(seg['start_x'] for seg in group)
            total_end = max(seg['end_x'] for seg in group)
            total_span = total_end - total_start
            
            groups.append({
                'segments': group,
                'count': len(group),
                'total_span': total_span,
                'start_x': total_start,
                'end_x': total_end,
                'y': sum(seg['y'] for seg in group) / len(group),
                'total_segment_length': sum(seg['length'] for seg in group)
            })
        
        i += len(group)
    
    print(f"  Grouped into {len(groups)} potential scale bars:")
    
    for i, group in enumerate(groups):
        print(f"    Group {i+1}: {group['total_span']:.1f}px span, {group['count']} segments")
    
    return groups, horizontal_segments

def test_smart_scale_detection():
    """Test the smart scale bar detection."""
    
    print("="*60)
    print("SMART SCALE BAR DETECTION")
    print("="*60)
    
    from image_preprocessing import load_image
    
    # Load image
    img_path = project_root / "sample_images" / "hollow_fiber_sample.jpg"
    image = load_image(str(img_path))
    
    print(f"Image shape: {image.shape}")
    
    # Step 1: Detect the actual scale bar strip
    strip_start, strip_end = detect_scale_bar_strip(image)
    
    # Step 2: Extract right-focused scale region
    scale_region, x_offset, y_offset = extract_right_focused_scale_region(image, strip_start, strip_end)
    
    # Step 3: Detect scale bar in focused region
    groups, all_segments = detect_scale_bar_in_focused_region(scale_region)
    
    # Step 4: Evaluate results
    print(f"\n📊 EVALUATION:")
    
    best_group = None
    best_error = float('inf')
    manual_scale = 0.444  # Your measurement
    
    for group in groups:
        if group['total_span'] > 50:  # Reasonable minimum
            # Convert back to full image coordinates for scale calculation
            full_image_span = group['total_span']  # Same in pixels
            
            implied_scale = 400.0 / full_image_span
            error = abs(implied_scale - manual_scale) / manual_scale
            
            print(f"  Group with {full_image_span:.1f}px span:")
            print(f"    Implied scale: {implied_scale:.4f} μm/pixel")
            print(f"    Error vs manual: {error*100:.1f}%")
            
            if error < best_error:
                best_error = error
                best_group = group
    
    if best_group:
        best_span = best_group['total_span']
        best_scale = 400.0 / best_span
        
        print(f"\n✅ BEST MATCH:")
        print(f"  Total span: {best_span:.1f} pixels")
        print(f"  Calculated scale: {best_scale:.4f} μm/pixel")
        print(f"  Your manual: {manual_scale:.4f} μm/pixel")
        print(f"  Error: {best_error*100:.1f}%")
        
        if best_error < 0.1:
            print(f"  🎯 EXCELLENT MATCH!")
        elif best_error < 0.2:
            print(f"  ✅ GOOD MATCH!")
        else:
            print(f"  ⚠️ Reasonable but could be better")
    
    # Visualize
    visualize_smart_detection(image, scale_region, groups, all_segments, 
                             strip_start, strip_end, x_offset, y_offset, best_group)
    
    return best_group

def visualize_smart_detection(image, scale_region, groups, all_segments,
                             strip_start, strip_end, x_offset, y_offset, best_group):
    """Visualize the smart scale detection process."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Full image with detected strip
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].axhline(y=strip_start, color='red', linewidth=2, alpha=0.7)
    axes[0, 0].axhline(y=strip_end, color='red', linewidth=2, alpha=0.7)
    axes[0, 0].axvline(x=x_offset, color='blue', linewidth=2, alpha=0.7)
    axes[0, 0].set_title(f'Full Image\nStrip: rows {strip_start}-{strip_end}\nRight focus from col {x_offset}')
    axes[0, 0].axis('off')
    
    # Extracted scale region
    axes[0, 1].imshow(scale_region, cmap='gray')
    axes[0, 1].set_title(f'Focused Scale Region\n{scale_region.shape[1]}×{scale_region.shape[0]} pixels')
    axes[0, 1].axis('off')
    
    # All detected segments
    overlay1 = cv2.cvtColor(scale_region, cv2.COLOR_GRAY2RGB)
    
    colors = [(255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 128, 0)]
    
    for group_idx, group in enumerate(groups):
        color = colors[group_idx % len(colors)]
        for seg in group['segments']:
            x1, y1, x2, y2 = [int(x) for x in seg['line']]
            cv2.line(overlay1, (x1, y1), (x2, y2), color, 2)
    
    axes[0, 2].imshow(overlay1)
    axes[0, 2].set_title(f'All Groups\n({len(groups)} found)')
    axes[0, 2].axis('off')
    
    # Best group highlighted
    overlay2 = cv2.cvtColor(scale_region, cv2.COLOR_GRAY2RGB)
    
    if best_group:
        # Draw segments
        for seg in best_group['segments']:
            x1, y1, x2, y2 = [int(x) for x in seg['line']]
            cv2.line(overlay2, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Draw total span
        y_avg = int(best_group['y'])
        cv2.line(overlay2, 
                (int(best_group['start_x']), y_avg), 
                (int(best_group['end_x']), y_avg), 
                (255, 0, 0), 2)
        
        # Label
        mid_x = int((best_group['start_x'] + best_group['end_x']) / 2)
        cv2.putText(overlay2, f"{best_group['total_span']:.0f}px", 
                   (mid_x-30, y_avg-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    axes[1, 0].imshow(overlay2)
    axes[1, 0].set_title('Best Group' if best_group else 'No Best Group')
    axes[1, 0].axis('off')
    
    # Scale region analysis
    row_means = np.mean(scale_region, axis=1)
    axes[1, 1].plot(row_means)
    axes[1, 1].set_title('Row Mean Intensities')
    axes[1, 1].set_xlabel('Row')
    axes[1, 1].set_ylabel('Mean Intensity')
    axes[1, 1].grid(True)
    
    # Results summary
    axes[1, 2].axis('off')
    axes[1, 2].set_title('Results Summary')
    
    if best_group:
        manual_scale = 0.444
        calc_scale = 400.0 / best_group['total_span']
        error = abs(calc_scale - manual_scale) / manual_scale * 100
        
        summary = f"""SMART DETECTION RESULTS:

Strip Detection:
  Rows: {strip_start} to {strip_end}
  Height: {strip_end - strip_start} pixels

Region Focus:
  Right {((image.shape[1] - x_offset) / image.shape[1] * 100):.0f}% of image
  Size: {scale_region.shape[1]}×{scale_region.shape[0]}

Best Scale Bar:
  Span: {best_group['total_span']:.0f} pixels
  Segments: {best_group['count']}
  Calculated: {calc_scale:.4f} μm/px
  Manual: {manual_scale:.4f} μm/px
  Error: {error:.1f}%

Status: {'✅ GOOD' if error < 20 else '⚠️ NEEDS WORK'}"""
    else:
        summary = "No suitable scale bar detected"
    
    axes[1, 2].text(0.05, 0.95, summary, transform=axes[1, 2].transAxes,
                    verticalalignment='top', fontsize=9, fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(project_root / 'analysis_results' / 'smart_scale_detection.png', 
                dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    best_result = test_smart_scale_detection()
    
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("This approach:")
    print("1. Finds the actual scale bar strip automatically")
    print("2. Focuses on the right side where scale bars typically are")
    print("3. Uses erosion to avoid edge artifacts")
    print("4. Should give much more accurate scale bar detection")
    
    if best_result:
        span = best_result['total_span']
        scale = 400.0 / span
        print(f"\nBest detection: {span:.0f} pixels → {scale:.4f} μm/pixel")