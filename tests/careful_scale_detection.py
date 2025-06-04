#!/usr/bin/env python3
"""
Careful Scale Detection - Better strip detection with minimal erosion
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "modules"))

def analyze_bottom_region(image):
    """
    Carefully analyze the bottom region to find the scale bar strip.
    """
    
    print("🔍 ANALYZING BOTTOM REGION")
    
    height, width = image.shape
    
    # Look at bottom 40% to be safe
    bottom_start = int(height * 0.6)
    bottom_region = image[bottom_start:, :]
    
    # Calculate row-wise statistics
    row_means = np.mean(bottom_region, axis=1)
    row_mins = np.min(bottom_region, axis=1)
    row_maxs = np.max(bottom_region, axis=1)
    row_stds = np.std(bottom_region, axis=1)
    
    print(f"  Bottom region: rows {bottom_start} to {height}")
    print(f"  Region shape: {bottom_region.shape}")
    print(f"  Mean intensity range: {row_means.min():.1f} to {row_means.max():.1f}")
    
    # Find the darkest region (likely the scale bar strip)
    darkest_threshold = np.percentile(row_means, 20)  # Bottom 20% of intensities
    dark_rows = row_means < darkest_threshold
    
    print(f"  Darkest threshold: {darkest_threshold:.1f}")
    print(f"  Dark rows found: {np.sum(dark_rows)}")
    
    # Find contiguous dark regions
    dark_regions = []
    in_region = False
    region_start = 0
    
    for i, is_dark in enumerate(dark_rows):
        if is_dark and not in_region:
            region_start = i
            in_region = True
        elif not is_dark and in_region:
            region_end = i
            region_length = region_end - region_start
            if region_length > 10:  # At least 10 rows
                dark_regions.append({
                    'start': region_start + bottom_start,  # Convert to full image coords
                    'end': region_end + bottom_start,
                    'length': region_length,
                    'mean_intensity': np.mean(row_means[region_start:region_end])
                })
            in_region = False
    
    # Handle case where dark region extends to bottom
    if in_region:
        region_end = len(dark_rows)
        region_length = region_end - region_start
        if region_length > 10:
            dark_regions.append({
                'start': region_start + bottom_start,
                'end': region_end + bottom_start,
                'length': region_length,
                'mean_intensity': np.mean(row_means[region_start:])
            })
    
    print(f"  Found {len(dark_regions)} dark regions:")
    for i, region in enumerate(dark_regions):
        print(f"    Region {i+1}: rows {region['start']}-{region['end']} ({region['length']} px)")
        print(f"               mean intensity: {region['mean_intensity']:.1f}")
    
    return dark_regions, bottom_region, row_means, bottom_start

def select_scale_strip(dark_regions, image_height):
    """
    Select the most likely scale bar strip from detected dark regions.
    """
    
    print(f"🎯 SELECTING SCALE STRIP")
    
    if not dark_regions:
        print(f"  No dark regions found, using conservative bottom area")
        # Fallback: bottom 10% with minimal erosion
        strip_start = int(image_height * 0.9)
        strip_end = image_height - 5  # Just 5 pixel erosion
        return strip_start, strip_end
    
    # Score regions based on:
    # 1. Position (closer to bottom = higher score)
    # 2. Size (reasonable size = higher score)
    # 3. Darkness (darker = higher score)
    
    best_region = None
    best_score = -1
    
    for region in dark_regions:
        # Position score (closer to bottom is better)
        position_score = region['start'] / image_height
        
        # Size score (prefer regions 20-100 pixels tall)
        ideal_size = 50
        size_score = 1.0 - abs(region['length'] - ideal_size) / ideal_size
        size_score = max(0, size_score)
        
        # Darkness score (darker is better, but not too dark which might be artifacts)
        darkness_score = 1.0 - (region['mean_intensity'] / 255.0)
        
        total_score = position_score * 0.4 + size_score * 0.3 + darkness_score * 0.3
        
        print(f"  Region {region['start']}-{region['end']}:")
        print(f"    Position score: {position_score:.3f}")
        print(f"    Size score: {size_score:.3f}")
        print(f"    Darkness score: {darkness_score:.3f}")
        print(f"    Total score: {total_score:.3f}")
        
        if total_score > best_score:
            best_score = total_score
            best_region = region
    
    if best_region:
        # Use minimal erosion (just 2-3 pixels)
        erosion = min(3, best_region['length'] // 10)
        strip_start = best_region['start'] + erosion
        strip_end = best_region['end'] - erosion
        
        print(f"  Selected region: {best_region['start']}-{best_region['end']}")
        print(f"  After minimal erosion ({erosion}px): {strip_start}-{strip_end}")
        
        return strip_start, strip_end
    else:
        # Fallback
        print(f"  No suitable region, using fallback")
        strip_start = int(image_height * 0.9)
        strip_end = image_height - 5
        return strip_start, strip_end

def extract_scale_region_careful(image, strip_start, strip_end):
    """
    Extract scale region with focus on right side but less aggressive.
    """
    
    print(f"📏 EXTRACTING SCALE REGION")
    
    height, width = image.shape
    
    # Focus on right side but less aggressively (right 50% instead of 60%)
    right_focus_fraction = 0.5
    left_boundary = int(width * (1 - right_focus_fraction))
    
    scale_region = image[strip_start:strip_end, left_boundary:]
    
    print(f"  Strip region: rows {strip_start} to {strip_end} ({strip_end - strip_start} px high)")
    print(f"  Right focus: columns {left_boundary} to {width} (right {right_focus_fraction*100:.0f}%)")
    print(f"  Final scale region: {scale_region.shape[1]} × {scale_region.shape[0]} pixels")
    
    return scale_region, left_boundary

def detect_scale_bar_careful(scale_region):
    """
    Careful scale bar detection with conservative parameters.
    """
    
    print(f"🔍 DETECTING SCALE BAR (CAREFUL MODE)")
    
    # Very gentle preprocessing
    smoothed = cv2.GaussianBlur(scale_region, (3, 3), 0)
    
    # Multiple edge detection strategies
    
    # Strategy 1: Conservative Canny
    edges1 = cv2.Canny(smoothed, 30, 120, apertureSize=3)
    
    # Strategy 2: Aggressive Canny for faint lines
    edges2 = cv2.Canny(smoothed, 10, 80, apertureSize=3)
    
    # Strategy 3: Sobel edge detection
    sobel_x = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
    edges3 = np.sqrt(sobel_x**2 + sobel_y**2).astype(np.uint8)
    edges3 = (edges3 > 30).astype(np.uint8) * 255
    
    all_detections = []
    
    # Test each edge detection method
    for i, (edges, name) in enumerate([(edges1, "Conservative"), (edges2, "Aggressive"), (edges3, "Sobel")]):
        
        # Morphological operations to connect lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        edges_processed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges_processed, 1, np.pi/180, 
                               threshold=8,       # Very low threshold
                               minLineLength=15,  # Short minimum
                               maxLineGap=40)     # Large gap tolerance
        
        segments = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Check horizontality
                if abs(x2 - x1) < 5:  # Vertical line, skip
                    continue
                    
                angle = np.arctan2(abs(y2 - y1), abs(x2 - x1)) * 180 / np.pi
                if angle <= 20:  # More lenient angle tolerance
                    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    
                    if length >= 10:
                        segments.append({
                            'start_x': min(x1, x2),
                            'end_x': max(x1, x2),
                            'y': (y1 + y2) / 2,
                            'length': length,
                            'line': (x1, y1, x2, y2),
                            'method': name
                        })
        
        print(f"  {name} method: {len(segments)} segments")
        all_detections.extend(segments)
    
    print(f"  Total segments found: {len(all_detections)}")
    
    # Group segments by Y level
    groups = []
    y_tolerance = 15
    
    all_detections.sort(key=lambda x: x['y'])
    
    i = 0
    while i < len(all_detections):
        current_y = all_detections[i]['y']
        group = []
        
        for j in range(i, len(all_detections)):
            if abs(all_detections[j]['y'] - current_y) <= y_tolerance:
                group.append(all_detections[j])
            else:
                break
        
        if len(group) >= 1:
            group.sort(key=lambda x: x['start_x'])
            
            total_start = min(seg['start_x'] for seg in group)
            total_end = max(seg['end_x'] for seg in group)
            total_span = total_end - total_start
            
            # Filter out very short spans
            if total_span >= 30:
                groups.append({
                    'segments': group,
                    'count': len(group),
                    'total_span': total_span,
                    'start_x': total_start,
                    'end_x': total_end,
                    'y': sum(seg['y'] for seg in group) / len(group),
                    'methods': list(set(seg['method'] for seg in group))
                })
        
        i += len(group)
    
    print(f"  Grouped into {len(groups)} potential scale bars")
    
    for i, group in enumerate(groups):
        print(f"    Group {i+1}: {group['total_span']:.1f}px span, {group['count']} segments")
        print(f"              methods: {group['methods']}")
    
    return groups

def test_careful_detection():
    """Test the careful scale detection."""
    
    print("="*60)
    print("CAREFUL SCALE BAR DETECTION")
    print("="*60)
    
    from image_preprocessing import load_image
    
    # Load image
    img_path = project_root / "sample_images" / "hollow_fiber_sample.jpg"
    image = load_image(str(img_path))
    
    print(f"Image shape: {image.shape}")
    
    # Step 1: Analyze bottom region
    dark_regions, bottom_region, row_means, bottom_start = analyze_bottom_region(image)
    
    # Step 2: Select scale strip
    strip_start, strip_end = select_scale_strip(dark_regions, image.shape[0])
    
    # Step 3: Extract scale region
    scale_region, x_offset = extract_scale_region_careful(image, strip_start, strip_end)
    
    # Step 4: Detect scale bar
    groups = detect_scale_bar_careful(scale_region)
    
    # Step 5: Evaluate results
    print(f"\n📊 EVALUATION:")
    
    manual_scale = 0.444
    best_group = None
    best_error = float('inf')
    
    for group in groups:
        span = group['total_span']
        if span > 0:
            implied_scale = 400.0 / span
            error = abs(implied_scale - manual_scale) / manual_scale
            
            print(f"  {span:.1f}px span → {implied_scale:.4f} μm/px (error: {error*100:.1f}%)")
            
            if error < best_error:
                best_error = error
                best_group = group
    
    if best_group:
        print(f"\n✅ BEST MATCH:")
        print(f"  Span: {best_group['total_span']:.1f} pixels")
        print(f"  Scale: {400.0 / best_group['total_span']:.4f} μm/pixel")
        print(f"  Error: {best_error*100:.1f}%")
        print(f"  Methods: {best_group['methods']}")
    
    # Visualize
    visualize_careful_detection(image, dark_regions, bottom_region, row_means, 
                               bottom_start, strip_start, strip_end, 
                               scale_region, x_offset, groups, best_group)

def visualize_careful_detection(image, dark_regions, bottom_region, row_means, 
                               bottom_start, strip_start, strip_end,
                               scale_region, x_offset, groups, best_group):
    """Visualize the careful detection process."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Full image with detected regions
    axes[0, 0].imshow(image, cmap='gray')
    
    # Show all dark regions
    for region in dark_regions:
        axes[0, 0].axhline(y=region['start'], color='yellow', linewidth=1, alpha=0.7)
        axes[0, 0].axhline(y=region['end'], color='yellow', linewidth=1, alpha=0.7)
    
    # Show selected strip
    axes[0, 0].axhline(y=strip_start, color='red', linewidth=2)
    axes[0, 0].axhline(y=strip_end, color='red', linewidth=2)
    axes[0, 0].axvline(x=x_offset, color='blue', linewidth=2, alpha=0.7)
    
    axes[0, 0].set_title(f'Strip Detection\nSelected: {strip_start}-{strip_end}')
    axes[0, 0].axis('off')
    
    # Bottom region analysis
    axes[0, 1].plot(row_means)
    axes[0, 1].set_title('Row Mean Intensities\n(Bottom Region)')
    axes[0, 1].set_xlabel('Row (relative to bottom region)')
    axes[0, 1].set_ylabel('Mean Intensity')
    axes[0, 1].grid(True)
    
    # Selected strip intensity profile
    if strip_start < strip_end:
        strip_profile = np.mean(image[strip_start:strip_end, :], axis=1)
        axes[0, 2].plot(strip_profile)
        axes[0, 2].set_title(f'Selected Strip Profile\n({strip_end - strip_start} rows)')
        axes[0, 2].set_xlabel('Row')
        axes[0, 2].set_ylabel('Mean Intensity')
        axes[0, 2].grid(True)
    
    # Scale region
    axes[1, 0].imshow(scale_region, cmap='gray')
    axes[1, 0].set_title(f'Scale Region\n{scale_region.shape[1]}×{scale_region.shape[0]}')
    axes[1, 0].axis('off')
    
    # Detection results
    if groups:
        overlay = cv2.cvtColor(scale_region, cv2.COLOR_GRAY2RGB)
        
        colors = [(255, 255, 0), (0, 255, 255), (255, 0, 255)]
        
        for i, group in enumerate(groups):
            color = colors[i % len(colors)]
            
            for seg in group['segments']:
                x1, y1, x2, y2 = [int(x) for x in seg['line']]
                cv2.line(overlay, (x1, y1), (x2, y2), color, 2)
            
            # Draw span
            y_avg = int(group['y'])
            cv2.line(overlay, (int(group['start_x']), y_avg), 
                    (int(group['end_x']), y_avg), (255, 0, 0), 1)
        
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title(f'Detected Groups ({len(groups)})')
    else:
        axes[1, 1].imshow(scale_region, cmap='gray')
        axes[1, 1].set_title('No Groups Detected')
    axes[1, 1].axis('off')
    
    # Results summary
    axes[1, 2].axis('off')
    
    if best_group:
        manual_scale = 0.444
        calc_scale = 400.0 / best_group['total_span']
        error = abs(calc_scale - manual_scale) / manual_scale * 100
        
        summary = f"""CAREFUL DETECTION RESULTS:

Dark Regions Found: {len(dark_regions)}
Selected Strip: {strip_start}-{strip_end}
Strip Height: {strip_end - strip_start} px

Best Scale Bar:
  Span: {best_group['total_span']:.0f} pixels
  Segments: {best_group['count']}
  Methods: {', '.join(best_group['methods'])}
  
Scale Calculation:
  Calculated: {calc_scale:.4f} μm/px
  Manual: {manual_scale:.4f} μm/px
  Error: {error:.1f}%

Status: {'✅ GOOD' if error < 20 else '⚠️ CHECK'}"""
    else:
        summary = "No scale bar detected"
    
    axes[1, 2].text(0.05, 0.95, summary, transform=axes[1, 2].transAxes,
                    verticalalignment='top', fontsize=9, fontfamily='monospace')
    axes[1, 2].set_title('Summary')
    
    plt.tight_layout()
    plt.savefig(project_root / 'analysis_results' / 'careful_scale_detection.png', 
                dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    test_careful_detection()
    
    print(f"\n" + "="*60)
    print("This careful approach should:")
    print("1. Better identify the actual scale bar strip")
    print("2. Use minimal erosion to preserve the scale bar")
    print("3. Try multiple detection methods")
    print("4. Give more accurate scale measurements")