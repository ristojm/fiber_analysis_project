#!/usr/bin/env python3
"""
Fix Broken Scale Bar Detection
Handle scale bars that are split by text in the middle
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "modules"))

def detect_broken_scale_bar(scale_region):
    """
    Detect scale bars that are broken by text in the middle.
    
    This handles the common SEM scale bar format:
    ═══════════  400.0μm  ═══════════
    """
    
    print("🔍 DETECTING BROKEN SCALE BAR")
    
    # Find all horizontal line segments
    edges = cv2.Canny(scale_region, 50, 150, apertureSize=3)
    
    # Use HoughLinesP to find line segments
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=20, maxLineGap=10)
    
    horizontal_segments = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Check if line is approximately horizontal
            if abs(y2 - y1) <= 3:  # Allow small vertical deviation
                length = abs(x2 - x1)
                if length > 20:  # Minimum segment length
                    
                    # Calculate line properties
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    horizontal_segments.append({
                        'start_x': min(x1, x2),
                        'end_x': max(x1, x2),
                        'y': int(center_y),
                        'length': length,
                        'center_x': center_x,
                        'line': (x1, y1, x2, y2)
                    })
    
    print(f"Found {len(horizontal_segments)} horizontal line segments")
    
    # Group segments that could be parts of the same scale bar
    scale_bar_candidates = []
    
    # Look for segments on the same horizontal line (same Y coordinate)
    y_tolerance = 5
    
    for i, seg1 in enumerate(horizontal_segments):
        for j, seg2 in enumerate(horizontal_segments[i+1:], i+1):
            
            # Check if segments are on the same horizontal line
            if abs(seg1['y'] - seg2['y']) <= y_tolerance:
                
                # Check if there's a reasonable gap between them (for text)
                gap_start = max(seg1['end_x'], seg2['end_x']) 
                gap_end = min(seg1['start_x'], seg2['start_x'])
                gap_size = abs(gap_end - gap_start)
                
                if 10 <= gap_size <= 200:  # Reasonable gap for text
                    
                    # Calculate total scale bar properties
                    total_start = min(seg1['start_x'], seg2['start_x'])
                    total_end = max(seg1['end_x'], seg2['end_x'])
                    total_length = total_end - total_start
                    
                    scale_bar_candidates.append({
                        'segments': [seg1, seg2],
                        'total_length': total_length,
                        'total_start': total_start,
                        'total_end': total_end,
                        'y': seg1['y'],
                        'gap_size': gap_size,
                        'left_segment': seg1 if seg1['center_x'] < seg2['center_x'] else seg2,
                        'right_segment': seg2 if seg1['center_x'] < seg2['center_x'] else seg1
                    })
    
    print(f"Found {len(scale_bar_candidates)} potential broken scale bars")
    
    for i, candidate in enumerate(scale_bar_candidates):
        print(f"  Candidate {i+1}:")
        print(f"    Total length: {candidate['total_length']:.1f} pixels")
        print(f"    Left segment: {candidate['left_segment']['length']:.1f} pixels")
        print(f"    Right segment: {candidate['right_segment']['length']:.1f} pixels")
        print(f"    Gap size: {candidate['gap_size']:.1f} pixels")
        print(f"    Y position: {candidate['y']}")
    
    # Return the longest candidate (most likely to be the scale bar)
    if scale_bar_candidates:
        best_candidate = max(scale_bar_candidates, key=lambda x: x['total_length'])
        return best_candidate, horizontal_segments
    
    return None, horizontal_segments

def test_broken_scale_detection():
    """Test the broken scale bar detection on your image."""
    
    print("="*60)
    print("TESTING BROKEN SCALE BAR DETECTION")
    print("="*60)
    
    from image_preprocessing import load_image
    from scale_detection import ScaleBarDetector
    
    # Load image and extract scale region
    img_path = project_root / "sample_images" / "hollow_fiber_sample.jpg"
    image = load_image(str(img_path))
    
    detector = ScaleBarDetector()
    scale_region, y_offset = detector.extract_scale_region(image)
    
    print(f"Scale region shape: {scale_region.shape}")
    
    # Test broken scale bar detection
    broken_scale, all_segments = detect_broken_scale_bar(scale_region)
    
    if broken_scale:
        total_length = broken_scale['total_length']
        
        print(f"\n✅ BROKEN SCALE BAR DETECTED:")
        print(f"  Total length: {total_length:.1f} pixels")
        print(f"  Left segment: {broken_scale['left_segment']['length']:.1f} pixels")
        print(f"  Right segment: {broken_scale['right_segment']['length']:.1f} pixels")
        print(f"  Text gap: {broken_scale['gap_size']:.1f} pixels")
        
        # Calculate scale with correct total length
        scale_value = 400.0  # from your scale bar
        calculated_scale = scale_value / total_length
        
        print(f"\n📏 CORRECTED SCALE CALCULATION:")
        print(f"  Scale value: {scale_value} μm")
        print(f"  Total bar length: {total_length:.1f} pixels")
        print(f"  Calculated scale: {calculated_scale:.4f} μm/pixel")
        
        # Compare with your manual measurement
        manual_scale = 0.444
        print(f"  Your manual: {manual_scale:.4f} μm/pixel")
        print(f"  Difference: {abs(calculated_scale - manual_scale):.4f} μm/pixel")
        print(f"  Error: {abs(calculated_scale - manual_scale) / manual_scale * 100:.1f}%")
        
        if abs(calculated_scale - manual_scale) / manual_scale < 0.1:
            print(f"  ✅ EXCELLENT MATCH! (< 10% error)")
        elif abs(calculated_scale - manual_scale) / manual_scale < 0.2:
            print(f"  ✅ GOOD MATCH! (< 20% error)")
        else:
            print(f"  ⚠️ Still some discrepancy - need further investigation")
        
        # Test porosity with corrected scale
        test_corrected_porosity(image, calculated_scale)
        
    else:
        print(f"❌ No broken scale bar detected")
        print(f"Found {len(all_segments)} individual segments:")
        for i, seg in enumerate(all_segments):
            print(f"  Segment {i+1}: {seg['length']:.1f} pixels at y={seg['y']}")
    
    # Visualize the detection
    visualize_broken_scale_detection(scale_region, broken_scale, all_segments)

def visualize_broken_scale_detection(scale_region, broken_scale, all_segments):
    """Visualize the broken scale bar detection."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original scale region
    axes[0].imshow(scale_region, cmap='gray')
    axes[0].set_title('Scale Region')
    axes[0].axis('off')
    
    # All detected segments
    overlay1 = cv2.cvtColor(scale_region, cv2.COLOR_GRAY2RGB)
    
    for i, seg in enumerate(all_segments):
        x1, y1, x2, y2 = seg['line']
        color = (255, 255, 0)  # Yellow for all segments
        cv2.line(overlay1, (x1, y1), (x2, y2), color, 2)
        
        # Add length label
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        cv2.putText(overlay1, f"{seg['length']:.0f}px", (mid_x-20, mid_y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    axes[1].imshow(overlay1)
    axes[1].set_title(f'All Segments ({len(all_segments)} found)')
    axes[1].axis('off')
    
    # Best broken scale bar
    overlay2 = cv2.cvtColor(scale_region, cv2.COLOR_GRAY2RGB)
    
    if broken_scale:
        # Draw the two segments of the broken scale bar
        left_seg = broken_scale['left_segment']
        right_seg = broken_scale['right_segment']
        
        # Left segment in green
        x1, y1, x2, y2 = left_seg['line']
        cv2.line(overlay2, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Right segment in green
        x1, y1, x2, y2 = right_seg['line']
        cv2.line(overlay2, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Draw total span in red
        total_y = broken_scale['y']
        cv2.line(overlay2, 
                (int(broken_scale['total_start']), total_y), 
                (int(broken_scale['total_end']), total_y), 
                (255, 0, 0), 1)
        
        # Add total length label
        mid_x = int((broken_scale['total_start'] + broken_scale['total_end']) / 2)
        cv2.putText(overlay2, f"Total: {broken_scale['total_length']:.0f}px", 
                   (mid_x-40, total_y-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        title = f"Broken Scale Bar\nTotal: {broken_scale['total_length']:.0f}px"
    else:
        title = "No Broken Scale Bar Found"
    
    axes[2].imshow(overlay2)
    axes[2].set_title(title)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(project_root / 'analysis_results' / 'broken_scale_detection.png', 
                dpi=150, bbox_inches='tight')
    plt.show()

def test_corrected_porosity(image, corrected_scale):
    """Test porosity analysis with the corrected scale."""
    
    print(f"\n" + "="*50)
    print("TESTING POROSITY WITH CORRECTED SCALE")
    print("="*50)
    
    try:
        from fiber_type_detection import FiberTypeDetector
        from porosity_analysis import PorosityAnalyzer
        
        # Get fiber mask
        detector = FiberTypeDetector()
        _, _, analysis_data = detector.classify_fiber_type(image)
        fiber_mask = analysis_data.get('fiber_mask')
        
        if fiber_mask is not None:
            # Run porosity with corrected scale
            analyzer = PorosityAnalyzer()
            results = analyzer.analyze_porosity(
                image=image,
                fiber_mask=fiber_mask,
                scale_factor=corrected_scale,
                fiber_type='hollow_fiber'
            )
            
            porosity_metrics = results['porosity_metrics']
            
            print(f"🎯 POROSITY WITH CORRECTED SCALE:")
            print(f"  Scale factor: {corrected_scale:.4f} μm/pixel")
            print(f"  Total Porosity: {porosity_metrics['total_porosity_percent']:.2f}%")
            print(f"  Pore Count: {porosity_metrics['pore_count']}")
            print(f"  Average Pore Size: {porosity_metrics['average_pore_size_um2']:.3f} μm²")
            print(f"  Fiber Area: {porosity_metrics['fiber_area_um2']:.1f} μm²")
            
            # Size distribution
            size_dist = results['size_distribution']
            if size_dist['sizes_um2']:
                stats = size_dist['statistics']
                print(f"  Mean pore diameter: {stats['mean_diameter_um']:.3f} μm")
                print(f"  Median pore diameter: {stats['median_diameter_um']:.3f} μm")
        
    except Exception as e:
        print(f"❌ Porosity test failed: {e}")

if __name__ == "__main__":
    test_broken_scale_detection()
    
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("This should detect the complete scale bar including both segments")
    print("separated by the '400.0μm' text, giving a much longer total length")
    print("that should match your manual measurement much better.")