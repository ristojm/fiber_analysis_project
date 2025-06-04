#!/usr/bin/env python3
"""
Multi-Segment Scale Bar Detection
Handle scale bars with multiple breaks/segments
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "modules"))

def detect_multi_segment_scale_bar(scale_region):
    """
    Detect scale bars that may have multiple segments/breaks.
    """
    
    print("🔍 DETECTING MULTI-SEGMENT SCALE BAR")
    
    # Enhanced edge detection
    edges = cv2.Canny(scale_region, 30, 150, apertureSize=3)
    
    # Use HoughLinesP with more permissive parameters
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15, minLineLength=15, maxLineGap=20)
    
    horizontal_segments = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Check if line is approximately horizontal (allow more tolerance)
            angle = np.arctan2(abs(y2 - y1), abs(x2 - x1)) * 180 / np.pi
            if angle <= 10:  # Within 10 degrees of horizontal
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if length > 10:  # Minimum segment length
                    
                    horizontal_segments.append({
                        'start_x': min(x1, x2),
                        'end_x': max(x1, x2),
                        'y': (y1 + y2) / 2,
                        'length': length,
                        'center_x': (x1 + x2) / 2,
                        'line': (x1, y1, x2, y2),
                        'angle': angle
                    })
    
    print(f"Found {len(horizontal_segments)} horizontal line segments")
    
    # Group segments by horizontal position (same Y level)
    y_tolerance = 15  # Increased tolerance
    scale_bar_groups = []
    
    # Sort segments by Y position
    horizontal_segments.sort(key=lambda x: x['y'])
    
    i = 0
    while i < len(horizontal_segments):
        current_y = horizontal_segments[i]['y']
        group = []
        
        # Collect all segments at roughly the same Y level
        for j in range(i, len(horizontal_segments)):
            if abs(horizontal_segments[j]['y'] - current_y) <= y_tolerance:
                group.append(horizontal_segments[j])
            else:
                break
        
        if len(group) >= 1:  # At least 1 segment (could be multiple)
            # Sort group by X position
            group.sort(key=lambda x: x['start_x'])
            
            # Calculate total span
            total_start = min(seg['start_x'] for seg in group)
            total_end = max(seg['end_x'] for seg in group)
            total_length = total_end - total_start
            
            # Calculate gaps between segments
            gaps = []
            for k in range(len(group) - 1):
                gap_start = group[k]['end_x']
                gap_end = group[k + 1]['start_x']
                gap_size = gap_end - gap_start
                gaps.append(gap_size)
            
            scale_bar_groups.append({
                'segments': group,
                'segment_count': len(group),
                'total_length': total_length,
                'total_start': total_start,
                'total_end': total_end,
                'average_y': sum(seg['y'] for seg in group) / len(group),
                'gaps': gaps,
                'total_segment_length': sum(seg['length'] for seg in group),
                'total_gap_length': sum(gaps) if gaps else 0
            })
        
        i += len(group)
    
    print(f"Found {len(scale_bar_groups)} potential scale bar groups:")
    
    for i, group in enumerate(scale_bar_groups):
        print(f"  Group {i+1}:")
        print(f"    Segments: {group['segment_count']}")
        print(f"    Total span: {group['total_length']:.1f} pixels")
        print(f"    Segment length sum: {group['total_segment_length']:.1f} pixels")
        print(f"    Gap length sum: {group['total_gap_length']:.1f} pixels")
        print(f"    Y position: {group['average_y']:.1f}")
        
        # Check if this could match your manual measurement
        if group['total_length'] > 0:
            implied_scale = 400.0 / group['total_length']
            manual_scale = 0.444
            error_percent = abs(implied_scale - manual_scale) / manual_scale * 100
            print(f"    Implied scale: {implied_scale:.4f} μm/pixel")
            print(f"    Error vs manual: {error_percent:.1f}%")
            
            if error_percent < 20:
                print(f"    ✅ GOOD MATCH!")
            elif error_percent < 50:
                print(f"    ⚠️ Reasonable match")
            else:
                print(f"    ❌ Poor match")
    
    # Return the group with the best match to manual measurement
    if scale_bar_groups:
        best_group = None
        best_error = float('inf')
        manual_scale = 0.444
        
        for group in scale_bar_groups:
            if group['total_length'] > 50:  # Reasonable minimum length
                implied_scale = 400.0 / group['total_length']
                error = abs(implied_scale - manual_scale) / manual_scale
                
                if error < best_error:
                    best_error = error
                    best_group = group
        
        return best_group, scale_bar_groups, horizontal_segments
    
    return None, scale_bar_groups, horizontal_segments

def test_multi_segment_detection():
    """Test the multi-segment scale bar detection."""
    
    print("="*60)
    print("TESTING MULTI-SEGMENT SCALE BAR DETECTION")
    print("="*60)
    
    from image_preprocessing import load_image
    from scale_detection import ScaleBarDetector
    
    # Load image and extract scale region
    img_path = project_root / "sample_images" / "hollow_fiber_sample.jpg"
    image = load_image(str(img_path))
    
    detector = ScaleBarDetector()
    scale_region, y_offset = detector.extract_scale_region(image)
    
    print(f"Scale region shape: {scale_region.shape}")
    
    # Test multi-segment detection
    best_group, all_groups, all_segments = detect_multi_segment_scale_bar(scale_region)
    
    if best_group:
        total_length = best_group['total_length']
        
        print(f"\n✅ BEST SCALE BAR GROUP DETECTED:")
        print(f"  Total span: {total_length:.1f} pixels")
        print(f"  Number of segments: {best_group['segment_count']}")
        print(f"  Total segment length: {best_group['total_segment_length']:.1f} pixels")
        print(f"  Total gap length: {best_group['total_gap_length']:.1f} pixels")
        
        # Calculate scale with correct total length
        scale_value = 400.0
        calculated_scale = scale_value / total_length
        
        print(f"\n📏 CORRECTED SCALE CALCULATION:")
        print(f"  Scale value: {scale_value} μm")
        print(f"  Total span: {total_length:.1f} pixels")
        print(f"  Calculated scale: {calculated_scale:.4f} μm/pixel")
        
        # Compare with manual measurement
        manual_scale = 0.444
        error_percent = abs(calculated_scale - manual_scale) / manual_scale * 100
        print(f"  Your manual: {manual_scale:.4f} μm/pixel")
        print(f"  Error: {error_percent:.1f}%")
        
        if error_percent < 10:
            print(f"  ✅ EXCELLENT MATCH!")
        elif error_percent < 20:
            print(f"  ✅ GOOD MATCH!")
        elif error_percent < 50:
            print(f"  ⚠️ Reasonable match - may need fine-tuning")
        else:
            print(f"  ❌ Still significant error - investigate further")
            
            # Diagnostic: what length would give the correct scale?
            correct_length = scale_value / manual_scale
            print(f"  For manual scale to be correct, total length should be: {correct_length:.1f} pixels")
    
    else:
        print(f"❌ No suitable scale bar group found")
        if all_groups:
            print(f"Available groups:")
            for i, group in enumerate(all_groups):
                print(f"  Group {i+1}: {group['total_length']:.1f} pixels, {group['segment_count']} segments")
    
    # Visualize the detection
    visualize_multi_segment_detection(scale_region, best_group, all_groups, all_segments)

def visualize_multi_segment_detection(scale_region, best_group, all_groups, all_segments):
    """Visualize the multi-segment scale bar detection."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Original scale region
    axes[0, 0].imshow(scale_region, cmap='gray')
    axes[0, 0].set_title('Original Scale Region')
    axes[0, 0].axis('off')
    
    # All detected segments
    overlay1 = cv2.cvtColor(scale_region, cv2.COLOR_GRAY2RGB)
    
    colors = [(255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 128, 0)]  # Different colors for groups
    
    for group_idx, group in enumerate(all_groups):
        color = colors[group_idx % len(colors)]
        
        for seg in group['segments']:
            x1, y1, x2, y2 = [int(x) for x in seg['line']]
            cv2.line(overlay1, (x1, y1), (x2, y2), color, 2)
    
    axes[0, 1].imshow(overlay1)
    axes[0, 1].set_title(f'All Groups ({len(all_groups)} found)')
    axes[0, 1].axis('off')
    
    # Best group highlighted
    overlay2 = cv2.cvtColor(scale_region, cv2.COLOR_GRAY2RGB)
    
    if best_group:
        # Draw all segments in the best group
        for seg in best_group['segments']:
            x1, y1, x2, y2 = [int(x) for x in seg['line']]
            cv2.line(overlay2, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Draw total span
        total_y = int(best_group['average_y'])
        cv2.line(overlay2, 
                (int(best_group['total_start']), total_y), 
                (int(best_group['total_end']), total_y), 
                (255, 0, 0), 2)
        
        # Add label
        mid_x = int((best_group['total_start'] + best_group['total_end']) / 2)
        cv2.putText(overlay2, f"Total: {best_group['total_length']:.0f}px", 
                   (mid_x-60, total_y-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        title = f"Best Group: {best_group['total_length']:.0f}px span"
    else:
        title = "No Best Group Found"
    
    axes[1, 0].imshow(overlay2)
    axes[1, 0].set_title(title)
    axes[1, 0].axis('off')
    
    # Summary statistics
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Detection Summary')
    
    summary_text = f"DETECTION SUMMARY:\n\n"
    summary_text += f"Total segments found: {len(all_segments)}\n"
    summary_text += f"Groups found: {len(all_groups)}\n\n"
    
    if best_group:
        manual_scale = 0.444
        calc_scale = 400.0 / best_group['total_length']
        error = abs(calc_scale - manual_scale) / manual_scale * 100
        
        summary_text += f"BEST GROUP:\n"
        summary_text += f"  Span: {best_group['total_length']:.1f} px\n"
        summary_text += f"  Segments: {best_group['segment_count']}\n"
        summary_text += f"  Calculated: {calc_scale:.4f} μm/px\n"
        summary_text += f"  Manual: {manual_scale:.4f} μm/px\n"
        summary_text += f"  Error: {error:.1f}%\n\n"
        
        if error < 20:
            summary_text += f"✅ GOOD MATCH!"
        else:
            summary_text += f"⚠️ Needs investigation"
    else:
        summary_text += f"❌ No suitable group found"
    
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                    verticalalignment='top', fontsize=10, fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(project_root / 'analysis_results' / 'multi_segment_scale_detection.png', 
                dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    test_multi_segment_detection()
    
    print(f"\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("This enhanced detection should find all segments that make up")
    print("the complete scale bar, even if there are multiple breaks.")
    print("Look for a group that gives ~900 pixels total span to match")
    print("your manual measurement of 0.444 μm/pixel.")