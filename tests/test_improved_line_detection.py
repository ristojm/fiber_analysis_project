#!/usr/bin/env python3
"""
Test the improved line detection specifically
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Setup paths
project_root = Path(__file__).parent.parent
if (project_root / "modules").exists():
    sys.path.insert(0, str(project_root / "modules"))
else:
    sys.path.insert(0, str(project_root))

def test_line_detection_debug():
    """Test line detection with detailed debugging."""
    
    print("🔍 TESTING IMPROVED LINE DETECTION")
    print("=" * 50)
    
    # Import the cleaned module
    try:
        from scale_detection import ScaleBarDetector
        print("✅ Module imported successfully")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return
    
    # Test with first working image
    sample_dir = project_root / "sample_images"
    if not sample_dir.exists():
        sample_dir = project_root / "../sample_images"
    
    test_image = sample_dir / "hollow_fiber_sample.jpg"
    
    if not test_image.exists():
        print(f"❌ Test image not found: {test_image}")
        return
    
    print(f"📸 Testing with: {test_image.name}")
    
    # Load image
    image = cv2.imread(str(test_image), cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("❌ Could not load image")
        return
    
    print(f"✅ Image loaded: {image.shape}")
    
    # Create detector
    detector = ScaleBarDetector()
    
    # Extract scale region
    scale_region, y_offset = detector.extract_scale_region(image)
    print(f"✅ Scale region extracted: {scale_region.shape}")
    
    # Find text (we know this works)
    text_candidates = detector.find_scale_text(scale_region)
    print(f"✅ Found {len(text_candidates)} text candidates")
    
    if text_candidates:
        best_text = text_candidates[0]
        print(f"   Best text: '{best_text['text']}' -> {best_text['value']} {best_text['unit']}")
        print(f"   Text center: ({best_text['center_x']}, {best_text['center_y']})")
        
        # Now test the improved line detection
        print("\n🔧 Testing line detection...")
        scale_span = detector.find_scale_bar_lines(
            scale_region, 
            best_text['center_x'], 
            best_text['center_y']
        )
        
        if scale_span:
            print("✅ Line detection SUCCESS!")
            print(f"   Span: {scale_span['total_span']:.1f} pixels")
            print(f"   Segments: {scale_span['segment_count']}")
            print(f"   Text position: {scale_span['text_relative_pos']:.3f}")
            print(f"   Score: {scale_span['score']:.3f}")
            
            # Calculate final result
            micrometers_per_pixel = best_text['micrometers'] / scale_span['total_span']
            print(f"   Final scale: {micrometers_per_pixel:.4f} μm/pixel")
            
            # Visualize the result
            visualize_detection(scale_region, best_text, scale_span)
            
        else:
            print("❌ Line detection failed - debugging...")
            debug_line_detection(detector, scale_region, best_text)
    
    else:
        print("❌ No text found - cannot test line detection")

def debug_line_detection(detector, scale_region, text_info):
    """Debug why line detection failed."""
    
    print("\n🐛 DEBUGGING LINE DETECTION")
    print("-" * 30)
    
    height, width = scale_region.shape
    text_center_x = text_info['center_x']
    text_center_y = text_info['center_y']
    
    # Search region
    y_search_radius = 25
    search_y_min = max(0, text_center_y - y_search_radius)
    search_y_max = min(height, text_center_y + y_search_radius)
    search_region = scale_region[search_y_min:search_y_max, :]
    
    print(f"Search region: {search_region.shape}")
    print(f"Text position: ({text_center_x}, {text_center_y})")
    
    # Try different thresholds manually
    thresholds = [240, 220, 200, 180, 160]
    
    for thresh in thresholds:
        _, binary = cv2.threshold(search_region, thresh, 255, cv2.THRESH_BINARY)
        white_pixels = np.sum(binary > 0)
        
        print(f"Threshold {thresh}: {white_pixels} white pixels")
        
        if white_pixels > 100:  # If we have some white pixels
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"  Found {len(contours)} contours")
            
            for i, contour in enumerate(contours[:3]):  # Check first 3
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                print(f"    Contour {i}: size=({w}x{h}), aspect={aspect_ratio:.2f}")
    
    # Show the search region
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(scale_region, cmap='gray')
    plt.title('Full Scale Region')
    plt.axhline(y=text_center_y, color='red', linestyle='--', alpha=0.7)
    plt.axvline(x=text_center_x, color='red', linestyle='--', alpha=0.7)
    
    plt.subplot(1, 3, 2)
    plt.imshow(search_region, cmap='gray')
    plt.title('Search Region')
    
    plt.subplot(1, 3, 3)
    _, binary = cv2.threshold(search_region, 200, 255, cv2.THRESH_BINARY)
    plt.imshow(binary, cmap='gray')
    plt.title('Binary (thresh=200)')
    
    plt.tight_layout()
    plt.show()

def visualize_detection(scale_region, text_info, scale_span):
    """Visualize successful detection."""
    
    plt.figure(figsize=(12, 6))
    
    # Original scale region
    plt.subplot(1, 2, 1)
    plt.imshow(scale_region, cmap='gray')
    plt.title('Scale Region')
    
    # Mark text position
    plt.scatter(text_info['center_x'], text_info['center_y'], 
               color='red', s=100, marker='x', label='Text Center')
    
    # Mark detected span
    plt.axvline(x=scale_span['leftmost_x'], color='green', linestyle='--', alpha=0.7, label='Scale Bar')
    plt.axvline(x=scale_span['rightmost_x'], color='green', linestyle='--', alpha=0.7)
    
    plt.legend()
    
    # Overlay with detections
    plt.subplot(1, 2, 2)
    overlay = cv2.cvtColor(scale_region, cv2.COLOR_GRAY2RGB)
    
    # Draw scale span
    cv2.line(overlay, 
             (scale_span['leftmost_x'], int(scale_span['average_y'])), 
             (scale_span['rightmost_x'], int(scale_span['average_y'])), 
             (0, 255, 0), 3)
    
    # Draw text box
    bbox = text_info['bbox']
    pts = bbox.astype(np.int32)
    cv2.polylines(overlay, [pts], True, (255, 0, 0), 2)
    
    plt.imshow(overlay)
    plt.title('Detection Overlay')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main test function."""
    test_line_detection_debug()

if __name__ == "__main__":
    main()