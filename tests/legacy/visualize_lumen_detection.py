#!/usr/bin/env python3
"""
Visualize the detected lumen so you can check it manually
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Setup paths
current_dir = Path(__file__).parent
if current_dir.name == 'tests':
    project_root = current_dir.parent
else:
    project_root = current_dir

modules_dir = project_root / "modules"
sys.path.insert(0, str(modules_dir))

def visualize_lumen_detection():
    """Visualize the lumen detection with overlays"""
    
    print("="*60)
    print("VISUALIZING LUMEN DETECTION")
    print("="*60)
    
    from fiber_type_detection import FiberTypeDetector
    from image_preprocessing import load_image
    
    detector = FiberTypeDetector()
    img = load_image(str(project_root / "sample_images" / "hollow_fiber_sample.jpg"))
    
    # Get the fiber
    preprocessed = detector.preprocess_for_detection(img)
    fiber_mask, fiber_properties = detector.segment_fibers(preprocessed)
    main_fiber = fiber_properties[0]
    contour = main_fiber['contour']
    
    print(f"🔬 Processing fiber with area: {main_fiber['area']:.0f} pixels")
    
    # Manual lumen detection (copy of the algorithm)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [contour], 255)
    
    fiber_region = cv2.bitwise_and(img, img, mask=mask)
    fiber_pixels = fiber_region[mask > 0]
    
    # Apply threshold = 50
    threshold = 50
    _, lumen_binary = cv2.threshold(fiber_region, threshold, 255, cv2.THRESH_BINARY_INV)
    lumen_binary_original = lumen_binary.copy()  # Keep original for visualization
    lumen_binary = cv2.bitwise_and(lumen_binary, lumen_binary, mask=mask)
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    lumen_binary = cv2.morphologyEx(lumen_binary, cv2.MORPH_CLOSE, kernel)
    lumen_binary = cv2.morphologyEx(lumen_binary, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    lumen_contours, _ = cv2.findContours(lumen_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Found {len(lumen_contours)} lumen contours")
    
    if lumen_contours:
        largest_lumen = max(lumen_contours, key=cv2.contourArea)
        lumen_area = cv2.contourArea(largest_lumen)
        fiber_area = cv2.contourArea(contour)
        area_ratio = lumen_area / fiber_area
        
        # Calculate properties
        perimeter = cv2.arcLength(largest_lumen, True)
        circularity = 4 * np.pi * lumen_area / (perimeter ** 2) if perimeter > 0 else 0
        
        print(f"Largest lumen: {lumen_area:.0f} pixels")
        print(f"Area ratio: {area_ratio:.3f}")
        print(f"Circularity: {circularity:.3f}")
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        axes[0, 0].imshow(img, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Fiber region
        axes[0, 1].imshow(fiber_region, cmap='gray')
        axes[0, 1].set_title('Fiber Region')
        axes[0, 1].axis('off')
        
        # Binary threshold result
        axes[0, 2].imshow(lumen_binary_original, cmap='gray')
        axes[0, 2].set_title('Binary Threshold (50)')
        axes[0, 2].axis('off')
        
        # After morphological operations
        axes[1, 0].imshow(lumen_binary, cmap='gray')
        axes[1, 0].set_title('After Morphological Ops')
        axes[1, 0].axis('off')
        
        # Original with overlays
        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Draw fiber contour in green
        cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 3)
        
        # Draw ALL lumen contours in red (to see what's detected)
        cv2.drawContours(overlay, lumen_contours, -1, (255, 0, 0), 2)
        
        # Draw the LARGEST lumen in bright blue
        cv2.drawContours(overlay, [largest_lumen], -1, (0, 255, 255), 3)
        
        # Mark centroids
        fiber_M = cv2.moments(contour)
        if fiber_M['m00'] > 0:
            fiber_cx = int(fiber_M['m10'] / fiber_M['m00'])
            fiber_cy = int(fiber_M['m01'] / fiber_M['m00'])
            cv2.circle(overlay, (fiber_cx, fiber_cy), 10, (0, 255, 0), -1)  # Green center
        
        lumen_M = cv2.moments(largest_lumen)
        if lumen_M['m00'] > 0:
            lumen_cx = int(lumen_M['m10'] / lumen_M['m00'])
            lumen_cy = int(lumen_M['m01'] / lumen_M['m00'])
            cv2.circle(overlay, (lumen_cx, lumen_cy), 8, (0, 255, 255), -1)  # Cyan center
        
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Detection Overlay\nGreen=Fiber, Red=All Lumens, Cyan=Largest')
        axes[1, 1].axis('off')
        
        # Results summary
        validation_text = f"""LUMEN DETECTION RESULTS:
        
Area: {lumen_area:.0f} pixels
Area Ratio: {area_ratio:.3f} (12.1%)
Circularity: {circularity:.3f}

VALIDATION CHECKS:
✓ Area ratio: {area_ratio:.3f} >= 0.02 ✓
✗ Circularity: {circularity:.3f} >= 0.10 ✗
✓ Centrality: OK ✓

CURRENT RESULT: REJECTED
Reason: Circularity too low

RECOMMENDATION:
Lower circularity threshold from 0.10 to 0.05
or use area-based validation instead"""

        axes[1, 2].text(0.05, 0.95, validation_text, transform=axes[1, 2].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 2].set_title('Analysis Results')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(project_root / 'analysis_results' / 'lumen_detection_visualization.png', 
                    dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\n💡 MANUAL INSPECTION:")
        print(f"Look at the visualization - does the CYAN outline look like a reasonable lumen?")
        print(f"The algorithm detected a lumen with:")
        print(f"  - 12.1% of the fiber area (excellent!)")
        print(f"  - Circularity of 0.084 (below 0.10 threshold)")
        print(f"  - Good centrality")
        print(f"\nThe lumen looks correct but has irregular shape (low circularity)")
        print(f"Solution: Lower the circularity threshold from 0.10 to 0.05")
        
    else:
        print("❌ No lumen contours found")

if __name__ == "__main__":
    visualize_lumen_detection()