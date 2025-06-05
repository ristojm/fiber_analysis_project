#!/usr/bin/env python3
"""
Debug why lumen validation is failing
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Setup paths
current_dir = Path(__file__).parent
if current_dir.name == 'tests':
    project_root = current_dir.parent
else:
    project_root = current_dir

modules_dir = project_root / "modules"
sys.path.insert(0, str(modules_dir))

def debug_lumen_validation():
    """Debug step by step why lumen validation fails"""
    
    print("="*60)
    print("DEBUGGING LUMEN VALIDATION")
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
    
    print(f"🔬 MANUAL LUMEN DETECTION DEBUG:")
    print(f"Fiber area: {main_fiber['area']:.0f} pixels")
    
    # Step 1: Create fiber mask
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [contour], 255)
    
    # Step 2: Extract fiber region
    fiber_region = cv2.bitwise_and(img, img, mask=mask)
    fiber_pixels = fiber_region[mask > 0]
    
    print(f"Fiber pixels: min={fiber_pixels.min()}, max={fiber_pixels.max()}, mean={fiber_pixels.mean():.1f}")
    
    # Step 3: Apply threshold = 50
    threshold = 50
    _, lumen_binary = cv2.threshold(fiber_region, threshold, 255, cv2.THRESH_BINARY_INV)
    lumen_binary = cv2.bitwise_and(lumen_binary, lumen_binary, mask=mask)
    
    # Step 4: Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    lumen_binary = cv2.morphologyEx(lumen_binary, cv2.MORPH_CLOSE, kernel)
    lumen_binary = cv2.morphologyEx(lumen_binary, cv2.MORPH_OPEN, kernel)
    
    # Step 5: Find contours
    lumen_contours, _ = cv2.findContours(lumen_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Lumen contours found: {len(lumen_contours)}")
    
    if not lumen_contours:
        print("❌ No lumen contours found after morphological operations")
        return
    
    # Step 6: Analyze largest lumen
    largest_lumen = max(lumen_contours, key=cv2.contourArea)
    lumen_area = cv2.contourArea(largest_lumen)
    fiber_area = cv2.contourArea(contour)
    area_ratio = lumen_area / fiber_area
    
    print(f"\n📊 LUMEN PROPERTIES:")
    print(f"Lumen area: {lumen_area:.0f} pixels")
    print(f"Area ratio: {area_ratio:.3f}")
    
    # Step 7: Calculate lumen properties (like the algorithm does)
    perimeter = cv2.arcLength(largest_lumen, True)
    circularity = 4 * np.pi * lumen_area / (perimeter ** 2) if perimeter > 0 else 0
    
    # Centroid
    M = cv2.moments(largest_lumen)
    if M['m00'] > 0:
        lumen_cx = M['m10'] / M['m00']
        lumen_cy = M['m01'] / M['m00']
    else:
        lumen_cx = lumen_cy = 0
    
    print(f"Lumen circularity: {circularity:.3f}")
    print(f"Lumen centroid: ({lumen_cx:.1f}, {lumen_cy:.1f})")
    
    # Step 8: Check validation criteria
    print(f"\n🔍 VALIDATION CHECKS:")
    
    # Check 1: Area ratio
    min_area_ratio = 0.02  # 2%
    max_area_ratio = 0.6   # 60%
    area_ok = min_area_ratio <= area_ratio <= max_area_ratio
    print(f"Area ratio check: {area_ratio:.3f} between {min_area_ratio} and {max_area_ratio} = {area_ok}")
    
    # Check 2: Circularity
    min_circularity = 0.1  # Very lenient
    circularity_ok = circularity >= min_circularity
    print(f"Circularity check: {circularity:.3f} >= {min_circularity} = {circularity_ok}")
    
    # Check 3: Centrality
    fiber_moments = cv2.moments(contour)
    if fiber_moments['m00'] > 0:
        fiber_cx = fiber_moments['m10'] / fiber_moments['m00']
        fiber_cy = fiber_moments['m01'] / fiber_moments['m00']
        
        distance = np.sqrt((fiber_cx - lumen_cx)**2 + (fiber_cy - lumen_cy)**2)
        fiber_radius = np.sqrt(fiber_area / np.pi)
        
        max_distance = 0.5 * fiber_radius  # 50% tolerance
        centrality_ok = distance <= max_distance
        
        print(f"Centrality check: distance={distance:.1f} <= {max_distance:.1f} = {centrality_ok}")
    else:
        centrality_ok = False
        print(f"Centrality check: Could not calculate fiber centroid = False")
    
    # Final validation
    final_valid = area_ok and circularity_ok and centrality_ok
    print(f"\n🎯 FINAL VALIDATION: {final_valid}")
    
    if not final_valid:
        print("❌ Lumen rejected by validation")
        if not area_ok:
            print(f"  - Area ratio {area_ratio:.3f} out of range")
        if not circularity_ok:
            print(f"  - Circularity {circularity:.3f} too low")
        if not centrality_ok:
            print(f"  - Lumen too far from center")
    else:
        print("✅ Lumen should be accepted!")
        print("🐛 There might be a bug in the validation method")

if __name__ == "__main__":
    debug_lumen_validation()