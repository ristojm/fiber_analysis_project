#!/usr/bin/env python3
"""
Test improved lumen detection for the hollow fiber.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Setup paths - adjusted for tests/ folder location
project_root = Path(__file__).parent.parent  # Go up one level from tests/ to project root
modules_dir = project_root / "modules"
sys.path.insert(0, str(modules_dir))

def improved_lumen_detection(image, fiber_contour):
    """
    Improved lumen detection specifically for your SEM images.
    """
    print("🔍 Testing improved lumen detection...")
    
    # Create mask for fiber region
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [fiber_contour], 255)
    
    # Extract fiber region
    fiber_region = cv2.bitwise_and(image, image, mask=mask)
    fiber_pixels = fiber_region[mask > 0]
    
    if len(fiber_pixels) == 0:
        return False, {}, None
    
    print(f"  Fiber pixel stats:")
    print(f"    Min: {fiber_pixels.min()}")
    print(f"    Max: {fiber_pixels.max()}")
    print(f"    Mean: {fiber_pixels.mean():.1f}")
    print(f"    Std: {fiber_pixels.std():.1f}")
    
    # Try multiple approaches
    lumen_candidates = []
    
    # Method 1: Very low threshold (bottom 10%)
    threshold1 = np.percentile(fiber_pixels, 10)
    print(f"  Threshold 1 (10th percentile): {threshold1}")
    
    _, lumen_binary1 = cv2.threshold(fiber_region, threshold1, 255, cv2.THRESH_BINARY_INV)
    lumen_binary1 = cv2.bitwise_and(lumen_binary1, lumen_binary1, mask=mask)
    
    # Method 2: Fixed low threshold
    threshold2 = 50  # Very dark regions
    print(f"  Threshold 2 (fixed low): {threshold2}")
    
    _, lumen_binary2 = cv2.threshold(fiber_region, threshold2, 255, cv2.THRESH_BINARY_INV)
    lumen_binary2 = cv2.bitwise_and(lumen_binary2, lumen_binary2, mask=mask)
    
    # Method 3: Adaptive threshold within fiber
    print(f"  Threshold 3 (adaptive):")
    
    # Create a smaller mask for adaptive thresholding
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    eroded_mask = cv2.erode(mask, kernel, iterations=2)
    
    # Apply adaptive threshold only to eroded region
    fiber_region_eroded = cv2.bitwise_and(image, image, mask=eroded_mask)
    
    # Use a more aggressive adaptive threshold
    lumen_binary3 = cv2.adaptiveThreshold(fiber_region_eroded, 255, 
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 21, 10)
    lumen_binary3 = cv2.bitwise_and(lumen_binary3, lumen_binary3, mask=eroded_mask)
    
    # Combine all methods
    methods = [
        (lumen_binary1, "10th percentile"),
        (lumen_binary2, "Fixed low (50)"),
        (lumen_binary3, "Adaptive")
    ]
    
    # Visualize all methods
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original fiber region
    axes[0, 0].imshow(fiber_region, cmap='gray')
    axes[0, 0].set_title('Fiber Region')
    axes[0, 0].axis('off')
    
    # Show each method
    for i, (binary, name) in enumerate(methods):
        row = (i + 1) // 3
        col = (i + 1) % 3
        
        axes[row, col].imshow(binary, cmap='gray')
        axes[row, col].set_title(f'Method {i+1}: {name}')
        axes[row, col].axis('off')
        
        # Find contours and analyze
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            fiber_area = cv2.contourArea(fiber_contour)
            area_ratio = area / fiber_area if fiber_area > 0 else 0
            
            print(f"    {name}: Lumen area = {area:.0f}, ratio = {area_ratio:.3f}")
            
            if area > 1000 and area_ratio > 0.01:  # Reasonable lumen
                lumen_candidates.append({
                    'contour': largest_contour,
                    'area': area,
                    'area_ratio': area_ratio,
                    'method': name,
                    'binary': binary
                })
    
    # Histogram of fiber pixels
    axes[0, 1].hist(fiber_pixels, bins=50, alpha=0.7, color='blue')
    axes[0, 1].axvline(threshold1, color='red', linestyle='--', label=f'10th percentile: {threshold1}')
    axes[0, 1].axvline(threshold2, color='green', linestyle='--', label=f'Fixed: {threshold2}')
    axes[0, 1].set_title('Pixel Intensity Distribution')
    axes[0, 1].legend()
    axes[0, 1].set_xlabel('Pixel Intensity')
    axes[0, 1].set_ylabel('Count')
    
    # Best result
    if lumen_candidates:
        best = max(lumen_candidates, key=lambda x: x['area_ratio'])
        axes[0, 2].imshow(best['binary'], cmap='gray')
        axes[0, 2].set_title(f'Best Result: {best["method"]}\nRatio: {best["area_ratio"]:.3f}')
        axes[0, 2].axis('off')
        
        print(f"  🎯 Best lumen detection: {best['method']} (ratio: {best['area_ratio']:.3f})")
        
        plt.tight_layout()
        plt.savefig(project_root / 'analysis_results' / 'lumen_detection_debug.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return True, best, best['binary']
    else:
        axes[0, 2].text(0.5, 0.5, 'No lumen\ndetected', ha='center', va='center', 
                       transform=axes[0, 2].transAxes, fontsize=16)
        axes[0, 2].set_title('No Lumen Found')
        axes[0, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(project_root / 'analysis_results' / 'lumen_detection_debug.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("  ❌ No suitable lumen candidates found")
        return False, {}, None

def test_lumen_fix():
    """Test the improved lumen detection."""
    
    print("="*60)
    print("TESTING IMPROVED LUMEN DETECTION")
    print("="*60)
    
    # Load the hollow fiber image
    sample_dir = project_root / "sample_images"
    hollow_fiber_path = sample_dir / "hollow_fiber_sample.jpg"
    
    from image_preprocessing import load_image
    from fiber_type_detection import FiberTypeDetector
    
    img = load_image(str(hollow_fiber_path))
    detector = FiberTypeDetector()
    
    # Get the fiber segmentation
    fiber_mask, fiber_properties = detector.segment_fibers(detector.preprocess_for_detection(img))
    
    # Find the largest fiber (the main one)
    largest_fiber = max(fiber_properties, key=lambda x: x['area'])
    print(f"📏 Largest fiber area: {largest_fiber['area']:.0f} pixels")
    
    # Test original lumen detection
    print(f"\n🔴 Original lumen detection:")
    has_lumen_old, lumen_props_old = detector.detect_lumen(img, largest_fiber['contour'])
    print(f"  Result: {has_lumen_old}")
    if has_lumen_old:
        print(f"  Area ratio: {lumen_props_old.get('area_ratio', 0):.3f}")
    
    # Test improved lumen detection
    print(f"\n🟢 Improved lumen detection:")
    has_lumen_new, lumen_props_new, lumen_binary = improved_lumen_detection(img, largest_fiber['contour'])
    print(f"  Result: {has_lumen_new}")
    if has_lumen_new:
        print(f"  Area ratio: {lumen_props_new.get('area_ratio', 0):.3f}")
        print(f"  Method used: {lumen_props_new.get('method', 'Unknown')}")

if __name__ == "__main__":
    test_lumen_fix()