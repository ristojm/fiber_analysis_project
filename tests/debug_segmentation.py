#!/usr/bin/env python3
"""
Debug the fiber segmentation to see why it's detecting multiple small fibers
instead of one large fiber.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Setup paths
project_root = Path(__file__).parent.parent
modules_dir = project_root / "modules"
sys.path.insert(0, str(modules_dir))

def debug_segmentation():
    """Debug the segmentation process step by step."""
    
    print("="*60)
    print("DEBUGGING FIBER SEGMENTATION")
    print("="*60)
    
    # Load the problematic hollow fiber image
    sample_dir = project_root / "sample_images"
    hollow_fiber_path = sample_dir / "hollow_fiber_sample.jpg"
    
    if not hollow_fiber_path.exists():
        print("❌ Hollow fiber image not found")
        return
    
    from image_preprocessing import load_image
    from fiber_type_detection import FiberTypeDetector
    
    # Load and preprocess
    img = load_image(str(hollow_fiber_path))
    print(f"✓ Original image: {img.shape}")
    
    # Create detector
    detector = FiberTypeDetector()
    
    # Step 1: Preprocessing
    preprocessed = detector.preprocess_for_detection(img)
    print(f"✓ Preprocessed: {preprocessed.shape}")
    
    # Step 2: Segmentation
    fiber_mask, fiber_properties = detector.segment_fibers(preprocessed)
    print(f"✓ Fibers detected: {len(fiber_properties)}")
    
    # Step 3: Analyze each detected "fiber"
    print(f"\n📊 Detected Fiber Analysis:")
    for i, props in enumerate(fiber_properties):
        area = props['area']
        bbox = props['bounding_rect']
        circularity = props['circularity']
        
        print(f"  Fiber {i+1}:")
        print(f"    Area: {area:.0f} pixels")
        print(f"    Bounding box: {bbox}")
        print(f"    Circularity: {circularity:.3f}")
        
        # Calculate approximate diameter
        diameter_pixels = np.sqrt(area / np.pi) * 2
        print(f"    Approx diameter: {diameter_pixels:.1f} pixels")
    
    # Step 4: Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Original image
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Preprocessed image
    axes[1].imshow(preprocessed, cmap='gray')
    axes[1].set_title('Preprocessed')
    axes[1].axis('off')
    
    # Binary segmentation
    axes[2].imshow(fiber_mask, cmap='gray')
    axes[2].set_title('Fiber Mask')
    axes[2].axis('off')
    
    # Overlay with contours
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Draw all detected contours
    for i, props in enumerate(fiber_properties):
        contour = props['contour']
        color = (255, 0, 0) if props['area'] < 10000 else (0, 255, 0)  # Red for small, green for large
        cv2.drawContours(overlay, [contour], -1, color, 2)
        
        # Add labels
        M = cv2.moments(contour)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.putText(overlay, f'{i+1}', (cx-10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    axes[3].imshow(overlay)
    axes[3].set_title('Detected Contours\n(Red=Small, Green=Large)')
    axes[3].axis('off')
    
    # Show the largest fiber only
    if fiber_properties:
        largest_fiber = max(fiber_properties, key=lambda x: x['area'])
        largest_mask = np.zeros_like(fiber_mask)
        cv2.fillPoly(largest_mask, [largest_fiber['contour']], 255)
        
        axes[4].imshow(largest_mask, cmap='gray')
        axes[4].set_title(f'Largest Fiber Only\n(Area: {largest_fiber["area"]:.0f})')
        axes[4].axis('off')
        
        # Test lumen detection on largest fiber
        has_lumen, lumen_props = detector.detect_lumen(img, largest_fiber['contour'])
        
        info_text = f"Largest Fiber Analysis:\n"
        info_text += f"Area: {largest_fiber['area']:.0f} pixels\n"
        info_text += f"Circularity: {largest_fiber['circularity']:.3f}\n"
        info_text += f"Has lumen: {has_lumen}\n"
        
        if has_lumen:
            info_text += f"Lumen area ratio: {lumen_props.get('area_ratio', 0):.3f}\n"
            info_text += f"Lumen circularity: {lumen_props.get('circularity', 0):.3f}\n"
        
        axes[5].text(0.05, 0.95, info_text, transform=axes[5].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[5].set_title('Analysis Results')
        axes[5].axis('off')
    
    plt.tight_layout()
    plt.savefig(project_root / 'analysis_results' / 'segmentation_debug.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Step 5: Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if len(fiber_properties) > 3:
        print("  🔧 Too many small fibers detected - segmentation needs improvement")
        print("  📝 Consider:")
        print("     - Larger minimum area threshold")
        print("     - Better morphological operations")
        print("     - Different thresholding method")
    
    # Check if the largest fiber should be hollow
    if fiber_properties:
        largest = max(fiber_properties, key=lambda x: x['area'])
        if largest['area'] > 100000:  # Large fiber
            has_lumen, _ = detector.detect_lumen(img, largest['contour'])
            if not has_lumen:
                print("  🔍 Large fiber not detecting lumen - check lumen detection algorithm")
                print("  📝 Possible issues:")
                print("     - Threshold too strict")
                print("     - Lumen not dark enough")
                print("     - Morphological operations removing lumen")

if __name__ == "__main__":
    debug_segmentation()