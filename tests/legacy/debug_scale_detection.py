#!/usr/bin/env python3
"""
Debug Scale Detection - Find out why it's giving wrong values
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "modules"))

def debug_scale_detection():
    """Debug the scale detection step by step."""
    
    print("="*60)
    print("DEBUGGING SCALE DETECTION")
    print("="*60)
    
    from image_preprocessing import load_image
    from scale_detection import ScaleBarDetector
    
    # Load your hollow fiber image
    img_path = project_root / "sample_images" / "hollow_fiber_sample.jpg"
    image = load_image(str(img_path))
    
    print(f"Image loaded: {image.shape}")
    print(f"Your manual measurement: 2.2525 pixels/μm = 0.444 μm/pixel")
    
    # Create detector and run full detection
    detector = ScaleBarDetector()
    
    print(f"\n🔍 STEP 1: Extract scale region")
    scale_region, y_offset = detector.extract_scale_region(image)
    print(f"Scale region shape: {scale_region.shape}")
    print(f"Y offset: {y_offset}")
    
    print(f"\n🔍 STEP 2: Detect scale bar lines")
    bar_candidates = detector.detect_scale_bar_line(scale_region)
    print(f"Found {len(bar_candidates)} bar candidates")
    
    for i, candidate in enumerate(bar_candidates[:3]):  # Show top 3
        print(f"  Candidate {i+1}:")
        print(f"    Length: {candidate['length']} pixels")
        print(f"    Thickness: {candidate['thickness']} pixels")
        print(f"    Aspect ratio: {candidate['aspect_ratio']:.2f}")
        print(f"    Position: {candidate['bbox']}")
    
    print(f"\n🔍 STEP 3: Extract text from scale region")
    
    # Try both OCR methods
    text_lines_tesseract = detector.extract_scale_text_pytesseract(scale_region, bar_candidates)
    text_lines_easyocr = detector.extract_scale_text_easyocr(scale_region, bar_candidates)
    
    print(f"Tesseract OCR results:")
    for line in text_lines_tesseract:
        print(f"  '{line}'")
    
    print(f"EasyOCR results:")  
    for line in text_lines_easyocr:
        print(f"  '{line}'")
    
    # Combine all text
    all_text = text_lines_tesseract + text_lines_easyocr
    
    print(f"\n🔍 STEP 4: Parse scale information")
    scale_info = detector.parse_scale_text(all_text)
    
    if scale_info:
        print(f"Parsed scale info:")
        print(f"  Value: {scale_info['value']}")
        print(f"  Unit: {scale_info['unit']}")
        print(f"  Micrometers: {scale_info['micrometers']}")
        print(f"  Original text: '{scale_info['original_text']}'")
    else:
        print("❌ No scale information could be parsed from text")
    
    print(f"\n🔍 STEP 5: Calculate pixel scale")
    if bar_candidates and scale_info:
        best_bar = bar_candidates[0]
        bar_length_pixels = best_bar['length']
        micrometers_per_pixel = detector.calculate_pixel_scale(scale_info, bar_length_pixels)
        
        print(f"Best bar length: {bar_length_pixels} pixels")
        print(f"Scale value: {scale_info['micrometers']} μm")
        print(f"Calculated: {micrometers_per_pixel:.4f} μm/pixel")
        print(f"Expected: 0.444 μm/pixel")
        print(f"Error factor: {micrometers_per_pixel / 0.444:.1f}x")
        
        # Check if the issue is bar length or scale value
        correct_scale_value = bar_length_pixels * 0.444  # What the scale should say
        print(f"\nDiagnostic:")
        print(f"  If bar length ({bar_length_pixels} px) is correct:")
        print(f"    Scale should read: {correct_scale_value:.1f} μm")
        print(f"    Actually reads: {scale_info['micrometers']:.1f} μm")
        
        if scale_info['micrometers'] > correct_scale_value * 2:
            print(f"  🔍 ISSUE: Scale value seems too large - OCR error?")
        elif bar_length_pixels < 50:
            print(f"  🔍 ISSUE: Bar length seems too small - detection error?")
    
    print(f"\n🔍 STEP 6: Visualize detection")
    visualize_scale_detection(image, scale_region, bar_candidates, all_text, y_offset)

def visualize_scale_detection(image, scale_region, bar_candidates, text_lines, y_offset):
    """Visualize what the scale detection is seeing."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Full image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Full Image')
    
    # Highlight scale region
    axes[0, 0].axhline(y=y_offset, color='red', linewidth=2, alpha=0.7)
    axes[0, 0].text(10, y_offset-20, 'Scale Region', color='red', fontsize=12, weight='bold')
    axes[0, 0].axis('off')
    
    # Scale region only
    axes[0, 1].imshow(scale_region, cmap='gray')
    axes[0, 1].set_title('Scale Region')
    axes[0, 1].axis('off')
    
    # Scale region with detected bars
    if bar_candidates:
        overlay = cv2.cvtColor(scale_region, cv2.COLOR_GRAY2RGB)
        
        for i, candidate in enumerate(bar_candidates[:3]):
            x, y, w, h = candidate['bbox']
            color = (0, 255, 0) if i == 0 else (255, 255, 0)  # Green for best, yellow for others
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            
            # Add label
            cv2.putText(overlay, f"{w}px", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title('Detected Scale Bars')
    else:
        axes[1, 0].imshow(scale_region, cmap='gray')
        axes[1, 0].set_title('No Scale Bars Detected')
    axes[1, 0].axis('off')
    
    # Text results
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Detection Results')
    
    result_text = "DETECTED TEXT:\n"
    for line in text_lines:
        result_text += f"  '{line}'\n"
    
    result_text += f"\nDETECTED BARS:\n"
    for i, candidate in enumerate(bar_candidates[:3]):
        result_text += f"  Bar {i+1}: {candidate['length']} pixels\n"
    
    result_text += f"\nMANUAL MEASUREMENT:\n"
    result_text += f"  Expected: 0.444 μm/pixel\n"
    result_text += f"  Or: 2.253 pixels/μm\n"
    
    axes[1, 1].text(0.05, 0.95, result_text, transform=axes[1, 1].transAxes,
                    verticalalignment='top', fontsize=10, fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(project_root / 'analysis_results' / 'scale_detection_debug.png', 
                dpi=150, bbox_inches='tight')
    plt.show()

def test_manual_scale():
    """Test analysis with your correct manual scale."""
    
    print(f"\n" + "="*60)
    print("TESTING WITH CORRECT MANUAL SCALE")
    print("="*60)
    
    from image_preprocessing import load_image
    from fiber_type_detection import FiberTypeDetector
    from porosity_analysis import PorosityAnalyzer
    
    # Load image and get fiber mask
    img_path = project_root / "sample_images" / "hollow_fiber_sample.jpg"
    image = load_image(str(img_path))
    
    detector = FiberTypeDetector()
    fiber_type, confidence, analysis_data = detector.classify_fiber_type(image)
    fiber_mask = analysis_data.get('fiber_mask')
    
    # Use your correct manual scale
    correct_scale_factor = 0.444  # μm/pixel (your measurement)
    
    print(f"Using manual scale: {correct_scale_factor} μm/pixel")
    print(f"Fiber type: {fiber_type} (confidence: {confidence:.3f})")
    
    # Run porosity analysis with correct scale
    analyzer = PorosityAnalyzer()
    results = analyzer.analyze_porosity(
        image=image,
        fiber_mask=fiber_mask,
        scale_factor=correct_scale_factor,
        fiber_type=fiber_type
    )
    
    porosity_metrics = results['porosity_metrics']
    
    print(f"\n🎯 CORRECTED POROSITY RESULTS:")
    print(f"   Total Porosity: {porosity_metrics['total_porosity_percent']:.2f}%")
    print(f"   Pore Count: {porosity_metrics['pore_count']}")
    print(f"   Average Pore Size: {porosity_metrics['average_pore_size_um2']:.3f} μm²")
    print(f"   Fiber Area: {porosity_metrics['fiber_area_um2']:.1f} μm²")
    print(f"   Pore Density: {porosity_metrics['pore_density_per_mm2']:.1f} pores/mm²")
    
    # Size distribution
    size_dist = results['size_distribution']
    if size_dist['sizes_um2']:
        stats = size_dist['statistics']
        print(f"\n📊 CORRECTED SIZE DISTRIBUTION:")
        print(f"   Mean diameter: {stats['mean_diameter_um']:.3f} μm")
        print(f"   Median diameter: {stats['median_diameter_um']:.3f} μm")
        print(f"   Size range: {min(size_dist['diameters_um']):.3f} - {max(size_dist['diameters_um']):.3f} μm")

if __name__ == "__main__":
    debug_scale_detection()
    test_manual_scale()
    
    print(f"\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Check what text the OCR is reading from the scale bar")
    print("2. Check what bar length is being detected")
    print("3. See if the issue is OCR misreading or bar detection error")
    print("4. Use manual scale for now: 0.444 μm/pixel")