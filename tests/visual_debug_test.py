#!/usr/bin/env python3
"""
Visual Debug Test Script for SEM Fiber Analysis
Shows popup windows at each processing step to visually examine results.

Usage:
    python visual_debug_test.py path/to/image.jpg
"""

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Add project paths (same as multiprocessing script)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "modules"))

print(f"🔧 Visual Debug Test Script")
print(f"   Project root: {project_root}")

# Import modules (same as multiprocessing script)
try:
    from modules.scale_detection import ScaleBarDetector
    from modules.fiber_type_detection import FiberTypeDetector
    from modules.image_preprocessing import load_image
    from modules.porosity_analysis import PorosityAnalyzer, analyze_fiber_porosity
    from modules.crumbly_detection import CrumblyDetector
    print("✅ All modules imported successfully")
except ImportError as e:
    print(f"❌ Module import failed: {e}")
    sys.exit(1)

# Try enhanced preprocessing
try:
    from modules.image_preprocessing import preprocess_for_analysis
    HAS_ENHANCED_PREPROCESSING = True
    print("✅ Enhanced preprocessing available")
except ImportError:
    from modules.image_preprocessing import enhance_contrast, denoise_image, normalize_image
    HAS_ENHANCED_PREPROCESSING = False
    print("⚠️ Using fallback preprocessing")

def preprocess_image_for_debug(image):
    """Same preprocessing as multiprocessing script."""
    try:
        if HAS_ENHANCED_PREPROCESSING:
            return preprocess_for_analysis(image, silent=True)
        else:
            enhanced = enhance_contrast(image, method='clahe')
            denoised = denoise_image(enhanced, method='bilateral')
            normalized = normalize_image(denoised)
            return normalized
    except Exception as e:
        print(f"⚠️ Preprocessing error: {e}")
        return image

def show_step_by_step_analysis(image_path: str):
    """Show visual analysis step by step with popup windows."""
    
    print(f"\n🔍 VISUAL ANALYSIS: {Path(image_path).name}")
    print("=" * 60)
    
    # Step 1: Load and show original image
    print("📸 Step 1: Loading original image...")
    original_image = load_image(image_path)
    if original_image is None:
        print("❌ Failed to load image")
        return
    
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title(f'Original Image\n{Path(image_path).name}')
    plt.axis('off')
    
    # Step 2: Preprocessing
    print("🔧 Step 2: Preprocessing...")
    preprocessed = preprocess_image_for_debug(original_image)
    
    plt.subplot(2, 3, 2)
    plt.imshow(preprocessed, cmap='gray')
    plt.title('Preprocessed Image')
    plt.axis('off')
    
    # Step 3: Scale detection
    print("📏 Step 3: Scale detection...")
    scale_factor = 1.0
    try:
        scale_detector = ScaleBarDetector(use_enhanced_detection=True)
        scale_result = scale_detector.detect_scale_bar(
            original_image, debug=False, save_debug_image=False, output_dir=None
        )
        if scale_result and scale_result.get('scale_detected', False):
            scale_factor = scale_result.get('micrometers_per_pixel', 1.0)
            print(f"   ✅ Scale detected: {scale_factor:.4f} μm/pixel")
        else:
            print(f"   ⚠️ Scale detection failed, using 1.0 μm/pixel")
    except Exception as e:
        print(f"   ❌ Scale detection error: {e}")
    
    # Step 4: Fiber detection
    print("🧬 Step 4: Fiber detection...")
    try:
        fiber_detector = FiberTypeDetector()
        fiber_type, confidence, fiber_analysis_data = fiber_detector.classify_fiber_type(preprocessed)
        
        # Extract fiber mask (same as multiprocessing script)
        fiber_mask = fiber_analysis_data.get('fiber_mask') if fiber_analysis_data else None
        
        if fiber_mask is not None and isinstance(fiber_mask, np.ndarray):
            if fiber_mask.dtype != np.uint8:
                fiber_mask = (fiber_mask > 0).astype(np.uint8) * 255
            
            mask_area = np.sum(fiber_mask > 0)
            print(f"   ✅ Fiber mask extracted: {mask_area:,} pixels")
            print(f"   Fiber type: {fiber_type} (confidence: {confidence:.3f})")
        else:
            print(f"   ❌ No valid fiber mask found")
            fiber_mask = np.zeros(preprocessed.shape[:2], dtype=np.uint8)
            
    except Exception as e:
        print(f"   ❌ Fiber detection failed: {e}")
        fiber_mask = np.zeros(preprocessed.shape[:2], dtype=np.uint8)
    
    # Show fiber mask
    plt.subplot(2, 3, 3)
    # Create colored overlay
    overlay = np.zeros((*original_image.shape, 3), dtype=np.uint8)
    overlay[fiber_mask > 0] = [0, 255, 0]  # Green for fiber
    
    # Blend with original
    if len(original_image.shape) == 2:
        orig_color = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    else:
        orig_color = original_image
    
    blended = cv2.addWeighted(orig_color, 0.7, overlay, 0.3, 0)
    plt.imshow(blended)
    plt.title(f'Fiber Segmentation\n{np.sum(fiber_mask > 0):,} pixels')
    plt.axis('off')
    
    # Step 5: Porosity analysis
    print("💧 Step 5: Porosity analysis...")
    porosity_result = None
    try:
        porosity_result = analyze_fiber_porosity(preprocessed, fiber_mask, scale_factor)
        
        if porosity_result and porosity_result.get('success', False):
            porosity_metrics = porosity_result.get('porosity_metrics', {})
            total_porosity = porosity_metrics.get('total_porosity_percent', 0)
            pore_count = porosity_metrics.get('pore_count', 0)
            avg_pore_size = porosity_metrics.get('average_pore_size_um2', 0)
            
            print(f"   ✅ Porosity analysis successful:")
            print(f"     Total porosity: {total_porosity:.2f}%")
            print(f"     Pore count: {pore_count}")
            print(f"     Average pore size: {avg_pore_size:.2f} μm²")
        else:
            print(f"   ❌ Porosity analysis failed")
            
    except Exception as e:
        print(f"   ❌ Porosity analysis error: {e}")
    
    # Show porosity visualization if available
    plt.subplot(2, 3, 4)
    if porosity_result and 'visualization_data' in porosity_result:
        # If porosity result has visualization data, use it
        pore_overlay = porosity_result['visualization_data'].get('pore_overlay')
        if pore_overlay is not None:
            plt.imshow(pore_overlay)
            plt.title('Detected Pores')
        else:
            plt.imshow(preprocessed, cmap='gray')
            plt.title('Porosity Analysis\n(No visualization data)')
    else:
        # Create simple pore visualization
        # Find dark regions in fiber mask as potential pores
        fiber_region = preprocessed.copy()
        fiber_region[fiber_mask == 0] = 255  # Set non-fiber to white
        
        # Simple threshold for pores
        pore_threshold = np.percentile(fiber_region[fiber_mask > 0], 30)
        pore_candidates = (fiber_region < pore_threshold) & (fiber_mask > 0)
        
        # Show pore candidates
        plt.imshow(pore_candidates, cmap='hot')
        plt.title(f'Pore Candidates\n(Threshold: {pore_threshold:.0f})')
    plt.axis('off')
    
    # Step 6: Crumbly texture analysis
    print("🧩 Step 6: Crumbly texture analysis...")
    try:
        crumbly_detector = CrumblyDetector(porosity_aware=True)
        fiber_mask_bool = fiber_mask > 127
        
        # Pass porosity data (same as multiprocessing script)
        porosity_data = {'porosity_metrics': porosity_result.get('porosity_metrics', {})} if porosity_result else {}
        
        crumbly_result = crumbly_detector.analyze_crumbly_texture(
            preprocessed, fiber_mask_bool, None, scale_factor, 
            debug=False, porosity_data=porosity_data
        )
        
        if crumbly_result and 'classification' in crumbly_result:
            classification = crumbly_result['classification']
            confidence = crumbly_result.get('confidence', 0.0)
            crumbly_score = crumbly_result.get('crumbly_score', 0.5)
            
            # Get evidence
            porous_evidence = crumbly_result.get('porous_evidence', 0)
            crumbly_evidence = crumbly_result.get('crumbly_evidence', 0)
            intermediate_evidence = crumbly_result.get('intermediate_evidence', 0)
            
            print(f"   ✅ Crumbly analysis result:")
            print(f"     Classification: {classification}")
            print(f"     Confidence: {confidence:.3f}")
            print(f"     Crumbly score: {crumbly_score:.3f}")
            print(f"     Evidence - Porous: {porous_evidence:.3f}, Crumbly: {crumbly_evidence:.3f}, Intermediate: {intermediate_evidence:.3f}")
        else:
            print(f"   ❌ Crumbly analysis failed")
            classification = "error"
            confidence = 0.0
            
    except Exception as e:
        print(f"   ❌ Crumbly analysis error: {e}")
        classification = "error"
        confidence = 0.0
    
    # Show final classification
    plt.subplot(2, 3, 5)
    plt.imshow(preprocessed, cmap='gray')
    plt.title(f'Final Classification\n{classification} (conf: {confidence:.2f})')
    plt.axis('off')
    
    # Summary text
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    summary_text = f"ANALYSIS SUMMARY\n\n"
    summary_text += f"Image: {Path(image_path).name}\n"
    summary_text += f"Scale: {scale_factor:.4f} μm/pixel\n"
    summary_text += f"Fiber mask: {np.sum(fiber_mask > 0):,} pixels\n\n"
    
    if porosity_result:
        porosity_metrics = porosity_result.get('porosity_metrics', {})
        summary_text += f"POROSITY:\n"
        summary_text += f"  Total: {porosity_metrics.get('total_porosity_percent', 0):.2f}%\n"
        summary_text += f"  Pores: {porosity_metrics.get('pore_count', 0)}\n"
        summary_text += f"  Avg size: {porosity_metrics.get('average_pore_size_um2', 0):.1f} μm²\n\n"
    
    summary_text += f"CLASSIFICATION:\n"
    summary_text += f"  Result: {classification}\n"
    summary_text += f"  Confidence: {confidence:.3f}\n"
    
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue'))
    
    plt.tight_layout()
    plt.suptitle(f'Visual Analysis: {Path(image_path).name}', fontsize=14)
    
    print(f"\n📊 Analysis complete! Close the window to continue...")
    plt.show()

def create_detailed_porosity_view(image_path: str):
    """Create detailed porosity analysis view."""
    
    print(f"\n🔍 DETAILED POROSITY VIEW: {Path(image_path).name}")
    print("=" * 60)
    
    # Load and preprocess
    original_image = load_image(image_path)
    preprocessed = preprocess_image_for_debug(original_image)
    
    # Get fiber mask
    fiber_detector = FiberTypeDetector()
    fiber_type, confidence, fiber_analysis_data = fiber_detector.classify_fiber_type(preprocessed)
    fiber_mask = fiber_analysis_data.get('fiber_mask') if fiber_analysis_data else None
    
    if fiber_mask is not None:
        if fiber_mask.dtype != np.uint8:
            fiber_mask = (fiber_mask > 0).astype(np.uint8) * 255
    else:
        fiber_mask = np.zeros(preprocessed.shape[:2], dtype=np.uint8)
    
    # Scale detection
    scale_detector = ScaleBarDetector(use_enhanced_detection=True)
    scale_result = scale_detector.detect_scale_bar(original_image)
    scale_factor = scale_result.get('micrometers_per_pixel', 1.0) if scale_result.get('scale_detected', False) else 1.0
    
    # Detailed porosity analysis
    print("💧 Running detailed porosity analysis...")
    
    plt.figure(figsize=(16, 10))
    
    # Original image
    plt.subplot(2, 4, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Preprocessed image
    plt.subplot(2, 4, 2)
    plt.imshow(preprocessed, cmap='gray')
    plt.title('Preprocessed')
    plt.axis('off')
    
    # Fiber mask
    plt.subplot(2, 4, 3)
    plt.imshow(fiber_mask, cmap='gray')
    plt.title(f'Fiber Mask\n{np.sum(fiber_mask > 0):,} pixels')
    plt.axis('off')
    
    # Fiber region only
    plt.subplot(2, 4, 4)
    fiber_region = preprocessed.copy()
    fiber_region[fiber_mask == 0] = 0
    plt.imshow(fiber_region, cmap='gray')
    plt.title('Fiber Region Only')
    plt.axis('off')
    
    # Different pore detection thresholds
    fiber_pixels = fiber_region[fiber_mask > 0]
    
    if len(fiber_pixels) > 0:
        for i, percentile in enumerate([20, 30, 40, 50]):
            plt.subplot(2, 4, 5 + i)
            
            threshold = np.percentile(fiber_pixels, percentile)
            pore_candidates = (fiber_region < threshold) & (fiber_mask > 0)
            
            # Clean up with morphology
            kernel = np.ones((3, 3), np.uint8)
            pore_candidates = cv2.morphologyEx(pore_candidates.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            pore_candidates = cv2.morphologyEx(pore_candidates, cv2.MORPH_CLOSE, kernel)
            
            # Find contours and count
            contours, _ = cv2.findContours(pore_candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_pores = [c for c in contours if cv2.contourArea(c) > 10]  # Min 10 pixels
            
            plt.imshow(pore_candidates, cmap='hot')
            plt.title(f'{percentile}th Percentile\nThreshold: {threshold:.0f}\nPores: {len(valid_pores)}')
            plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle(f'Detailed Porosity Analysis: {Path(image_path).name}', fontsize=14)
    plt.show()

def main():
    """Main function for visual debug test."""
    
    parser = argparse.ArgumentParser(description='Visual Debug Test for SEM Fiber Analysis')
    parser.add_argument('image_path', help='Path to image file to analyze')
    parser.add_argument('--detailed-porosity', action='store_true', 
                       help='Show detailed porosity analysis view')
    
    args = parser.parse_args()
    
    image_path = Path(args.image_path)
    
    if not image_path.exists():
        print(f"❌ Image file not found: {image_path}")
        return 1
    
    print(f"🔍 Starting visual debug analysis...")
    print(f"   Image: {image_path}")
    
    try:
        # Main step-by-step analysis
        show_step_by_step_analysis(str(image_path))
        
        # Optional detailed porosity view
        if args.detailed_porosity:
            create_detailed_porosity_view(str(image_path))
            
        print(f"✅ Visual analysis complete!")
        return 0
        
    except KeyboardInterrupt:
        print(f"\n🛑 Analysis interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())