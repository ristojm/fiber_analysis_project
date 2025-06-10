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
from typing import Dict

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

# ===== MODULE AVAILABILITY TRACKING =====
MODULES_LOADED = {}
POROSITY_TYPE = None

# Track which modules are available (same pattern as multiprocessing workflow)
try:
    from modules.scale_detection import ScaleBarDetector
    from modules.fiber_type_detection import FiberTypeDetector
    from modules.image_preprocessing import load_image
    MODULES_LOADED['core'] = True
    print("✅ Core modules loaded")
except ImportError as e:
    print(f"❌ Core modules failed: {e}")
    MODULES_LOADED['core'] = False

# Porosity analysis
try:
    from modules.porosity_analysis import PorosityAnalyzer, analyze_fiber_porosity
    MODULES_LOADED['porosity_analysis'] = True
    POROSITY_TYPE = 'fast_refined'
    print("✅ Porosity analysis loaded")
except ImportError:
    try:
        from modules.porosity_analysis import quick_porosity_check
        MODULES_LOADED['porosity_analysis'] = True
        POROSITY_TYPE = 'basic'
        print("✅ Basic porosity analysis loaded")
    except ImportError:
        MODULES_LOADED['porosity_analysis'] = False
        POROSITY_TYPE = None
        print("❌ No porosity analysis available")

# Crumbly detection
try:
    from modules.crumbly_detection import CrumblyDetector
    MODULES_LOADED['crumbly_detection'] = True
    print("✅ Crumbly detection loaded")
except ImportError as e:
    print(f"❌ Crumbly detection failed: {e}")
    MODULES_LOADED['crumbly_detection'] = False

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
    """Same preprocessing as multiprocessing script with scale bar removal."""
    try:
        if HAS_ENHANCED_PREPROCESSING:
            temp_processed = preprocess_for_analysis(image, silent=True)
        else:
            enhanced = enhance_contrast(image, method='clahe')
            denoised = denoise_image(enhanced, method='bilateral')
            temp_processed = normalize_image(denoised)
        
        # Remove scale bar region
        from modules.image_preprocessing import remove_scale_bar_region
        main_region, scale_bar_region = remove_scale_bar_region(temp_processed)
        return main_region
        
    except Exception as e:
        print(f"⚠️ Preprocessing error: {e}")
        # Fallback: manual crop of bottom 15% where scale bars typically are
        height = image.shape[0]
        crop_height = int(height * 0.95)  # Remove bottom 5%
        return image[:crop_height, :]

def create_optimal_fiber_mask(image: np.ndarray, fiber_analysis_data: dict, debug: bool = False) -> np.ndarray:
    """
    Create the optimal fiber mask that:
    1. Uses the precise fiber contour (no background noise)
    2. Excludes the lumen for hollow fibers
    3. Only includes the actual fiber wall for analysis
    """
    
    if debug:
        print(f"   🔧 Creating optimal fiber mask...")
    
    if not fiber_analysis_data:
        return np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Get individual results to find the best fiber
    individual_results = fiber_analysis_data.get('individual_results', [])
    
    if individual_results and len(individual_results) > 0:
        # Find the largest fiber
        largest_fiber_result = max(individual_results, key=lambda x: x['fiber_properties']['area'])
        
        # Get fiber contour
        fiber_props = largest_fiber_result.get('fiber_properties', {})
        fiber_contour = fiber_props.get('contour')
        
        if fiber_contour is not None:
            # Create base mask from contour (precise boundary, no background noise)
            fiber_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(fiber_mask, [fiber_contour], 255)
            
            # Check if this fiber has a lumen
            has_lumen = largest_fiber_result.get('has_lumen', False)
            
            if has_lumen:
                if debug:
                    print(f"   🕳️ Hollow fiber detected, excluding lumen...")
                
                # Get lumen properties if available
                lumen_props = largest_fiber_result.get('lumen_properties', {})
                lumen_contour = lumen_props.get('contour')
                
                if lumen_contour is not None:
                    # Method 1: Use detected lumen contour
                    lumen_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(lumen_mask, [lumen_contour], 255)
                    # Remove lumen from fiber mask
                    fiber_mask[lumen_mask > 0] = 0
                    
                    if debug:
                        lumen_area = np.sum(lumen_mask > 0)
                        print(f"   ✅ Lumen excluded using contour: {lumen_area:,} pixels")
                else:
                    # Method 2: Detect lumen using intensity
                    fiber_region = image.copy()
                    fiber_region[fiber_mask == 0] = 255  # Set non-fiber to white
                    
                    # Find very dark regions within fiber (likely lumen)
                    if np.sum(fiber_mask > 0) > 0:
                        lumen_threshold = np.percentile(fiber_region[fiber_mask > 0], 10)  # Bottom 10%
                        potential_lumen = (fiber_region < lumen_threshold) & (fiber_mask > 0)
                        
                        # Clean up lumen detection with morphology
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                        potential_lumen = cv2.morphologyEx(potential_lumen.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
                        potential_lumen = cv2.morphologyEx(potential_lumen, cv2.MORPH_OPEN, kernel)
                        
                        # Remove detected lumen from fiber mask
                        fiber_mask[potential_lumen > 0] = 0
                        
                        if debug:
                            lumen_area = np.sum(potential_lumen > 0)
                            print(f"   ✅ Lumen excluded using intensity: {lumen_area:,} pixels")
            
            final_area = np.sum(fiber_mask > 0)
            if debug:
                print(f"   ✅ Final analysis mask: {final_area:,} pixels")
            
            return fiber_mask
    
    # Fallback to general mask
    if debug:
        print(f"   ⚠️ Using fallback general mask")
    
    fiber_mask = fiber_analysis_data.get('fiber_mask')
    if fiber_mask is not None and isinstance(fiber_mask, np.ndarray):
        if fiber_mask.dtype != np.uint8:
            fiber_mask = (fiber_mask > 0).astype(np.uint8) * 255
        return fiber_mask
    else:
        return np.zeros(image.shape[:2], dtype=np.uint8)

def show_step_by_step_analysis(image_path: str):
    """Show visual analysis step by step with popup windows - FIXED ORDER."""
    
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
    
    # Step 2: Scale detection FIRST (needs original image with scale bar)
    print("📏 Step 2: Scale detection...")
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
    
    # Step 3: Preprocessing with scale bar removal
    print("🔧 Step 3: Preprocessing with scale bar removal...")
    preprocessed = preprocess_image_for_debug(original_image)
    
    print(f"   Original shape: {original_image.shape}")
    print(f"   Preprocessed shape: {preprocessed.shape}")
    
    plt.subplot(2, 3, 2)
    plt.imshow(preprocessed, cmap='gray')
    plt.title('Preprocessed Image\n(Scale bar removed)')
    plt.axis('off')
    
    # Step 4: Fiber detection (on clean preprocessed image)
    print("🧬 Step 4: Fiber detection...")
    try:
        fiber_detector = FiberTypeDetector()
        fiber_type, confidence, fiber_analysis_data = fiber_detector.classify_fiber_type(preprocessed)
        
        # Extract fiber mask using the proper extraction logic
        fiber_mask = create_optimal_fiber_mask(preprocessed, fiber_analysis_data, debug=True)
        
        mask_area = np.sum(fiber_mask > 0)
        print(f"   ✅ Fiber mask extracted: {mask_area:,} pixels")
        print(f"   Fiber type: {fiber_type} (confidence: {confidence:.3f})")
        print(f"   Mask shape: {fiber_mask.shape} (should match preprocessed)")
        
        # Create visualization overlay (dimensions now guaranteed to match)
        overlay = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2RGB)
        overlay[fiber_mask > 0] = [0, 255, 0]  # Green for fiber
        
        plt.subplot(2, 3, 3)
        plt.imshow(overlay)
        plt.title(f'Fiber Segmentation\n{mask_area:,} pixels')
        plt.axis('off')
        
    except Exception as e:
        print(f"   ❌ Fiber detection error: {e}")
        fiber_mask = np.zeros(preprocessed.shape[:2], dtype=np.uint8)
        fiber_type = "error"
        confidence = 0.0
        mask_area = 0
    
    # Step 5: Porosity analysis (on clean image and matching mask)
    print("🔬 Step 5: Porosity analysis...")
    porosity_result = None
    try:
        if MODULES_LOADED.get('porosity_analysis', False):
            if POROSITY_TYPE == 'fast_refined':
                porosity_result = analyze_fiber_porosity(preprocessed, fiber_mask, scale_factor)
            elif POROSITY_TYPE == 'basic':
                porosity_result = quick_porosity_check(preprocessed, fiber_mask, scale_factor)
            else:
                # Fallback
                porosity_analyzer = PorosityAnalyzer()
                porosity_result = porosity_analyzer.analyze_fiber_porosity(
                    preprocessed, fiber_mask, scale_factor
                )
            
            if porosity_result:
                porosity_metrics = porosity_result.get('porosity_metrics', {})
                total_porosity = porosity_metrics.get('total_porosity_percent', 0)
                pore_count = porosity_metrics.get('pore_count', 0)
                avg_pore_size = porosity_metrics.get('average_pore_size_um2', 0)
                
                print(f"   ✅ Porosity: {total_porosity:.2f}%")
                print(f"   ✅ Pore count: {pore_count}")
                print(f"   ✅ Avg pore size: {avg_pore_size:.1f} μm²")
        else:
            print("   ⚠️ Porosity analysis not available")
    except Exception as e:
        print(f"   ❌ Porosity analysis error: {e}")
                
    # Show porosity visualization
    plt.subplot(2, 3, 4)
    if porosity_result and 'visualization_data' in porosity_result:
        pore_overlay = porosity_result['visualization_data'].get('pore_overlay')
        if pore_overlay is not None:
            plt.imshow(pore_overlay)
            plt.title('Detected Pores')
        else:
            plt.imshow(preprocessed, cmap='gray')
            plt.title('Porosity Analysis\n(No visualization data)')
    else:
        # Simple pore visualization fallback
        fiber_region = preprocessed.copy()
        fiber_region[fiber_mask == 0] = 255  # Set non-fiber to white
        
        if np.sum(fiber_mask > 0) > 0:
            pore_threshold = np.percentile(fiber_region[fiber_mask > 0], 30)
            pore_candidates = (fiber_region < pore_threshold) & (fiber_mask > 0)
            plt.imshow(pore_candidates, cmap='hot')
            plt.title(f'Pore Candidates\n(Threshold: {pore_threshold:.0f})')
        else:
            plt.imshow(preprocessed, cmap='gray')
            plt.title('No Fiber Detected')
    plt.axis('off')
    
    # Step 6: Crumbly texture analysis
    print("🧩 Step 6: Crumbly texture analysis...")
    try:
        if MODULES_LOADED.get('crumbly_detection', False):
            crumbly_detector = CrumblyDetector(porosity_aware=True)
            fiber_mask_bool = fiber_mask > 127
            
            # Pass porosity data
            porosity_data = {'porosity_metrics': porosity_result.get('porosity_metrics', {})} if porosity_result else {}
            
            crumbly_result = crumbly_detector.analyze_crumbly_texture(
                preprocessed, fiber_mask_bool, None, scale_factor, 
                debug=False, porosity_data=porosity_data
            )
            
            if crumbly_result and 'classification' in crumbly_result:
                classification = crumbly_result['classification']
                confidence_score = crumbly_result.get('confidence', 0.0)
                crumbly_score = crumbly_result.get('crumbly_score', 0.5)
                
                print(f"   ✅ Classification: {classification}")
                print(f"   ✅ Confidence: {confidence_score:.3f}")
                print(f"   ✅ Crumbly score: {crumbly_score:.3f}")
                
                # Show texture analysis result on FIBER REGION ONLY
                plt.subplot(2, 3, 5)
                
                # Create texture visualization
                texture_vis = preprocessed.copy()
                texture_vis[fiber_mask == 0] = 128  # Set background to gray
                
                # If crumbly result has visualization data, use it
                if 'visualization_data' in crumbly_result:
                    vis_data = crumbly_result['visualization_data']
                    if 'texture_overlay' in vis_data:
                        plt.imshow(vis_data['texture_overlay'])
                    else:
                        plt.imshow(texture_vis, cmap='gray')
                else:
                    # Create basic texture visualization
                    # Highlight texture features within fiber mask
                    edges = cv2.Canny(texture_vis, 50, 150)
                    edges_masked = edges.copy()
                    edges_masked[fiber_mask == 0] = 0
                    
                    # Create colored overlay
                    texture_overlay = cv2.cvtColor(texture_vis, cv2.COLOR_GRAY2RGB)
                    texture_overlay[edges_masked > 0] = [255, 255, 0]  # Yellow for edges
                    plt.imshow(texture_overlay)
                
                plt.title(f'Texture Analysis\n{classification} ({confidence_score:.2f})\nFiber region only')
                plt.axis('off')
                
                # Show summary
                plt.subplot(2, 3, 6)
                plt.text(0.1, 0.8, f"Image: {Path(image_path).name}", fontsize=10, weight='bold')
                plt.text(0.1, 0.7, f"Scale: {scale_factor:.4f} μm/pixel", fontsize=9)
                plt.text(0.1, 0.6, f"Fiber type: {fiber_type}", fontsize=9)
                plt.text(0.1, 0.5, f"Confidence: {confidence:.3f}", fontsize=9)
                if porosity_result:
                    plt.text(0.1, 0.4, f"Porosity: {total_porosity:.2f}%", fontsize=9)
                    plt.text(0.1, 0.3, f"Pores: {pore_count}", fontsize=9)
                plt.text(0.1, 0.2, f"Texture: {classification}", fontsize=9)
                plt.text(0.1, 0.1, f"Crumbly score: {crumbly_score:.3f}", fontsize=9)
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                plt.axis('off')
                plt.title('Analysis Summary')
            else:
                print("   ❌ Crumbly analysis failed")
        else:
            print("   ⚠️ Crumbly detection not available")
    except Exception as e:
        print(f"   ❌ Crumbly analysis error: {e}")
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 60)
    print("✅ Visual analysis complete!")

# Add this function to your visual_debug_test.py file (before show_step_by_step_analysis)

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
            pore_candidates_clean = cv2.morphologyEx(pore_candidates.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            pore_candidates_clean = cv2.morphologyEx(pore_candidates_clean, cv2.MORPH_CLOSE, kernel)
            
            # Find contours and count ACTUAL pores
            contours, _ = cv2.findContours(pore_candidates_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_pores = [c for c in contours if cv2.contourArea(c) > 10]  # Min 10 pixels
            
            # Create visualization showing ONLY the valid pores, not all dark regions
            pore_visualization = np.zeros_like(fiber_region)
            for contour in valid_pores:
                cv2.fillPoly(pore_visualization, [contour], 255)
            
            plt.imshow(pore_visualization, cmap='hot')
            plt.title(f'{percentile}th Percentile\nThreshold: {threshold:.0f}\nValid Pores: {len(valid_pores)}')
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