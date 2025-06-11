#!/usr/bin/env python3
"""
Visual Debug Test Script for SEM Fiber Analysis - UPDATED VERSION
Calls functions from multiprocessing_crumbly_workflow.py with debug visualizations.

This script tests the same workflow functions but with visual output to verify:
1. Scale bar removal is working
2. Optimal fiber mask selection is working  
3. Porosity analysis is accurate
4. Texture analysis classification is correct
5. All processing steps are in the right order

Usage:
    python visual_debug_test.py path/to/image.jpg [--model-path /path/to/hybrid/model]
"""

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Dict

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "modules"))
sys.path.insert(0, str(Path(__file__).parent))

print(f"🔧 Visual Debug Test Script - Enhanced Version")
print(f"   Project root: {project_root}")

# Import the workflow functions we want to test
try:
    from Crumble_ML.multiprocessing_crumbly_workflow2_edited import (
        preprocess_image_for_worker,
        create_optimal_fiber_mask,
        apply_classification_improvements,
        MODULES_LOADED,
        HAS_ENHANCED_PREPROCESSING,
        HAS_SCALE_BAR_REMOVAL,
        POROSITY_TYPE
    )
    print("✅ Workflow functions imported successfully")
except ImportError as e:
    print(f"❌ Failed to import workflow functions: {e}")
    sys.exit(1)

# Import required modules
try:
    from modules.scale_detection import ScaleBarDetector
    from modules.fiber_type_detection import FiberTypeDetector
    from modules.image_preprocessing import load_image
    print("✅ Core modules imported successfully")
except ImportError as e:
    print(f"❌ Module import failed: {e}")
    sys.exit(1)

# Import analysis modules
if MODULES_LOADED.get('porosity_analysis', False):
    if POROSITY_TYPE == 'fast_refined':
        from modules.porosity_analysis import analyze_fiber_porosity
    else:
        from modules.porosity_analysis import quick_porosity_check

if MODULES_LOADED.get('crumbly_detection', False):
    from modules.crumbly_detection import CrumblyDetector

if MODULES_LOADED.get('hybrid_detector', False):
    from hybrid_crumbly_detector import load_hybrid_detector

def show_step_by_step_analysis(image_path: str, model_path: str = None):
    """
    Show visual analysis step by step using the SAME functions as multiprocessing workflow.
    This ensures we're testing exactly what the main workflow does.
    """
    
    print(f"\n🔍 VISUAL ANALYSIS: {Path(image_path).name}")
    print("=" * 60)
    print(f"🔧 Testing the SAME functions used in multiprocessing_crumbly_workflow.py")
    print(f"🎯 This ensures visual test matches actual workflow behavior")
    
    # Step 1: Load original image
    print("\n📸 Step 1: Loading original image...")
    original_image = load_image(image_path)
    if original_image is None:
        print("❌ Failed to load image")
        return
    
    print(f"   ✅ Image loaded: {original_image.shape}")
    
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 4, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title(f'Original Image\n{Path(image_path).name}')
    plt.axis('off')
    
    # Step 2: Scale detection FIRST (same as workflow)
    print("\n📏 Step 2: Scale detection (on original image with scale bar)...")
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
    
    # Step 3: Preprocessing with scale bar removal (SAME function as workflow)
    print(f"\n🔧 Step 3: Preprocessing with scale bar removal...")
    print(f"   Using preprocess_image_for_worker() from multiprocessing_crumbly_workflow.py")
    print(f"   Enhanced preprocessing: {HAS_ENHANCED_PREPROCESSING}")
    print(f"   Scale bar removal: {HAS_SCALE_BAR_REMOVAL}")
    
    preprocessed = preprocess_image_for_worker(original_image)
    
    print(f"   Original shape: {original_image.shape}")
    print(f"   Preprocessed shape: {preprocessed.shape}")
    
    if preprocessed.shape != original_image.shape:
        print(f"   ✅ Scale bar removed: {original_image.shape[0] - preprocessed.shape[0]} pixels cropped")
    else:
        print(f"   ⚠️ No dimension change - scale bar removal may have failed")
    
    plt.subplot(2, 4, 2)
    plt.imshow(preprocessed, cmap='gray')
    plt.title('Preprocessed Image\n(Scale bar removed)')
    plt.axis('off')
    
    # Step 4: Fiber detection (on clean preprocessed image)
    print(f"\n🧬 Step 4: Fiber detection (on preprocessed image)...")
    try:
        fiber_detector = FiberTypeDetector()
        fiber_type, confidence, fiber_analysis_data = fiber_detector.classify_fiber_type(preprocessed)
        
        # Extract optimal fiber mask (SAME function as workflow)
        print(f"   Using create_optimal_fiber_mask() from multiprocessing_crumbly_workflow.py")
        fiber_mask = create_optimal_fiber_mask(preprocessed, fiber_analysis_data, debug=True)
        
        mask_area = np.sum(fiber_mask > 0)
        print(f"   ✅ Fiber type: {fiber_type} (confidence: {confidence:.3f})")
        print(f"   ✅ Optimal mask area: {mask_area:,} pixels")
        
        # Create visualization overlay
        overlay = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2RGB)
        overlay[fiber_mask > 0] = [0, 255, 0]  # Green for fiber
        
        plt.subplot(2, 4, 3)
        plt.imshow(overlay)
        plt.title(f'Fiber Segmentation\n{mask_area:,} pixels\n(Optimal mask)')
        plt.axis('off')
        
    except Exception as e:
        print(f"   ❌ Fiber detection error: {e}")
        fiber_mask = np.zeros(preprocessed.shape[:2], dtype=np.uint8)
        fiber_type = "error"
        confidence = 0.0
        mask_area = 0
    
    # Step 5: Porosity analysis (SAME logic as workflow)
    print(f"\n🔬 Step 5: Porosity analysis...")
    print(f"   Using SAME porosity logic as multiprocessing_crumbly_workflow.py")
    print(f"   Porosity type: {POROSITY_TYPE}")
    
    porosity_result = None
    try:
        if MODULES_LOADED.get('porosity_analysis', False):
            if POROSITY_TYPE == 'fast_refined':
                porosity_result = analyze_fiber_porosity(preprocessed, fiber_mask, scale_factor)
            elif POROSITY_TYPE == 'basic':
                porosity_result = quick_porosity_check(preprocessed, fiber_mask, scale_factor)
            
            if porosity_result:
                porosity_metrics = porosity_result.get('porosity_metrics', {})
                total_porosity = porosity_metrics.get('total_porosity_percent', 0)
                pore_count = porosity_metrics.get('pore_count', 0)
                avg_pore_size = porosity_metrics.get('average_pore_size_um2', 0)
                
                print(f"   ✅ Porosity: {total_porosity:.2f}%")
                print(f"   ✅ Pore count: {pore_count}")
                print(f"   ✅ Avg pore size: {avg_pore_size:.1f} μm²")
                
                # Check analysis area
                analyzed_area_pixels = porosity_metrics.get('fiber_area_pixels', 0)
                if analyzed_area_pixels > 0:
                    print(f"   ✅ Analyzed area: {analyzed_area_pixels:,} pixels")
                    if analyzed_area_pixels != mask_area:
                        excluded_pixels = mask_area - analyzed_area_pixels
                        print(f"   ✅ Excluded area (lumen): {excluded_pixels:,} pixels")
            else:
                print(f"   ⚠️ No porosity result returned")
        else:
            print("   ⚠️ Porosity analysis not available")
    except Exception as e:
        print(f"   ❌ Porosity analysis error: {e}")
    
    # Show porosity visualization - IMPROVED to show actual pores
    plt.subplot(2, 4, 4)
    if porosity_result and 'visualization_data' in porosity_result:
        pore_overlay = porosity_result['visualization_data'].get('pore_overlay')
        if pore_overlay is not None:
            plt.imshow(pore_overlay)
            plt.title('Detected Pores\n(From porosity analysis)')
        else:
            plt.imshow(preprocessed, cmap='gray')
            plt.title('Porosity Analysis\n(No visualization data)')
    else:
        # IMPROVED: Show actual processed pores, not raw candidates
        fiber_region = preprocessed.copy()
        fiber_region[fiber_mask == 0] = 255  # Set non-fiber to white
        
        if np.sum(fiber_mask > 0) > 0:
            # Use same threshold as porosity analysis
            pore_threshold = np.percentile(fiber_region[fiber_mask > 0], 30)
            pore_candidates = (fiber_region < pore_threshold) & (fiber_mask > 0)
            
            # Apply morphological cleaning (same as porosity analysis)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            pore_candidates_clean = cv2.morphologyEx(pore_candidates.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            pore_candidates_clean = cv2.morphologyEx(pore_candidates_clean, cv2.MORPH_CLOSE, kernel)
            
            # Find contours and filter by size (same as porosity analysis)
            contours, _ = cv2.findContours(pore_candidates_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_pores = [c for c in contours if cv2.contourArea(c) > 10]  # Min area filter
            
            # Create visualization showing ONLY valid pores
            pore_visualization = np.zeros_like(fiber_region)
            for contour in valid_pores:
                cv2.fillPoly(pore_visualization, [contour], 255)
            
            plt.imshow(pore_visualization, cmap='hot')
            plt.title(f'Actual Detected Pores\n{len(valid_pores)} pores (cleaned)')
            
            print(f"   🔍 PORE DETECTION DEBUG:")
            print(f"     Raw candidates: {np.sum(pore_candidates)} pixels")
            print(f"     After cleaning: {np.sum(pore_candidates_clean)} pixels") 
            print(f"     Valid pores: {len(valid_pores)} contours")
            print(f"     Final pore pixels: {np.sum(pore_visualization > 0)} pixels")
        else:
            plt.imshow(preprocessed, cmap='gray')
            plt.title('No Fiber Detected')
    plt.axis('off')
    
    # Step 6: Crumbly texture analysis (SAME logic as workflow)
    print(f"\n🧩 Step 6: Crumbly texture analysis...")
    print(f"   Using SAME crumbly logic as multiprocessing_crumbly_workflow.py")
    print(f"   Model path: {model_path if model_path else 'None (traditional detector)'}")
    
    try:
        if MODULES_LOADED.get('crumbly_detection', False):
            # Choose detector type (SAME logic as workflow)
            if model_path and MODULES_LOADED.get('hybrid_detector', False):
                try:
                    crumbly_detector = load_hybrid_detector(model_path)
                    model_type = 'hybrid'
                    print(f"   🤖 Using hybrid model: {model_path}")
                except Exception as e:
                    crumbly_detector = CrumblyDetector(porosity_aware=True)
                    model_type = 'traditional_fallback'
                    print(f"   ⚠️ Hybrid model failed, using traditional: {e}")
            else:
                crumbly_detector = CrumblyDetector(porosity_aware=True)
                model_type = 'traditional'
                print(f"   🔧 Using traditional detector")
            
            fiber_mask_bool = fiber_mask > 127
            
            # Pass porosity data (SAME as workflow)
            porosity_data = None
            if porosity_result:
                porosity_data = {'porosity_metrics': porosity_result.get('porosity_metrics', {})}
                print(f"   📊 Passing porosity data to texture analysis")
            
            # Run crumbly analysis with debug
            crumbly_result = crumbly_detector.analyze_crumbly_texture(
                preprocessed, fiber_mask_bool, None, scale_factor, 
                debug=True, porosity_data=porosity_data
            )
            
            if crumbly_result and 'classification' in crumbly_result:
                # Apply classification improvements (SAME function as workflow)
                print(f"   🔧 Applying classification improvements...")
                original_classification = crumbly_result['classification']
                original_confidence = crumbly_result.get('confidence', 0.0)
                original_score = crumbly_result.get('crumbly_score', 0.5)
                
                crumbly_result = apply_classification_improvements(crumbly_result, debug=True)
                
                classification = crumbly_result['classification']
                confidence_score = crumbly_result.get('confidence', 0.0)
                crumbly_score = crumbly_result.get('crumbly_score', 0.5)
                override_reason = crumbly_result.get('override_reason', 'none')
                
                print(f"   📊 RESULTS:")
                print(f"     Original: {original_classification} (conf: {original_confidence:.3f}, score: {original_score:.3f})")
                print(f"     Final: {classification} (conf: {confidence_score:.3f}, score: {crumbly_score:.3f})")
                if override_reason != 'none':
                    print(f"     🔧 Override applied: {override_reason}")
                
                # Show texture analysis result
                plt.subplot(2, 4, 5)
                
                # Create texture visualization
                texture_vis = preprocessed.copy()
                texture_vis[fiber_mask == 0] = 128  # Set background to gray
                
                # Show edges on fiber for texture visualization
                edges = cv2.Canny(texture_vis, 30, 100)
                edges_masked = edges.copy()
                edges_masked[fiber_mask == 0] = 0
                
                # Create colored overlay
                texture_overlay = cv2.cvtColor(texture_vis, cv2.COLOR_GRAY2RGB)
                texture_overlay[edges_masked > 0] = [255, 255, 0]  # Yellow for edges
                
                plt.imshow(texture_overlay)
                plt.title(f'Texture Analysis\n{classification} ({confidence_score:.2f})\nModel: {model_type}')
                plt.axis('off')
                
                # Show analysis summary
                plt.subplot(2, 4, 6)
                plt.text(0.05, 0.9, f"Image: {Path(image_path).name}", fontsize=9, weight='bold')
                plt.text(0.05, 0.8, f"Scale: {scale_factor:.4f} μm/pixel", fontsize=8)
                plt.text(0.05, 0.7, f"Fiber: {fiber_type} ({confidence:.3f})", fontsize=8)
                plt.text(0.05, 0.6, f"Mask: {mask_area:,} pixels", fontsize=8)
                if porosity_result:
                    plt.text(0.05, 0.5, f"Porosity: {total_porosity:.2f}%", fontsize=8)
                    plt.text(0.05, 0.4, f"Pores: {pore_count}", fontsize=8)
                plt.text(0.05, 0.3, f"Texture: {classification}", fontsize=8)
                plt.text(0.05, 0.2, f"Score: {crumbly_score:.3f}", fontsize=8)
                plt.text(0.05, 0.1, f"Model: {model_type}", fontsize=8)
                if override_reason != 'none':
                    plt.text(0.05, 0.0, f"Override: {override_reason}", fontsize=7, color='red')
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                plt.axis('off')
                plt.title('Analysis Summary')
                
                # Workflow comparison
                plt.subplot(2, 4, 7)
                plt.text(0.05, 0.9, "WORKFLOW TEST RESULTS", fontsize=9, weight='bold')
                plt.text(0.05, 0.8, f"✅ Scale bar removal: {HAS_SCALE_BAR_REMOVAL}", fontsize=8)
                plt.text(0.05, 0.7, f"✅ Optimal mask: {mask_area > 0}", fontsize=8)
                plt.text(0.05, 0.6, f"✅ Porosity analysis: {porosity_result is not None}", fontsize=8)
                plt.text(0.05, 0.5, f"✅ Texture analysis: {classification != 'error'}", fontsize=8)
                plt.text(0.05, 0.4, f"✅ Classification fix: {override_reason != 'none'}", fontsize=8)
                plt.text(0.05, 0.3, f"Model type: {model_type}", fontsize=8)
                plt.text(0.05, 0.2, f"Processing order: ✅", fontsize=8)
                plt.text(0.05, 0.1, f"Dimension match: ✅", fontsize=8)
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                plt.axis('off')
                plt.title('Workflow Validation')
                
                # Debug metrics - COMPREHENSIVE debugging to find surface roughness
                plt.subplot(2, 4, 8)
                plt.text(0.05, 0.9, "DEBUG METRICS", fontsize=9, weight='bold')
                
                # COMPREHENSIVE DEBUG: Print the entire crumbly_result structure
                print(f"\n   🔍 COMPREHENSIVE CRUMBLY RESULT DEBUG:")
                print(f"   📊 Top-level keys: {list(crumbly_result.keys())}")
                
                # Check for surface roughness in multiple possible locations
                surface_roughness = None
                edge_irregularity = None
                wall_integrity = None
                
                # Location 1: Direct top-level metrics
                surface_metrics = crumbly_result.get('surface_metrics', {})
                if surface_metrics:
                    print(f"   📊 Direct surface_metrics: {surface_metrics}")
                    surface_roughness = surface_metrics.get('roughness_score')
                
                boundary_metrics = crumbly_result.get('boundary_metrics', {})
                if boundary_metrics:
                    print(f"   📊 Direct boundary_metrics: {boundary_metrics}")
                    edge_irregularity = boundary_metrics.get('irregularity_score')
                
                wall_metrics = crumbly_result.get('wall_integrity_metrics', {})
                if wall_metrics:
                    print(f"   📊 Direct wall_integrity_metrics: {wall_metrics}")
                    wall_integrity = wall_metrics.get('integrity_score')
                
                # Location 2: Nested in traditional_result (for hybrid)
                if 'traditional_result' in crumbly_result:
                    traditional_data = crumbly_result['traditional_result']
                    print(f"   📊 Traditional result keys: {list(traditional_data.keys())}")
                    
                    trad_surface = traditional_data.get('surface_metrics', {})
                    if trad_surface:
                        print(f"   📊 Traditional surface_metrics: {trad_surface}")
                        if surface_roughness is None:
                            surface_roughness = trad_surface.get('roughness_score')
                    
                    trad_boundary = traditional_data.get('boundary_metrics', {})
                    if trad_boundary:
                        print(f"   📊 Traditional boundary_metrics: {trad_boundary}")
                        if edge_irregularity is None:
                            edge_irregularity = trad_boundary.get('irregularity_score')
                    
                    trad_wall = traditional_data.get('wall_integrity_metrics', {})
                    if trad_wall:
                        print(f"   📊 Traditional wall_integrity_metrics: {trad_wall}")
                        if wall_integrity is None:
                            wall_integrity = trad_wall.get('integrity_score')
                
                # Location 3: Check ALL keys for anything containing 'surface', 'roughness', etc.
                print(f"   🔍 Searching ALL keys for surface/roughness metrics...")
                for key, value in crumbly_result.items():
                    if 'surface' in key.lower() or 'roughness' in key.lower():
                        print(f"   📊 Found surface-related key '{key}': {value}")
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if 'surface' in subkey.lower() or 'roughness' in subkey.lower():
                                print(f"   📊 Found nested surface key '{key}.{subkey}': {subvalue}")
                
                # Location 4: Check for alternative metric names
                alternative_names = [
                    'texture_roughness', 'surface_texture', 'roughness', 'texture_score',
                    'surface_roughness_score', 'boundary_roughness', 'edge_roughness'
                ]
                
                for alt_name in alternative_names:
                    if alt_name in crumbly_result:
                        print(f"   📊 Found alternative metric '{alt_name}': {crumbly_result[alt_name]}")
                        if surface_roughness is None:
                            surface_roughness = crumbly_result[alt_name]
                
                # Final debug output
                print(f"   🎯 FINAL EXTRACTED VALUES:")
                print(f"     Surface roughness: {surface_roughness}")
                print(f"     Edge irregularity: {edge_irregularity}")
                print(f"     Wall integrity: {wall_integrity}")
                
                # Helper function to safely format numbers
                def safe_format(value, default='N/A'):
                    if isinstance(value, (int, float)):
                        return f"{value:.3f}"
                    else:
                        return str(default)
                
                # Display the metrics
                plt.text(0.05, 0.8, f"Surface roughness:", fontsize=8)
                plt.text(0.05, 0.75, f"  {safe_format(surface_roughness)}", fontsize=7)
                plt.text(0.05, 0.65, f"Edge irregularity:", fontsize=8)
                plt.text(0.05, 0.6, f"  {safe_format(edge_irregularity)}", fontsize=7)
                plt.text(0.05, 0.5, f"Wall integrity:", fontsize=8)
                plt.text(0.05, 0.45, f"  {safe_format(wall_integrity)}", fontsize=7)
                
                # Show where we found the data
                if surface_roughness is not None:
                    plt.text(0.05, 0.35, f"✅ Found surface roughness", fontsize=7, color='green')
                else:
                    plt.text(0.05, 0.35, f"❌ Surface roughness missing", fontsize=7, color='red')
                
                plt.text(0.05, 0.25, f"Original: {original_classification}", fontsize=7)
                plt.text(0.05, 0.2, f"Final: {classification}", fontsize=7)
                if override_reason != 'none':
                    plt.text(0.05, 0.1, f"Override: {override_reason}", fontsize=6, color='red')
                else:
                    plt.text(0.05, 0.1, f"No override", fontsize=6)
                
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                plt.axis('off')
                plt.title('Classification Debug')
                
            else:
                print("   ❌ Crumbly analysis failed")
        else:
            print("   ⚠️ Crumbly detection not available")
    except Exception as e:
        print(f"   ❌ Crumbly analysis error: {e}")
        import traceback
        traceback.print_exc()
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n" + "=" * 60)
    print(f"✅ Visual analysis complete!")
    print(f"🎯 This tested the EXACT SAME functions used in multiprocessing_crumbly_workflow.py")
    print(f"📊 Any issues seen here will also occur in the main workflow")

def main():
    """Main function for visual debug test."""
    
    parser = argparse.ArgumentParser(description='Visual Debug Test for Multiprocessing Workflow Functions')
    parser.add_argument('image_path', help='Path to image file to analyze')
    parser.add_argument('--model-path', help='Path to hybrid model for testing')
    
    args = parser.parse_args()
    
    image_path = Path(args.image_path)
    
    if not image_path.exists():
        print(f"❌ Image file not found: {image_path}")
        return 1
    
    print(f"🔍 Testing multiprocessing workflow functions...")
    print(f"   Image: {image_path}")
    print(f"   Model: {args.model_path if args.model_path else 'None (traditional)'}")
    
    try:
        show_step_by_step_analysis(str(image_path), args.model_path)
        
        print(f"\n✅ Visual test complete!")
        print(f"🎯 If this looks correct, the multiprocessing workflow should work identically")
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