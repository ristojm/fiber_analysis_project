#!/usr/bin/env python3
"""
Test Script for Crumbly Detection Module
Integrates with existing SEM Fiber Analysis System to test crumbly texture detection
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from datetime import datetime

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "modules"))

# Import existing modules
try:
    from modules.image_preprocessing import load_image, preprocess_pipeline
    from modules.scale_detection import ScaleBarDetector
    from modules.fiber_type_detection import FiberTypeDetector
    print("✅ Existing modules loaded successfully")
except ImportError as e:
    print(f"❌ Could not import existing modules: {e}")
    sys.exit(1)

# Import the new crumbly detection module
try:
    from modules.crumbly_detection import CrumblyDetector, detect_crumbly_texture
    print("✅ Crumbly detection module loaded successfully")
except ImportError as e:
    print(f"❌ Could not import crumbly detection module: {e}")
    print("Make sure crumbly_detection.py is in the modules/ directory")
    sys.exit(1)

def find_sample_images():
    """Find sample images for testing crumbly detection"""
    print("\n📁 Looking for sample images...")
    
    sample_dirs = ["sample_images", "test", "images"]
    sample_extensions = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp"]
    
    found_images = []
    
    for sample_dir in sample_dirs:
        sample_path = Path(sample_dir)
        if sample_path.exists():
            print(f"   Found directory: {sample_path}")
            
            for ext in sample_extensions:
                image_files = list(sample_path.glob(ext))
                image_files.extend(list(sample_path.glob(ext.upper())))
                found_images.extend(image_files)
    
    # Remove duplicates and sort
    found_images = sorted(set(found_images))
    
    if found_images:
        print(f"✅ Found {len(found_images)} images")
        for img in found_images:
            print(f"   - {img.name}")
        return found_images
    else:
        print("❌ No sample images found!")
        return []

def create_test_output_dir():
    """Create output directory for test results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("crumbly_test_results") / f"test_session_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Test output directory: {output_dir}")
    return output_dir

def test_single_image_crumbly_detection(image_path, output_dir, verbose=True):
    """Test crumbly detection on a single image using the full pipeline"""
    
    if verbose:
        print(f"\n🔬 Testing Crumbly Detection: {Path(image_path).name}")
        print("-" * 60)
    
    start_time = time.time()
    
    try:
        # Step 1: Load and preprocess image
        if verbose:
            print("   📸 Step 1: Loading and preprocessing image...")
        
        # Load original image (for scale detection)
        original_image = load_image(str(image_path))
        if original_image is None:
            return False, {'error': 'Could not load image'}
        
        # Preprocess image (removes scale bar)
        preprocessing_result = preprocess_pipeline(str(image_path))
        if not preprocessing_result.get('preprocessing_complete', False):
            return False, {'error': 'Preprocessing failed'}
        
        processed_image = preprocessing_result['processed']
        
        # Step 2: Scale detection (use original image)
        if verbose:
            print("   📏 Step 2: Detecting scale...")
        
        scale_detector = ScaleBarDetector(use_enhanced_detection=True)
        scale_result = scale_detector.detect_scale_bar(original_image, debug=False)
        
        scale_factor = scale_result.get('micrometers_per_pixel', 1.0) if scale_result.get('scale_detected', False) else 1.0
        
        if verbose:
            if scale_result.get('scale_detected', False):
                print(f"      ✅ Scale detected: {scale_factor:.4f} μm/pixel")
            else:
                print(f"      ⚠️ Scale not detected, using default: {scale_factor} μm/pixel")
        
        # Step 3: Enhanced fiber type detection with oval fitting
        if verbose:
            print("   🧬 Step 3: Enhanced fiber detection with oval fitting...")
        
        fiber_detector = FiberTypeDetector()
        fiber_type, fiber_confidence, fiber_analysis_data = fiber_detector.classify_fiber_type(
            processed_image, scale_factor
        )
        
        if verbose:
            print(f"      ✅ Fiber type: {fiber_type} (confidence: {fiber_confidence:.3f})")
            print(f"      Total fibers detected: {fiber_analysis_data.get('total_fibers', 0)}")
            print(f"      Hollow fibers: {fiber_analysis_data.get('hollow_fibers', 0)}")
        
        # Get fiber mask and individual fiber results
        fiber_mask = fiber_analysis_data.get('fiber_mask', np.zeros_like(processed_image, dtype=bool))
        individual_results = fiber_analysis_data.get('individual_results', [])
        
        if np.sum(fiber_mask) < 1000:
            return False, {'error': 'Insufficient fiber area detected'}
        
        # Step 4: Crumbly texture detection - ANALYZE ONLY THE MAIN FIBER
        if verbose:
            print("   🕳️ Step 4: Analyzing crumbly texture on main fiber...")
        
        # Initialize crumbly detector
        crumbly_detector = CrumblyDetector(porosity_aware=True)
        
        # Find the main fiber (largest area)
        best_fiber_result = None
        best_fiber_crumbly_result = None
        
        if individual_results:
            # Sort fibers by area and take the largest one
            sorted_fibers = sorted(individual_results, 
                                 key=lambda x: x.get('fiber_properties', {}).get('area', 0), 
                                 reverse=True)
            
            # Only process the largest fiber
            main_fiber = sorted_fibers[0]
            fiber_props = main_fiber.get('fiber_properties', {})
            fiber_contour = fiber_props.get('contour')
            
            if fiber_contour is not None:
                if verbose:
                    print(f"      Analyzing main fiber (area: {fiber_props.get('area', 0):.0f} pixels)")
                
                # Create individual fiber mask
                individual_fiber_mask = np.zeros(processed_image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(individual_fiber_mask, [fiber_contour], 255)
                individual_fiber_mask = individual_fiber_mask.astype(bool)
                
                # Get lumen mask if this is a hollow fiber
                lumen_mask = None
                if main_fiber.get('has_lumen', False):
                    lumen_props = main_fiber.get('lumen_properties', {})
                    lumen_contour = lumen_props.get('contour')
                    if lumen_contour is not None:
                        lumen_mask = np.zeros(processed_image.shape[:2], dtype=np.uint8)
                        cv2.fillPoly(lumen_mask, [lumen_contour], 255)
                        lumen_mask = lumen_mask.astype(bool)
                
                # Run crumbly analysis on the main fiber
                best_fiber_crumbly_result = crumbly_detector.analyze_crumbly_texture(
                    processed_image, 
                    individual_fiber_mask, 
                    lumen_mask, 
                    scale_factor,
                    debug=False  # Disable debug visualization to prevent blocking
                )
                
                # Add fiber metadata
                best_fiber_crumbly_result['fiber_id'] = 0
                best_fiber_crumbly_result['fiber_area_pixels'] = np.sum(individual_fiber_mask)
                best_fiber_crumbly_result['has_lumen'] = main_fiber.get('has_lumen', False)
                
                # Add oval fitting context if available
                if fiber_props.get('oval_fitted', False):
                    best_fiber_crumbly_result['oval_fitted'] = True
                    best_fiber_crumbly_result['oval_diameter_um'] = fiber_props.get('oval_mean_diameter_um', 0)
                    best_fiber_crumbly_result['oval_fit_quality'] = fiber_props.get('oval_fit_quality', 0)
                else:
                    best_fiber_crumbly_result['oval_fitted'] = False
                
                if verbose:
                    classification = best_fiber_crumbly_result.get('classification')
                    confidence = best_fiber_crumbly_result.get('confidence', 0)
                    crumbly_score = best_fiber_crumbly_result.get('crumbly_score', 0)
                    print(f"      Result: {classification} (confidence: {confidence:.3f}, score: {crumbly_score:.3f})")
        
        processing_time = time.time() - start_time
        
        # Compile results
        test_result = {
            'image_path': str(image_path),
            'image_name': Path(image_path).name,
            'success': True,
            'processing_time': processing_time,
            'scale_factor': scale_factor,
            'scale_detected': scale_result.get('scale_detected', False),
            'fiber_type': fiber_type,
            'fiber_confidence': fiber_confidence,
            'total_fibers_detected': len(individual_results),
            'best_fiber_result': best_fiber_crumbly_result
        }
        
        # Save visualization
        if best_fiber_crumbly_result:
            save_path = create_simple_visualization(
                processed_image, fiber_mask, best_fiber_crumbly_result, 
                output_dir, Path(image_path).stem
            )
            test_result['visualization_path'] = str(save_path)
        
        if verbose:
            print(f"   ✅ Crumbly detection completed in {processing_time:.2f}s")
            if best_fiber_crumbly_result:
                best_classification = best_fiber_crumbly_result['classification']
                best_confidence = best_fiber_crumbly_result.get('confidence', 0)
                print(f"   🎯 Final result: {best_classification} (confidence: {best_confidence:.3f})")
        
        return True, test_result
        
    except Exception as e:
        processing_time = time.time() - start_time
        if verbose:
            print(f"   ❌ Error: {e}")
        
        return False, {
            'image_path': str(image_path),
            'image_name': Path(image_path).name,
            'success': False,
            'error': str(e),
            'processing_time': processing_time
        }

def create_simple_visualization(processed_image, fiber_mask, crumbly_result, 
                              output_dir, image_stem):
    """Create simple visualization of crumbly detection results"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Original processed image
    axes[0].imshow(processed_image, cmap='gray')
    axes[0].set_title('Processed Image')
    axes[0].axis('off')
    
    # 2. Fiber mask
    axes[1].imshow(fiber_mask, cmap='gray')
    axes[1].set_title('Detected Fiber')
    axes[1].axis('off')
    
    # 3. Results text
    classification = crumbly_result.get('classification', 'unknown')
    confidence = crumbly_result.get('confidence', 0)
    crumbly_score = crumbly_result.get('crumbly_score', 0)
    
    results_text = f"CRUMBLY DETECTION RESULTS\n\n"
    results_text += f"Classification: {classification.upper()}\n"
    results_text += f"Confidence: {confidence:.3f}\n"
    results_text += f"Crumbly Score: {crumbly_score:.3f}\n\n"
    
    # Add key metrics
    if 'pore_metrics' in crumbly_result:
        pore_metrics = crumbly_result['pore_metrics']
        results_text += f"Pore Analysis:\n"
        results_text += f"  Count: {pore_metrics.get('pore_count', 0)}\n"
        results_text += f"  Organization: {pore_metrics.get('organized_porosity_score', 0):.3f}\n"
        results_text += f"  Circularity: {pore_metrics.get('mean_pore_circularity', 0):.3f}\n\n"
    
    if 'wall_integrity_metrics' in crumbly_result:
        wall_metrics = crumbly_result['wall_integrity_metrics']
        results_text += f"Wall Integrity:\n"
        results_text += f"  Score: {wall_metrics.get('wall_integrity_score', 0):.3f}\n"
    
    # Color code based on classification
    if classification == 'crumbly':
        bbox_color = 'lightcoral'
    elif classification == 'porous':
        bbox_color = 'lightgreen'
    elif classification == 'intermediate':
        bbox_color = 'lightyellow'
    else:
        bbox_color = 'lightgray'
    
    axes[2].text(0.05, 0.95, results_text, transform=axes[2].transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=bbox_color))
    axes[2].set_title('Analysis Results')
    axes[2].axis('off')
    
    plt.suptitle(f'Crumbly Texture Detection - {image_stem}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save visualization
    save_path = output_dir / f'crumbly_result_{image_stem}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()  # IMPORTANT: Close the figure to prevent memory issues
    
    return save_path

def run_crumbly_detection_tests():
    """Main test function"""
    
    print("🧪 CRUMBLY TEXTURE DETECTION MODULE TEST")
    print("=" * 80)
    
    # Find sample images
    sample_images = find_sample_images()
    if not sample_images:
        print("❌ No sample images found for testing!")
        return False
    
    # Create output directory
    output_dir = create_test_output_dir()
    
    # Test results
    test_results = []
    successful_tests = 0
    
    # Test each image (limit to first 3 for testing)
    num_test_images = min(3, len(sample_images))
    for i, image_path in enumerate(sample_images[:num_test_images]):
        print(f"\n{'='*20} TEST {i+1}/{num_test_images} {'='*20}")
        
        success, result = test_single_image_crumbly_detection(image_path, output_dir, verbose=True)
        test_results.append(result)
        
        if success:
            successful_tests += 1
        
        # Show individual results summary
        if success and result.get('best_fiber_result'):
            best_result = result['best_fiber_result']
            print(f"\n🎯 RESULT SUMMARY:")
            print(f"   Classification: {best_result.get('classification', 'unknown')}")
            print(f"   Confidence: {best_result.get('confidence', 0):.3f}")
            print(f"   Crumbly Score: {best_result.get('crumbly_score', 0):.3f}")
            
            if best_result.get('oval_fitted', False):
                print(f"   Oval Diameter: {best_result.get('oval_diameter_um', 0):.1f} μm")
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"🎯 FINAL TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total tests: {len(test_results)}")
    print(f"Successful: {successful_tests}")
    print(f"Success rate: {successful_tests/len(test_results)*100:.1f}%")
    print(f"Output directory: {output_dir}")
    
    # Show classification results
    successful_results = [r for r in test_results if r.get('success', False)]
    if successful_results:
        print(f"\n📊 CLASSIFICATION RESULTS:")
        for result in successful_results:
            image_name = result['image_name']
            best_fiber = result.get('best_fiber_result', {})
            classification = best_fiber.get('classification', 'unknown')
            confidence = best_fiber.get('confidence', 0)
            
            print(f"   {image_name}: {classification} (confidence: {confidence:.3f})")
    
    # Check for classification distribution
    classifications = {}
    for result in successful_results:
        best_fiber = result.get('best_fiber_result', {})
        classification = best_fiber.get('classification', 'unknown')
        classifications[classification] = classifications.get(classification, 0) + 1
    
    print(f"\n🔍 TEXTURE CLASSIFICATION BREAKDOWN:")
    for cls, count in classifications.items():
        print(f"   {cls}: {count}")
    
    if successful_tests > 0:
        print(f"\n✅ Crumbly detection module is working!")
        print(f"   Check {output_dir} for visualization files")
    else:
        print(f"\n❌ All tests failed - check the error messages above")
    
    return successful_tests > 0

if __name__ == "__main__":
    success = run_crumbly_detection_tests()
    sys.exit(0 if success else 1)