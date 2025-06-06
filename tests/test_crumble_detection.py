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
            print(f"      Total fibers: {fiber_analysis_data.get('total_fibers', 0)}")
            print(f"      Hollow fibers: {fiber_analysis_data.get('hollow_fibers', 0)}")
        
        # Get fiber mask and individual fiber results
        fiber_mask = fiber_analysis_data.get('fiber_mask', np.zeros_like(processed_image, dtype=bool))
        individual_results = fiber_analysis_data.get('individual_results', [])
        
        if np.sum(fiber_mask) < 1000:
            return False, {'error': 'Insufficient fiber area detected'}
        
        # Step 4: Crumbly texture detection
        if verbose:
            print("   🕳️ Step 4: Analyzing crumbly texture...")
        
        # Initialize crumbly detector
        crumbly_detector = CrumblyDetector(porosity_aware=True)
        
        # Analyze each detected fiber individually
        fiber_crumbly_results = []
        
        for i, fiber_result in enumerate(individual_results):
            fiber_props = fiber_result.get('fiber_properties', {})
            fiber_contour = fiber_props.get('contour')
            
            if fiber_contour is None:
                continue
            
            # Create individual fiber mask
            individual_fiber_mask = np.zeros(processed_image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(individual_fiber_mask, [fiber_contour], 255)
            individual_fiber_mask = individual_fiber_mask.astype(bool)
            
            # Get lumen mask if this is a hollow fiber
            lumen_mask = None
            if fiber_result.get('has_lumen', False):
                lumen_props = fiber_result.get('lumen_properties', {})
                lumen_contour = lumen_props.get('contour')
                if lumen_contour is not None:
                    lumen_mask = np.zeros(processed_image.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(lumen_mask, [lumen_contour], 255)
                    lumen_mask = lumen_mask.astype(bool)
            
            # Run crumbly analysis on this individual fiber
            crumbly_result = crumbly_detector.analyze_crumbly_texture(
                processed_image, 
                individual_fiber_mask, 
                lumen_mask, 
                scale_factor,
                debug=True
            )
            
            # Add fiber ID and metadata
            crumbly_result['fiber_id'] = i
            crumbly_result['fiber_area_pixels'] = np.sum(individual_fiber_mask)
            crumbly_result['has_lumen'] = fiber_result.get('has_lumen', False)
            
            # Add oval fitting context if available
            if fiber_props.get('oval_fitted', False):
                crumbly_result['oval_fitted'] = True
                crumbly_result['oval_diameter_um'] = fiber_props.get('oval_mean_diameter_um', 0)
                crumbly_result['oval_fit_quality'] = fiber_props.get('oval_fit_quality', 0)
            else:
                crumbly_result['oval_fitted'] = False
            
            fiber_crumbly_results.append(crumbly_result)
            
            if verbose and crumbly_result.get('classification'):
                classification = crumbly_result['classification']
                confidence = crumbly_result.get('confidence', 0)
                crumbly_score = crumbly_result.get('crumbly_score', 0)
                print(f"      Fiber {i}: {classification} (confidence: {confidence:.3f}, score: {crumbly_score:.3f})")
        
        # Overall analysis using the largest/best fiber
        overall_crumbly_result = None
        if fiber_crumbly_results:
            # Use the fiber with highest confidence
            best_fiber_result = max(fiber_crumbly_results, key=lambda x: x.get('confidence', 0))
            
            # Also run analysis on the complete fiber mask for comparison
            overall_crumbly_result = crumbly_detector.analyze_crumbly_texture(
                processed_image, 
                fiber_mask, 
                None,  # No lumen mask for overall analysis
                scale_factor,
                debug=True  # Enable debug visualization for overall analysis
            )
        
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
            'total_fibers': len(individual_results),
            'individual_fiber_results': fiber_crumbly_results,
            'overall_result': overall_crumbly_result,
            'best_fiber_result': best_fiber_result if fiber_crumbly_results else None
        }
        
        # Save detailed visualization
        if overall_crumbly_result and fiber_crumbly_results:
            save_path = create_detailed_visualization(
                processed_image, fiber_analysis_data, fiber_crumbly_results, 
                overall_crumbly_result, output_dir, Path(image_path).stem
            )
            test_result['visualization_path'] = str(save_path)
        
        if verbose:
            print(f"   ✅ Crumbly detection completed in {processing_time:.2f}s")
            if best_fiber_result:
                best_classification = best_fiber_result['classification']
                best_confidence = best_fiber_result.get('confidence', 0)
                print(f"   🎯 Best result: {best_classification} (confidence: {best_confidence:.3f})")
        
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

def create_detailed_visualization(processed_image, fiber_analysis_data, 
                                fiber_crumbly_results, overall_result, 
                                output_dir, image_stem):
    """Create comprehensive visualization of crumbly detection results"""
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    # 1. Original processed image
    axes[0].imshow(processed_image, cmap='gray')
    axes[0].set_title('Processed Image')
    axes[0].axis('off')
    
    # 2. Fiber segmentation with oval fitting
    fiber_mask = fiber_analysis_data.get('fiber_mask', np.zeros_like(processed_image))
    individual_results = fiber_analysis_data.get('individual_results', [])
    
    overlay = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
    
    # Color-code fibers by crumbly classification
    for i, (fiber_result, crumbly_result) in enumerate(zip(individual_results, fiber_crumbly_results)):
        fiber_props = fiber_result.get('fiber_properties', {})
        contour = fiber_props.get('contour')
        
        if contour is not None:
            classification = crumbly_result.get('classification', 'unknown')
            
            # Color mapping
            if classification == 'crumbly':
                color = (255, 0, 0)  # Red for crumbly
            elif classification == 'smooth':
                color = (0, 255, 0)  # Green for smooth
            elif classification == 'intermediate':
                color = (255, 255, 0)  # Yellow for intermediate
            else:
                color = (128, 128, 128)  # Gray for unknown
            
            cv2.drawContours(overlay, [contour], -1, color, 2)
            
            # Add fiber ID
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.putText(overlay, str(i), (cx-10, cy+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    axes[1].imshow(overlay)
    axes[1].set_title('Crumbly Classification\n(Red=Crumbly, Green=Smooth, Yellow=Intermediate)')
    axes[1].axis('off')
    
    # 3. Oval fitting visualization
    oval_overlay = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
    
    for fiber_result in individual_results:
        fiber_props = fiber_result.get('fiber_properties', {})
        
        # Draw oval if fitted
        if fiber_props.get('oval_fitted', False) and 'ellipse_params' in fiber_props:
            ellipse_params = fiber_props['ellipse_params']
            if ellipse_params is not None:
                cv2.ellipse(oval_overlay, ellipse_params, (0, 255, 255), 2)  # Cyan for fiber oval
        
        # Draw lumen oval if available
        if fiber_result.get('has_lumen', False):
            lumen_props = fiber_result.get('lumen_properties', {})
            if lumen_props.get('oval_fitted', False) and 'ellipse_params' in lumen_props:
                ellipse_params = lumen_props['ellipse_params']
                if ellipse_params is not None:
                    cv2.ellipse(oval_overlay, ellipse_params, (255, 0, 255), 2)  # Magenta for lumen oval
    
    axes[2].imshow(oval_overlay)
    axes[2].set_title('Oval Fitting\n(Cyan=Fiber, Magenta=Lumen)')
    axes[2].axis('off')
    
    # 4. Crumbly scores bar chart
    if fiber_crumbly_results:
        fiber_ids = [f"F{r['fiber_id']}" for r in fiber_crumbly_results]
        crumbly_scores = [r.get('crumbly_score', 0) for r in fiber_crumbly_results]
        classifications = [r.get('classification', 'unknown') for r in fiber_crumbly_results]
        
        colors = ['red' if c == 'crumbly' else 'green' if c == 'smooth' else 'yellow' 
                 for c in classifications]
        
        bars = axes[3].bar(fiber_ids, crumbly_scores, color=colors, alpha=0.7)
        axes[3].set_title('Crumbly Scores by Fiber')
        axes[3].set_ylabel('Crumbly Score')
        axes[3].set_ylim(0, 1)
        axes[3].axhline(y=0.6, color='red', linestyle='--', alpha=0.5, label='Crumbly threshold')
        axes[3].axhline(y=0.3, color='green', linestyle='--', alpha=0.5, label='Smooth threshold')
        
        # Add value labels on bars
        for bar, score in zip(bars, crumbly_scores):
            axes[3].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                        f'{score:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 5. Confidence scores
    if fiber_crumbly_results:
        confidences = [r.get('confidence', 0) for r in fiber_crumbly_results]
        
        bars = axes[4].bar(fiber_ids, confidences, color=colors, alpha=0.7)
        axes[4].set_title('Classification Confidence')
        axes[4].set_ylabel('Confidence')
        axes[4].set_ylim(0, 1)
        
        # Add value labels
        for bar, conf in zip(bars, confidences):
            axes[4].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                        f'{conf:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 6. Texture metrics summary
    if fiber_crumbly_results:
        metrics_text = "TEXTURE ANALYSIS SUMMARY\n\n"
        
        for i, result in enumerate(fiber_crumbly_results):
            classification = result.get('classification', 'unknown')
            confidence = result.get('confidence', 0)
            crumbly_score = result.get('crumbly_score', 0)
            
            metrics_text += f"Fiber {i}: {classification}\n"
            metrics_text += f"  Score: {crumbly_score:.3f}\n"
            metrics_text += f"  Confidence: {confidence:.3f}\n"
            
            # Add oval fitting info if available
            if result.get('oval_fitted', False):
                diameter = result.get('oval_diameter_um', 0)
                quality = result.get('oval_fit_quality', 0)
                metrics_text += f"  Oval: {diameter:.1f}μm (Q:{quality:.2f})\n"
            
            metrics_text += "\n"
        
        # Overall result
        if overall_result:
            overall_class = overall_result.get('classification', 'unknown')
            overall_conf = overall_result.get('confidence', 0)
            overall_score = overall_result.get('crumbly_score', 0)
            
            metrics_text += f"OVERALL: {overall_class}\n"
            metrics_text += f"Score: {overall_score:.3f}\n"
            metrics_text += f"Confidence: {overall_conf:.3f}"
        
        axes[5].text(0.05, 0.95, metrics_text, transform=axes[5].transAxes,
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue'))
        axes[5].set_title('Analysis Summary')
        axes[5].axis('off')
    
    # 7. Scale and processing info
    scale_factor = overall_result.get('scale_factor', 1.0) if overall_result else 1.0
    oval_summary = fiber_analysis_data.get('oval_fitting_summary', {})
    
    info_text = f"PROCESSING INFORMATION\n\n"
    info_text += f"Scale Factor: {scale_factor:.4f} μm/pixel\n"
    info_text += f"Total Fibers: {len(individual_results)}\n"
    info_text += f"Fibers Analyzed: {len(fiber_crumbly_results)}\n\n"
    
    if oval_summary:
        info_text += f"Oval Fitting:\n"
        info_text += f"  Success Rate: {oval_summary.get('fiber_fit_success_rate', 0):.1%}\n"
        info_text += f"  Avg Quality: {oval_summary.get('fiber_avg_fit_quality', 0):.2f}\n"
        info_text += f"  Avg Diameter: {oval_summary.get('fiber_avg_mean_diameter_um', 0):.1f} μm\n"
    
    axes[6].text(0.05, 0.95, info_text, transform=axes[6].transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow'))
    axes[6].set_title('Processing Info')
    axes[6].axis('off')
    
    # 8. Hide last subplot
    axes[7].axis('off')
    
    plt.suptitle(f'Crumbly Texture Detection Results - {image_stem}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save visualization
    save_path = output_dir / f'crumbly_analysis_{image_stem}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
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
    
    # Test each image
    for i, image_path in enumerate(sample_images[:3]):  # Limit to first 3 images for testing
        print(f"\n{'='*20} TEST {i+1}/{min(3, len(sample_images))} {'='*20}")
        
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
    
    # Check for your expected results
    crumbly_count = 0
    smooth_count = 0
    
    for result in successful_results:
        best_fiber = result.get('best_fiber_result', {})
        classification = best_fiber.get('classification', 'unknown')
        
        if classification == 'crumbly':
            crumbly_count += 1
        elif classification == 'smooth':
            smooth_count += 1
    
    print(f"\n🔍 TEXTURE CLASSIFICATION BREAKDOWN:")
    print(f"   Crumbly fibers: {crumbly_count}")
    print(f"   Smooth fibers: {smooth_count}")
    print(f"   Other/Unknown: {len(successful_results) - crumbly_count - smooth_count}")
    
    if successful_tests > 0:
        print(f"\n✅ Crumbly detection module is working!")
        print(f"   Check {output_dir} for detailed visualizations")
        print(f"   Look for 'crumbly_analysis_*.png' files")
    else:
        print(f"\n❌ All tests failed - check the error messages above")
    
    return successful_tests > 0

if __name__ == "__main__":
    success = run_crumbly_detection_tests()
    sys.exit(0 if success else 1)