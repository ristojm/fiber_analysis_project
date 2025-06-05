#!/usr/bin/env python3
"""
Test Scale Detection on Multiple SEM Images
Batch testing to verify robustness across different scale formats and conditions
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
from datetime import datetime

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "modules"))

# Import our improved detection functions
from complete_scale_bar_detection import (
    find_scale_text_simple, 
    find_complete_scale_bar_span,
    parse_scale_text_simple
)

def test_scale_detection_batch():
    """Test scale detection on all available SEM images."""
    
    print("="*80)
    print("BATCH SCALE DETECTION TESTING")
    print("="*80)
    
    # Find all image files
    sample_dir = project_root / "sample_images"
    if not sample_dir.exists():
        print(f"❌ Sample images directory not found: {sample_dir}")
        return
    
    # Look for common SEM image formats
    image_extensions = ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg', '*.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(sample_dir.glob(ext))
        image_files.extend(sample_dir.glob(ext.upper()))
    
    if not image_files:
        print(f"❌ No image files found in {sample_dir}")
        print("Supported formats: .tif, .tiff, .png, .jpg, .jpeg, .bmp")
        return
    
    print(f"Found {len(image_files)} image files:")
    for img_file in image_files:
        print(f"  - {img_file.name}")
    
    # Test each image
    results = []
    
    for i, img_path in enumerate(image_files):
        print(f"\n{'='*60}")
        print(f"TESTING IMAGE {i+1}/{len(image_files)}: {img_path.name}")
        print('='*60)
        
        try:
            result = test_single_image(img_path)
            results.append(result)
        
        except Exception as e:
            print(f"❌ Error processing {img_path.name}: {e}")
            results.append({
                'filename': img_path.name,
                'success': False,
                'error': str(e),
                'scale_detected': False
            })
    
    # Generate summary report
    generate_batch_report(results)
    
    return results

def test_single_image(img_path):
    """Test scale detection on a single image."""
    
    from image_preprocessing import load_image
    from scale_detection import ScaleBarDetector
    
    # Load image
    try:
        image = load_image(str(img_path))
        print(f"✅ Image loaded: {image.shape}")
    except Exception as e:
        raise Exception(f"Failed to load image: {e}")
    
    # Extract scale region
    detector = ScaleBarDetector()
    scale_region, y_offset = detector.extract_scale_region(image)
    print(f"✅ Scale region extracted: {scale_region.shape}")
    
    # Find scale text
    text_candidates = find_scale_text_simple(scale_region)
    
    result = {
        'filename': img_path.name,
        'image_shape': image.shape,
        'scale_region_shape': scale_region.shape,
        'text_candidates_found': len(text_candidates),
        'text_candidates': text_candidates[:3],  # Store top 3
        'success': False,
        'scale_detected': False,
        'scale_value': None,
        'scale_unit': None,
        'span_pixels': None,
        'calculated_scale': None,
        'error': None
    }
    
    if not text_candidates:
        result['error'] = "No valid scale text found"
        print("❌ No valid scale text found")
        return result
    
    # Use best text candidate
    best_text = text_candidates[0]
    print(f"✅ Best text: '{best_text['text']}' = {best_text['micrometers']} μm")
    
    result['scale_value'] = best_text['value']
    result['scale_unit'] = best_text['unit']
    result['scale_micrometers'] = best_text['micrometers']
    
    # Find complete scale bar span
    best_span, all_segments = find_complete_scale_bar_span(
        scale_region, 
        best_text['center_x'], 
        best_text['center_y'], 
        best_text['bbox'],
        debug=False  # Reduce output for batch processing
    )
    
    if best_span:
        span_pixels = best_span['total_span']
        calculated_scale = best_text['micrometers'] / span_pixels
        
        result.update({
            'success': True,
            'scale_detected': True,
            'span_pixels': span_pixels,
            'calculated_scale': calculated_scale,
            'segments_found': len(all_segments),
            'segments_in_best_span': best_span['segment_count'],
            'text_centrality': best_span['text_relative_pos'],
            'span_score': best_span['score']
        })
        
        print(f"✅ Scale bar detected:")
        print(f"   Text: {best_text['value']} {best_text['unit']}")
        print(f"   Span: {span_pixels:.1f} pixels")
        print(f"   Scale: {calculated_scale:.4f} μm/pixel")
        print(f"   Segments: {best_span['segment_count']}")
        print(f"   Score: {best_span['score']:.3f}")
        
    else:
        result['error'] = "Scale bar lines not detected"
        print("❌ Scale bar lines not detected")
    
    return result

def generate_batch_report(results):
    """Generate a comprehensive report of batch testing results."""
    
    print(f"\n{'='*80}")
    print("BATCH TESTING SUMMARY REPORT")
    print('='*80)
    
    total_images = len(results)
    successful = sum(1 for r in results if r['success'])
    text_detected = sum(1 for r in results if r.get('text_candidates_found', 0) > 0)
    
    print(f"Total images tested: {total_images}")
    print(f"Text detection success: {text_detected}/{total_images} ({text_detected/total_images*100:.1f}%)")
    print(f"Complete detection success: {successful}/{total_images} ({successful/total_images*100:.1f}%)")
    
    # Detailed results table
    print(f"\n{'IMAGE':<25} {'TEXT':<15} {'SCALE':<20} {'STATUS':<15}")
    print('-' * 80)
    
    for result in results:
        filename = result['filename'][:24]
        
        if result.get('scale_value'):
            scale_text = f"{result['scale_value']}{result['scale_unit']}"
        else:
            scale_text = "N/A"
        
        if result['success']:
            scale_info = f"{result['calculated_scale']:.4f} μm/px"
            status = "✅ SUCCESS"
        elif result.get('text_candidates_found', 0) > 0:
            scale_info = "Text found"
            status = "⚠️ NO LINES"
        else:
            scale_info = "No text"
            status = "❌ FAILED"
        
        print(f"{filename:<25} {scale_text:<15} {scale_info:<20} {status}")
    
    # Analysis by scale value ranges
    if successful > 0:
        print(f"\n📊 SCALE VALUE ANALYSIS:")
        successful_results = [r for r in results if r['success']]
        
        scale_values = [r['scale_micrometers'] for r in successful_results]
        calculated_scales = [r['calculated_scale'] for r in successful_results]
        
        print(f"Scale value range: {min(scale_values):.1f} - {max(scale_values):.1f} μm")
        print(f"Calculated scale range: {min(calculated_scales):.4f} - {max(calculated_scales):.4f} μm/pixel")
        
        # Group by scale ranges
        nano_range = sum(1 for v in scale_values if v < 1)
        micro_range = sum(1 for v in scale_values if 1 <= v < 1000)
        milli_range = sum(1 for v in scale_values if v >= 1000)
        
        print(f"Scale distribution:")
        print(f"  Nanometer range (<1μm): {nano_range}")
        print(f"  Micrometer range (1-999μm): {micro_range}")
        print(f"  Millimeter range (≥1000μm): {milli_range}")
    
    # Common failure modes
    print(f"\n🔍 FAILURE ANALYSIS:")
    failures = [r for r in results if not r['success']]
    
    if failures:
        error_types = {}
        for failure in failures:
            error = failure.get('error', 'Unknown error')
            error_types[error] = error_types.get(error, 0) + 1
        
        for error, count in error_types.items():
            print(f"  {error}: {count} images")
    else:
        print("  No failures! 🎉")
    
    # Save detailed report
    save_detailed_report(results)

def save_detailed_report(results):
    """Save detailed results to a file."""
    
    output_dir = project_root / 'analysis_results'
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f'scale_detection_batch_test_{timestamp}.txt'
    
    with open(report_file, 'w') as f:
        f.write("SEM Scale Detection Batch Test Report\n")
        f.write("="*50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for result in results:
            f.write(f"Image: {result['filename']}\n")
            f.write(f"  Image shape: {result.get('image_shape', 'N/A')}\n")
            f.write(f"  Text candidates: {result.get('text_candidates_found', 0)}\n")
            
            if result['success']:
                f.write(f"  ✅ SUCCESS\n")
                f.write(f"  Scale text: {result['scale_value']} {result['scale_unit']}\n")
                f.write(f"  Span: {result['span_pixels']:.1f} pixels\n")
                f.write(f"  Calculated scale: {result['calculated_scale']:.6f} μm/pixel\n")
                f.write(f"  Segments: {result['segments_in_best_span']}\n")
                f.write(f"  Text centrality: {result['text_centrality']:.3f}\n")
                f.write(f"  Score: {result['span_score']:.3f}\n")
            else:
                f.write(f"  ❌ FAILED: {result.get('error', 'Unknown error')}\n")
            
            f.write("\n")
    
    print(f"\n💾 Detailed report saved: {report_file}")

def visualize_batch_results(results):
    """Create visualizations of batch test results."""
    
    successful_results = [r for r in results if r['success']]
    
    if len(successful_results) < 2:
        print("Need at least 2 successful results for visualization")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Scale value distribution
    scale_values = [r['scale_micrometers'] for r in successful_results]
    axes[0, 0].hist(scale_values, bins=min(10, len(scale_values)), alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Scale Bar Value (μm)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Distribution of Scale Bar Values')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Calculated scale distribution
    calculated_scales = [r['calculated_scale'] for r in successful_results]
    axes[0, 1].hist(calculated_scales, bins=min(10, len(calculated_scales)), alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Calculated Scale (μm/pixel)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Distribution of Pixel Scales')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Scale value vs pixel scale
    axes[1, 0].scatter(scale_values, calculated_scales, alpha=0.7)
    axes[1, 0].set_xlabel('Scale Bar Value (μm)')
    axes[1, 0].set_ylabel('Calculated Scale (μm/pixel)')
    axes[1, 0].set_title('Scale Value vs Pixel Scale')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Success rate summary
    total = len(results)
    successful = len(successful_results)
    text_only = sum(1 for r in results if r.get('text_candidates_found', 0) > 0 and not r['success'])
    failed = total - successful - text_only
    
    labels = ['Complete Success', 'Text Only', 'Complete Failure']
    sizes = [successful, text_only, failed]
    colors = ['green', 'orange', 'red']
    
    axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('Detection Success Rate')
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = project_root / 'analysis_results'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = output_dir / f'batch_test_visualization_{timestamp}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Visualization saved: {plot_file}")

def test_known_image_first():
    """Test the known working image first to verify the system."""
    
    print("🔧 TESTING KNOWN WORKING IMAGE FIRST...")
    
    known_image = project_root / "sample_images" / "hollow_fiber_sample.jpg"
    
    if not known_image.exists():
        print(f"❌ Known test image not found: {known_image}")
        return False
    
    try:
        result = test_single_image(known_image)
        
        if result['success']:
            print(f"✅ Known image test PASSED")
            print(f"   Scale: {result['calculated_scale']:.4f} μm/pixel")
            return True
        else:
            print(f"❌ Known image test FAILED: {result.get('error', 'Unknown')}")
            return False
    
    except Exception as e:
        print(f"❌ Known image test ERROR: {e}")
        return False

if __name__ == "__main__":
    # First test the known working image
    if test_known_image_first():
        print("\nProceeding with batch testing...\n")
        
        # Run batch testing
        results = test_scale_detection_batch()
        
        # Create visualizations if we have results
        if results and any(r['success'] for r in results):
            print("\nCreating visualizations...")
            visualize_batch_results(results)
        
        print(f"\n🎯 BATCH TESTING COMPLETE!")
        print("Check the analysis_results folder for detailed reports and visualizations.")
    
    else:
        print("\n❌ Known image test failed - check the algorithm before batch testing")
        print("Make sure 'hollow_fiber_sample.jpg' is in the sample_images folder")