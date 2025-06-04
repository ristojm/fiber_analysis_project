#!/usr/bin/env python3
"""
Batch Scale Detection Testing Script
Test the improved scale detection on multiple SEM images
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "modules"))

def find_image_files(directory):
    """Find all SEM image files in directory."""
    
    image_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp']
    image_files = []
    
    directory = Path(directory)
    if not directory.exists():
        return []
    
    for ext in image_extensions:
        image_files.extend(directory.glob(f'*{ext}'))
        image_files.extend(directory.glob(f'*{ext.upper()}'))
    
    # Remove duplicates that might occur from case variations
    unique_files = []
    seen_names = set()
    
    for file_path in sorted(image_files):
        # Use lowercase filename for comparison to avoid duplicates
        file_key = file_path.name.lower()
        if file_key not in seen_names:
            seen_names.add(file_key)
            unique_files.append(file_path)
    
    print(f"🔍 Found image files:")
    for i, img_file in enumerate(unique_files, 1):
        print(f"   {i}. {img_file.name}")
    
    return unique_files

def test_single_image(image_path, verbose=False):
    """Test scale detection on a single image."""
    
    try:
        from modules.image_preprocessing import load_image
        from modules.scale_detection import detect_scale_bar
        
        # Load image
        image = load_image(str(image_path))
        
        if verbose:
            print(f"\n📸 Testing: {image_path.name}")
            print(f"   Image size: {image.shape}")
        
        # Run enhanced scale detection
        result = detect_scale_bar(image, use_enhanced=True)
        
        # Prepare result summary
        test_result = {
            'filename': image_path.name,
            'image_width': image.shape[1],
            'image_height': image.shape[0],
            'success': result['scale_detected'],
            'method': result.get('method_used', 'unknown'),
            'scale_micrometers_per_pixel': result.get('micrometers_per_pixel', 0.0),
            'scale_text_detected': None,
            'scale_value': None,
            'scale_unit': None,
            'bar_length_pixels': result.get('bar_length_pixels', 0),
            'detection_score': result.get('detection_score', 0.0),
            'error_message': result.get('error', None)
        }
        
        if result['scale_detected']:
            scale_info = result.get('scale_info', {})
            test_result.update({
                'scale_text_detected': scale_info.get('original_text', ''),
                'scale_value': scale_info.get('value', 0),
                'scale_unit': scale_info.get('unit', ''),
            })
            
            if verbose:
                print(f"   ✅ SUCCESS: {scale_info.get('value', 0)} {scale_info.get('unit', '')}")
                print(f"      Scale: {result['micrometers_per_pixel']:.4f} μm/pixel")
                print(f"      Bar length: {result.get('bar_length_pixels', 0):.0f} pixels")
                if 'detection_score' in result:
                    print(f"      Score: {result['detection_score']:.3f}")
        else:
            if verbose:
                print(f"   ❌ FAILED: {result.get('error', 'Unknown error')}")
        
        return test_result
        
    except Exception as e:
        error_result = {
            'filename': image_path.name,
            'success': False,
            'error_message': str(e)
        }
        
        if verbose:
            print(f"   💥 ERROR: {e}")
        
        return error_result

def run_batch_test(image_directory, output_dir=None, verbose=True):
    """Run batch scale detection test on all images in directory."""
    
    print("🧪 BATCH SCALE DETECTION TEST")
    print("=" * 50)
    
    # Setup
    image_dir = Path(image_directory)
    if output_dir is None:
        output_dir = project_root / 'analysis_results'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    # Find images
    image_files = find_image_files(image_dir)
    
    if not image_files:
        print(f"❌ No images found in {image_dir}")
        print("Supported formats: .tif, .tiff, .png, .jpg, .jpeg, .bmp")
        return
    
    print(f"📁 Testing {len(image_files)} images from: {image_dir}")
    print(f"📊 Results will be saved to: {output_dir}")
    
    # Test each image
    results = []
    successful = 0
    
    for i, image_path in enumerate(image_files, 1):
        if verbose:
            print(f"\n[{i}/{len(image_files)}]", end="")
        
        result = test_single_image(image_path, verbose=verbose)
        results.append(result)
        
        if result.get('success', False):
            successful += 1
    
    # Generate summary
    print(f"\n" + "=" * 50)
    print("📊 BATCH TEST SUMMARY")
    print("=" * 50)
    print(f"Total images: {len(image_files)}")
    print(f"Successful detections: {successful}")
    print(f"Success rate: {successful/len(image_files)*100:.1f}%")
    
    # Analyze results
    if successful > 0:
        successful_results = [r for r in results if r.get('success', False)]
        
        scales = [r['scale_micrometers_per_pixel'] for r in successful_results if r.get('scale_micrometers_per_pixel', 0) > 0]
        scale_values = [r['scale_value'] for r in successful_results if r.get('scale_value')]
        
        if scales:
            print(f"\n📏 Scale Analysis:")
            print(f"   Scale range: {min(scales):.4f} - {max(scales):.4f} μm/pixel")
            print(f"   Mean scale: {np.mean(scales):.4f} μm/pixel")
            print(f"   Median scale: {np.median(scales):.4f} μm/pixel")
        
        if scale_values:
            print(f"\n🔢 Scale Value Analysis:")
            print(f"   Value range: {min(scale_values)} - {max(scale_values)}")
            print(f"   Common values: {list(set(scale_values))}")
    
    # Show failures
    failures = [r for r in results if not r.get('success', False)]
    if failures:
        print(f"\n❌ Failed Images ({len(failures)}):")
        for failure in failures[:5]:  # Show first 5
            error = failure.get('error_message', 'Unknown error')
            print(f"   {failure['filename']}: {error}")
        if len(failures) > 5:
            print(f"   ... and {len(failures) - 5} more")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as CSV
    df = pd.DataFrame(results)
    csv_file = output_dir / f'batch_scale_test_{timestamp}.csv'
    df.to_csv(csv_file, index=False)
    print(f"\n💾 Detailed results saved to: {csv_file}")
    
    # Save summary report
    report_file = output_dir / f'batch_scale_summary_{timestamp}.txt'
    with open(report_file, 'w', encoding='utf-8') as f:  # FIX: Add UTF-8 encoding
        f.write("SEM Scale Detection Batch Test Report\n")
        f.write("=" * 40 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test directory: {image_dir}\n\n")
        
        f.write(f"Summary:\n")
        f.write(f"  Total images: {len(image_files)}\n")
        f.write(f"  Successful: {successful}\n")
        f.write(f"  Failed: {len(image_files) - successful}\n")
        f.write(f"  Success rate: {successful/len(image_files)*100:.1f}%\n\n")
        
        if successful > 0:
            successful_results = [r for r in results if r.get('success', False)]
            scales = [r['scale_micrometers_per_pixel'] for r in successful_results if r.get('scale_micrometers_per_pixel', 0) > 0]
            
            if scales:
                f.write(f"Scale Statistics:\n")
                f.write(f"  Range: {min(scales):.4f} - {max(scales):.4f} um/pixel\n")  # FIX: Use 'um' instead of μ
                f.write(f"  Mean: {np.mean(scales):.4f} um/pixel\n")
                f.write(f"  Median: {np.median(scales):.4f} um/pixel\n")
                f.write(f"  Std Dev: {np.std(scales):.4f} um/pixel\n\n")
        
        f.write("Individual Results:\n")
        f.write("-" * 20 + "\n")
        for result in results:
            status = "SUCCESS" if result.get('success', False) else "FAILED"  # FIX: Remove emoji
            f.write(f"{result['filename']}: {status}\n")
            
            if result.get('success', False):
                f.write(f"  Scale: {result.get('scale_micrometers_per_pixel', 0):.4f} um/pixel\n")
                f.write(f"  Text: {result.get('scale_text_detected', 'N/A')}\n")
            else:
                f.write(f"  Error: {result.get('error_message', 'Unknown')}\n")
            f.write("\n")
    
    print(f"📄 Summary report saved to: {report_file}")
    
    # Create visualization if we have successful results
    if successful > 1:
        create_results_visualization(results, output_dir, timestamp)
    
    return results

def create_results_visualization(results, output_dir, timestamp):
    """Create visualization of batch test results."""
    
    successful_results = [r for r in results if r.get('success', False)]
    
    if len(successful_results) < 2:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Success rate pie chart
    success_count = len(successful_results)
    fail_count = len(results) - success_count
    
    axes[0, 0].pie([success_count, fail_count], 
                   labels=['Success', 'Failed'], 
                   colors=['green', 'red'],
                   autopct='%1.1f%%')
    axes[0, 0].set_title('Detection Success Rate')
    
    # 2. Scale distribution histogram
    scales = [r['scale_micrometers_per_pixel'] for r in successful_results 
              if r.get('scale_micrometers_per_pixel', 0) > 0]
    
    if scales:
        axes[0, 1].hist(scales, bins=min(10, len(scales)), alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Scale (μm/pixel)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Scale Distribution')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Scale values bar chart
    scale_values = [r['scale_value'] for r in successful_results if r.get('scale_value')]
    if scale_values:
        from collections import Counter
        value_counts = Counter(scale_values)
        
        axes[1, 0].bar(range(len(value_counts)), list(value_counts.values()))
        axes[1, 0].set_xlabel('Scale Values')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Scale Value Frequency')
        axes[1, 0].set_xticks(range(len(value_counts)))
        axes[1, 0].set_xticklabels([f"{v}" for v in value_counts.keys()], rotation=45)
    
    # 4. Results summary text
    axes[1, 1].axis('off')
    
    summary_text = f"""BATCH TEST RESULTS

Total Images: {len(results)}
Successful: {len(successful_results)}
Success Rate: {len(successful_results)/len(results)*100:.1f}%

Scale Statistics:
"""
    
    if scales:
        summary_text += f"""  Min: {min(scales):.4f} μm/px
  Max: {max(scales):.4f} μm/px  
  Mean: {np.mean(scales):.4f} μm/px
  Median: {np.median(scales):.4f} μm/px"""
    
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                    verticalalignment='top', fontsize=11, fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save visualization
    viz_file = output_dir / f'batch_test_visualization_{timestamp}.png'
    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Visualization saved to: {viz_file}")

def main():
    """Main function with command line interface."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch test SEM scale detection')
    parser.add_argument('--input', '-i', 
                       default=str(project_root / 'sample_images'),
                       help='Input directory containing SEM images')
    parser.add_argument('--output', '-o',
                       default=str(project_root / 'analysis_results'),
                       help='Output directory for results')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--single', '-s',
                       help='Test single image file')
    
    args = parser.parse_args()
    
    if args.single:
        # Test single image
        image_path = Path(args.single)
        if not image_path.exists():
            print(f"❌ Image not found: {image_path}")
            return
        
        print("🔍 SINGLE IMAGE TEST")
        print("=" * 30)
        result = test_single_image(image_path, verbose=True)
        
        if result.get('success', False):
            print(f"\n✅ SUCCESS!")
        else:
            print(f"\n❌ FAILED!")
    
    else:
        # Run batch test
        results = run_batch_test(args.input, args.output, args.verbose)
        
        print(f"\n🎯 BATCH TEST COMPLETE!")
        print(f"Check {args.output} for detailed results.")

if __name__ == "__main__":
    main()