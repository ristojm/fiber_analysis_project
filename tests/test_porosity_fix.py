#!/usr/bin/env python3
"""
Test Porosity Fix - Run from tests/ folder
Tests the fixed porosity analysis against the original buggy version

Run from tests/ folder:
    cd tests
    python test_porosity_fix.py
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Setup paths to access parent directory modules
test_dir = Path(__file__).parent
project_root = test_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "modules"))

print(f"🧪 POROSITY FIX TEST")
print(f"Test directory: {test_dir}")
print(f"Project root: {project_root}")
print("=" * 60)

# Import modules from parent directory
try:
    from modules.scale_detection import detect_scale_bar
    from modules.fiber_type_detection import FiberTypeDetector
    from modules.image_preprocessing import load_image
    from modules.porosity_analysis import EnhancedPorosityAnalyzer
    print("✅ Successfully imported modules from parent directory")
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    sys.exit(1)

class FixedPorosityAnalyzer:
    """
    Fixed porosity analyzer that correctly calculates areas and detects pores.
    """
    
    def __init__(self, config=None):
        """Initialize with corrected default parameters."""
        self.config = {
            'pore_detection': {
                'adaptive_threshold': True,
                'intensity_percentile': 25,  # Use 25th percentile as base threshold
                'min_pore_area_pixels': 20,  # Minimum 20 pixels
                'max_pore_area_ratio': 0.1,  # Max 10% of fiber area
                'morphology_iterations': 2,
                'remove_border_pores': True,
            },
            'segmentation': {
                'gaussian_blur': 2,  # Slight blur before thresholding
            },
            'hollow_fiber': {
                'exclude_lumen': True,  # Critical: exclude lumen from analysis
                'lumen_buffer_pixels': 5,  # Small buffer around lumen edge
            }
        }
        
        if config:
            self._update_config(config)
    
    def _update_config(self, new_config):
        """Update configuration recursively."""
        for key, value in new_config.items():
            if key in self.config and isinstance(self.config[key], dict):
                self.config[key].update(value)
            else:
                self.config[key] = value
    
    def analyze_fiber_porosity_fixed(self, 
                                   image, 
                                   fiber_analysis_data,
                                   scale_factor=1.0):
        """
        Fixed porosity analysis that correctly handles individual fibers and hollow fiber lumens.
        """
        
        print(f"\n🔧 FIXED POROSITY ANALYSIS")
        print(f"   Scale factor: {scale_factor:.4f} μm/pixel")
        
        # Get individual fiber results
        individual_results = fiber_analysis_data.get('individual_results', [])
        if not individual_results:
            return {
                'error': 'No individual fiber results found',
                'porosity_metrics': {'total_porosity_percent': 0.0}
            }
        
        # Analyze each fiber individually with correct area calculation
        all_pore_results = []
        total_wall_area_pixels = 0
        total_pore_area_pixels = 0
        
        for i, fiber_result in enumerate(individual_results):
            fiber_props = fiber_result.get('fiber_properties', {})
            fiber_contour = fiber_props.get('contour')
            has_lumen = fiber_result.get('has_lumen', False)
            
            if fiber_contour is None:
                continue
            
            print(f"\n   Analyzing Fiber {i+1}:")
            
            # Create individual fiber mask
            fiber_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(fiber_mask, [fiber_contour], 255)
            
            # Calculate correct analysis area (exclude lumen for hollow fibers)
            analysis_mask = fiber_mask.copy()
            fiber_area_pixels = np.sum(fiber_mask > 0)
            
            if has_lumen and self.config['hollow_fiber']['exclude_lumen']:
                lumen_props = fiber_result.get('lumen_properties', {})
                lumen_contour = lumen_props.get('contour')
                
                if lumen_contour is not None:
                    # Create lumen mask
                    lumen_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(lumen_mask, [lumen_contour], 255)
                    
                    # Subtract lumen from analysis area
                    analysis_mask = cv2.bitwise_and(analysis_mask, cv2.bitwise_not(lumen_mask))
                    
                    lumen_area_pixels = np.sum(lumen_mask > 0)
                    wall_area_pixels = fiber_area_pixels - lumen_area_pixels
                    
                    print(f"     Hollow fiber detected:")
                    print(f"     Total fiber area: {fiber_area_pixels:,} pixels ({fiber_area_pixels * scale_factor**2:,.0f} μm²)")
                    print(f"     Lumen area: {lumen_area_pixels:,} pixels ({lumen_area_pixels * scale_factor**2:,.0f} μm²)")
                    print(f"     Wall area for analysis: {wall_area_pixels:,} pixels ({wall_area_pixels * scale_factor**2:,.0f} μm²)")
                else:
                    wall_area_pixels = fiber_area_pixels
                    print(f"     Hollow fiber but no lumen contour found, using full area")
            else:
                wall_area_pixels = fiber_area_pixels
                print(f"     Solid fiber area: {wall_area_pixels:,} pixels ({wall_area_pixels * scale_factor**2:,.0f} μm²)")
            
            # Detect pores in this fiber using improved algorithm
            fiber_pores = self._detect_pores_improved(image, analysis_mask, scale_factor)
            
            # Add fiber information to each pore
            for pore in fiber_pores:
                pore['fiber_id'] = i
                pore['fiber_wall_area_pixels'] = wall_area_pixels
                pore['fiber_wall_area_um2'] = wall_area_pixels * (scale_factor ** 2)
            
            all_pore_results.extend(fiber_pores)
            total_wall_area_pixels += wall_area_pixels
            
            pore_area_pixels = sum(pore['area_pixels'] for pore in fiber_pores)
            total_pore_area_pixels += pore_area_pixels
            
            fiber_porosity = (pore_area_pixels / wall_area_pixels * 100) if wall_area_pixels > 0 else 0
            
            print(f"     Pores detected: {len(fiber_pores)}")
            print(f"     Pore area: {pore_area_pixels:,} pixels ({pore_area_pixels * scale_factor**2:,.0f} μm²)")
            print(f"     Fiber porosity: {fiber_porosity:.2f}%")
        
        # Calculate corrected overall metrics
        total_wall_area_um2 = total_wall_area_pixels * (scale_factor ** 2)
        total_pore_area_um2 = total_pore_area_pixels * (scale_factor ** 2)
        overall_porosity = (total_pore_area_pixels / total_wall_area_pixels * 100) if total_wall_area_pixels > 0 else 0
        
        print(f"\n🎯 CORRECTED OVERALL RESULTS:")
        print(f"   Total wall area: {total_wall_area_pixels:,} pixels ({total_wall_area_um2:,.0f} μm²)")
        print(f"   Total pore area: {total_pore_area_pixels:,} pixels ({total_pore_area_um2:,.0f} μm²)")
        print(f"   CORRECTED POROSITY: {overall_porosity:.2f}%")
        print(f"   Total pores: {len(all_pore_results)}")
        
        # Create comprehensive results
        porosity_metrics = self._calculate_corrected_metrics(
            all_pore_results, total_wall_area_um2, total_pore_area_um2, overall_porosity
        )
        
        return {
            'porosity_metrics': porosity_metrics,
            'individual_pores': all_pore_results,
            'method': 'fixed_individual_fiber_analysis',
            'scale_factor': scale_factor,
            'fibers_analyzed': len(individual_results)
        }
    
    def _detect_pores_improved(self, image, analysis_mask, scale_factor):
        """
        Improved pore detection specifically tuned for SEM images.
        """
        
        # Extract region for analysis
        masked_image = cv2.bitwise_and(image, image, mask=analysis_mask)
        region_pixels = masked_image[analysis_mask > 0]
        
        if len(region_pixels) == 0:
            return []
        
        # Calculate adaptive threshold based on image statistics
        if self.config['pore_detection']['adaptive_threshold']:
            # Use percentile-based threshold - more robust for SEM images
            percentile = self.config['pore_detection']['intensity_percentile']
            base_threshold = np.percentile(region_pixels, percentile)
            
            # Adjust based on image contrast
            region_std = np.std(region_pixels)
            if region_std < 20:  # Low contrast image
                threshold = base_threshold * 0.8
            else:  # Good contrast
                threshold = base_threshold * 0.9
                
            print(f"     Adaptive threshold: {threshold:.1f} (base: {base_threshold:.1f}, std: {region_std:.1f})")
        else:
            # Fallback to basic threshold
            threshold = np.mean(region_pixels) - np.std(region_pixels)
            print(f"     Basic threshold: {threshold:.1f}")
        
        # Apply threshold to create pore mask
        pore_mask = (masked_image < threshold) & (analysis_mask > 0)
        
        # Apply Gaussian blur if specified
        blur_size = self.config['segmentation']['gaussian_blur']
        if blur_size > 0:
            pore_mask = pore_mask.astype(np.uint8) * 255
            pore_mask = cv2.GaussianBlur(pore_mask, (blur_size*2+1, blur_size*2+1), 0)
            pore_mask = pore_mask > 127
        
        # Morphological cleanup
        iterations = self.config['pore_detection']['morphology_iterations']
        if iterations > 0:
            # Close small gaps
            kernel_close = np.ones((3, 3), np.uint8)
            pore_mask = cv2.morphologyEx(pore_mask.astype(np.uint8), cv2.MORPH_CLOSE, 
                                       kernel_close, iterations=iterations)
            
            # Remove small noise
            kernel_open = np.ones((2, 2), np.uint8)
            pore_mask = cv2.morphologyEx(pore_mask, cv2.MORPH_OPEN, 
                                       kernel_open, iterations=1)
        
        # Find pore contours
        contours, _ = cv2.findContours(pore_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and analyze pores
        min_area_pixels = max(self.config['pore_detection']['min_pore_area_pixels'], 
                            int(10 / (scale_factor ** 2)))  # At least 10 μm² in real units
        
        max_area_pixels = int(np.sum(analysis_mask > 0) * self.config['pore_detection']['max_pore_area_ratio'])
        
        pores = []
        for contour in contours:
            area_pixels = cv2.contourArea(contour)
            
            # Size filtering
            if area_pixels < min_area_pixels or area_pixels > max_area_pixels:
                continue
            
            # Calculate properties
            pore_props = self._calculate_pore_properties(contour, area_pixels, scale_factor)
            
            # Shape filtering - remove very elongated objects (likely artifacts)
            if pore_props['aspect_ratio'] > 5:  # Very elongated
                continue
            
            # Circularity filtering - keep reasonable shapes
            if pore_props['circularity'] < 0.1:  # Very irregular
                continue
            
            pores.append(pore_props)
        
        print(f"     Threshold: {threshold:.1f}, Raw detections: {len(contours)}, Filtered pores: {len(pores)}")
        
        return pores
    
    def _calculate_pore_properties(self, contour, area_pixels, scale_factor):
        """Calculate comprehensive pore properties."""
        
        # Basic measurements
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        
        # Convert to real units
        area_um2 = area_pixels * (scale_factor ** 2)
        perimeter_um = perimeter * scale_factor
        equivalent_diameter_um = 2 * np.sqrt(area_um2 / np.pi)
        
        # Shape descriptors
        circularity = 4 * np.pi * area_pixels / (perimeter ** 2) if perimeter > 0 else 0
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1
        extent = area_pixels / (w * h) if w * h > 0 else 0
        
        return {
            'contour': contour,
            'area_pixels': area_pixels,
            'area_um2': area_um2,
            'perimeter_pixels': perimeter,
            'perimeter_um': perimeter_um,
            'equivalent_diameter_um': equivalent_diameter_um,
            'centroid_x': cx,
            'centroid_y': cy,
            'bbox': (x, y, w, h),
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'extent': extent,
            'radius_pixels': radius,
            'radius_um': radius * scale_factor
        }
    
    def _calculate_corrected_metrics(self, pores, total_wall_area_um2, total_pore_area_um2, overall_porosity):
        """Calculate comprehensive porosity metrics."""
        
        if not pores:
            return {
                'total_porosity_percent': 0.0,
                'pore_count': 0,
                'total_pore_area_um2': 0.0,
                'wall_area_um2': total_wall_area_um2,
                'average_pore_size_um2': 0.0
            }
        
        # Size statistics
        pore_areas = [pore['area_um2'] for pore in pores]
        pore_diameters = [pore['equivalent_diameter_um'] for pore in pores]
        
        metrics = {
            'total_porosity_percent': overall_porosity,
            'pore_count': len(pores),
            'total_pore_area_um2': total_pore_area_um2,
            'wall_area_um2': total_wall_area_um2,  # CORRECT: wall area, not total mask
            'average_pore_size_um2': np.mean(pore_areas),
            'median_pore_size_um2': np.median(pore_areas),
            'std_pore_size_um2': np.std(pore_areas),
            'min_pore_size_um2': np.min(pore_areas),
            'max_pore_size_um2': np.max(pore_areas),
            'mean_pore_diameter_um': np.mean(pore_diameters),
            'median_pore_diameter_um': np.median(pore_diameters),
            'pore_density_per_mm2': len(pores) / (total_wall_area_um2 / 1e6) if total_wall_area_um2 > 0 else 0,
        }
        
        # Size distribution
        size_bins = [0, 10, 50, 100, 500, 1000, 5000, float('inf')]
        size_labels = ['<10', '10-50', '50-100', '100-500', '500-1000', '1000-5000', '>5000']
        
        for i, (bin_start, bin_end, label) in enumerate(zip(size_bins[:-1], size_bins[1:], size_labels)):
            count = len([p for p in pore_areas if bin_start <= p < bin_end])
            metrics[f'pores_{label.replace("<", "under_").replace(">", "over_").replace("-", "_to_")}_um2'] = count
        
        return metrics

def create_comparison_visualization(image, original_result, fixed_result, scale_factor, save_path):
    """Create side-by-side comparison of original vs fixed analysis."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    axes[0,0].imshow(image, cmap='gray')
    axes[0,0].set_title('Original SEM Image')
    axes[0,0].axis('off')
    
    # Original results text
    orig_metrics = original_result.get('porosity_metrics', {})
    orig_text = f"ORIGINAL ANALYSIS:\n\n"
    orig_text += f"Porosity: {orig_metrics.get('total_porosity_percent', 0):.2f}%\n"
    orig_text += f"Pore count: {orig_metrics.get('pore_count', 0)}\n"
    orig_text += f"Fiber area: {orig_metrics.get('fiber_area_um2', 0):,.0f} μm²\n"
    orig_text += f"Avg pore size: {orig_metrics.get('average_pore_size_um2', 0):.1f} μm²\n\n"
    orig_text += f"❌ PROBLEM: Using total mask area\ninstead of wall area"
    
    axes[0,1].text(0.05, 0.95, orig_text, transform=axes[0,1].transAxes,
                  fontsize=11, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcoral'))
    axes[0,1].set_title('Original Analysis (WRONG)')
    axes[0,1].axis('off')
    
    # Fixed results text
    fixed_metrics = fixed_result.get('porosity_metrics', {})
    fixed_text = f"FIXED ANALYSIS:\n\n"
    fixed_text += f"Porosity: {fixed_metrics.get('total_porosity_percent', 0):.2f}%\n"
    fixed_text += f"Pore count: {fixed_metrics.get('pore_count', 0)}\n"
    fixed_text += f"Wall area: {fixed_metrics.get('wall_area_um2', 0):,.0f} μm²\n"
    fixed_text += f"Avg pore size: {fixed_metrics.get('average_pore_size_um2', 0):.1f} μm²\n\n"
    fixed_text += f"✅ CORRECT: Using actual wall area\nwith improved pore detection"
    
    axes[0,2].text(0.05, 0.95, fixed_text, transform=axes[0,2].transAxes,
                  fontsize=11, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen'))
    axes[0,2].set_title('Fixed Analysis (CORRECT)')
    axes[0,2].axis('off')
    
    # Pore size distribution comparison
    if 'individual_pores' in fixed_result:
        fixed_pores = fixed_result['individual_pores']
        fixed_areas = [p['area_um2'] for p in fixed_pores]
        
        if fixed_areas:
            axes[1,0].hist(fixed_areas, bins=30, alpha=0.7, color='green', edgecolor='black')
            axes[1,0].set_xlabel('Pore Size (μm²)')
            axes[1,0].set_ylabel('Count')
            axes[1,0].set_title('Fixed: Pore Size Distribution')
            axes[1,0].set_xlim(0, max(fixed_areas))
    
    # Detection visualization
    if 'individual_pores' in fixed_result:
        pore_overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        for pore in fixed_result['individual_pores']:
            if 'contour' in pore:
                # Color by size: green=small, yellow=medium, red=large
                area = pore['area_um2']
                if area < 50:
                    color = (0, 255, 0)  # Green
                elif area < 500:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 0, 255)  # Red
                
                cv2.drawContours(pore_overlay, [pore['contour']], -1, color, 2)
        
        axes[1,1].imshow(pore_overlay)
        axes[1,1].set_title('Fixed: Detected Pores\n(Green<50, Yellow<500, Red>500 μm²)')
        axes[1,1].axis('off')
    
    # Comparison bar chart
    categories = ['Porosity %', 'Pore Count']
    original_values = [orig_metrics.get('total_porosity_percent', 0), orig_metrics.get('pore_count', 0)/10]  # Scale pore count for display
    fixed_values = [fixed_metrics.get('total_porosity_percent', 0), fixed_metrics.get('pore_count', 0)/10]
    
    x = np.arange(len(categories))
    width = 0.35
    
    axes[1,2].bar(x - width/2, original_values, width, label='Original (Wrong)', color='lightcoral')
    axes[1,2].bar(x + width/2, fixed_values, width, label='Fixed (Correct)', color='lightgreen')
    
    axes[1,2].set_xlabel('Metric')
    axes[1,2].set_ylabel('Value')
    axes[1,2].set_title('Original vs Fixed Comparison\n(Pore count scaled /10)')
    axes[1,2].set_xticks(x)
    axes[1,2].set_xticklabels(categories)
    axes[1,2].legend()
    
    # Add value labels on bars
    for i, (orig, fixed) in enumerate(zip(original_values, fixed_values)):
        if i == 0:  # Porosity percentage
            axes[1,2].text(i - width/2, orig + max(original_values)*0.01, f'{orig:.1f}%', 
                          ha='center', va='bottom', fontweight='bold')
            axes[1,2].text(i + width/2, fixed + max(fixed_values)*0.01, f'{fixed:.1f}%', 
                          ha='center', va='bottom', fontweight='bold')
        else:  # Pore count
            axes[1,2].text(i - width/2, orig + max(original_values)*0.01, f'{int(orig*10)}', 
                          ha='center', va='bottom', fontweight='bold')
            axes[1,2].text(i + width/2, fixed + max(fixed_values)*0.01, f'{int(fixed*10)}', 
                          ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Comparison visualization saved: {save_path}")

def test_fixed_porosity():
    """Test the fixed porosity analysis on the sample image."""
    
    # Setup test results directory
    results_dir = test_dir / "test_results"
    results_dir.mkdir(exist_ok=True)
    
    # Look for the sample image
    image_paths = [
        project_root / "sample_images" / "28d_001.jpg",
        project_root / "28d_001.jpg",
        project_root / "sample_images" / "28d_001.tif",
    ]
    
    image_path = None
    for path in image_paths:
        if path.exists():
            image_path = str(path)
            break
    
    if not image_path:
        print("❌ Could not find 28d_001.jpg in expected locations:")
        for path in image_paths:
            print(f"   {path}")
        return
    
    print(f"📸 Loading image: {image_path}")
    
    # Load and analyze image
    image = load_image(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    print(f"✅ Image loaded: {image.shape}")
    
    # Get scale factor
    print("\n📏 Detecting scale...")
    scale_result = detect_scale_bar(image, debug=False)
    scale_factor = scale_result.get('micrometers_per_pixel', 1.0) if scale_result.get('scale_detected') else 1.0
    print(f"   Scale factor: {scale_factor:.4f} μm/pixel")
    
    # Detect fibers
    print("\n🧬 Detecting fibers...")
    detector = FiberTypeDetector()
    fiber_type, confidence, analysis_data = detector.classify_fiber_type(image)
    print(f"   Fiber type: {fiber_type} (confidence: {confidence:.3f})")
    
    # Original analysis (buggy)
    print("\n🐛 Running original (buggy) analysis...")
    try:
        original_analyzer = EnhancedPorosityAnalyzer()
        fiber_mask = analysis_data.get('fiber_mask', np.zeros_like(image))
        original_result = original_analyzer.analyze_fiber_porosity(
            image, fiber_mask.astype(np.uint8), scale_factor, fiber_type, analysis_data
        )
        print("✅ Original analysis completed")
    except Exception as e:
        print(f"❌ Original analysis failed: {e}")
        original_result = {
            'porosity_metrics': {
                'total_porosity_percent': 0.0,
                'pore_count': 0,
                'fiber_area_um2': 0.0,
                'average_pore_size_um2': 0.0
            }
        }
    
    # Fixed analysis
    print("\n🔧 Running fixed analysis...")
    try:
        fixed_analyzer = FixedPorosityAnalyzer()
        fixed_result = fixed_analyzer.analyze_fiber_porosity_fixed(
            image, analysis_data, scale_factor
        )
        print("✅ Fixed analysis completed")
    except Exception as e:
        print(f"❌ Fixed analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create comparison visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_path = results_dir / f"porosity_comparison_{timestamp}.png"
    create_comparison_visualization(image, original_result, fixed_result, scale_factor, str(viz_path))
    
    # Save detailed results
    results_path = results_dir / f"porosity_test_results_{timestamp}.json"
    
    # Prepare results for JSON (remove numpy objects)
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    test_results = {
        'timestamp': timestamp,
        'image_path': image_path,
        'scale_factor': scale_factor,
        'fiber_type': fiber_type,
        'original_analysis': clean_for_json(original_result),
        'fixed_analysis': clean_for_json(fixed_result)
    }
    
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"💾 Test results saved: {results_path}")
    
    # Print comparison summary
    orig_porosity = original_result.get('porosity_metrics', {}).get('total_porosity_percent', 0)
    fixed_porosity = fixed_result.get('porosity_metrics', {}).get('total_porosity_percent', 0)
    
    print(f"\n🎯 COMPARISON SUMMARY:")
    print(f"Original (buggy): {orig_porosity:.2f}% porosity")
    print(f"Fixed (correct): {fixed_porosity:.2f}% porosity")
    if orig_porosity > 0:
        improvement = fixed_porosity / orig_porosity
        print(f"Improvement factor: {improvement:.1f}x")
    
    orig_count = original_result.get('porosity_metrics', {}).get('pore_count', 0)
    fixed_count = fixed_result.get('porosity_metrics', {}).get('pore_count', 0)
    print(f"Original pore count: {orig_count}")
    print(f"Fixed pore count: {fixed_count}")
    
    print(f"\n✅ Test completed! Check {results_dir} for detailed results.")
    
    return fixed_result

if __name__ == "__main__":
    result = test_fixed_porosity()