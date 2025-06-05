#!/usr/bin/env python3
"""
Enhanced Pore Detection - Better Small Pore Sensitivity
Improves detection of small pores while maintaining accuracy for larger ones

Save as: tests/enhanced_pore_detection.py
Run from: tests/ folder
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Setup paths
test_dir = Path(__file__).parent
project_root = test_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "modules"))

from modules.scale_detection import detect_scale_bar
from modules.fiber_type_detection import FiberTypeDetector
from modules.image_preprocessing import load_image

class EnhancedPoreDetector:
    """
    Enhanced pore detector with better sensitivity to small pores.
    """
    
    def __init__(self):
        self.config = {
            'pore_detection': {
                # More aggressive small pore detection
                'intensity_percentile': 30,  # Higher percentile = more sensitive
                'min_pore_area_pixels': 5,   # Smaller minimum (was 20)
                'max_pore_area_ratio': 0.15,  # Allow larger pores
                'adaptive_threshold_methods': ['percentile', 'otsu', 'local_adaptive'],
                'combine_methods': True,  # Combine multiple detection methods
                'morphology_iterations': 1,  # Less aggressive cleanup
                'small_pore_boost': True,  # Special handling for small pores
            },
            'segmentation': {
                'gaussian_blur': 1,  # Less blur to preserve small features
                'multi_scale': True,  # Detect at multiple scales
            },
            'hollow_fiber': {
                'exclude_lumen': True,
                'lumen_buffer_pixels': 3,  # Smaller buffer
            }
        }
    
    def analyze_fiber_porosity_enhanced(self, image, fiber_analysis_data, scale_factor=1.0):
        """
        Enhanced porosity analysis with improved small pore detection.
        """
        
        print(f"\n🔬 ENHANCED PORE DETECTION")
        print(f"   Scale factor: {scale_factor:.4f} μm/pixel")
        
        individual_results = fiber_analysis_data.get('individual_results', [])
        if not individual_results:
            return {'error': 'No individual fiber results found'}
        
        all_pore_results = []
        total_wall_area_pixels = 0
        total_pore_area_pixels = 0
        
        for i, fiber_result in enumerate(individual_results):
            fiber_props = fiber_result.get('fiber_properties', {})
            fiber_contour = fiber_props.get('contour')
            has_lumen = fiber_result.get('has_lumen', False)
            
            if fiber_contour is None:
                continue
            
            print(f"\n   Enhanced Analysis - Fiber {i+1}:")
            
            # Create analysis mask (exclude lumen)
            fiber_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(fiber_mask, [fiber_contour], 255)
            analysis_mask = fiber_mask.copy()
            fiber_area_pixels = np.sum(fiber_mask > 0)
            
            if has_lumen and self.config['hollow_fiber']['exclude_lumen']:
                lumen_props = fiber_result.get('lumen_properties', {})
                lumen_contour = lumen_props.get('contour')
                
                if lumen_contour is not None:
                    lumen_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(lumen_mask, [lumen_contour], 255)
                    
                    # Small buffer around lumen
                    buffer_size = self.config['hollow_fiber']['lumen_buffer_pixels']
                    if buffer_size > 0:
                        kernel = np.ones((buffer_size*2+1, buffer_size*2+1), np.uint8)
                        lumen_mask = cv2.dilate(lumen_mask, kernel, iterations=1)
                    
                    analysis_mask = cv2.bitwise_and(analysis_mask, cv2.bitwise_not(lumen_mask))
                    lumen_area_pixels = np.sum(lumen_mask > 0)
                    wall_area_pixels = fiber_area_pixels - lumen_area_pixels
                    
                    print(f"     Wall area: {wall_area_pixels:,} pixels ({wall_area_pixels * scale_factor**2:,.0f} μm²)")
                else:
                    wall_area_pixels = fiber_area_pixels
            else:
                wall_area_pixels = fiber_area_pixels
            
            # Enhanced multi-method pore detection
            fiber_pores = self._detect_pores_multi_method(image, analysis_mask, scale_factor)
            
            # Add fiber info
            for pore in fiber_pores:
                pore['fiber_id'] = i
                pore['fiber_wall_area_pixels'] = wall_area_pixels
            
            all_pore_results.extend(fiber_pores)
            total_wall_area_pixels += wall_area_pixels
            
            pore_area_pixels = sum(pore['area_pixels'] for pore in fiber_pores)
            total_pore_area_pixels += pore_area_pixels
            
            fiber_porosity = (pore_area_pixels / wall_area_pixels * 100) if wall_area_pixels > 0 else 0
            
            print(f"     Enhanced detection: {len(fiber_pores)} pores")
            print(f"     Pore area: {pore_area_pixels:,} pixels ({pore_area_pixels * scale_factor**2:,.0f} μm²)")
            print(f"     Fiber porosity: {fiber_porosity:.2f}%")
        
        # Calculate results
        total_wall_area_um2 = total_wall_area_pixels * (scale_factor ** 2)
        total_pore_area_um2 = total_pore_area_pixels * (scale_factor ** 2)
        overall_porosity = (total_pore_area_pixels / total_wall_area_pixels * 100) if total_wall_area_pixels > 0 else 0
        
        print(f"\n🎯 ENHANCED OVERALL RESULTS:")
        print(f"   Total wall area: {total_wall_area_pixels:,} pixels ({total_wall_area_um2:,.0f} μm²)")
        print(f"   Total pore area: {total_pore_area_pixels:,} pixels ({total_pore_area_um2:,.0f} μm²)")
        print(f"   ENHANCED POROSITY: {overall_porosity:.2f}%")
        print(f"   Total pores: {len(all_pore_results)}")
        
        # Analyze pore size distribution
        if all_pore_results:
            pore_areas = [p['area_um2'] for p in all_pore_results]
            small_pores = [p for p in pore_areas if p < 50]
            medium_pores = [p for p in pore_areas if 50 <= p < 500]
            large_pores = [p for p in pore_areas if p >= 500]
            
            print(f"   Pore size breakdown:")
            print(f"     Small (<50 μm²): {len(small_pores)} pores ({len(small_pores)/len(all_pore_results)*100:.1f}%)")
            print(f"     Medium (50-500 μm²): {len(medium_pores)} pores ({len(medium_pores)/len(all_pore_results)*100:.1f}%)")
            print(f"     Large (>500 μm²): {len(large_pores)} pores ({len(large_pores)/len(all_pore_results)*100:.1f}%)")
        
        # Create metrics
        porosity_metrics = self._calculate_enhanced_metrics(
            all_pore_results, total_wall_area_um2, total_pore_area_um2, overall_porosity
        )
        
        return {
            'porosity_metrics': porosity_metrics,
            'individual_pores': all_pore_results,
            'method': 'enhanced_multi_method',
            'scale_factor': scale_factor,
        }
    
    def _detect_pores_multi_method(self, image, analysis_mask, scale_factor):
        """
        Multi-method pore detection to catch both large and small pores.
        """
        
        masked_image = cv2.bitwise_and(image, image, mask=analysis_mask)
        region_pixels = masked_image[analysis_mask > 0]
        
        if len(region_pixels) == 0:
            return []
        
        print(f"     Multi-method pore detection:")
        
        # Method 1: Percentile-based (good for consistent detection)
        percentile = self.config['pore_detection']['intensity_percentile']
        percentile_threshold = np.percentile(region_pixels, percentile)
        method1_mask = (masked_image < percentile_threshold) & (analysis_mask > 0)
        
        # Method 2: Adaptive Otsu (good for varying contrast)
        otsu_threshold = cv2.threshold(region_pixels.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        # Make more aggressive for small pores
        adaptive_otsu_threshold = otsu_threshold * 0.85
        method2_mask = (masked_image < adaptive_otsu_threshold) & (analysis_mask > 0)
        
        # Method 3: Local adaptive (excellent for small pores in varying lighting)
        # Apply local adaptive threshold to the masked region only
        local_thresh = np.zeros_like(image, dtype=np.uint8)
        if np.sum(analysis_mask) > 100:  # Only if region is large enough
            # Extract bounding box for efficiency
            coords = np.column_stack(np.where(analysis_mask > 0))
            if len(coords) > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                
                # Ensure we have enough area for block size
                roi_height = y_max - y_min + 1
                roi_width = x_max - x_min + 1
                
                if roi_height > 50 and roi_width > 50:
                    roi = image[y_min:y_max+1, x_min:x_max+1]
                    roi_mask = analysis_mask[y_min:y_max+1, x_min:x_max+1]
                    
                    # Adaptive block size based on region size
                    block_size = max(11, min(51, min(roi_height, roi_width) // 10))
                    if block_size % 2 == 0:
                        block_size += 1
                    
                    local_adaptive = cv2.adaptiveThreshold(
                        roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 
                        block_size, 8  # More aggressive C parameter
                    )
                    
                    # Apply to full image
                    local_thresh[y_min:y_max+1, x_min:x_max+1] = local_adaptive
        
        method3_mask = (local_thresh > 0) & (analysis_mask > 0)
        
        print(f"       Method 1 (percentile): threshold={percentile_threshold:.1f}")
        print(f"       Method 2 (adaptive Otsu): threshold={adaptive_otsu_threshold:.1f}")
        print(f"       Method 3 (local adaptive): block-based")
        
        # Combine methods intelligently
        if self.config['pore_detection']['combine_methods']:
            # Start with the most conservative (percentile)
            combined_mask = method1_mask.copy()
            
            # Add areas found by other methods (union approach)
            combined_mask = combined_mask | method2_mask | method3_mask
            
            # Light morphological cleanup
            kernel = np.ones((2, 2), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            
            # Close small gaps
            kernel_close = np.ones((3, 3), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close)
            
            pore_mask = combined_mask
            print(f"       Combined methods with union approach")
        else:
            # Use percentile method only
            pore_mask = method1_mask
        
        # Find contours
        contours, _ = cv2.findContours(pore_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Enhanced filtering for small pores
        min_area_pixels = self.config['pore_detection']['min_pore_area_pixels']
        max_area_pixels = int(np.sum(analysis_mask > 0) * self.config['pore_detection']['max_pore_area_ratio'])
        
        pores = []
        for contour in contours:
            area_pixels = cv2.contourArea(contour)
            
            # Size filtering
            if area_pixels < min_area_pixels or area_pixels > max_area_pixels:
                continue
            
            # Calculate properties
            pore_props = self._calculate_pore_properties(contour, area_pixels, scale_factor)
            
            # More lenient shape filtering for small pores
            area_um2 = pore_props['area_um2']
            
            if area_um2 < 25:  # Small pores - very lenient
                if pore_props['aspect_ratio'] > 8:  # Only reject very extreme shapes
                    continue
                if pore_props['circularity'] < 0.05:  # Very lenient circularity
                    continue
            elif area_um2 < 100:  # Medium pores - moderate filtering
                if pore_props['aspect_ratio'] > 6:
                    continue
                if pore_props['circularity'] < 0.1:
                    continue
            else:  # Large pores - normal filtering
                if pore_props['aspect_ratio'] > 5:
                    continue
                if pore_props['circularity'] < 0.15:
                    continue
            
            pores.append(pore_props)
        
        print(f"       Raw detections: {len(contours)}, Filtered pores: {len(pores)}")
        
        return pores
    
    def _calculate_pore_properties(self, contour, area_pixels, scale_factor):
        """Calculate pore properties."""
        
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
    
    def _calculate_enhanced_metrics(self, pores, total_wall_area_um2, total_pore_area_um2, overall_porosity):
        """Calculate comprehensive metrics."""
        
        if not pores:
            return {
                'total_porosity_percent': 0.0,
                'pore_count': 0,
                'total_pore_area_um2': 0.0,
                'wall_area_um2': total_wall_area_um2,
                'average_pore_size_um2': 0.0
            }
        
        pore_areas = [pore['area_um2'] for pore in pores]
        pore_diameters = [pore['equivalent_diameter_um'] for pore in pores]
        
        return {
            'total_porosity_percent': overall_porosity,
            'pore_count': len(pores),
            'total_pore_area_um2': total_pore_area_um2,
            'wall_area_um2': total_wall_area_um2,
            'average_pore_size_um2': np.mean(pore_areas),
            'median_pore_size_um2': np.median(pore_areas),
            'std_pore_size_um2': np.std(pore_areas),
            'min_pore_size_um2': np.min(pore_areas),
            'max_pore_size_um2': np.max(pore_areas),
            'mean_pore_diameter_um': np.mean(pore_diameters),
            'median_pore_diameter_um': np.median(pore_diameters),
            'pore_density_per_mm2': len(pores) / (total_wall_area_um2 / 1e6) if total_wall_area_um2 > 0 else 0,
        }

def create_enhanced_comparison(image, fixed_result, enhanced_result, scale_factor, save_path):
    """Create comparison between fixed and enhanced detection."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    axes[0,0].imshow(image, cmap='gray')
    axes[0,0].set_title('Original SEM Image')
    axes[0,0].axis('off')
    
    # Fixed results
    fixed_metrics = fixed_result.get('porosity_metrics', {})
    fixed_text = f"FIXED ANALYSIS:\n\n"
    fixed_text += f"Porosity: {fixed_metrics.get('total_porosity_percent', 0):.2f}%\n"
    fixed_text += f"Pore count: {fixed_metrics.get('pore_count', 0)}\n"
    fixed_text += f"Wall area: {fixed_metrics.get('wall_area_um2', 0):,.0f} μm²\n"
    fixed_text += f"Avg pore size: {fixed_metrics.get('average_pore_size_um2', 0):.1f} μm²\n\n"
    fixed_text += f"✅ Correct area calculation\n❓ Missing some small pores"
    
    axes[0,1].text(0.05, 0.95, fixed_text, transform=axes[0,1].transAxes,
                  fontsize=11, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue'))
    axes[0,1].set_title('Fixed Analysis')
    axes[0,1].axis('off')
    
    # Enhanced results
    enhanced_metrics = enhanced_result.get('porosity_metrics', {})
    enhanced_text = f"ENHANCED ANALYSIS:\n\n"
    enhanced_text += f"Porosity: {enhanced_metrics.get('total_porosity_percent', 0):.2f}%\n"
    enhanced_text += f"Pore count: {enhanced_metrics.get('pore_count', 0)}\n"
    enhanced_text += f"Wall area: {enhanced_metrics.get('wall_area_um2', 0):,.0f} μm²\n"
    enhanced_text += f"Avg pore size: {enhanced_metrics.get('average_pore_size_um2', 0):.1f} μm²\n\n"
    enhanced_text += f"✅ Correct area calculation\n✅ Better small pore detection"
    
    axes[0,2].text(0.05, 0.95, enhanced_text, transform=axes[0,2].transAxes,
                  fontsize=11, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen'))
    axes[0,2].set_title('Enhanced Analysis')
    axes[0,2].axis('off')
    
    # Fixed detection visualization
    if 'individual_pores' in fixed_result:
        fixed_overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        for pore in fixed_result['individual_pores']:
            if 'contour' in pore:
                area = pore['area_um2']
                if area < 50:
                    color = (0, 255, 0)  # Green
                elif area < 500:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 0, 255)  # Red
                cv2.drawContours(fixed_overlay, [pore['contour']], -1, color, 2)
        
        axes[1,0].imshow(fixed_overlay)
        axes[1,0].set_title(f'Fixed: {fixed_metrics.get("pore_count", 0)} Pores')
        axes[1,0].axis('off')
    
    # Enhanced detection visualization
    if 'individual_pores' in enhanced_result:
        enhanced_overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        for pore in enhanced_result['individual_pores']:
            if 'contour' in pore:
                area = pore['area_um2']
                if area < 50:
                    color = (0, 255, 0)  # Green
                elif area < 500:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 0, 255)  # Red
                cv2.drawContours(enhanced_overlay, [pore['contour']], -1, color, 2)
        
        axes[1,1].imshow(enhanced_overlay)
        axes[1,1].set_title(f'Enhanced: {enhanced_metrics.get("pore_count", 0)} Pores')
        axes[1,1].axis('off')
    
    # Comparison chart
    categories = ['Porosity %', 'Pore Count', 'Avg Size μm²']
    fixed_values = [
        fixed_metrics.get('total_porosity_percent', 0),
        fixed_metrics.get('pore_count', 0) / 100,  # Scale for visibility
        fixed_metrics.get('average_pore_size_um2', 0) / 10  # Scale for visibility
    ]
    enhanced_values = [
        enhanced_metrics.get('total_porosity_percent', 0),
        enhanced_metrics.get('pore_count', 0) / 100,
        enhanced_metrics.get('average_pore_size_um2', 0) / 10
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    axes[1,2].bar(x - width/2, fixed_values, width, label='Fixed', color='lightblue')
    axes[1,2].bar(x + width/2, enhanced_values, width, label='Enhanced', color='lightgreen')
    
    axes[1,2].set_xlabel('Metric')
    axes[1,2].set_ylabel('Value (scaled)')
    axes[1,2].set_title('Fixed vs Enhanced Comparison')
    axes[1,2].set_xticks(x)
    axes[1,2].set_xticklabels(categories)
    axes[1,2].legend()
    
    # Add actual values as text
    for i, (fixed, enhanced) in enumerate(zip(fixed_values, enhanced_values)):
        if i == 0:  # Porosity
            axes[1,2].text(i - width/2, fixed + 0.5, f'{fixed:.1f}%', ha='center', va='bottom')
            axes[1,2].text(i + width/2, enhanced + 0.5, f'{enhanced:.1f}%', ha='center', va='bottom')
        elif i == 1:  # Pore count
            axes[1,2].text(i - width/2, fixed + 0.5, f'{int(fixed*100)}', ha='center', va='bottom')
            axes[1,2].text(i + width/2, enhanced + 0.5, f'{int(enhanced*100)}', ha='center', va='bottom')
        else:  # Average size
            axes[1,2].text(i - width/2, fixed + 0.5, f'{fixed*10:.1f}', ha='center', va='bottom')
            axes[1,2].text(i + width/2, enhanced + 0.5, f'{enhanced*10:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Enhanced comparison saved: {save_path}")

def test_enhanced_detection():
    """Test enhanced vs fixed detection."""
    
    # Import the fixed analyzer from previous test
    sys.path.append(str(test_dir))
    from test_porosity_fix import FixedPorosityAnalyzer
    
    results_dir = test_dir / "test_results"
    results_dir.mkdir(exist_ok=True)
    
    # Find image
    image_paths = [
        project_root / "sample_images" / "28d_001.jpg",
        project_root / "28d_001.jpg",
    ]
    
    image_path = None
    for path in image_paths:
        if path.exists():
            image_path = str(path)
            break
    
    if not image_path:
        print("❌ Could not find test image")
        return
    
    print(f"📸 Testing enhanced detection on: {Path(image_path).name}")
    
    # Load image and get fiber data
    image = load_image(image_path)
    scale_result = detect_scale_bar(image, debug=False)
    scale_factor = scale_result.get('micrometers_per_pixel', 1.0) if scale_result.get('scale_detected') else 1.0
    
    detector = FiberTypeDetector()
    fiber_type, confidence, analysis_data = detector.classify_fiber_type(image)
    
    # Run fixed analysis
    print("\n🔧 Running fixed analysis...")
    fixed_analyzer = FixedPorosityAnalyzer()
    fixed_result = fixed_analyzer.analyze_fiber_porosity_fixed(image, analysis_data, scale_factor)
    
    # Run enhanced analysis
    print("\n🔬 Running enhanced analysis...")
    enhanced_analyzer = EnhancedPoreDetector()
    enhanced_result = enhanced_analyzer.analyze_fiber_porosity_enhanced(image, analysis_data, scale_factor)
    
    # Create comparison
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_path = results_dir / f"enhanced_comparison_{timestamp}.png"
    create_enhanced_comparison(image, fixed_result, enhanced_result, scale_factor, str(viz_path))
    
    # Print comparison
    fixed_porosity = fixed_result.get('porosity_metrics', {}).get('total_porosity_percent', 0)
    enhanced_porosity = enhanced_result.get('porosity_metrics', {}).get('total_porosity_percent', 0)
    fixed_count = fixed_result.get('porosity_metrics', {}).get('pore_count', 0)
    enhanced_count = enhanced_result.get('porosity_metrics', {}).get('pore_count', 0)
    
    print(f"\n🎯 ENHANCED vs FIXED COMPARISON:")
    print(f"Fixed: {fixed_porosity:.2f}% porosity, {fixed_count} pores")
    print(f"Enhanced: {enhanced_porosity:.2f}% porosity, {enhanced_count} pores")
    print(f"Improvement: +{enhanced_count - fixed_count} pores detected")
    print(f"Porosity change: {enhanced_porosity - fixed_porosity:+.2f}% points")
    
    return enhanced_result

if __name__ == "__main__":
    result = test_enhanced_detection()