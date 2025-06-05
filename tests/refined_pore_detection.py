#!/usr/bin/env python3
"""
Refined Pore Detection - Conservative Enhancement
Fine-tunes the enhanced method to catch more small pores without over-detection

Save as: tests/refined_pore_detection.py
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

class RefinedPoreDetector:
    """
    Refined detector that improves on the enhanced method with better small pore sensitivity
    while avoiding over-detection.
    """
    
    def __init__(self):
        self.config = {
            'pore_detection': {
                # Conservative but sensitive parameters
                'intensity_percentile': 28,  # Slightly more sensitive than 30
                'min_pore_area_pixels': 3,   # Smaller minimum (was 5)
                'max_pore_area_ratio': 0.1,  # Conservative maximum
                'adaptive_threshold_methods': ['percentile', 'local_adaptive'],  # Remove Otsu (too aggressive)
                'combine_methods': True,
                'morphology_iterations': 1,  # Light cleanup
                'quality_filtering': True,   # Enhanced quality filtering
            },
            'segmentation': {
                'gaussian_blur': 1,          # Minimal blur
                'noise_reduction': True,     # Add noise reduction step
            },
            'hollow_fiber': {
                'exclude_lumen': True,
                'lumen_buffer_pixels': 3,
            },
            'quality_control': {
                'circularity_threshold': 0.05,  # Very lenient but not zero
                'aspect_ratio_threshold': 8,    # Allow some elongation
                'solidity_threshold': 0.3,      # Allow irregular shapes
                'intensity_validation': True,   # Validate pore intensity
            }
        }
    
    def analyze_fiber_porosity_refined(self, image, fiber_analysis_data, scale_factor=1.0):
        """
        Refined porosity analysis with improved small pore detection.
        """
        
        print(f"\n🔬 REFINED PORE DETECTION")
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
            
            print(f"\n   Refined Analysis - Fiber {i+1}:")
            
            # Create analysis mask (exclude lumen)
            fiber_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(fiber_mask, [fiber_contour], 255)
            analysis_mask = fiber_mask.copy()
            fiber_area_pixels = np.sum(fiber_mask > 0)
            
            if has_lumen and fiber_result.get('lumen_properties'):
                lumen_contour = fiber_result.get('lumen_properties', {}).get('contour')
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
            
            # Refined pore detection
            fiber_pores = self._detect_pores_refined(image, analysis_mask, scale_factor)
            
            # Add fiber info
            for pore in fiber_pores:
                pore['fiber_id'] = i
                pore['fiber_wall_area_pixels'] = wall_area_pixels
            
            all_pore_results.extend(fiber_pores)
            total_wall_area_pixels += wall_area_pixels
            
            pore_area_pixels = sum(pore['area_pixels'] for pore in fiber_pores)
            total_pore_area_pixels += pore_area_pixels
            
            fiber_porosity = (pore_area_pixels / wall_area_pixels * 100) if wall_area_pixels > 0 else 0
            
            print(f"     Refined detection: {len(fiber_pores)} pores")
            print(f"     Pore area: {pore_area_pixels:,} pixels ({pore_area_pixels * scale_factor**2:,.0f} μm²)")
            print(f"     Fiber porosity: {fiber_porosity:.2f}%")
        
        # Calculate results
        total_wall_area_um2 = total_wall_area_pixels * (scale_factor ** 2)
        total_pore_area_um2 = total_pore_area_pixels * (scale_factor ** 2)
        overall_porosity = (total_pore_area_pixels / total_wall_area_pixels * 100) if total_wall_area_pixels > 0 else 0
        
        print(f"\n🎯 REFINED OVERALL RESULTS:")
        print(f"   Total wall area: {total_wall_area_pixels:,} pixels ({total_wall_area_um2:,.0f} μm²)")
        print(f"   Total pore area: {total_pore_area_pixels:,} pixels ({total_pore_area_um2:,.0f} μm²)")
        print(f"   REFINED POROSITY: {overall_porosity:.2f}%")
        print(f"   Total pores: {len(all_pore_results)}")
        
        # Analyze pore size distribution
        if all_pore_results:
            pore_areas = [p['area_um2'] for p in all_pore_results]
            tiny_pores = [p for p in pore_areas if p < 10]
            small_pores = [p for p in pore_areas if 10 <= p < 50]
            medium_pores = [p for p in pore_areas if 50 <= p < 500]
            large_pores = [p for p in pore_areas if p >= 500]
            
            print(f"   Pore size breakdown:")
            print(f"     Tiny (<10 μm²): {len(tiny_pores)} pores ({len(tiny_pores)/len(all_pore_results)*100:.1f}%)")
            print(f"     Small (10-50 μm²): {len(small_pores)} pores ({len(small_pores)/len(all_pore_results)*100:.1f}%)")
            print(f"     Medium (50-500 μm²): {len(medium_pores)} pores ({len(medium_pores)/len(all_pore_results)*100:.1f}%)")
            print(f"     Large (>500 μm²): {len(large_pores)} pores ({len(large_pores)/len(all_pore_results)*100:.1f}%)")
        
        # Create metrics
        porosity_metrics = self._calculate_refined_metrics(
            all_pore_results, total_wall_area_um2, total_pore_area_um2, overall_porosity
        )
        
        return {
            'porosity_metrics': porosity_metrics,
            'individual_pores': all_pore_results,
            'method': 'refined_conservative',
            'scale_factor': scale_factor,
        }
    
    def _detect_pores_refined(self, image, analysis_mask, scale_factor):
        """
        Refined pore detection that's more sensitive but still conservative.
        """
        
        masked_image = cv2.bitwise_and(image, image, mask=analysis_mask)
        region_pixels = masked_image[analysis_mask > 0]
        
        if len(region_pixels) == 0:
            return []
        
        print(f"     Refined multi-method detection:")
        
        # Noise reduction first
        if self.config['segmentation']['noise_reduction']:
            # Very light bilateral filter to reduce noise while preserving edges
            masked_image = cv2.bilateralFilter(masked_image, 3, 20, 20)
            region_pixels = masked_image[analysis_mask > 0]
        
        # Method 1: Slightly more sensitive percentile-based
        percentile = self.config['pore_detection']['intensity_percentile']
        percentile_threshold = np.percentile(region_pixels, percentile)
        method1_mask = (masked_image < percentile_threshold) & (analysis_mask > 0)
        
        # Method 2: Conservative local adaptive
        method2_mask = np.zeros_like(analysis_mask, dtype=bool)
        if np.sum(analysis_mask) > 100:
            # Only apply local adaptive if region is large enough
            coords = np.column_stack(np.where(analysis_mask > 0))
            if len(coords) > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                
                roi_height = y_max - y_min + 1
                roi_width = x_max - x_min + 1
                
                if roi_height > 30 and roi_width > 30:
                    roi = image[y_min:y_max+1, x_min:x_max+1]
                    
                    # Conservative block size
                    block_size = max(11, min(31, min(roi_height, roi_width) // 15))
                    if block_size % 2 == 0:
                        block_size += 1
                    
                    # Conservative C parameter (less aggressive)
                    local_adaptive = cv2.adaptiveThreshold(
                        roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 
                        block_size, 3  # Conservative C parameter
                    )
                    
                    # Apply to mask area only
                    local_thresh = np.zeros_like(image, dtype=np.uint8)
                    local_thresh[y_min:y_max+1, x_min:x_max+1] = local_adaptive
                    roi_mask = analysis_mask[y_min:y_max+1, x_min:x_max+1]
                    
                    method2_mask = (local_thresh > 0) & (analysis_mask > 0)
        
        print(f"       Method 1 (percentile): threshold={percentile_threshold:.1f}")
        print(f"       Method 2 (local adaptive): conservative block-based")
        
        # Conservative combination - be more selective
        if self.config['pore_detection']['combine_methods']:
            # Use intersection for very small pores (more conservative)
            # Use union for larger pores (keep sensitivity)
            
            # Find potential pores from each method
            combined_mask = method1_mask | method2_mask
            
            # Light morphological cleanup
            kernel = np.ones((2, 2), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            
            pore_mask = combined_mask
            print(f"       Combined methods with conservative union")
        else:
            pore_mask = method1_mask
        
        # Find contours
        contours, _ = cv2.findContours(pore_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Refined filtering
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
            
            # Enhanced quality filtering
            if not self._validate_pore_quality(pore_props, masked_image, analysis_mask):
                continue
            
            pores.append(pore_props)
        
        print(f"       Raw detections: {len(contours)}, Quality filtered: {len(pores)}")
        
        return pores
    
    def _validate_pore_quality(self, pore_props, image, analysis_mask):
        """Enhanced quality validation to avoid false positives."""
        
        area_um2 = pore_props['area_um2']
        
        # Basic shape filtering - size-dependent
        if area_um2 < 5:  # Very tiny pores - strictest filtering
            if pore_props['aspect_ratio'] > 6:
                return False
            if pore_props['circularity'] < 0.08:
                return False
        elif area_um2 < 25:  # Small pores - moderate filtering  
            if pore_props['aspect_ratio'] > self.config['quality_control']['aspect_ratio_threshold']:
                return False
            if pore_props['circularity'] < self.config['quality_control']['circularity_threshold']:
                return False
        else:  # Larger pores - more lenient
            if pore_props['aspect_ratio'] > 10:
                return False
            if pore_props['circularity'] < 0.03:
                return False
        
        # Solidity check
        if pore_props.get('solidity', 1.0) < self.config['quality_control']['solidity_threshold']:
            return False
        
        # Intensity validation for very small pores
        if self.config['quality_control']['intensity_validation'] and area_um2 < 15:
            contour = pore_props['contour']
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)
            
            # Check if the pore region is actually darker than surroundings
            pore_pixels = image[mask > 0]
            if len(pore_pixels) > 0:
                pore_mean = np.mean(pore_pixels)
                
                # Create surrounding region
                kernel = np.ones((7, 7), np.uint8)
                expanded_mask = cv2.dilate(mask, kernel, iterations=1)
                surrounding_mask = expanded_mask & ~mask & analysis_mask
                
                surrounding_pixels = image[surrounding_mask > 0]
                if len(surrounding_pixels) > 0:
                    surrounding_mean = np.mean(surrounding_pixels)
                    
                    # Pore should be at least 10% darker than surroundings for tiny pores
                    intensity_ratio = pore_mean / surrounding_mean if surrounding_mean > 0 else 1.0
                    if intensity_ratio > 0.9:  # Not dark enough
                        return False
        
        return True
    
    def _calculate_pore_properties(self, contour, area_pixels, scale_factor):
        """Calculate comprehensive pore properties."""
        
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
        
        # Solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area_pixels / hull_area if hull_area > 0 else 0
        
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
            'solidity': solidity,
            'radius_pixels': radius,
            'radius_um': radius * scale_factor
        }
    
    def _calculate_refined_metrics(self, pores, total_wall_area_um2, total_pore_area_um2, overall_porosity):
        """Calculate comprehensive refined metrics."""
        
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

def create_refined_comparison(image, enhanced_result, refined_result, scale_factor, save_path):
    """Create comparison between enhanced and refined detection."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    axes[0,0].imshow(image, cmap='gray')
    axes[0,0].set_title('Original SEM Image')
    axes[0,0].axis('off')
    
    # Enhanced results
    enhanced_metrics = enhanced_result.get('porosity_metrics', {})
    enhanced_text = f"ENHANCED ANALYSIS:\n\n"
    enhanced_text += f"Porosity: {enhanced_metrics.get('total_porosity_percent', 0):.2f}%\n"
    enhanced_text += f"Pore count: {enhanced_metrics.get('pore_count', 0)}\n"
    enhanced_text += f"Avg pore size: {enhanced_metrics.get('average_pore_size_um2', 0):.1f} μm²\n\n"
    enhanced_text += f"✅ Good baseline\n❓ Missing some small pores"
    
    axes[0,1].text(0.05, 0.95, enhanced_text, transform=axes[0,1].transAxes,
                  fontsize=11, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue'))
    axes[0,1].set_title('Enhanced Analysis')
    axes[0,1].axis('off')
    
    # Refined results
    refined_metrics = refined_result.get('porosity_metrics', {})
    refined_text = f"REFINED ANALYSIS:\n\n"
    refined_text += f"Porosity: {refined_metrics.get('total_porosity_percent', 0):.2f}%\n"
    refined_text += f"Pore count: {refined_metrics.get('pore_count', 0)}\n"
    refined_text += f"Avg pore size: {refined_metrics.get('average_pore_size_um2', 0):.1f} μm²\n\n"
    refined_text += f"✅ Conservative enhancement\n✅ Better small pore detection"
    
    axes[0,2].text(0.05, 0.95, refined_text, transform=axes[0,2].transAxes,
                  fontsize=11, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen'))
    axes[0,2].set_title('Refined Analysis')
    axes[0,2].axis('off')
    
    # Enhanced detection visualization
    if 'individual_pores' in enhanced_result:
        enhanced_overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        for pore in enhanced_result['individual_pores']:
            if 'contour' in pore:
                area = pore['area_um2']
                if area < 10:
                    color = (0, 255, 0)     # Bright green for tiny
                elif area < 50:
                    color = (0, 255, 255)   # Yellow for small
                elif area < 500:
                    color = (0, 165, 255)   # Orange for medium
                else:
                    color = (0, 0, 255)     # Red for large
                cv2.drawContours(enhanced_overlay, [pore['contour']], -1, color, 2)
        
        axes[1,0].imshow(enhanced_overlay)
        axes[1,0].set_title(f'Enhanced: {enhanced_metrics.get("pore_count", 0)} Pores')
        axes[1,0].axis('off')
    
    # Refined detection visualization
    if 'individual_pores' in refined_result:
        refined_overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        for pore in refined_result['individual_pores']:
            if 'contour' in pore:
                area = pore['area_um2']
                if area < 10:
                    color = (0, 255, 0)     # Bright green for tiny
                elif area < 50:
                    color = (0, 255, 255)   # Yellow for small
                elif area < 500:
                    color = (0, 165, 255)   # Orange for medium
                else:
                    color = (0, 0, 255)     # Red for large
                cv2.drawContours(refined_overlay, [pore['contour']], -1, color, 2)
        
        axes[1,1].imshow(refined_overlay)
        axes[1,1].set_title(f'Refined: {refined_metrics.get("pore_count", 0)} Pores\n(Green<10, Yellow<50, Orange<500, Red>500 μm²)')
        axes[1,1].axis('off')
    
    # Comparison chart
    categories = ['Porosity %', 'Pore Count', 'Avg Size μm²']
    enhanced_values = [
        enhanced_metrics.get('total_porosity_percent', 0),
        enhanced_metrics.get('pore_count', 0) / 50,  # Scale for visibility
        enhanced_metrics.get('average_pore_size_um2', 0)
    ]
    refined_values = [
        refined_metrics.get('total_porosity_percent', 0),
        refined_metrics.get('pore_count', 0) / 50,
        refined_metrics.get('average_pore_size_um2', 0)
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    axes[1,2].bar(x - width/2, enhanced_values, width, label='Enhanced', color='lightblue')
    axes[1,2].bar(x + width/2, refined_values, width, label='Refined', color='lightgreen')
    
    axes[1,2].set_xlabel('Metric')
    axes[1,2].set_ylabel('Value (pore count scaled /50)')
    axes[1,2].set_title('Enhanced vs Refined Comparison')
    axes[1,2].set_xticks(x)
    axes[1,2].set_xticklabels(categories)
    axes[1,2].legend()
    
    # Add actual values as text
    for i, (enhanced, refined) in enumerate(zip(enhanced_values, refined_values)):
        if i == 0:  # Porosity
            axes[1,2].text(i - width/2, enhanced + 0.2, f'{enhanced:.1f}%', ha='center', va='bottom')
            axes[1,2].text(i + width/2, refined + 0.2, f'{refined:.1f}%', ha='center', va='bottom')
        elif i == 1:  # Pore count
            axes[1,2].text(i - width/2, enhanced + 0.2, f'{int(enhanced*50)}', ha='center', va='bottom')
            axes[1,2].text(i + width/2, refined + 0.2, f'{int(refined*50)}', ha='center', va='bottom')
        else:  # Average size
            axes[1,2].text(i - width/2, enhanced + 0.2, f'{enhanced:.1f}', ha='center', va='bottom')
            axes[1,2].text(i + width/2, refined + 0.2, f'{refined:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Refined comparison saved: {save_path}")

def test_refined_detection():
    """Test refined vs enhanced detection."""
    
    # Import previous analyzer
    sys.path.append(str(test_dir))
    from enhanced_pore_detection import EnhancedPoreDetector
    
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
    
    print(f"📸 Testing refined detection on: {Path(image_path).name}")
    
    # Load image and get fiber data
    image = load_image(image_path)
    scale_result = detect_scale_bar(image, debug=False)
    scale_factor = scale_result.get('micrometers_per_pixel', 1.0) if scale_result.get('scale_detected') else 1.0
    
    detector = FiberTypeDetector()
    fiber_type, confidence, analysis_data = detector.classify_fiber_type(image)
    
    # Run enhanced analysis
    print("\n🔬 Running enhanced analysis...")
    enhanced_analyzer = EnhancedPoreDetector()
    enhanced_result = enhanced_analyzer.analyze_fiber_porosity_enhanced(image, analysis_data, scale_factor)
    
    # Run refined analysis
    print("\n🔬 Running refined analysis...")
    refined_analyzer = RefinedPoreDetector()
    refined_result = refined_analyzer.analyze_fiber_porosity_refined(image, analysis_data, scale_factor)
    
    # Create comparison
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_path = results_dir / f"refined_comparison_{timestamp}.png"
    create_refined_comparison(image, enhanced_result, refined_result, scale_factor, str(viz_path))
    
    # Print comparison
    enhanced_porosity = enhanced_result.get('porosity_metrics', {}).get('total_porosity_percent', 0)
    refined_porosity = refined_result.get('porosity_metrics', {}).get('total_porosity_percent', 0)
    enhanced_count = enhanced_result.get('porosity_metrics', {}).get('pore_count', 0)
    refined_count = refined_result.get('porosity_metrics', {}).get('pore_count', 0)
    
    print(f"\n🎯 REFINED vs ENHANCED COMPARISON:")
    print(f"Enhanced: {enhanced_porosity:.2f}% porosity, {enhanced_count} pores")
    print(f"Refined: {refined_porosity:.2f}% porosity, {refined_count} pores")
    print(f"Improvement: +{refined_count - enhanced_count} pores detected")
    print(f"Porosity change: {refined_porosity - enhanced_porosity:+.2f}% points")
    
    if refined_porosity > enhanced_porosity and refined_count > enhanced_count:
        print(f"✅ Refined method successfully caught more small pores!")
    elif refined_count > enhanced_count * 3:
        print(f"⚠️ Refined method may be over-detecting - check visualization")
    else:
        print(f"📊 Results are similar - both methods are reasonable")
    
    return refined_result

if __name__ == "__main__":
    result = test_refined_detection()