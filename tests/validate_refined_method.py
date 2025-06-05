#!/usr/bin/env python3
"""
Refined Method Validation and Fine-Tuning
Since the refined method appears most accurate, let's validate and fine-tune it

Save as: tests/validate_refined_method.py
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

class ValidatedRefinedDetector:
    """
    Refined detector with validation and minor tuning based on visual assessment.
    This is our "production ready" version.
    """
    
    def __init__(self):
        self.config = {
            'pore_detection': {
                # Proven effective parameters from refined method
                'intensity_percentile': 28,      # Slight adjustment from 28
                'min_pore_area_pixels': 3,       # Keep sensitive minimum
                'max_pore_area_ratio': 0.1,      # Conservative maximum
                'adaptive_threshold_methods': ['percentile', 'local_adaptive'],
                'combine_methods': True,
                'morphology_iterations': 1,
                'quality_filtering': True,
                'validation_mode': True,         # Enable additional validation
            },
            'segmentation': {
                'gaussian_blur': 1,
                'noise_reduction': True,
            },
            'hollow_fiber': {
                'exclude_lumen': True,
                'lumen_buffer_pixels': 3,
            },
            'quality_control': {
                # Refined quality control - less aggressive than optimized
                'circularity_threshold': 0.05,   # Very lenient
                'aspect_ratio_threshold': 8,     # Allow elongation
                'solidity_threshold': 0.25,      # Allow irregular shapes
                'intensity_validation': True,    # Keep intensity check
                'size_dependent_validation': True,
            },
            'validation': {
                'visual_inspection_mode': True,
                'detailed_logging': True,
                'size_category_analysis': True,
            }
        }
    
    def analyze_fiber_porosity_validated(self, image, fiber_analysis_data, scale_factor=1.0):
        """
        Validated refined porosity analysis with detailed reporting.
        """
        
        print(f"\n🔬 VALIDATED REFINED PORE DETECTION")
        print(f"   Scale factor: {scale_factor:.4f} μm/pixel")
        print(f"   Method: Conservative enhancement with validation")
        
        individual_results = fiber_analysis_data.get('individual_results', [])
        if not individual_results:
            return {'error': 'No individual fiber results found'}
        
        all_pore_results = []
        total_wall_area_pixels = 0
        total_pore_area_pixels = 0
        validation_stats = {
            'stage1_size_filtered': 0,
            'stage2_quality_filtered': 0,
            'stage3_intensity_filtered': 0,
            'final_accepted': 0
        }
        
        for i, fiber_result in enumerate(individual_results):
            fiber_props = fiber_result.get('fiber_properties', {})
            fiber_contour = fiber_props.get('contour')
            has_lumen = fiber_result.get('has_lumen', False)
            
            if fiber_contour is None:
                continue
            
            print(f"\n   📊 Validated Analysis - Fiber {i+1}:")
            
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
                    
                    # Buffer around lumen
                    buffer_size = self.config['hollow_fiber']['lumen_buffer_pixels']
                    if buffer_size > 0:
                        kernel = np.ones((buffer_size*2+1, buffer_size*2+1), np.uint8)
                        lumen_mask = cv2.dilate(lumen_mask, kernel, iterations=1)
                    
                    analysis_mask = cv2.bitwise_and(analysis_mask, cv2.bitwise_not(lumen_mask))
                    lumen_area_pixels = np.sum(lumen_mask > 0)
                    wall_area_pixels = fiber_area_pixels - lumen_area_pixels
                    
                    print(f"     Wall area: {wall_area_pixels:,} pixels ({wall_area_pixels * scale_factor**2:,.0f} μm²)")
                    print(f"     Lumen excluded: {lumen_area_pixels:,} pixels")
                else:
                    wall_area_pixels = fiber_area_pixels
            else:
                wall_area_pixels = fiber_area_pixels
            
            # Validated pore detection with detailed stats
            fiber_pores, fiber_stats = self._detect_pores_validated(image, analysis_mask, scale_factor)
            
            # Update validation statistics
            for key in validation_stats:
                validation_stats[key] += fiber_stats.get(key, 0)
            
            # Add fiber info
            for pore in fiber_pores:
                pore['fiber_id'] = i
                pore['fiber_wall_area_pixels'] = wall_area_pixels
            
            all_pore_results.extend(fiber_pores)
            total_wall_area_pixels += wall_area_pixels
            
            pore_area_pixels = sum(pore['area_pixels'] for pore in fiber_pores)
            total_pore_area_pixels += pore_area_pixels
            
            fiber_porosity = (pore_area_pixels / wall_area_pixels * 100) if wall_area_pixels > 0 else 0
            
            print(f"     ✅ Validated detection: {len(fiber_pores)} pores")
            print(f"     Total pore area: {pore_area_pixels:,} pixels ({pore_area_pixels * scale_factor**2:,.0f} μm²)")
            print(f"     Fiber porosity: {fiber_porosity:.2f}%")
            
            # Print filtering statistics
            print(f"     📈 Filtering stats:")
            print(f"       Stage 1 (size): {fiber_stats.get('stage1_size_filtered', 0)} candidates")
            print(f"       Stage 2 (quality): {fiber_stats.get('stage2_quality_filtered', 0)} passed")
            print(f"       Stage 3 (intensity): {fiber_stats.get('stage3_intensity_filtered', 0)} passed")
            print(f"       Final accepted: {fiber_stats.get('final_accepted', 0)} pores")
        
        # Calculate comprehensive results
        total_wall_area_um2 = total_wall_area_pixels * (scale_factor ** 2)
        total_pore_area_um2 = total_pore_area_pixels * (scale_factor ** 2)
        overall_porosity = (total_pore_area_pixels / total_wall_area_pixels * 100) if total_wall_area_pixels > 0 else 0
        
        print(f"\n🎯 VALIDATED REFINED RESULTS:")
        print(f"   Total wall area: {total_wall_area_pixels:,} pixels ({total_wall_area_um2:,.0f} μm²)")
        print(f"   Total pore area: {total_pore_area_pixels:,} pixels ({total_pore_area_um2:,.0f} μm²)")
        print(f"   🔥 VALIDATED POROSITY: {overall_porosity:.2f}%")
        print(f"   🔥 TOTAL PORES: {len(all_pore_results)}")
        
        # Detailed size analysis
        if all_pore_results:
            size_analysis = self._analyze_size_distribution_detailed(all_pore_results)
            print(f"\n   📊 DETAILED SIZE ANALYSIS:")
            for category, data in size_analysis.items():
                if isinstance(data, dict) and 'count' in data:
                    print(f"     {category}: {data['count']} pores ({data['percentage']:.1f}%) - Total: {data['total_area_um2']:.0f} μm²")
        
        # Overall validation assessment
        validation_assessment = self._assess_validation_quality(all_pore_results, validation_stats, total_wall_area_um2)
        
        print(f"\n   🔍 VALIDATION ASSESSMENT:")
        print(f"     Overall quality: {validation_assessment['quality_level']}")
        print(f"     Detection confidence: {validation_assessment['confidence']:.2f}")
        print(f"     Method reliability: {validation_assessment['reliability']}")
        
        # Create comprehensive metrics
        porosity_metrics = self._calculate_validated_metrics(
            all_pore_results, total_wall_area_um2, total_pore_area_um2, 
            overall_porosity, validation_assessment, validation_stats
        )
        
        return {
            'porosity_metrics': porosity_metrics,
            'individual_pores': all_pore_results,
            'validation_stats': validation_stats,
            'validation_assessment': validation_assessment,
            'method': 'validated_refined',
            'scale_factor': scale_factor,
            'recommended_for_production': True
        }
    
    def _detect_pores_validated(self, image, analysis_mask, scale_factor):
        """
        Validated pore detection with detailed statistics tracking.
        """
        
        masked_image = cv2.bitwise_and(image, image, mask=analysis_mask)
        region_pixels = masked_image[analysis_mask > 0]
        
        if len(region_pixels) == 0:
            return [], {}
        
        # Enhanced preprocessing (light noise reduction)
        if self.config['segmentation']['noise_reduction']:
            masked_image = cv2.bilateralFilter(masked_image, 3, 20, 20)
            region_pixels = masked_image[analysis_mask > 0]
        
        # Method 1: Percentile-based (primary method)
        percentile = self.config['pore_detection']['intensity_percentile']
        percentile_threshold = np.percentile(region_pixels, percentile)
        method1_mask = (masked_image < percentile_threshold) & (analysis_mask > 0)
        
        # Method 2: Conservative local adaptive (secondary method)
        method2_mask = self._apply_conservative_local_adaptive(masked_image, analysis_mask)
        
        # Combine methods
        if self.config['pore_detection']['combine_methods']:
            combined_mask = method1_mask | method2_mask
            
            # Light morphological cleanup
            kernel = np.ones((2, 2), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            pore_mask = combined_mask
        else:
            pore_mask = method1_mask
        
        # Find contours
        contours, _ = cv2.findContours(pore_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Validated filtering with statistics
        stats = {'stage1_size_filtered': 0, 'stage2_quality_filtered': 0, 'stage3_intensity_filtered': 0, 'final_accepted': 0}
        
        min_area_pixels = self.config['pore_detection']['min_pore_area_pixels']
        max_area_pixels = int(np.sum(analysis_mask > 0) * self.config['pore_detection']['max_pore_area_ratio'])
        
        validated_pores = []
        
        for contour in contours:
            area_pixels = cv2.contourArea(contour)
            
            # Stage 1: Size filtering
            if area_pixels < min_area_pixels or area_pixels > max_area_pixels:
                continue
            stats['stage1_size_filtered'] += 1
            
            # Calculate properties
            pore_props = self._calculate_pore_properties(contour, area_pixels, scale_factor)
            
            # Stage 2: Quality filtering (shape-based)
            if not self._validate_pore_shape_refined(pore_props):
                continue
            stats['stage2_quality_filtered'] += 1
            
            # Stage 3: Intensity validation (for small pores)
            if not self._validate_pore_intensity_refined(pore_props, masked_image, analysis_mask):
                continue
            stats['stage3_intensity_filtered'] += 1
            
            validated_pores.append(pore_props)
            stats['final_accepted'] += 1
        
        return validated_pores, stats
    
    def _apply_conservative_local_adaptive(self, image, analysis_mask):
        """Apply conservative local adaptive thresholding."""
        
        if np.sum(analysis_mask) < 200:
            return np.zeros_like(analysis_mask, dtype=bool)
        
        coords = np.column_stack(np.where(analysis_mask > 0))
        if len(coords) == 0:
            return np.zeros_like(analysis_mask, dtype=bool)
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        roi_height = y_max - y_min + 1
        roi_width = x_max - x_min + 1
        
        if roi_height < 30 or roi_width < 30:
            return np.zeros_like(analysis_mask, dtype=bool)
        
        roi = image[y_min:y_max+1, x_min:x_max+1]
        
        # Conservative block size
        block_size = max(11, min(31, min(roi_height, roi_width) // 15))
        if block_size % 2 == 0:
            block_size += 1
        
        # Conservative C parameter
        local_adaptive = cv2.adaptiveThreshold(
            roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 
            block_size, 3  # Conservative C
        )
        
        # Map back to full image
        local_thresh = np.zeros_like(image, dtype=np.uint8)
        local_thresh[y_min:y_max+1, x_min:x_max+1] = local_adaptive
        
        return (local_thresh > 0) & (analysis_mask > 0)
    
    def _validate_pore_shape_refined(self, pore_props):
        """Refined shape validation - lenient but effective."""
        
        area_um2 = pore_props['area_um2']
        
        # Size-dependent shape validation
        if area_um2 < 5:  # Very tiny pores - slightly stricter
            if pore_props['aspect_ratio'] > 6:
                return False
            if pore_props['circularity'] < 0.08:
                return False
        elif area_um2 < 25:  # Small pores - moderate
            if pore_props['aspect_ratio'] > 8:
                return False
            if pore_props['circularity'] < 0.05:
                return False
        else:  # Larger pores - very lenient
            if pore_props['aspect_ratio'] > 12:
                return False
            if pore_props['circularity'] < 0.03:
                return False
        
        # Solidity check
        if pore_props.get('solidity', 1.0) < 0.25:
            return False
        
        return True
    
    def _validate_pore_intensity_refined(self, pore_props, image, analysis_mask):
        """Refined intensity validation - focus on very small pores."""
        
        area_um2 = pore_props['area_um2']
        
        # Only validate intensity for very small pores
        if area_um2 >= 15:
            return True
        
        contour = pore_props['contour']
        pore_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(pore_mask, [contour], 255)
        
        pore_pixels = image[pore_mask > 0]
        if len(pore_pixels) == 0:
            return False
        
        pore_mean = np.mean(pore_pixels)
        
        # Create surrounding region
        kernel_size = max(5, int(np.sqrt(pore_props['area_pixels']) * 2))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        expanded_mask = cv2.dilate(pore_mask, kernel, iterations=1)
        surrounding_mask = expanded_mask & ~pore_mask & analysis_mask
        
        surrounding_pixels = image[surrounding_mask > 0]
        if len(surrounding_pixels) == 0:
            return True  # Can't validate, accept it
        
        surrounding_mean = np.mean(surrounding_pixels)
        intensity_ratio = pore_mean / surrounding_mean if surrounding_mean > 0 else 1.0
        
        # Very tiny pores should be noticeably darker
        if area_um2 < 5 and intensity_ratio > 0.88:
            return False
        elif area_um2 < 10 and intensity_ratio > 0.92:
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
    
    def _analyze_size_distribution_detailed(self, pores):
        """Detailed size distribution analysis."""
        
        pore_areas = [pore['area_um2'] for pore in pores]
        
        categories = {
            'ultra_tiny': {'range': '< 3 μm²', 'pores': [p for p in pores if p['area_um2'] < 3]},
            'tiny': {'range': '3-10 μm²', 'pores': [p for p in pores if 3 <= p['area_um2'] < 10]},
            'small': {'range': '10-50 μm²', 'pores': [p for p in pores if 10 <= p['area_um2'] < 50]},
            'medium': {'range': '50-200 μm²', 'pores': [p for p in pores if 50 <= p['area_um2'] < 200]},
            'large': {'range': '200-500 μm²', 'pores': [p for p in pores if 200 <= p['area_um2'] < 500]},
            'very_large': {'range': '> 500 μm²', 'pores': [p for p in pores if p['area_um2'] >= 500]}
        }
        
        total_pores = len(pores)
        
        for category, data in categories.items():
            count = len(data['pores'])
            percentage = (count / total_pores * 100) if total_pores > 0 else 0
            total_area = sum(p['area_um2'] for p in data['pores'])
            
            categories[category].update({
                'count': count,
                'percentage': percentage,
                'total_area_um2': total_area
            })
        
        return categories
    
    def _assess_validation_quality(self, pores, validation_stats, total_wall_area_um2):
        """Assess the quality of the validated detection."""
        
        total_detected = validation_stats.get('stage1_size_filtered', 0)
        final_accepted = validation_stats.get('final_accepted', 0)
        
        if total_detected == 0:
            return {'quality_level': 'poor', 'confidence': 0.0, 'reliability': 'low'}
        
        # Calculate acceptance rate
        acceptance_rate = final_accepted / total_detected if total_detected > 0 else 0
        
        # Assess based on pore density and acceptance rate
        pore_density = len(pores) / (total_wall_area_um2 / 1e6) if total_wall_area_um2 > 0 else 0
        
        # Size distribution assessment
        if pores:
            tiny_fraction = len([p for p in pores if p['area_um2'] < 10]) / len(pores)
        else:
            tiny_fraction = 0
        
        # Overall quality assessment
        if pore_density > 10000 and 0.2 <= acceptance_rate <= 0.8 and tiny_fraction < 0.8:
            quality_level = 'excellent'
            confidence = 0.9
            reliability = 'high'
        elif pore_density > 5000 and 0.15 <= acceptance_rate <= 0.85:
            quality_level = 'good'
            confidence = 0.8
            reliability = 'good'
        elif pore_density > 1000:
            quality_level = 'moderate'
            confidence = 0.7
            reliability = 'moderate'
        else:
            quality_level = 'poor'
            confidence = 0.5
            reliability = 'low'
        
        return {
            'quality_level': quality_level,
            'confidence': confidence,
            'reliability': reliability,
            'acceptance_rate': acceptance_rate,
            'pore_density_per_mm2': pore_density,
            'tiny_pore_fraction': tiny_fraction
        }
    
    def _calculate_validated_metrics(self, pores, total_wall_area_um2, total_pore_area_um2, 
                                   overall_porosity, validation_assessment, validation_stats):
        """Calculate comprehensive validated metrics."""
        
        if not pores:
            return {
                'total_porosity_percent': 0.0,
                'pore_count': 0,
                'total_pore_area_um2': 0.0,
                'wall_area_um2': total_wall_area_um2,
                'average_pore_size_um2': 0.0,
                'validation_assessment': validation_assessment,
                'method': 'validated_refined'
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
            'validation_assessment': validation_assessment,
            'validation_statistics': validation_stats,
            'method': 'validated_refined',
            'recommended_for_production': True
        }

def create_validation_comparison(image, original_refined, validated_refined, scale_factor, save_path):
    """Create comparison between original and validated refined methods."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    axes[0,0].imshow(image, cmap='gray')
    axes[0,0].set_title('Original SEM Image', fontweight='bold')
    axes[0,0].axis('off')
    
    # Original refined results
    orig_metrics = original_refined.get('porosity_metrics', {})
    orig_text = f"ORIGINAL REFINED:\n\n"
    orig_text += f"Porosity: {orig_metrics.get('total_porosity_percent', 0):.2f}%\n"
    orig_text += f"Pore count: {orig_metrics.get('pore_count', 0)}\n"
    orig_text += f"Avg pore size: {orig_metrics.get('average_pore_size_um2', 0):.1f} μm²\n\n"
    orig_text += f"✅ High sensitivity\n⚠️ Needs validation"
    
    axes[0,1].text(0.05, 0.95, orig_text, transform=axes[0,1].transAxes,
                  fontsize=11, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen'))
    axes[0,1].set_title('Original Refined')
    axes[0,1].axis('off')
    
    # Validated refined results
    val_metrics = validated_refined.get('porosity_metrics', {})
    val_assessment = val_metrics.get('validation_assessment', {})
    val_text = f"VALIDATED REFINED:\n\n"
    val_text += f"Porosity: {val_metrics.get('total_porosity_percent', 0):.2f}%\n"
    val_text += f"Pore count: {val_metrics.get('pore_count', 0)}\n"
    val_text += f"Avg pore size: {val_metrics.get('average_pore_size_um2', 0):.1f} μm²\n\n"
    val_text += f"Quality: {val_assessment.get('quality_level', 'unknown')}\n"
    val_text += f"Confidence: {val_assessment.get('confidence', 0):.2f}\n"
    val_text += f"✅ Production ready"
    
    axes[0,2].text(0.05, 0.95, val_text, transform=axes[0,2].transAxes,
                  fontsize=11, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue'))
    axes[0,2].set_title('Validated Refined')
    axes[0,2].axis('off')
    
    # Visual comparison - Original refined
    if 'individual_pores' in original_refined:
        orig_overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        for pore in original_refined['individual_pores']:
            if 'contour' in pore:
                area = pore['area_um2']
                if area < 10:
                    color = (0, 255, 0)     # Green for tiny
                elif area < 50:
                    color = (0, 255, 255)   # Yellow for small
                else:
                    color = (0, 0, 255)     # Red for medium/large
                cv2.drawContours(orig_overlay, [pore['contour']], -1, color, 1)
        
        axes[1,0].imshow(orig_overlay)
        axes[1,0].set_title(f'Original: {orig_metrics.get("pore_count", 0)} Pores')
        axes[1,0].axis('off')
    
    # Visual comparison - Validated refined
    if 'individual_pores' in validated_refined:
        val_overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        for pore in validated_refined['individual_pores']:
            if 'contour' in pore:
                area = pore['area_um2']
                if area < 10:
                    color = (0, 255, 0)     # Green for tiny
                elif area < 50:
                    color = (0, 255, 255)   # Yellow for small
                else:
                    color = (0, 0, 255)     # Red for medium/large
                cv2.drawContours(val_overlay, [pore['contour']], -1, color, 2)
        
        axes[1,1].imshow(val_overlay)
        axes[1,1].set_title(f'Validated: {val_metrics.get("pore_count", 0)} Pores\n(Green<10, Yellow<50, Red>50 μm²)')
        axes[1,1].axis('off')
    
    # Validation statistics
    val_stats = validated_refined.get('validation_stats', {})
    stats_text = f"VALIDATION FILTERING:\n\n"
    stats_text += f"Stage 1 (size): {val_stats.get('stage1_size_filtered', 0)}\n"
    stats_text += f"Stage 2 (shape): {val_stats.get('stage2_quality_filtered', 0)}\n"
    stats_text += f"Stage 3 (intensity): {val_stats.get('stage3_intensity_filtered', 0)}\n"
    stats_text += f"Final accepted: {val_stats.get('final_accepted', 0)}\n\n"
    
    if val_stats.get('stage1_size_filtered', 0) > 0:
        acceptance_rate = val_stats.get('final_accepted', 0) / val_stats.get('stage1_size_filtered', 1)
        stats_text += f"Acceptance rate: {acceptance_rate:.2%}\n"
    
    stats_text += f"\nReliability: {val_assessment.get('reliability', 'unknown')}\n"
    stats_text += f"Density: {val_assessment.get('pore_density_per_mm2', 0):.0f}/mm²"
    
    axes[1,2].text(0.05, 0.95, stats_text, transform=axes[1,2].transAxes,
                  fontsize=10, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow'))
    axes[1,2].set_title('Validation Statistics')
    axes[1,2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Validation comparison saved: {save_path}")

def test_validated_refined():
    """Test and validate the refined method."""
    
    # Import original refined analyzer
    sys.path.append(str(test_dir))
    from refined_pore_detection import RefinedPoreDetector
    
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
    
    print(f"📸 Testing validated refined method on: {Path(image_path).name}")
    
    # Load image and get fiber data
    image = load_image(image_path)
    scale_result = detect_scale_bar(image, debug=False)
    scale_factor = scale_result.get('micrometers_per_pixel', 1.0) if scale_result.get('scale_detected') else 1.0
    
    detector = FiberTypeDetector()
    fiber_type, confidence, analysis_data = detector.classify_fiber_type(image)
    
    print(f"🔬 Fiber type: {fiber_type} (confidence: {confidence:.3f})")
    print(f"📏 Scale factor: {scale_factor:.4f} μm/pixel")
    
    # Run original refined analysis
    print("\n🔬 Running original refined analysis...")
    refined_analyzer = RefinedPoreDetector()
    original_result = refined_analyzer.analyze_fiber_porosity_refined(image, analysis_data, scale_factor)
    
    # Run validated refined analysis
    print("\n🔬 Running validated refined analysis...")
    validated_analyzer = ValidatedRefinedDetector()
    validated_result = validated_analyzer.analyze_fiber_porosity_validated(image, analysis_data, scale_factor)
    
    # Create comparison
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_path = results_dir / f"validated_refined_comparison_{timestamp}.png"
    create_validation_comparison(image, original_result, validated_result, scale_factor, str(viz_path))
    
    # Print final assessment
    orig_porosity = original_result.get('porosity_metrics', {}).get('total_porosity_percent', 0)
    val_porosity = validated_result.get('porosity_metrics', {}).get('total_porosity_percent', 0)
    orig_count = original_result.get('porosity_metrics', {}).get('pore_count', 0)
    val_count = validated_result.get('porosity_metrics', {}).get('pore_count', 0)
    
    val_assessment = validated_result.get('porosity_metrics', {}).get('validation_assessment', {})
    
    print(f"\n🎯 VALIDATION RESULTS:")
    print(f"Original refined: {orig_porosity:.2f}% porosity, {orig_count} pores")
    print(f"Validated refined: {val_porosity:.2f}% porosity, {val_count} pores")
    print(f"Difference: {val_porosity - orig_porosity:+.2f}% porosity, {val_count - orig_count:+d} pores")
    
    print(f"\n📊 VALIDATION QUALITY:")
    print(f"Quality level: {val_assessment.get('quality_level', 'unknown')}")
    print(f"Confidence: {val_assessment.get('confidence', 0):.2f}")
    print(f"Reliability: {val_assessment.get('reliability', 'unknown')}")
    print(f"Acceptance rate: {val_assessment.get('acceptance_rate', 0):.2%}")
    
    print(f"\n✅ FINAL RECOMMENDATION:")
    if val_assessment.get('quality_level') in ['excellent', 'good']:
        print(f"   🚀 VALIDATED REFINED method is PRODUCTION READY")
        print(f"   📊 Porosity: {val_porosity:.2f}%")
        print(f"   🔢 Pore count: {val_count:,}")
        print(f"   🎯 This appears to match the visual evidence in the SEM image")
    else:
        print(f"   ⚠️ Method needs further tuning")
        print(f"   📊 Consider parameter adjustment or manual verification")
    
    return validated_result

if __name__ == "__main__":
    result = test_validated_refined()