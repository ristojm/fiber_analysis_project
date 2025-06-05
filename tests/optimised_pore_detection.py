#!/usr/bin/env python3
"""
Optimized Pore Detection - Balanced Enhancement
Fine-tunes the refined method with better quality control and validation

Save as: tests/optimized_pore_detection.py
Run from: tests/ folder
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import ndimage

# Setup paths
test_dir = Path(__file__).parent
project_root = test_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "modules"))

from modules.scale_detection import detect_scale_bar
from modules.fiber_type_detection import FiberTypeDetector
from modules.image_preprocessing import load_image

class OptimizedPoreDetector:
    """
    Optimized detector with balanced sensitivity and precision.
    Incorporates multi-stage validation and adaptive thresholding.
    """
    
    def __init__(self):
        self.config = {
            'pore_detection': {
                # Optimized sensitivity parameters
                'intensity_percentile': 27,      # Slightly more conservative
                'min_pore_area_pixels': 4,       # Keep small minimum
                'max_pore_area_ratio': 0.08,     # Slightly more conservative
                'adaptive_methods': ['percentile', 'local_adaptive', 'gradient'],
                'combine_strategy': 'weighted_union',  # More sophisticated combination
                'morphology_iterations': 1,
                'multi_stage_validation': True,
            },
            'segmentation': {
                'gaussian_blur': 0.8,            # Very minimal blur
                'edge_preservation': True,       # Preserve important edges
                'noise_reduction_strength': 'light',
            },
            'hollow_fiber': {
                'exclude_lumen': True,
                'lumen_buffer_pixels': 2,        # Reduced buffer
                'wall_analysis_only': True,
            },
            'quality_control': {
                # Size-dependent quality thresholds
                'tiny_pore_thresholds': {       # < 10 μm²
                    'min_circularity': 0.12,
                    'max_aspect_ratio': 5,
                    'min_solidity': 0.4,
                    'intensity_validation': True,
                    'context_validation': True,
                },
                'small_pore_thresholds': {      # 10-50 μm²
                    'min_circularity': 0.08,
                    'max_aspect_ratio': 8,
                    'min_solidity': 0.3,
                    'intensity_validation': True,
                    'context_validation': False,
                },
                'medium_pore_thresholds': {     # 50-500 μm²
                    'min_circularity': 0.05,
                    'max_aspect_ratio': 10,
                    'min_solidity': 0.25,
                    'intensity_validation': False,
                    'context_validation': False,
                },
                'large_pore_thresholds': {      # > 500 μm²
                    'min_circularity': 0.03,
                    'max_aspect_ratio': 15,
                    'min_solidity': 0.2,
                    'intensity_validation': False,
                    'context_validation': False,
                },
            }
        }
    
    def analyze_fiber_porosity_optimized(self, image, fiber_analysis_data, scale_factor=1.0):
        """
        Optimized porosity analysis with multi-stage validation.
        """
        
        print(f"\n🔬 OPTIMIZED PORE DETECTION")
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
            
            print(f"\n   Optimized Analysis - Fiber {i+1}:")
            
            # Create analysis mask (exclude lumen with smaller buffer)
            fiber_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(fiber_mask, [fiber_contour], 255)
            analysis_mask = fiber_mask.copy()
            fiber_area_pixels = np.sum(fiber_mask > 0)
            
            if has_lumen and fiber_result.get('lumen_properties'):
                lumen_contour = fiber_result.get('lumen_properties', {}).get('contour')
                if lumen_contour is not None:
                    lumen_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(lumen_mask, [lumen_contour], 255)
                    
                    # Smaller, more precise buffer around lumen
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
            
            # Multi-stage pore detection
            fiber_pores = self._detect_pores_optimized(image, analysis_mask, scale_factor)
            
            # Add fiber info
            for pore in fiber_pores:
                pore['fiber_id'] = i
                pore['fiber_wall_area_pixels'] = wall_area_pixels
            
            all_pore_results.extend(fiber_pores)
            total_wall_area_pixels += wall_area_pixels
            
            pore_area_pixels = sum(pore['area_pixels'] for pore in fiber_pores)
            total_pore_area_pixels += pore_area_pixels
            
            fiber_porosity = (pore_area_pixels / wall_area_pixels * 100) if wall_area_pixels > 0 else 0
            
            print(f"     Optimized detection: {len(fiber_pores)} pores")
            print(f"     Pore area: {pore_area_pixels:,} pixels ({pore_area_pixels * scale_factor**2:,.0f} μm²)")
            print(f"     Fiber porosity: {fiber_porosity:.2f}%")
        
        # Calculate results with quality assessment
        total_wall_area_um2 = total_wall_area_pixels * (scale_factor ** 2)
        total_pore_area_um2 = total_pore_area_pixels * (scale_factor ** 2)
        overall_porosity = (total_pore_area_pixels / total_wall_area_pixels * 100) if total_wall_area_pixels > 0 else 0
        
        print(f"\n🎯 OPTIMIZED OVERALL RESULTS:")
        print(f"   Total wall area: {total_wall_area_pixels:,} pixels ({total_wall_area_um2:,.0f} μm²)")
        print(f"   Total pore area: {total_pore_area_pixels:,} pixels ({total_pore_area_um2:,.0f} μm²)")
        print(f"   OPTIMIZED POROSITY: {overall_porosity:.2f}%")
        print(f"   Total pores: {len(all_pore_results)}")
        
        # Enhanced pore size analysis
        if all_pore_results:
            size_analysis = self._analyze_pore_size_distribution(all_pore_results)
            print(f"   {size_analysis['size_summary']}")
        
        # Quality assessment
        quality_metrics = self._assess_detection_quality(all_pore_results, total_wall_area_um2)
        
        # Create comprehensive metrics
        porosity_metrics = self._calculate_optimized_metrics(
            all_pore_results, total_wall_area_um2, total_pore_area_um2, 
            overall_porosity, quality_metrics
        )
        
        return {
            'porosity_metrics': porosity_metrics,
            'individual_pores': all_pore_results,
            'quality_assessment': quality_metrics,
            'method': 'optimized_multi_stage',
            'scale_factor': scale_factor,
        }
    
    def _detect_pores_optimized(self, image, analysis_mask, scale_factor):
        """
        Multi-stage optimized pore detection with weighted combination.
        """
        
        masked_image = cv2.bitwise_and(image, image, mask=analysis_mask)
        region_pixels = masked_image[analysis_mask > 0]
        
        if len(region_pixels) == 0:
            return []
        
        print(f"     Multi-stage optimized detection:")
        
        # Enhanced preprocessing
        processed_image = self._preprocess_for_detection(masked_image, analysis_mask)
        
        # Multi-method detection
        detection_masks = self._apply_multiple_detection_methods(processed_image, region_pixels, analysis_mask)
        
        # Weighted combination strategy
        combined_mask = self._combine_detection_masks(detection_masks, analysis_mask)
        
        # Find and validate contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Multi-stage filtering
        validated_pores = self._multi_stage_pore_validation(
            contours, masked_image, analysis_mask, scale_factor
        )
        
        print(f"       Raw detections: {len(contours)}, Multi-stage filtered: {len(validated_pores)}")
        
        return validated_pores
    
    def _preprocess_for_detection(self, masked_image, analysis_mask):
        """Enhanced preprocessing that preserves important features."""
        
        # Very light noise reduction while preserving edges
        if self.config['segmentation']['edge_preservation']:
            # Use edge-preserving filter
            processed = cv2.edgePreservingFilter(masked_image, flags=1, sigma_s=30, sigma_r=0.4)
        else:
            processed = masked_image.copy()
        
        # Minimal gaussian blur if needed
        blur_sigma = self.config['segmentation']['gaussian_blur']
        if blur_sigma > 0:
            processed = cv2.GaussianBlur(processed, (3, 3), blur_sigma)
        
        return processed
    
    def _apply_multiple_detection_methods(self, image, region_pixels, analysis_mask):
        """Apply multiple detection methods and return individual masks."""
        
        detection_masks = {}
        
        # Method 1: Percentile-based (most reliable)
        percentile = self.config['pore_detection']['intensity_percentile']
        percentile_threshold = np.percentile(region_pixels, percentile)
        detection_masks['percentile'] = {
            'mask': (image < percentile_threshold) & (analysis_mask > 0),
            'weight': 0.4,
            'confidence': 0.9
        }
        
        # Method 2: Local adaptive (good for varying illumination)
        local_mask = self._apply_local_adaptive(image, analysis_mask)
        detection_masks['local_adaptive'] = {
            'mask': local_mask,
            'weight': 0.35,
            'confidence': 0.8
        }
        
        # Method 3: Gradient-based (edge-sensitive)
        gradient_mask = self._apply_gradient_detection(image, analysis_mask)
        detection_masks['gradient'] = {
            'mask': gradient_mask,
            'weight': 0.25,
            'confidence': 0.7
        }
        
        print(f"       Applied {len(detection_masks)} detection methods")
        for method, data in detection_masks.items():
            pore_count = np.sum(data['mask'])
            print(f"         {method}: {pore_count:,} potential pore pixels")
        
        return detection_masks
    
    def _apply_local_adaptive(self, image, analysis_mask):
        """Apply conservative local adaptive thresholding."""
        
        # Only apply if region is large enough
        if np.sum(analysis_mask) < 200:
            return np.zeros_like(analysis_mask, dtype=bool)
        
        coords = np.column_stack(np.where(analysis_mask > 0))
        if len(coords) == 0:
            return np.zeros_like(analysis_mask, dtype=bool)
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        roi_height = y_max - y_min + 1
        roi_width = x_max - x_min + 1
        
        if roi_height < 40 or roi_width < 40:
            return np.zeros_like(analysis_mask, dtype=bool)
        
        roi = image[y_min:y_max+1, x_min:x_max+1]
        
        # Adaptive block size based on ROI size
        block_size = max(11, min(51, min(roi_height, roi_width) // 12))
        if block_size % 2 == 0:
            block_size += 1
        
        # Conservative parameters
        local_adaptive = cv2.adaptiveThreshold(
            roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 
            block_size, 4  # Slightly more conservative
        )
        
        # Map back to full image
        local_thresh = np.zeros_like(image, dtype=np.uint8)
        local_thresh[y_min:y_max+1, x_min:x_max+1] = local_adaptive
        
        return (local_thresh > 0) & (analysis_mask > 0)
    
    def _apply_gradient_detection(self, image, analysis_mask):
        """Apply gradient-based edge detection for pore boundaries."""
        
        # Calculate gradients
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Threshold gradient
        gradient_threshold = np.percentile(gradient_magnitude[analysis_mask > 0], 70)
        edge_mask = gradient_magnitude > gradient_threshold
        
        # Fill enclosed regions
        filled_mask = ndimage.binary_fill_holes(edge_mask)
        
        # Apply morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        cleaned_mask = cv2.morphologyEx(filled_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        return (cleaned_mask > 0) & (analysis_mask > 0)
    
    def _combine_detection_masks(self, detection_masks, analysis_mask):
        """Combine multiple detection masks using weighted strategy."""
        
        if self.config['pore_detection']['combine_strategy'] == 'weighted_union':
            # Create weighted combination
            combined_score = np.zeros(analysis_mask.shape, dtype=np.float32)
            
            for method, data in detection_masks.items():
                mask = data['mask']
                weight = data['weight']
                confidence = data['confidence']
                
                # Add weighted contribution
                combined_score += mask.astype(np.float32) * weight * confidence
            
            # Adaptive threshold for final mask
            # Higher threshold for areas with fewer contributing methods
            threshold = 0.3  # Require at least 30% confidence
            final_mask = combined_score > threshold
            
            # Light morphological cleanup
            kernel = np.ones((2, 2), np.uint8)
            final_mask = cv2.morphologyEx(final_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            
            print(f"       Weighted combination threshold: {threshold}")
            
        else:
            # Fallback to simple union
            final_mask = np.zeros_like(analysis_mask, dtype=bool)
            for method, data in detection_masks.items():
                final_mask |= data['mask']
        
        return final_mask.astype(np.uint8)
    
    def _multi_stage_pore_validation(self, contours, image, analysis_mask, scale_factor):
        """Multi-stage validation with size-dependent criteria."""
        
        validated_pores = []
        
        # Size filtering
        min_area_pixels = self.config['pore_detection']['min_pore_area_pixels']
        max_area_pixels = int(np.sum(analysis_mask > 0) * self.config['pore_detection']['max_pore_area_ratio'])
        
        stage1_passed = 0
        stage2_passed = 0
        stage3_passed = 0
        
        for contour in contours:
            area_pixels = cv2.contourArea(contour)
            
            # Stage 1: Basic size filtering
            if area_pixels < min_area_pixels or area_pixels > max_area_pixels:
                continue
            stage1_passed += 1
            
            # Calculate properties
            pore_props = self._calculate_pore_properties(contour, area_pixels, scale_factor)
            area_um2 = pore_props['area_um2']
            
            # Stage 2: Size-dependent shape validation
            if not self._validate_pore_shape(pore_props, area_um2):
                continue
            stage2_passed += 1
            
            # Stage 3: Context validation (for tiny pores)
            if not self._validate_pore_context(pore_props, image, analysis_mask, area_um2):
                continue
            stage3_passed += 1
            
            # Add quality score
            pore_props['quality_score'] = self._calculate_quality_score(pore_props, area_um2)
            validated_pores.append(pore_props)
        
        print(f"         Stage 1 (size): {stage1_passed}/{len(contours)}")
        print(f"         Stage 2 (shape): {stage2_passed}/{stage1_passed}")
        print(f"         Stage 3 (context): {stage3_passed}/{stage2_passed}")
        
        return validated_pores
    
    def _validate_pore_shape(self, pore_props, area_um2):
        """Size-dependent shape validation."""
        
        # Determine size category and get appropriate thresholds
        if area_um2 < 10:
            thresholds = self.config['quality_control']['tiny_pore_thresholds']
        elif area_um2 < 50:
            thresholds = self.config['quality_control']['small_pore_thresholds']
        elif area_um2 < 500:
            thresholds = self.config['quality_control']['medium_pore_thresholds']
        else:
            thresholds = self.config['quality_control']['large_pore_thresholds']
        
        # Apply thresholds
        if pore_props['circularity'] < thresholds['min_circularity']:
            return False
        
        if pore_props['aspect_ratio'] > thresholds['max_aspect_ratio']:
            return False
        
        if pore_props.get('solidity', 1.0) < thresholds['min_solidity']:
            return False
        
        return True
    
    def _validate_pore_context(self, pore_props, image, analysis_mask, area_um2):
        """Context validation for tiny and small pores."""
        
        # Determine if context validation is needed
        if area_um2 < 10:
            thresholds = self.config['quality_control']['tiny_pore_thresholds']
        elif area_um2 < 50:
            thresholds = self.config['quality_control']['small_pore_thresholds']
        else:
            return True  # Skip context validation for larger pores
        
        if not (thresholds.get('intensity_validation', False) or thresholds.get('context_validation', False)):
            return True
        
        contour = pore_props['contour']
        pore_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(pore_mask, [contour], 255)
        
        # Intensity validation
        if thresholds.get('intensity_validation', False):
            pore_pixels = image[pore_mask > 0]
            if len(pore_pixels) == 0:
                return False
            
            pore_mean = np.mean(pore_pixels)
            
            # Check surrounding region
            kernel_size = max(5, int(np.sqrt(pore_props['area_pixels']) * 2))
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            expanded_mask = cv2.dilate(pore_mask, kernel, iterations=1)
            surrounding_mask = expanded_mask & ~pore_mask & analysis_mask
            
            surrounding_pixels = image[surrounding_mask > 0]
            if len(surrounding_pixels) > 0:
                surrounding_mean = np.mean(surrounding_pixels)
                intensity_ratio = pore_mean / surrounding_mean if surrounding_mean > 0 else 1.0
                
                # Tiny pores should be significantly darker
                if area_um2 < 5 and intensity_ratio > 0.85:
                    return False
                elif area_um2 < 10 and intensity_ratio > 0.9:
                    return False
        
        # Context validation (neighboring pores)
        if thresholds.get('context_validation', False) and area_um2 < 10:
            # Check if this tiny pore is isolated or part of a cluster
            # Isolated tiny pores are more likely to be noise
            kernel = np.ones((15, 15), np.uint8)
            neighborhood = cv2.dilate(pore_mask, kernel, iterations=1)
            
            # This would require checking against other detected pores
            # For now, we'll be more lenient with tiny pores in clustered regions
            pass
        
        return True
    
    def _calculate_quality_score(self, pore_props, area_um2):
        """Calculate a quality score for the pore (0-1)."""
        
        score = 0.5  # Base score
        
        # Circularity contribution
        circularity_score = min(1.0, pore_props['circularity'] / 0.8)
        score += 0.3 * circularity_score
        
        # Aspect ratio contribution (lower is better)
        aspect_score = max(0.0, 1.0 - (pore_props['aspect_ratio'] - 1.0) / 4.0)
        score += 0.2 * aspect_score
        
        # Size appropriateness (medium sizes score higher)
        if 20 <= area_um2 <= 200:
            size_score = 1.0
        elif 10 <= area_um2 <= 500:
            size_score = 0.8
        elif 5 <= area_um2 <= 1000:
            size_score = 0.6
        else:
            size_score = 0.4
        
        score = min(1.0, score + 0.1 * size_score)
        
        return score
    
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
    
    def _analyze_pore_size_distribution(self, pores):
        """Analyze pore size distribution with detailed breakdown."""
        
        pore_areas = [pore['area_um2'] for pore in pores]
        
        # Size categories
        tiny = [p for p in pore_areas if p < 5]
        very_small = [p for p in pore_areas if 5 <= p < 15]
        small = [p for p in pore_areas if 15 <= p < 50]
        medium = [p for p in pore_areas if 50 <= p < 200]
        large = [p for p in pore_areas if 200 <= p < 500]
        very_large = [p for p in pore_areas if p >= 500]
        
        total = len(pore_areas)
        
        size_summary = f"Size distribution:"
        size_summary += f"\n     Tiny (<5 μm²): {len(tiny)} ({len(tiny)/total*100:.1f}%)"
        size_summary += f"\n     Very small (5-15 μm²): {len(very_small)} ({len(very_small)/total*100:.1f}%)"
        size_summary += f"\n     Small (15-50 μm²): {len(small)} ({len(small)/total*100:.1f}%)"
        size_summary += f"\n     Medium (50-200 μm²): {len(medium)} ({len(medium)/total*100:.1f}%)"
        size_summary += f"\n     Large (200-500 μm²): {len(large)} ({len(large)/total*100:.1f}%)"
        size_summary += f"\n     Very large (>500 μm²): {len(very_large)} ({len(very_large)/total*100:.1f}%)"
        
        return {
            'size_summary': size_summary,
            'categories': {
                'tiny': len(tiny),
                'very_small': len(very_small),
                'small': len(small),
                'medium': len(medium),
                'large': len(large),
                'very_large': len(very_large)
            }
        }
    
    def _assess_detection_quality(self, pores, total_wall_area_um2):
        """Assess the quality of the detection results."""
        
        if not pores:
            return {'overall_quality': 'poor', 'confidence': 0.0}
        
        # Calculate quality metrics
        quality_scores = [pore.get('quality_score', 0.5) for pore in pores]
        avg_quality = np.mean(quality_scores)
        
        # Size distribution analysis
        pore_areas = [pore['area_um2'] for pore in pores]
        tiny_fraction = len([p for p in pore_areas if p < 10]) / len(pore_areas)
        
        # Assess overall quality
        if avg_quality > 0.8 and tiny_fraction < 0.7:
            overall_quality = 'excellent'
            confidence = 0.9
        elif avg_quality > 0.7 and tiny_fraction < 0.8:
            overall_quality = 'good'
            confidence = 0.8
        elif avg_quality > 0.6:
            overall_quality = 'moderate'
            confidence = 0.7
        elif tiny_fraction > 0.9:
            overall_quality = 'questionable_tiny_pores'
            confidence = 0.5
        else:
            overall_quality = 'poor'
            confidence = 0.4
        
        return {
            'overall_quality': overall_quality,
            'confidence': confidence,
            'avg_quality_score': avg_quality,
            'tiny_pore_fraction': tiny_fraction,
            'total_pores': len(pores),
            'pore_density_per_mm2': len(pores) / (total_wall_area_um2 / 1e6) if total_wall_area_um2 > 0 else 0
        }
    
    def _calculate_optimized_metrics(self, pores, total_wall_area_um2, total_pore_area_um2, 
                                   overall_porosity, quality_metrics):
        """Calculate comprehensive optimized metrics."""
        
        if not pores:
            return {
                'total_porosity_percent': 0.0,
                'pore_count': 0,
                'total_pore_area_um2': 0.0,
                'wall_area_um2': total_wall_area_um2,
                'average_pore_size_um2': 0.0,
                'detection_quality': quality_metrics
            }
        
        pore_areas = [pore['area_um2'] for pore in pores]
        pore_diameters = [pore['equivalent_diameter_um'] for pore in pores]
        quality_scores = [pore.get('quality_score', 0.5) for pore in pores]
        
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
            'average_quality_score': np.mean(quality_scores),
            'detection_quality': quality_metrics,
            'method': 'optimized_multi_stage'
        }

def create_three_way_comparison(image, enhanced_result, refined_result, optimized_result, scale_factor, save_path):
    """Create three-way comparison between enhanced, refined, and optimized detection."""
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
    # Row 1: Method summaries
    for col, (result, title, color) in enumerate([
        (enhanced_result, 'Enhanced Analysis', 'lightblue'),
        (refined_result, 'Refined Analysis', 'lightgreen'),
        (optimized_result, 'Optimized Analysis', 'lightyellow')
    ]):
        metrics = result.get('porosity_metrics', {})
        quality = result.get('quality_assessment', {})
        
        text = f"{title.upper()}:\n\n"
        text += f"Porosity: {metrics.get('total_porosity_percent', 0):.2f}%\n"
        text += f"Pore count: {metrics.get('pore_count', 0)}\n"
        text += f"Avg pore size: {metrics.get('average_pore_size_um2', 0):.1f} μm²\n\n"
        
        if 'detection_quality' in metrics:
            dq = metrics['detection_quality']
            text += f"Quality: {dq.get('overall_quality', 'unknown')}\n"
            text += f"Confidence: {dq.get('confidence', 0):.2f}\n"
            text += f"Avg score: {dq.get('avg_quality_score', 0):.2f}"
        
        axes[0, col].text(0.05, 0.95, text, transform=axes[0, col].transAxes,
                         fontsize=10, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle="round,pad=0.5", facecolor=color))
        axes[0, col].set_title(title, fontweight='bold')
        axes[0, col].axis('off')
    
    # Row 2: Visual comparisons
    for col, (result, title) in enumerate([
        (enhanced_result, 'Enhanced'),
        (refined_result, 'Refined'),
        (optimized_result, 'Optimized')
    ]):
        if 'individual_pores' in result:
            overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            pore_count = 0
            
            for pore in result['individual_pores']:
                if 'contour' in pore:
                    area = pore['area_um2']
                    pore_count += 1
                    
                    # Color by size
                    if area < 10:
                        color = (0, 255, 0)     # Green for tiny
                    elif area < 50:
                        color = (0, 255, 255)   # Yellow for small
                    elif area < 200:
                        color = (0, 165, 255)   # Orange for medium
                    else:
                        color = (0, 0, 255)     # Red for large
                    
                    # Thickness based on quality (if available)
                    thickness = 2
                    if 'quality_score' in pore:
                        thickness = max(1, int(pore['quality_score'] * 3))
                    
                    cv2.drawContours(overlay, [pore['contour']], -1, color, thickness)
            
            axes[1, col].imshow(overlay)
            axes[1, col].set_title(f'{title}: {pore_count} Pores')
            axes[1, col].axis('off')
    
    # Row 3: Detailed analysis
    
    # Size distribution comparison
    methods = ['Enhanced', 'Refined', 'Optimized']
    results = [enhanced_result, refined_result, optimized_result]
    colors = ['lightblue', 'lightgreen', 'lightyellow']
    
    # Pore counts
    pore_counts = [r.get('porosity_metrics', {}).get('pore_count', 0) for r in results]
    porosities = [r.get('porosity_metrics', {}).get('total_porosity_percent', 0) for r in results]
    avg_sizes = [r.get('porosity_metrics', {}).get('average_pore_size_um2', 0) for r in results]
    
    # Bar chart comparison
    x = np.arange(len(methods))
    width = 0.25
    
    axes[2, 0].bar(x - width, pore_counts, width, label='Pore Count (/10)', color='skyblue')
    axes[2, 0].bar(x, porosities, width, label='Porosity %', color='lightgreen')
    axes[2, 0].bar(x + width, [s/10 for s in avg_sizes], width, label='Avg Size (/10 μm²)', color='lightyellow')
    
    axes[2, 0].set_xlabel('Method')
    axes[2, 0].set_ylabel('Value')
    axes[2, 0].set_title('Quantitative Comparison')
    axes[2, 0].set_xticks(x)
    axes[2, 0].set_xticklabels(methods)
    axes[2, 0].legend()
    
    # Add value labels on bars
    for i, (count, porosity, size) in enumerate(zip(pore_counts, porosities, avg_sizes)):
        axes[2, 0].text(i - width, count + 50, f'{count}', ha='center', va='bottom', fontsize=8)
        axes[2, 0].text(i, porosity + 0.5, f'{porosity:.1f}%', ha='center', va='bottom', fontsize=8)
        axes[2, 0].text(i + width, size/10 + 0.5, f'{size:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Quality comparison (if available)
    quality_data = []
    for result in results:
        dq = result.get('porosity_metrics', {}).get('detection_quality', {})
        quality_data.append({
            'confidence': dq.get('confidence', 0),
            'avg_quality': dq.get('avg_quality_score', 0),
            'tiny_fraction': dq.get('tiny_pore_fraction', 0)
        })
    
    if any(qd['confidence'] > 0 for qd in quality_data):
        quality_metrics = ['Confidence', 'Avg Quality', 'Tiny Fraction']
        enhanced_quality = [quality_data[0]['confidence'], quality_data[0]['avg_quality'], quality_data[0]['tiny_fraction']]
        refined_quality = [quality_data[1]['confidence'], quality_data[1]['avg_quality'], quality_data[1]['tiny_fraction']]
        optimized_quality = [quality_data[2]['confidence'], quality_data[2]['avg_quality'], quality_data[2]['tiny_fraction']]
        
        x_qual = np.arange(len(quality_metrics))
        axes[2, 1].bar(x_qual - width, enhanced_quality, width, label='Enhanced', color='lightblue')
        axes[2, 1].bar(x_qual, refined_quality, width, label='Refined', color='lightgreen')
        axes[2, 1].bar(x_qual + width, optimized_quality, width, label='Optimized', color='lightyellow')
        
        axes[2, 1].set_xlabel('Quality Metric')
        axes[2, 1].set_ylabel('Score (0-1)')
        axes[2, 1].set_title('Quality Assessment')
        axes[2, 1].set_xticks(x_qual)
        axes[2, 1].set_xticklabels(quality_metrics)
        axes[2, 1].legend()
        axes[2, 1].set_ylim(0, 1)
    else:
        axes[2, 1].axis('off')
    
    # Summary recommendations
    summary_text = "OPTIMIZATION SUMMARY:\n\n"
    
    opt_count = optimized_result.get('porosity_metrics', {}).get('pore_count', 0)
    ref_count = refined_result.get('porosity_metrics', {}).get('pore_count', 0)
    enh_count = enhanced_result.get('porosity_metrics', {}).get('pore_count', 0)
    
    opt_quality = optimized_result.get('porosity_metrics', {}).get('detection_quality', {})
    
    if opt_quality.get('overall_quality') in ['excellent', 'good']:
        summary_text += "✅ OPTIMIZED method recommended\n"
        summary_text += f"• High quality detection\n"
        summary_text += f"• {opt_count} pores detected\n"
        summary_text += f"• Confidence: {opt_quality.get('confidence', 0):.2f}\n\n"
    elif ref_count > enh_count * 3 and opt_count < ref_count * 1.5:
        summary_text += "✅ OPTIMIZED method recommended\n"
        summary_text += f"• Balanced sensitivity/precision\n"
        summary_text += f"• Reduced over-detection vs refined\n\n"
    elif opt_quality.get('overall_quality') == 'questionable_tiny_pores':
        summary_text += "⚠️ ENHANCED method recommended\n"
        summary_text += f"• Too many tiny pores detected\n"
        summary_text += f"• May need parameter adjustment\n\n"
    else:
        summary_text += "📊 Manual review recommended\n"
        summary_text += f"• Compare all three methods\n"
        summary_text += f"• Consider sample characteristics\n\n"
    
    summary_text += f"Porosity range: {min(porosities):.1f}% - {max(porosities):.1f}%\n"
    summary_text += f"Pore count range: {min(pore_counts)} - {max(pore_counts)}"
    
    axes[2, 2].text(0.05, 0.95, summary_text, transform=axes[2, 2].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcyan'))
    axes[2, 2].set_title('Analysis Summary', fontweight='bold')
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Three-way comparison saved: {save_path}")

def test_optimized_detection():
    """Test optimized vs refined vs enhanced detection."""
    
    # Import previous analyzers
    sys.path.append(str(test_dir))
    from enhanced_pore_detection import EnhancedPoreDetector
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
    
    print(f"📸 Testing optimized detection on: {Path(image_path).name}")
    
    # Load image and get fiber data
    image = load_image(image_path)
    scale_result = detect_scale_bar(image, debug=False)
    scale_factor = scale_result.get('micrometers_per_pixel', 1.0) if scale_result.get('scale_detected') else 1.0
    
    detector = FiberTypeDetector()
    fiber_type, confidence, analysis_data = detector.classify_fiber_type(image)
    
    # Run all three analyses
    print("\n🔬 Running enhanced analysis...")
    enhanced_analyzer = EnhancedPoreDetector()
    enhanced_result = enhanced_analyzer.analyze_fiber_porosity_enhanced(image, analysis_data, scale_factor)
    
    print("\n🔬 Running refined analysis...")
    refined_analyzer = RefinedPoreDetector()
    refined_result = refined_analyzer.analyze_fiber_porosity_refined(image, analysis_data, scale_factor)
    
    print("\n🔬 Running optimized analysis...")
    optimized_analyzer = OptimizedPoreDetector()
    optimized_result = optimized_analyzer.analyze_fiber_porosity_optimized(image, analysis_data, scale_factor)
    
    # Create three-way comparison
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_path = results_dir / f"optimized_three_way_comparison_{timestamp}.png"
    create_three_way_comparison(image, enhanced_result, refined_result, optimized_result, scale_factor, str(viz_path))
    
    # Print comparison
    enhanced_porosity = enhanced_result.get('porosity_metrics', {}).get('total_porosity_percent', 0)
    refined_porosity = refined_result.get('porosity_metrics', {}).get('total_porosity_percent', 0)
    optimized_porosity = optimized_result.get('porosity_metrics', {}).get('total_porosity_percent', 0)
    
    enhanced_count = enhanced_result.get('porosity_metrics', {}).get('pore_count', 0)
    refined_count = refined_result.get('porosity_metrics', {}).get('pore_count', 0)
    optimized_count = optimized_result.get('porosity_metrics', {}).get('pore_count', 0)
    
    optimized_quality = optimized_result.get('porosity_metrics', {}).get('detection_quality', {})
    
    print(f"\n🎯 THREE-WAY COMPARISON RESULTS:")
    print(f"Enhanced:  {enhanced_porosity:.2f}% porosity, {enhanced_count} pores")
    print(f"Refined:   {refined_porosity:.2f}% porosity, {refined_count} pores")
    print(f"Optimized: {optimized_porosity:.2f}% porosity, {optimized_count} pores")
    
    print(f"\nOptimized Quality Assessment:")
    print(f"Overall Quality: {optimized_quality.get('overall_quality', 'unknown')}")
    print(f"Confidence: {optimized_quality.get('confidence', 0):.2f}")
    print(f"Average Quality Score: {optimized_quality.get('avg_quality_score', 0):.2f}")
    
    # Recommendation
    if optimized_quality.get('overall_quality') in ['excellent', 'good']:
        print(f"\n✅ RECOMMENDATION: Use OPTIMIZED method")
        print(f"   Reason: High quality detection with good balance")
    elif optimized_count < refined_count * 0.7:
        print(f"\n⚠️  RECOMMENDATION: Consider REFINED method")
        print(f"   Reason: Optimized may be too conservative")
    elif optimized_quality.get('overall_quality') == 'questionable_tiny_pores':
        print(f"\n⚠️  RECOMMENDATION: Use ENHANCED method")
        print(f"   Reason: Too many questionable tiny pores detected")
    else:
        print(f"\n📊 RECOMMENDATION: Manual review required")
        print(f"   Reason: Results vary significantly between methods")
    
    return optimized_result

if __name__ == "__main__":
    result = test_optimized_detection()