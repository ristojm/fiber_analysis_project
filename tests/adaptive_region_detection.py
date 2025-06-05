#!/usr/bin/env python3
"""
Adaptive Region-Based Pore Detection
Handles different surface types (face vs side) with specialized detection

Save as: tests/adaptive_region_detection.py
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
from skimage import filters, morphology

# Setup paths
test_dir = Path(__file__).parent
project_root = test_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "modules"))

from modules.scale_detection import detect_scale_bar
from modules.fiber_type_detection import FiberTypeDetector
from modules.image_preprocessing import load_image

class AdaptiveRegionDetector:
    """
    Adaptive detector that handles different surface regions with specialized algorithms.
    """
    
    def __init__(self):
        self.config = {
            'region_detection': {
                'intensity_analysis': True,  # Analyze intensity patterns
                'gradient_analysis': True,   # Use gradient to find edges
                'morphological_analysis': True,  # Use shape analysis
                'min_face_area_ratio': 0.1,  # Minimum 10% of fiber for face region
            },
            'face_detection': {
                'intensity_percentiles': [20, 25, 30],  # Multiple thresholds to try
                'local_adaptive': True,     # Use local adaptive thresholding
                'morphology_iterations': 1, # Light cleanup
                'min_pore_area_pixels': 3,  # Very small minimum
                'small_pore_emphasis': True, # Special handling for small face pores
            },
            'side_detection': {
                'intensity_percentiles': [25, 30, 35],  # Different thresholds for side
                'edge_enhancement': True,   # Enhance edges for side pores
                'morphology_iterations': 2, # More cleanup for noisy side
                'min_pore_area_pixels': 5,  # Slightly larger minimum
                'shadow_compensation': True, # Compensate for shadow effects
            },
            'combination': {
                'weight_face': 1.0,        # Equal weight
                'weight_side': 1.0,        # Equal weight
                'overlap_handling': 'union', # How to handle overlapping detections
            }
        }
    
    def analyze_fiber_porosity_adaptive(self, image, fiber_analysis_data, scale_factor=1.0):
        """
        Adaptive porosity analysis that handles face and side regions separately.
        """
        
        print(f"\n🎯 ADAPTIVE REGION-BASED DETECTION")
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
            
            print(f"\n   Adaptive Analysis - Fiber {i+1}:")
            
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
                    kernel = np.ones((7, 7), np.uint8)
                    lumen_mask = cv2.dilate(lumen_mask, kernel, iterations=1)
                    
                    analysis_mask = cv2.bitwise_and(analysis_mask, cv2.bitwise_not(lumen_mask))
                    lumen_area_pixels = np.sum(lumen_mask > 0)
                    wall_area_pixels = fiber_area_pixels - lumen_area_pixels
                    
                    print(f"     Wall area: {wall_area_pixels:,} pixels ({wall_area_pixels * scale_factor**2:,.0f} μm²)")
                else:
                    wall_area_pixels = fiber_area_pixels
            else:
                wall_area_pixels = fiber_area_pixels
            
            # Detect different surface regions
            face_mask, side_mask = self._detect_surface_regions(image, analysis_mask)
            
            # Analyze each region separately
            face_pores = self._detect_face_pores(image, face_mask, scale_factor)
            side_pores = self._detect_side_pores(image, side_mask, scale_factor)
            
            # Combine results
            combined_pores = self._combine_region_results(face_pores, side_pores, analysis_mask)
            
            # Add fiber info
            for pore in combined_pores:
                pore['fiber_id'] = i
                pore['fiber_wall_area_pixels'] = wall_area_pixels
            
            all_pore_results.extend(combined_pores)
            total_wall_area_pixels += wall_area_pixels
            
            pore_area_pixels = sum(pore['area_pixels'] for pore in combined_pores)
            total_pore_area_pixels += pore_area_pixels
            
            fiber_porosity = (pore_area_pixels / wall_area_pixels * 100) if wall_area_pixels > 0 else 0
            
            print(f"     Face pores: {len(face_pores)}")
            print(f"     Side pores: {len(side_pores)}")
            print(f"     Combined pores: {len(combined_pores)}")
            print(f"     Fiber porosity: {fiber_porosity:.2f}%")
        
        # Calculate results
        total_wall_area_um2 = total_wall_area_pixels * (scale_factor ** 2)
        total_pore_area_um2 = total_pore_area_pixels * (scale_factor ** 2)
        overall_porosity = (total_pore_area_pixels / total_wall_area_pixels * 100) if total_wall_area_pixels > 0 else 0
        
        print(f"\n🎯 ADAPTIVE OVERALL RESULTS:")
        print(f"   Total wall area: {total_wall_area_pixels:,} pixels ({total_wall_area_um2:,.0f} μm²)")
        print(f"   Total pore area: {total_pore_area_pixels:,} pixels ({total_pore_area_um2:,.0f} μm²)")
        print(f"   ADAPTIVE POROSITY: {overall_porosity:.2f}%")
        print(f"   Total pores: {len(all_pore_results)}")
        
        # Create metrics
        porosity_metrics = self._calculate_adaptive_metrics(
            all_pore_results, total_wall_area_um2, total_pore_area_um2, overall_porosity
        )
        
        return {
            'porosity_metrics': porosity_metrics,
            'individual_pores': all_pore_results,
            'method': 'adaptive_region_based',
            'scale_factor': scale_factor,
            'surface_analysis': {
                'face_pores': len([p for p in all_pore_results if p.get('surface_type') == 'face']),
                'side_pores': len([p for p in all_pore_results if p.get('surface_type') == 'side']),
            }
        }
    
    def _detect_surface_regions(self, image, analysis_mask):
        """
        Detect face vs side surface regions based on image characteristics.
        """
        
        print(f"     Detecting surface regions...")
        
        # Extract region for analysis
        masked_image = cv2.bitwise_and(image, image, mask=analysis_mask)
        
        # Method 1: Intensity-based detection
        # Face regions often have different average intensity
        region_pixels = masked_image[analysis_mask > 0]
        median_intensity = np.median(region_pixels)
        intensity_std = np.std(region_pixels)
        
        # Face mask: regions with intensity close to median (flat, consistent lighting)
        face_threshold_low = median_intensity - intensity_std * 0.3
        face_threshold_high = median_intensity + intensity_std * 0.3
        intensity_face_mask = ((masked_image >= face_threshold_low) & 
                              (masked_image <= face_threshold_high) & 
                              (analysis_mask > 0))
        
        # Method 2: Gradient-based detection
        # Side regions often have higher gradients (curved surfaces, edges)
        gradient_x = cv2.Sobel(masked_image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(masked_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        gradient_threshold = np.percentile(gradient_magnitude[analysis_mask > 0], 70)
        gradient_side_mask = (gradient_magnitude > gradient_threshold) & (analysis_mask > 0)
        
        # Method 3: Morphological analysis
        # Use distance transform to find interior vs edge regions
        distance = cv2.distanceTransform(analysis_mask, cv2.DIST_L2, 5)
        distance_threshold = np.percentile(distance[analysis_mask > 0], 60)
        morphological_face_mask = (distance > distance_threshold) & (analysis_mask > 0)
        
        # Combine methods for face detection
        face_mask = intensity_face_mask & morphological_face_mask
        
        # Ensure minimum face area
        face_area = np.sum(face_mask)
        min_face_area = np.sum(analysis_mask) * self.config['region_detection']['min_face_area_ratio']
        
        if face_area < min_face_area:
            # Expand face mask if too small
            kernel = np.ones((5, 5), np.uint8)
            face_mask = cv2.dilate(face_mask.astype(np.uint8), kernel, iterations=2) > 0
            face_mask = face_mask & (analysis_mask > 0)
        
        # Side mask is everything else
        side_mask = (analysis_mask > 0) & ~face_mask
        
        print(f"       Face region: {np.sum(face_mask):,} pixels")
        print(f"       Side region: {np.sum(side_mask):,} pixels")
        
        return face_mask, side_mask
    
    def _detect_face_pores(self, image, face_mask, scale_factor):
        """
        Specialized detection for face surface pores.
        """
        
        if np.sum(face_mask) == 0:
            return []
        
        print(f"       Detecting face pores...")
        
        masked_image = cv2.bitwise_and(image, image, mask=face_mask.astype(np.uint8))
        region_pixels = masked_image[face_mask > 0]
        
        if len(region_pixels) == 0:
            return []
        
        face_pores = []
        
        # Multiple threshold approach for face pores
        for percentile in self.config['face_detection']['intensity_percentiles']:
            threshold = np.percentile(region_pixels, percentile)
            
            # Create pore mask
            pore_mask = (masked_image < threshold) & face_mask
            
            # Light morphological cleanup
            if self.config['face_detection']['morphology_iterations'] > 0:
                kernel = np.ones((2, 2), np.uint8)
                pore_mask = cv2.morphologyEx(pore_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
                pore_mask = cv2.morphologyEx(pore_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(pore_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area_pixels = cv2.contourArea(contour)
                
                # Size filtering (very lenient for face pores)
                min_area = self.config['face_detection']['min_pore_area_pixels']
                max_area = np.sum(face_mask) * 0.1  # Max 10% of face area
                
                if area_pixels < min_area or area_pixels > max_area:
                    continue
                
                # Calculate properties
                pore_props = self._calculate_pore_properties(contour, area_pixels, scale_factor)
                pore_props['surface_type'] = 'face'
                pore_props['detection_threshold'] = threshold
                pore_props['detection_percentile'] = percentile
                
                # Very lenient shape filtering for face pores
                if pore_props['area_um2'] < 20:  # Very small pores
                    if pore_props['aspect_ratio'] > 10:  # Only reject extreme shapes
                        continue
                else:  # Larger face pores
                    if pore_props['aspect_ratio'] > 6:
                        continue
                    if pore_props['circularity'] < 0.08:
                        continue
                
                face_pores.append(pore_props)
        
        # Remove duplicates (pores detected with multiple thresholds)
        face_pores = self._remove_duplicate_pores(face_pores)
        
        print(f"         Face pores detected: {len(face_pores)}")
        return face_pores
    
    def _detect_side_pores(self, image, side_mask, scale_factor):
        """
        Specialized detection for side surface pores.
        """
        
        if np.sum(side_mask) == 0:
            return []
        
        print(f"       Detecting side pores...")
        
        masked_image = cv2.bitwise_and(image, image, mask=side_mask.astype(np.uint8))
        region_pixels = masked_image[side_mask > 0]
        
        if len(region_pixels) == 0:
            return []
        
        side_pores = []
        
        # Edge enhancement for side pores
        if self.config['side_detection']['edge_enhancement']:
            # Enhance edges to better detect side pores
            enhanced = cv2.bilateralFilter(masked_image, 5, 50, 50)
            masked_image = enhanced
            region_pixels = masked_image[side_mask > 0]
        
        # Multiple threshold approach for side pores
        for percentile in self.config['side_detection']['intensity_percentiles']:
            threshold = np.percentile(region_pixels, percentile)
            
            # Shadow compensation for side pores
            if self.config['side_detection']['shadow_compensation']:
                # Use local adaptive threshold for shadow areas
                local_thresh = cv2.adaptiveThreshold(
                    masked_image.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 11, 5
                )
                local_pore_mask = (local_thresh > 0) & side_mask
                
                # Combine with global threshold
                global_pore_mask = (masked_image < threshold) & side_mask
                pore_mask = global_pore_mask | local_pore_mask
            else:
                pore_mask = (masked_image < threshold) & side_mask
            
            # More aggressive morphological cleanup for side pores
            iterations = self.config['side_detection']['morphology_iterations']
            if iterations > 0:
                kernel = np.ones((3, 3), np.uint8)
                pore_mask = cv2.morphologyEx(pore_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
                pore_mask = cv2.morphologyEx(pore_mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
            
            # Find contours
            contours, _ = cv2.findContours(pore_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area_pixels = cv2.contourArea(contour)
                
                # Size filtering
                min_area = self.config['side_detection']['min_pore_area_pixels']
                max_area = np.sum(side_mask) * 0.15  # Max 15% of side area
                
                if area_pixels < min_area or area_pixels > max_area:
                    continue
                
                # Calculate properties
                pore_props = self._calculate_pore_properties(contour, area_pixels, scale_factor)
                pore_props['surface_type'] = 'side'
                pore_props['detection_threshold'] = threshold
                pore_props['detection_percentile'] = percentile
                
                # Moderate shape filtering for side pores
                if pore_props['aspect_ratio'] > 7:
                    continue
                if pore_props['circularity'] < 0.1:
                    continue
                
                side_pores.append(pore_props)
        
        # Remove duplicates
        side_pores = self._remove_duplicate_pores(side_pores)
        
        print(f"         Side pores detected: {len(side_pores)}")
        return side_pores
    
    def _remove_duplicate_pores(self, pores):
        """Remove duplicate pores detected with different thresholds."""
        
        if len(pores) <= 1:
            return pores
        
        # Group pores that are very close to each other
        unique_pores = []
        used = set()
        
        for i, pore1 in enumerate(pores):
            if i in used:
                continue
            
            # Find all similar pores
            similar_pores = [pore1]
            for j, pore2 in enumerate(pores[i+1:], i+1):
                if j in used:
                    continue
                
                # Check if pores overlap significantly
                distance = np.sqrt((pore1['centroid_x'] - pore2['centroid_x'])**2 + 
                                 (pore1['centroid_y'] - pore2['centroid_y'])**2)
                avg_radius = (pore1['radius_pixels'] + pore2['radius_pixels']) / 2
                
                if distance < avg_radius * 0.7:  # 70% overlap threshold
                    similar_pores.append(pore2)
                    used.add(j)
            
            # Keep the largest pore from the group
            best_pore = max(similar_pores, key=lambda p: p['area_pixels'])
            unique_pores.append(best_pore)
            used.add(i)
        
        return unique_pores
    
    def _combine_region_results(self, face_pores, side_pores, analysis_mask):
        """Combine face and side pore detection results."""
        
        # Simple union approach - combine all pores
        if self.config['combination']['overlap_handling'] == 'union':
            combined = face_pores + side_pores
            
            # Remove any overlaps between face and side detections
            combined = self._remove_duplicate_pores(combined)
            
            return combined
        
        # Could implement weighted combination or other strategies here
        return face_pores + side_pores
    
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
    
    def _calculate_adaptive_metrics(self, pores, total_wall_area_um2, total_pore_area_um2, overall_porosity):
        """Calculate comprehensive metrics with surface breakdown."""
        
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
        
        # Surface-specific metrics
        face_pores = [p for p in pores if p.get('surface_type') == 'face']
        side_pores = [p for p in pores if p.get('surface_type') == 'side']
        
        metrics = {
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
            
            # Surface-specific metrics
            'face_pore_count': len(face_pores),
            'side_pore_count': len(side_pores),
            'face_pore_area_um2': sum(p['area_um2'] for p in face_pores),
            'side_pore_area_um2': sum(p['area_um2'] for p in side_pores),
        }
        
        return metrics

def create_adaptive_visualization(image, enhanced_result, adaptive_result, scale_factor, save_path):
    """Create comparison between enhanced and adaptive detection."""
    
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
    enhanced_text += f"✅ Multi-method detection\n❓ Single threshold approach"
    
    axes[0,1].text(0.05, 0.95, enhanced_text, transform=axes[0,1].transAxes,
                  fontsize=11, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue'))
    axes[0,1].set_title('Enhanced Analysis')
    axes[0,1].axis('off')
    
    # Adaptive results
    adaptive_metrics = adaptive_result.get('porosity_metrics', {})
    surface_analysis = adaptive_result.get('surface_analysis', {})
    adaptive_text = f"ADAPTIVE ANALYSIS:\n\n"
    adaptive_text += f"Porosity: {adaptive_metrics.get('total_porosity_percent', 0):.2f}%\n"
    adaptive_text += f"Total pores: {adaptive_metrics.get('pore_count', 0)}\n"
    adaptive_text += f"Face pores: {surface_analysis.get('face_pores', 0)}\n"
    adaptive_text += f"Side pores: {surface_analysis.get('side_pores', 0)}\n\n"
    adaptive_text += f"✅ Region-adaptive detection\n✅ Face + side specialized"
    
    axes[0,2].text(0.05, 0.95, adaptive_text, transform=axes[0,2].transAxes,
                  fontsize=11, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen'))
    axes[0,2].set_title('Adaptive Analysis')
    axes[0,2].axis('off')
    
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
        
        axes[1,0].imshow(enhanced_overlay)
        axes[1,0].set_title(f'Enhanced: {enhanced_metrics.get("pore_count", 0)} Pores')
        axes[1,0].axis('off')
    
    # Adaptive detection visualization with surface types
    if 'individual_pores' in adaptive_result:
        adaptive_overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        for pore in adaptive_result['individual_pores']:
            if 'contour' in pore:
                surface_type = pore.get('surface_type', 'unknown')
                area = pore['area_um2']
                
                # Color by surface type AND size
                if surface_type == 'face':
                    if area < 50:
                        color = (0, 255, 0)    # Bright green for small face pores
                    else:
                        color = (0, 200, 0)    # Darker green for large face pores
                elif surface_type == 'side':
                    if area < 50:
                        color = (255, 0, 255)  # Magenta for small side pores
                    else:
                        color = (200, 0, 200)  # Darker magenta for large side pores
                else:
                    color = (128, 128, 128)    # Gray for unknown
                
                cv2.drawContours(adaptive_overlay, [pore['contour']], -1, color, 2)
        
        axes[1,1].imshow(adaptive_overlay)
        axes[1,1].set_title(f'Adaptive: {adaptive_metrics.get("pore_count", 0)} Pores\n(Green=Face, Magenta=Side)')
        axes[1,1].axis('off')
    
    # Comparison chart
    categories = ['Porosity %', 'Total Pores', 'Face Pores', 'Side Pores']
    enhanced_values = [
        enhanced_metrics.get('total_porosity_percent', 0),
        enhanced_metrics.get('pore_count', 0) / 100,  # Scale for display
        0,  # Enhanced doesn't separate face/side
        0
    ]
    adaptive_values = [
        adaptive_metrics.get('total_porosity_percent', 0),
        adaptive_metrics.get('pore_count', 0) / 100,
        surface_analysis.get('face_pores', 0) / 100,
        surface_analysis.get('side_pores', 0) / 100
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    axes[1,2].bar(x - width/2, enhanced_values, width, label='Enhanced', color='lightblue')
    axes[1,2].bar(x + width/2, adaptive_values, width, label='Adaptive', color='lightgreen')
    
    axes[1,2].set_xlabel('Metric')
    axes[1,2].set_ylabel('Value (pores scaled /100)')
    axes[1,2].set_title('Enhanced vs Adaptive Comparison')
    axes[1,2].set_xticks(x)
    axes[1,2].set_xticklabels(categories, rotation=45)
    axes[1,2].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Adaptive comparison saved: {save_path}")

def test_adaptive_detection():
    """Test adaptive vs enhanced detection."""
    
    # Import previous analyzers
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
    
    print(f"📸 Testing adaptive detection on: {Path(image_path).name}")
    
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
    
    # Run adaptive analysis
    print("\n🎯 Running adaptive analysis...")
    adaptive_analyzer = AdaptiveRegionDetector()
    adaptive_result = adaptive_analyzer.analyze_fiber_porosity_adaptive(image, analysis_data, scale_factor)
    
    # Create comparison
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_path = results_dir / f"adaptive_comparison_{timestamp}.png"
    create_adaptive_visualization(image, enhanced_result, adaptive_result, scale_factor, str(viz_path))
    
    # Print detailed comparison
    enhanced_porosity = enhanced_result.get('porosity_metrics', {}).get('total_porosity_percent', 0)
    adaptive_porosity = adaptive_result.get('porosity_metrics', {}).get('total_porosity_percent', 0)
    enhanced_count = enhanced_result.get('porosity_metrics', {}).get('pore_count', 0)
    adaptive_count = adaptive_result.get('porosity_metrics', {}).get('pore_count', 0)
    
    surface_analysis = adaptive_result.get('surface_analysis', {})
    face_pores = surface_analysis.get('face_pores', 0)
    side_pores = surface_analysis.get('side_pores', 0)
    
    print(f"\n🎯 ADAPTIVE vs ENHANCED COMPARISON:")
    print(f"Enhanced: {enhanced_porosity:.2f}% porosity, {enhanced_count} pores")
    print(f"Adaptive: {adaptive_porosity:.2f}% porosity, {adaptive_count} pores")
    print(f"  ├─ Face surface: {face_pores} pores")
    print(f"  └─ Side surface: {side_pores} pores")
    print(f"Improvement: +{adaptive_count - enhanced_count} pores detected")
    print(f"Porosity change: {adaptive_porosity - enhanced_porosity:+.2f}% points")
    
    return adaptive_result

if __name__ == "__main__":
    result = test_adaptive_detection()