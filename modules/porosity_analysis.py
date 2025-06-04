"""
SEM Fiber Analysis System - Porosity Analysis Module
Phase 2A: Comprehensive pore detection, measurement, and analysis
"""

import cv2
import numpy as np
from skimage import filters, morphology, measure, feature, segmentation
from scipy import ndimage, spatial
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt
import pandas as pd

class PorosityAnalyzer:
    """
    Comprehensive porosity analysis for fiber cross-sections.
    Detects, measures, and analyzes pores within fiber walls.
    """
    
    def __init__(self,
                 min_pore_area: int = 10,              # Minimum pore area in pixels
                 max_pore_area_ratio: float = 0.05,    # Max pore as fraction of fiber area
                 pore_circularity_threshold: float = 0.1,  # Very lenient for irregular pores
                 edge_exclusion_width: int = 5,        # Exclude pores near fiber edge
                 adaptive_threshold: bool = True):      # Use adaptive thresholding
        """
        Initialize porosity analyzer.
        
        Args:
            min_pore_area: Minimum pore area in pixels
            max_pore_area_ratio: Maximum pore area as fraction of fiber area
            pore_circularity_threshold: Minimum circularity for valid pores
            edge_exclusion_width: Width to exclude near fiber edges (pixels)
            adaptive_threshold: Whether to use adaptive thresholding
        """
        self.min_pore_area = min_pore_area
        self.max_pore_area_ratio = max_pore_area_ratio
        self.pore_circularity_threshold = pore_circularity_threshold
        self.edge_exclusion_width = edge_exclusion_width
        self.adaptive_threshold = adaptive_threshold
    
    def calculate_adaptive_pore_thresholds(self, fiber_area: float, scale_factor: float = 1.0) -> Dict:
        """
        Calculate adaptive thresholds for pore detection based on fiber characteristics.
        
        Args:
            fiber_area: Area of the fiber in pixels
            scale_factor: Micrometers per pixel conversion factor
            
        Returns:
            Dictionary of adaptive thresholds
        """
        # Scale-aware minimum pore area
        min_pore_area = max(self.min_pore_area, int(fiber_area * 0.0001))  # 0.01% of fiber
        
        # Maximum pore area
        max_pore_area = int(fiber_area * self.max_pore_area_ratio)
        
        # Adaptive morphological kernel size
        fiber_radius = np.sqrt(fiber_area / np.pi)
        kernel_size = max(3, min(15, int(fiber_radius / 50)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Edge exclusion width (scale-aware)
        edge_exclusion = max(self.edge_exclusion_width, int(fiber_radius / 30))
        
        thresholds = {
            'min_pore_area': min_pore_area,
            'max_pore_area': max_pore_area,
            'kernel_size': kernel_size,
            'edge_exclusion_width': edge_exclusion,
            'fiber_area': fiber_area,
            'fiber_radius': fiber_radius,
            'scale_factor': scale_factor
        }
        
        return thresholds
    
    def segment_pores(self, image: np.ndarray, fiber_mask: np.ndarray, 
                     exclude_lumen: bool = True, lumen_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Segment pores within the fiber wall region.
        
        Args:
            image: Original grayscale image
            fiber_mask: Binary mask of the fiber region
            exclude_lumen: Whether to exclude the central lumen from analysis
            lumen_mask: Optional binary mask of the lumen region
            
        Returns:
            Binary mask of detected pores
        """
        # Create analysis region (fiber minus lumen if specified)
        analysis_mask = fiber_mask.copy()
        
        if exclude_lumen and lumen_mask is not None:
            analysis_mask = cv2.bitwise_and(analysis_mask, cv2.bitwise_not(lumen_mask))
        
        # Extract the fiber wall region
        fiber_region = cv2.bitwise_and(image, image, mask=analysis_mask)
        
        # Get fiber pixels for statistics
        fiber_pixels = fiber_region[analysis_mask > 0]
        
        if len(fiber_pixels) == 0:
            return np.zeros_like(image, dtype=np.uint8)
        
        # Multiple thresholding approaches for robust pore detection
        pore_masks = []
        
        # Method 1: Percentile-based thresholding
        for percentile in [10, 15, 20]:
            threshold = np.percentile(fiber_pixels, percentile)
            _, pore_binary = cv2.threshold(fiber_region, threshold, 255, cv2.THRESH_BINARY_INV)
            pore_binary = cv2.bitwise_and(pore_binary, analysis_mask)
            pore_masks.append(pore_binary)
        
        # Method 2: Adaptive thresholding within fiber
        if self.adaptive_threshold:
            # Create a working region for adaptive threshold
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            eroded_mask = cv2.erode(analysis_mask, kernel, iterations=2)
            
            if np.sum(eroded_mask) > 100:  # Ensure enough pixels for adaptive threshold
                try:
                    adaptive_binary = cv2.adaptiveThreshold(
                        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                        cv2.THRESH_BINARY_INV, 15, 5
                    )
                    adaptive_binary = cv2.bitwise_and(adaptive_binary, eroded_mask)
                    pore_masks.append(adaptive_binary)
                except:
                    pass  # Skip if adaptive threshold fails
        
        # Method 3: Otsu thresholding on fiber region
        try:
            # Create a sub-image of just the fiber region for better Otsu performance
            fiber_bbox = cv2.boundingRect(np.where(analysis_mask > 0))
            if fiber_bbox[2] > 10 and fiber_bbox[3] > 10:  # Ensure reasonable size
                x, y, w, h = fiber_bbox
                fiber_sub = fiber_region[y:y+h, x:x+w]
                mask_sub = analysis_mask[y:y+h, x:x+w]
                
                # Apply Otsu to sub-region
                fiber_sub_pixels = fiber_sub[mask_sub > 0]
                if len(fiber_sub_pixels) > 100:
                    otsu_threshold = filters.threshold_otsu(fiber_sub_pixels)
                    _, otsu_binary = cv2.threshold(fiber_region, otsu_threshold, 255, cv2.THRESH_BINARY_INV)
                    otsu_binary = cv2.bitwise_and(otsu_binary, analysis_mask)
                    pore_masks.append(otsu_binary)
        except:
            pass  # Skip if Otsu fails
        
        # Combine results using intersection and union strategies
        if len(pore_masks) == 0:
            return np.zeros_like(image, dtype=np.uint8)
        
        # Conservative approach: intersection of multiple methods
        consensus_mask = pore_masks[0].copy()
        for mask in pore_masks[1:]:
            consensus_mask = cv2.bitwise_and(consensus_mask, mask)
        
        # Liberal approach: union of multiple methods
        union_mask = pore_masks[0].copy()
        for mask in pore_masks[1:]:
            union_mask = cv2.bitwise_or(union_mask, mask)
        
        # Hybrid approach: use consensus for large pores, union for small ones
        # This helps capture both obvious pores and subtle ones
        final_mask = consensus_mask.copy()
        
        # Add small pores from union that aren't noise
        union_only = cv2.bitwise_and(union_mask, cv2.bitwise_not(consensus_mask))
        
        # Filter small additions by size and circularity
        contours, _ = cv2.findContours(union_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.min_pore_area and area <= 100:  # Small pores only
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                if circularity >= self.pore_circularity_threshold:
                    cv2.fillPoly(final_mask, [contour], 255)
        
        return final_mask
    
    def filter_pores_by_criteria(self, pore_mask: np.ndarray, fiber_contour: np.ndarray, 
                                thresholds: Dict) -> Tuple[np.ndarray, List[Dict]]:
        """
        Filter detected pores by size, shape, and position criteria.
        
        Args:
            pore_mask: Binary mask of detected pores
            fiber_contour: Contour of the fiber
            thresholds: Adaptive thresholds dictionary
            
        Returns:
            Tuple of (filtered_pore_mask, pore_properties_list)
        """
        # Create edge exclusion mask
        fiber_mask = np.zeros_like(pore_mask)
        cv2.fillPoly(fiber_mask, [fiber_contour], 255)
        
        # Erode fiber mask to exclude edge regions
        edge_kernel_size = thresholds['edge_exclusion_width']
        edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_kernel_size, edge_kernel_size))
        inner_fiber_mask = cv2.erode(fiber_mask, edge_kernel, iterations=1)
        
        # Find pore contours
        contours, _ = cv2.findContours(pore_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_pores = []
        filtered_mask = np.zeros_like(pore_mask)
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Size filtering
            if area < thresholds['min_pore_area'] or area > thresholds['max_pore_area']:
                continue
            
            # Calculate pore properties
            pore_props = self._calculate_pore_properties(contour, area, thresholds)
            
            # Shape filtering
            if pore_props['circularity'] < self.pore_circularity_threshold:
                continue
            
            # Position filtering (exclude pores too close to fiber edge)
            pore_center = pore_props['centroid']
            if inner_fiber_mask[int(pore_center[1]), int(pore_center[0])] == 0:
                continue  # Pore is too close to edge
            
            # Valid pore - add to results
            pore_props['pore_id'] = i
            pore_props['contour'] = contour
            valid_pores.append(pore_props)
            
            # Add to filtered mask
            cv2.fillPoly(filtered_mask, [contour], 255)
        
        return filtered_mask, valid_pores
    
    def _calculate_pore_properties(self, contour: np.ndarray, area: float, thresholds: Dict) -> Dict:
        """
        Calculate comprehensive properties for a pore.
        
        Args:
            contour: Pore contour
            area: Pore area in pixels
            thresholds: Adaptive thresholds dictionary
            
        Returns:
            Dictionary of pore properties
        """
        # Basic geometric properties
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        
        # Shape descriptors
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        extent = area / (w * h) if w * h > 0 else 0
        solidity = area / cv2.contourArea(cv2.convexHull(contour)) if len(contour) >= 3 else 1.0
        
        # Size metrics
        equivalent_diameter = np.sqrt(4 * area / np.pi)  # Diameter of circle with same area
        
        # Convert to real units if scale factor available
        scale_factor = thresholds.get('scale_factor', 1.0)
        area_um2 = area * (scale_factor ** 2)
        equivalent_diameter_um = equivalent_diameter * scale_factor
        
        return {
            'area_pixels': area,
            'area_um2': area_um2,
            'perimeter_pixels': perimeter,
            'perimeter_um': perimeter * scale_factor,
            'centroid': (cx, cy),
            'radius_pixels': radius,
            'radius_um': radius * scale_factor,
            'equivalent_diameter_pixels': equivalent_diameter,
            'equivalent_diameter_um': equivalent_diameter_um,
            'bounding_rect': (x, y, w, h),
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'extent': extent,
            'solidity': solidity
        }
    
    def calculate_porosity_metrics(self, pore_properties: List[Dict], fiber_area: float, 
                                  thresholds: Dict) -> Dict:
        """
        Calculate comprehensive porosity metrics.
        
        Args:
            pore_properties: List of pore property dictionaries
            fiber_area: Total fiber area in pixels
            thresholds: Adaptive thresholds dictionary
            
        Returns:
            Dictionary of porosity metrics
        """
        if not pore_properties:
            return {
                'total_porosity': 0.0,
                'pore_count': 0,
                'pore_density': 0.0,
                'mean_pore_size': 0.0,
                'pore_size_std': 0.0,
                'size_distribution': {},
                'error': 'No valid pores detected'
            }
        
        # Extract areas for analysis
        pore_areas_pixels = [pore['area_pixels'] for pore in pore_properties]
        pore_areas_um2 = [pore['area_um2'] for pore in pore_properties]
        pore_diameters_um = [pore['equivalent_diameter_um'] for pore in pore_properties]
        
        # Basic metrics
        total_pore_area_pixels = sum(pore_areas_pixels)
        total_pore_area_um2 = sum(pore_areas_um2)
        
        # Porosity calculations
        total_porosity = total_pore_area_pixels / fiber_area if fiber_area > 0 else 0
        pore_count = len(pore_properties)
        
        # Scale-aware calculations
        scale_factor = thresholds.get('scale_factor', 1.0)
        fiber_area_um2 = fiber_area * (scale_factor ** 2)
        pore_density = pore_count / fiber_area_um2 if fiber_area_um2 > 0 else 0  # pores per μm²
        
        # Size statistics
        mean_pore_size_um2 = np.mean(pore_areas_um2)
        median_pore_size_um2 = np.median(pore_areas_um2)
        pore_size_std_um2 = np.std(pore_areas_um2)
        
        mean_pore_diameter_um = np.mean(pore_diameters_um)
        median_pore_diameter_um = np.median(pore_diameters_um)
        pore_diameter_std_um = np.std(pore_diameters_um)
        
        # Size distribution analysis
        size_distribution = self._analyze_size_distribution(pore_diameters_um)
        
        # Shape analysis
        circularities = [pore['circularity'] for pore in pore_properties]
        mean_circularity = np.mean(circularities)
        
        aspect_ratios = [pore['aspect_ratio'] for pore in pore_properties]
        mean_aspect_ratio = np.mean(aspect_ratios)
        
        return {
            # Basic porosity metrics
            'total_porosity': total_porosity,
            'total_porosity_percent': total_porosity * 100,
            'pore_count': pore_count,
            'pore_density_per_um2': pore_density,
            
            # Area statistics
            'total_pore_area_pixels': total_pore_area_pixels,
            'total_pore_area_um2': total_pore_area_um2,
            'mean_pore_area_um2': mean_pore_size_um2,
            'median_pore_area_um2': median_pore_size_um2,
            'pore_area_std_um2': pore_size_std_um2,
            
            # Diameter statistics
            'mean_pore_diameter_um': mean_pore_diameter_um,
            'median_pore_diameter_um': median_pore_diameter_um,
            'pore_diameter_std_um': pore_diameter_std_um,
            'min_pore_diameter_um': min(pore_diameters_um),
            'max_pore_diameter_um': max(pore_diameters_um),
            
            # Shape statistics
            'mean_circularity': mean_circularity,
            'mean_aspect_ratio': mean_aspect_ratio,
            
            # Distribution analysis
            'size_distribution': size_distribution,
            
            # Metadata
            'fiber_area_pixels': fiber_area,
            'fiber_area_um2': fiber_area_um2,
            'scale_factor': scale_factor,
            'analysis_parameters': thresholds
        }
    
    def _analyze_size_distribution(self, pore_diameters_um: List[float]) -> Dict:
        """
        Analyze pore size distribution with binning and statistics.
        
        Args:
            pore_diameters_um: List of pore diameters in micrometers
            
        Returns:
            Dictionary with distribution analysis
        """
        if not pore_diameters_um:
            return {}
        
        diameters = np.array(pore_diameters_um)
        
        # Define size categories (adjust based on typical pore sizes)
        size_bins = [0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, np.inf]
        size_labels = ['< 0.1 μm', '0.1-0.2 μm', '0.2-0.5 μm', '0.5-1.0 μm', 
                      '1.0-2.0 μm', '2.0-5.0 μm', '> 5.0 μm']
        
        # Bin the data
        bin_counts, _ = np.histogram(diameters, bins=size_bins)
        
        # Calculate percentages
        total_pores = len(diameters)
        percentages = (bin_counts / total_pores * 100) if total_pores > 0 else np.zeros_like(bin_counts)
        
        # Create distribution dictionary
        distribution = {}
        for i, label in enumerate(size_labels):
            distribution[label] = {
                'count': int(bin_counts[i]),
                'percentage': float(percentages[i])
            }
        
        # Additional statistics
        distribution['statistics'] = {
            'total_pores': total_pores,
            'mean_diameter': float(np.mean(diameters)),
            'median_diameter': float(np.median(diameters)),
            'std_diameter': float(np.std(diameters)),
            'min_diameter': float(np.min(diameters)),
            'max_diameter': float(np.max(diameters)),
            'q25_diameter': float(np.percentile(diameters, 25)),
            'q75_diameter': float(np.percentile(diameters, 75))
        }
        
        return distribution
    
    def analyze_fiber_porosity(self, image: np.ndarray, fiber_contour: np.ndarray,
                              lumen_contour: Optional[np.ndarray] = None,
                              scale_factor: float = 1.0) -> Dict:
        """
        Complete porosity analysis for a single fiber.
        
        Args:
            image: Original grayscale image
            fiber_contour: Contour of the fiber
            lumen_contour: Optional contour of the lumen (for hollow fibers)
            scale_factor: Micrometers per pixel conversion factor
            
        Returns:
            Comprehensive porosity analysis results
        """
        # Calculate fiber area
        fiber_area = cv2.contourArea(fiber_contour)
        
        # Calculate adaptive thresholds
        thresholds = self.calculate_adaptive_pore_thresholds(fiber_area, scale_factor)
        
        # Create fiber mask
        fiber_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(fiber_mask, [fiber_contour], 255)
        
        # Create lumen mask if provided
        lumen_mask = None
        if lumen_contour is not None:
            lumen_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(lumen_mask, [lumen_contour], 255)
        
        # Segment pores
        pore_mask = self.segment_pores(image, fiber_mask, 
                                      exclude_lumen=(lumen_mask is not None), 
                                      lumen_mask=lumen_mask)
        
        # Filter pores by criteria
        filtered_pore_mask, pore_properties = self.filter_pores_by_criteria(
            pore_mask, fiber_contour, thresholds
        )
        
        # Calculate porosity metrics
        porosity_metrics = self.calculate_porosity_metrics(
            pore_properties, fiber_area, thresholds
        )
        
        # Compile complete analysis
        analysis_results = {
            'porosity_metrics': porosity_metrics,
            'pore_properties': pore_properties,
            'masks': {
                'fiber_mask': fiber_mask,
                'lumen_mask': lumen_mask,
                'pore_mask': pore_mask,
                'filtered_pore_mask': filtered_pore_mask
            },
            'thresholds': thresholds,
            'scale_factor': scale_factor,
            'analysis_complete': True
        }
        
        return analysis_results

def visualize_porosity_analysis(image: np.ndarray, analysis_results: Dict, 
                               figsize: Tuple[int, int] = (20, 12)):
    """
    Comprehensive visualization of porosity analysis results.
    
    Args:
        image: Original image
        analysis_results: Results from analyze_fiber_porosity
        figsize: Figure size for visualization
    """
    fig, axes = plt.subplots(3, 4, figsize=figsize)
    
    masks = analysis_results['masks']
    porosity_metrics = analysis_results['porosity_metrics']
    pore_properties = analysis_results['pore_properties']
    
    # Row 1: Original image and masks
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(masks['fiber_mask'], cmap='gray')
    axes[0, 1].set_title('Fiber Mask')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(masks['pore_mask'], cmap='gray')
    axes[0, 2].set_title('Detected Pores (Raw)')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(masks['filtered_pore_mask'], cmap='gray')
    axes[0, 3].set_title('Filtered Pores')
    axes[0, 3].axis('off')
    
    # Row 2: Analysis overlays
    # Pore overlay on original image
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for pore in pore_properties:
        contour = pore['contour']
        cv2.drawContours(overlay, [contour], -1, (255, 0, 0), 2)  # Red pores
        # Mark centroids
        cx, cy = pore['centroid']
        cv2.circle(overlay, (int(cx), int(cy)), 3, (0, 255, 255), -1)  # Yellow centers
    
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title(f'Pore Detection Overlay\\n({len(pore_properties)} pores)')
    axes[1, 0].axis('off')
    
    # Pore size visualization (color-coded by size)
    if pore_properties:
        size_overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        max_area = max(pore['area_um2'] for pore in pore_properties)
        min_area = min(pore['area_um2'] for pore in pore_properties)
        
        for pore in pore_properties:
            contour = pore['contour']
            # Color based on size (blue = small, red = large)
            if max_area > min_area:
                size_ratio = (pore['area_um2'] - min_area) / (max_area - min_area)
            else:
                size_ratio = 0.5
            color = (int(255 * (1 - size_ratio)), 0, int(255 * size_ratio))
            cv2.drawContours(size_overlay, [contour], -1, color, -1)
        
        axes[1, 1].imshow(size_overlay)
        axes[1, 1].set_title('Pore Size Distribution\\n(Blue=Small, Red=Large)')
        axes[1, 1].axis('off')
    else:
        axes[1, 1].text(0.5, 0.5, 'No pores detected', ha='center', va='center',
                       transform=axes[1, 1].transAxes, fontsize=16)
        axes[1, 1].set_title('No Pores Detected')
        axes[1, 1].axis('off')
    
    # Porosity metrics summary
    metrics_text = f"""POROSITY ANALYSIS RESULTS:

Total Porosity: {porosity_metrics['total_porosity_percent']:.2f}%
Pore Count: {porosity_metrics['pore_count']}
Pore Density: {porosity_metrics['pore_density_per_um2']:.3f} pores/μm²

SIZE STATISTICS:
Mean Diameter: {porosity_metrics['mean_pore_diameter_um']:.3f} μm
Median Diameter: {porosity_metrics['median_pore_diameter_um']:.3f} μm
Size Range: {porosity_metrics['min_pore_diameter_um']:.3f} - {porosity_metrics['max_pore_diameter_um']:.3f} μm

SHAPE STATISTICS:
Mean Circularity: {porosity_metrics['mean_circularity']:.3f}
Mean Aspect Ratio: {porosity_metrics['mean_aspect_ratio']:.3f}"""

    axes[1, 2].text(0.05, 0.95, metrics_text, transform=axes[1, 2].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[1, 2].set_title('Porosity Metrics')
    axes[1, 2].axis('off')
    
    # Size distribution pie chart
    if 'size_distribution' in porosity_metrics and pore_properties:
        dist = porosity_metrics['size_distribution']
        labels = []
        sizes = []
        for label, data in dist.items():
            if label != 'statistics' and data['count'] > 0:
                labels.append(f"{label}\\n({data['count']})")
                sizes.append(data['count'])
        
        if sizes:
            axes[1, 3].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            axes[1, 3].set_title('Pore Size Distribution')
        else:
            axes[1, 3].text(0.5, 0.5, 'No size distribution', ha='center', va='center',
                           transform=axes[1, 3].transAxes)
            axes[1, 3].set_title('Size Distribution')
    
    # Row 3: Detailed analysis plots
    if pore_properties:
        # Pore area histogram
        areas_um2 = [pore['area_um2'] for pore in pore_properties]
        axes[2, 0].hist(areas_um2, bins=min(20, len(areas_um2)), alpha=0.7, edgecolor='black')
        axes[2, 0].set_xlabel('Pore Area (μm²)')
        axes[2, 0].set_ylabel('Frequency')
        axes[2, 0].set_title('Pore Area Distribution')
        
        # Pore diameter histogram
        diameters_um = [pore['equivalent_diameter_um'] for pore in pore_properties]
        axes[2, 1].hist(diameters_um, bins=min(20, len(diameters_um)), alpha=0.7, edgecolor='black')
        axes[2, 1].set_xlabel('Equivalent Diameter (μm)')
        axes[2, 1].set_ylabel('Frequency')
        axes[2, 1].set_title('Pore Diameter Distribution')
        
        # Circularity vs Size scatter
        circularities = [pore['circularity'] for pore in pore_properties]
        axes[2, 2].scatter(diameters_um, circularities, alpha=0.6)
        axes[2, 2].set_xlabel('Diameter (μm)')
        axes[2, 2].set_ylabel('Circularity')
        axes[2, 2].set_title('Pore Shape vs Size')
        
        # Cumulative size distribution
        sorted_diameters = np.sort(diameters_um)
        cumulative = np.arange(1, len(sorted_diameters) + 1) / len(sorted_diameters) * 100
        axes[2, 3].plot(sorted_diameters, cumulative, 'b-', linewidth=2)
        axes[2, 3].set_xlabel('Diameter (μm)')
        axes[2, 3].set_ylabel('Cumulative Percentage')
        axes[2, 3].set_title('Cumulative Size Distribution')
        axes[2, 3].grid(True, alpha=0.3)
    else:
        for i in range(4):
            axes[2, i].text(0.5, 0.5, 'No pores for analysis', ha='center', va='center',
                           transform=axes[2, i].transAxes)
            axes[2, i].set_title('No Data')
    
    plt.tight_layout()
    plt.show()

# Convenience function for quick porosity analysis
def analyze_porosity(image: np.ndarray, fiber_contour: np.ndarray, 
                    lumen_contour: Optional[np.ndarray] = None,
                    scale_factor: float = 1.0, **kwargs) -> Dict:
    """
    Convenience function for quick porosity analysis.
    
    Args:
        image: Input grayscale image
        fiber_contour: Contour of the fiber
        lumen_contour: Optional contour of the lumen
        scale_factor: Micrometers per pixel conversion factor
        **kwargs: Additional parameters for PorosityAnalyzer
        
    Returns:
        Porosity analysis results
    """
    analyzer = PorosityAnalyzer(**kwargs)
    return analyzer.analyze_fiber_porosity(image, fiber_contour, lumen_contour, scale_factor)