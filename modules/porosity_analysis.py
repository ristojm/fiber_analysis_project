"""
Porosity Analysis Module for SEM Fiber Analysis System

This module provides comprehensive porosity analysis for fiber samples including:
- Pore detection and segmentation
- Pore size distribution analysis
- Porosity quantification
- Statistical analysis and visualization

Author: Fiber Analysis Project
Date: 2025
"""

import numpy as np
import cv2
from skimage import (
    segmentation, filters, morphology, measure, 
    feature, restoration, exposure
)
from skimage.morphology import disk, remove_small_objects, binary_closing
from scipy import ndimage, spatial
from sklearn.cluster import DBSCAN
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class PorosityAnalyzer:
    """
    Comprehensive porosity analysis for SEM fiber images.
    
    This class handles pore detection, measurement, and statistical analysis
    for both hollow fibers and solid filaments.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the PorosityAnalyzer with configuration parameters.
        
        Args:
            config: Dictionary containing analysis parameters
        """
        self.default_config = {
            'pore_detection': {
                'min_pore_area': 5,  # minimum pore area in pixels
                'max_pore_area': 10000,  # maximum pore area in pixels
                'contrast_threshold': 0.3,  # contrast threshold for pore detection
                'gaussian_sigma': 1.0,  # Gaussian blur sigma
                'morphology_disk_size': 2,  # morphological operations disk size
                'remove_border_pores': True,  # remove pores touching image border
            },
            'segmentation': {
                'method': 'multi_otsu',  # 'otsu', 'multi_otsu', 'adaptive', 'watershed'
                'watershed_markers': 'distance',  # 'distance', 'h_maxima'
                'adaptive_block_size': 51,  # block size for adaptive thresholding
                'adaptive_c': 10,  # constant for adaptive thresholding
            },
            'analysis': {
                'circularity_weight': 0.3,  # weight for circularity in pore scoring
                'aspect_ratio_weight': 0.2,  # weight for aspect ratio in pore scoring
                'size_bins': 50,  # number of bins for size distribution
                'percentiles': [25, 50, 75, 90, 95],  # percentiles to calculate
            },
            'filtering': {
                'denoise_method': 'bilateral',  # 'bilateral', 'gaussian', 'median'
                'enhance_contrast': True,  # apply contrast enhancement
                'clahe_clip_limit': 2.0,  # CLAHE clip limit
                'clahe_tile_size': (8, 8),  # CLAHE tile grid size
            }
        }
        
        self.config = self.default_config.copy()
        if config:
            self._update_config(config)
        
        self.results = {}
        self.pore_data = None
        self.fiber_mask = None
        self.scale_factor = None
        
    def _update_config(self, new_config: Dict):
        """Recursively update configuration dictionary."""
        for key, value in new_config.items():
            if key in self.config and isinstance(self.config[key], dict):
                self.config[key].update(value)
            else:
                self.config[key] = value
    
    def analyze_porosity(self, 
                        image: np.ndarray, 
                        fiber_mask: np.ndarray,
                        scale_factor: float = 1.0,
                        fiber_type: str = 'hollow_fiber') -> Dict[str, Any]:
        """
        Main porosity analysis function.
        
        Args:
            image: Input SEM image (grayscale)
            fiber_mask: Binary mask of fiber regions
            scale_factor: Micrometers per pixel conversion factor
            fiber_type: Type of fiber ('hollow_fiber' or 'filament')
            
        Returns:
            Dictionary containing comprehensive porosity analysis results
        """
        self.fiber_mask = fiber_mask.astype(bool)
        self.scale_factor = scale_factor
        
        print("Starting porosity analysis...")
        
        # Step 1: Preprocess image
        processed_image = self._preprocess_image(image)
        
        # Step 2: Detect pores within fiber regions
        pore_mask, pore_labels = self._detect_pores(processed_image, fiber_mask)
        
        # Step 3: Extract pore properties
        pore_properties = self._extract_pore_properties(pore_labels, processed_image)
        
        # Step 4: Filter and validate pores
        valid_pores = self._filter_pores(pore_properties)
        
        # Step 5: Calculate porosity metrics
        porosity_metrics = self._calculate_porosity_metrics(valid_pores, fiber_mask)
        
        # Step 6: Analyze size distribution
        size_distribution = self._analyze_size_distribution(valid_pores)
        
        # Step 7: Spatial analysis
        spatial_analysis = self._analyze_spatial_distribution(valid_pores, fiber_mask)
        
        # Compile results
        self.results = {
            'porosity_metrics': porosity_metrics,
            'size_distribution': size_distribution,
            'spatial_analysis': spatial_analysis,
            'pore_properties': valid_pores,
            'pore_mask': pore_mask,
            'pore_labels': pore_labels,
            'fiber_type': fiber_type,
            'scale_factor': scale_factor,
            'config_used': self.config.copy()
        }
        
        self.pore_data = valid_pores
        
        print(f"Porosity analysis complete. Found {len(valid_pores)} valid pores.")
        return self.results
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for optimal pore detection.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Preprocessed image
        """
        processed = image.copy()
        
        # Denoise
        if self.config['filtering']['denoise_method'] == 'bilateral':
            processed = cv2.bilateralFilter(processed, 9, 75, 75)
        elif self.config['filtering']['denoise_method'] == 'gaussian':
            processed = filters.gaussian(processed, sigma=self.config['pore_detection']['gaussian_sigma'])
        elif self.config['filtering']['denoise_method'] == 'median':
            processed = filters.median(processed, disk(2))
        
        # Enhance contrast
        if self.config['filtering']['enhance_contrast']:
            clahe = cv2.createCLAHE(
                clipLimit=self.config['filtering']['clahe_clip_limit'],
                tileGridSize=self.config['filtering']['clahe_tile_size']
            )
            processed = clahe.apply(processed.astype(np.uint8))
        
        return processed
    
    def _detect_pores(self, image: np.ndarray, fiber_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect pores within fiber regions using multiple segmentation approaches.
        
        Args:
            image: Preprocessed image
            fiber_mask: Binary mask of fiber regions
            
        Returns:
            Tuple of (pore_mask, pore_labels)
        """
        # Restrict analysis to fiber regions only
        fiber_region = image * fiber_mask
        
        method = self.config['segmentation']['method']
        
        if method == 'multi_otsu':
            pore_mask = self._multi_otsu_segmentation(fiber_region, fiber_mask)
        elif method == 'otsu':
            pore_mask = self._otsu_segmentation(fiber_region, fiber_mask)
        elif method == 'adaptive':
            pore_mask = self._adaptive_segmentation(fiber_region, fiber_mask)
        elif method == 'watershed':
            pore_mask = self._watershed_segmentation(fiber_region, fiber_mask)
        else:
            raise ValueError(f"Unknown segmentation method: {method}")
        
        # Post-process pore mask
        pore_mask = self._postprocess_pore_mask(pore_mask, fiber_mask)
        
        # Label connected components
        pore_labels = measure.label(pore_mask)
        
        return pore_mask, pore_labels
    
    def _multi_otsu_segmentation(self, image: np.ndarray, fiber_mask: np.ndarray) -> np.ndarray:
        """Multi-Otsu thresholding for pore detection."""
        # Apply multi-Otsu to get multiple thresholds
        fiber_pixels = image[fiber_mask > 0]
        if len(fiber_pixels) == 0:
            return np.zeros_like(image, dtype=bool)
        
        # Use 3 classes: background, fiber material, pores
        try:
            thresholds = filters.threshold_multiotsu(fiber_pixels, classes=3)
            # Pores are typically the darkest regions
            pore_threshold = thresholds[0]
            pore_mask = (image < pore_threshold) & fiber_mask
        except:
            # Fallback to regular Otsu
            threshold = filters.threshold_otsu(fiber_pixels)
            pore_mask = (image < threshold * 0.7) & fiber_mask  # More aggressive threshold
        
        return pore_mask
    
    def _otsu_segmentation(self, image: np.ndarray, fiber_mask: np.ndarray) -> np.ndarray:
        """Standard Otsu thresholding for pore detection."""
        fiber_pixels = image[fiber_mask > 0]
        if len(fiber_pixels) == 0:
            return np.zeros_like(image, dtype=bool)
        
        threshold = filters.threshold_otsu(fiber_pixels)
        # Use a more aggressive threshold for pore detection
        pore_threshold = threshold * (1 - self.config['pore_detection']['contrast_threshold'])
        pore_mask = (image < pore_threshold) & fiber_mask
        
        return pore_mask
    
    def _adaptive_segmentation(self, image: np.ndarray, fiber_mask: np.ndarray) -> np.ndarray:
        """Adaptive thresholding for pore detection."""
        block_size = self.config['segmentation']['adaptive_block_size']
        C = self.config['segmentation']['adaptive_c']
        
        # Ensure block_size is odd
        if block_size % 2 == 0:
            block_size += 1
        
        adaptive_thresh = cv2.adaptiveThreshold(
            image.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, C
        )
        
        pore_mask = (adaptive_thresh > 0) & fiber_mask
        return pore_mask
    
    def _watershed_segmentation(self, image: np.ndarray, fiber_mask: np.ndarray) -> np.ndarray:
        """Watershed segmentation for pore detection."""
        # Start with basic thresholding
        threshold = filters.threshold_otsu(image[fiber_mask > 0]) if np.any(fiber_mask) else 0
        binary = (image < threshold * 0.8) & fiber_mask
        
        if not np.any(binary):
            return binary
        
        # Distance transform for markers
        if self.config['segmentation']['watershed_markers'] == 'distance':
            distance = ndimage.distance_transform_edt(binary)
            local_maxima = feature.peak_local_maxima(distance, min_distance=5, threshold_abs=0.3*distance.max())
            markers = np.zeros_like(binary, dtype=int)
            if len(local_maxima[0]) > 0:
                markers[local_maxima] = np.arange(1, len(local_maxima[0]) + 1)
        else:
            # H-maxima markers
            distance = ndimage.distance_transform_edt(binary)
            markers = morphology.h_maxima(distance, h=distance.max()*0.3)
            markers = measure.label(markers)
        
        # Watershed
        if np.max(markers) > 0:
            labels = segmentation.watershed(-distance, markers, mask=binary)
            pore_mask = labels > 0
        else:
            pore_mask = binary
        
        return pore_mask
    
    def _postprocess_pore_mask(self, pore_mask: np.ndarray, fiber_mask: np.ndarray) -> np.ndarray:
        """Post-process pore mask to remove artifacts and noise."""
        # Remove small objects
        min_area = self.config['pore_detection']['min_pore_area']
        pore_mask = remove_small_objects(pore_mask, min_size=min_area)
        
        # Morphological closing to fill small gaps
        disk_size = self.config['pore_detection']['morphology_disk_size']
        pore_mask = binary_closing(pore_mask, disk(disk_size))
        
        # Remove pores touching border if specified
        if self.config['pore_detection']['remove_border_pores']:
            pore_mask = segmentation.clear_border(pore_mask)
        
        # Ensure pores are within fiber regions
        pore_mask = pore_mask & fiber_mask
        
        return pore_mask
    
    def _extract_pore_properties(self, pore_labels: np.ndarray, image: np.ndarray) -> List[Dict]:
        """
        Extract detailed properties for each detected pore.
        
        Args:
            pore_labels: Labeled pore regions
            image: Original image for intensity measurements
            
        Returns:
            List of dictionaries containing pore properties
        """
        properties = measure.regionprops(pore_labels, intensity_image=image)
        pore_data = []
        
        for prop in properties:
            # Basic geometric properties
            area_pixels = prop.area
            area_um2 = area_pixels * (self.scale_factor ** 2)
            
            # Equivalent diameter (diameter of circle with same area)
            equiv_diameter_pixels = prop.equivalent_diameter
            equiv_diameter_um = equiv_diameter_pixels * self.scale_factor
            
            # Major and minor axis lengths
            major_axis_um = prop.major_axis_length * self.scale_factor
            minor_axis_um = prop.minor_axis_length * self.scale_factor
            
            # Shape descriptors
            aspect_ratio = major_axis_um / minor_axis_um if minor_axis_um > 0 else 1
            circularity = 4 * np.pi * area_pixels / (prop.perimeter ** 2) if prop.perimeter > 0 else 0
            solidity = prop.solidity
            extent = prop.extent
            
            # Position
            centroid_y, centroid_x = prop.centroid
            
            # Intensity properties
            mean_intensity = prop.mean_intensity
            max_intensity = prop.max_intensity
            min_intensity = prop.min_intensity
            
            # Bounding box
            bbox = prop.bbox  # (min_row, min_col, max_row, max_col)
            
            pore_info = {
                'label': prop.label,
                'area_pixels': area_pixels,
                'area_um2': area_um2,
                'equivalent_diameter_pixels': equiv_diameter_pixels,
                'equivalent_diameter_um': equiv_diameter_um,
                'major_axis_um': major_axis_um,
                'minor_axis_um': minor_axis_um,
                'aspect_ratio': aspect_ratio,
                'circularity': circularity,
                'solidity': solidity,
                'extent': extent,
                'centroid_x': centroid_x,
                'centroid_y': centroid_y,
                'mean_intensity': mean_intensity,
                'max_intensity': max_intensity,
                'min_intensity': min_intensity,
                'perimeter_pixels': prop.perimeter,
                'perimeter_um': prop.perimeter * self.scale_factor,
                'bbox': bbox,
                'orientation': prop.orientation,
            }
            
            pore_data.append(pore_info)
        
        return pore_data
    
    def _filter_pores(self, pore_properties: List[Dict]) -> List[Dict]:
        """
        Filter pores based on size and shape criteria to remove artifacts.
        
        Args:
            pore_properties: List of pore property dictionaries
            
        Returns:
            Filtered list of valid pores
        """
        valid_pores = []
        
        min_area = self.config['pore_detection']['min_pore_area'] * (self.scale_factor ** 2)
        max_area = self.config['pore_detection']['max_pore_area'] * (self.scale_factor ** 2)
        
        for pore in pore_properties:
            # Size filtering
            if not (min_area <= pore['area_um2'] <= max_area):
                continue
            
            # Shape filtering - remove very elongated or irregular shapes
            if pore['aspect_ratio'] > 5:  # Very elongated
                continue
            
            if pore['circularity'] < 0.1:  # Very irregular
                continue
            
            if pore['solidity'] < 0.5:  # Very non-convex
                continue
            
            valid_pores.append(pore)
        
        return valid_pores
    
    def _calculate_porosity_metrics(self, pore_data: List[Dict], fiber_mask: np.ndarray) -> Dict:
        """Calculate overall porosity metrics."""
        if not pore_data:
            return {
                'total_porosity_percent': 0.0,
                'pore_count': 0,
                'total_pore_area_um2': 0.0,
                'fiber_area_um2': 0.0,
                'average_pore_size_um2': 0.0,
                'pore_density_per_mm2': 0.0
            }
        
        # Calculate total areas
        total_pore_area_pixels = sum(pore['area_pixels'] for pore in pore_data)
        total_pore_area_um2 = sum(pore['area_um2'] for pore in pore_data)
        
        fiber_area_pixels = np.sum(fiber_mask)
        fiber_area_um2 = fiber_area_pixels * (self.scale_factor ** 2)
        
        # Calculate porosity percentage
        porosity_percent = (total_pore_area_pixels / fiber_area_pixels) * 100 if fiber_area_pixels > 0 else 0
        
        # Calculate other metrics
        pore_count = len(pore_data)
        avg_pore_size = total_pore_area_um2 / pore_count if pore_count > 0 else 0
        
        # Pore density (pores per mm²)
        fiber_area_mm2 = fiber_area_um2 / 1e6  # Convert µm² to mm²
        pore_density = pore_count / fiber_area_mm2 if fiber_area_mm2 > 0 else 0
        
        return {
            'total_porosity_percent': porosity_percent,
            'pore_count': pore_count,
            'total_pore_area_um2': total_pore_area_um2,
            'fiber_area_um2': fiber_area_um2,
            'average_pore_size_um2': avg_pore_size,
            'pore_density_per_mm2': pore_density
        }
    
    def _analyze_size_distribution(self, pore_data: List[Dict]) -> Dict:
        """Analyze pore size distribution."""
        if not pore_data:
            return {'sizes_um2': [], 'diameters_um': [], 'statistics': {}}
        
        sizes = [pore['area_um2'] for pore in pore_data]
        diameters = [pore['equivalent_diameter_um'] for pore in pore_data]
        
        # Calculate statistics
        percentiles = self.config['analysis']['percentiles']
        size_stats = {
            'mean_size_um2': np.mean(sizes),
            'median_size_um2': np.median(sizes),
            'std_size_um2': np.std(sizes),
            'min_size_um2': np.min(sizes),
            'max_size_um2': np.max(sizes),
            'mean_diameter_um': np.mean(diameters),
            'median_diameter_um': np.median(diameters),
            'std_diameter_um': np.std(diameters),
        }
        
        # Add percentiles
        for p in percentiles:
            size_stats[f'p{p}_size_um2'] = np.percentile(sizes, p)
            size_stats[f'p{p}_diameter_um'] = np.percentile(diameters, p)
        
        return {
            'sizes_um2': sizes,
            'diameters_um': diameters,
            'statistics': size_stats
        }
    
    def _analyze_spatial_distribution(self, pore_data: List[Dict], fiber_mask: np.ndarray) -> Dict:
        """Analyze spatial distribution of pores."""
        if len(pore_data) < 2:
            return {'spatial_uniformity': 1.0, 'clustering_coefficient': 0.0}
        
        # Extract centroid coordinates
        centroids = np.array([[pore['centroid_x'], pore['centroid_y']] for pore in pore_data])
        
        # Calculate nearest neighbor distances
        distances = spatial.distance_matrix(centroids, centroids)
        np.fill_diagonal(distances, np.inf)  # Ignore self-distances
        nearest_distances = np.min(distances, axis=1)
        
        # Spatial uniformity metric (coefficient of variation of nearest neighbor distances)
        spatial_uniformity = 1 / (1 + np.std(nearest_distances) / np.mean(nearest_distances))
        
        # Clustering analysis using DBSCAN
        try:
            # Use median nearest neighbor distance as eps
            eps = np.median(nearest_distances) * 1.5
            clustering = DBSCAN(eps=eps, min_samples=2).fit(centroids)
            n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
            clustering_coefficient = n_clusters / len(pore_data)
        except:
            clustering_coefficient = 0.0
        
        return {
            'spatial_uniformity': spatial_uniformity,
            'clustering_coefficient': clustering_coefficient,
            'nearest_neighbor_distances_um': nearest_distances * self.scale_factor,
            'mean_nn_distance_um': np.mean(nearest_distances) * self.scale_factor
        }
    
    def get_pore_dataframe(self) -> pd.DataFrame:
        """Convert pore data to pandas DataFrame for easy analysis."""
        if self.pore_data is None:
            return pd.DataFrame()
        
        return pd.DataFrame(self.pore_data)
    
    def export_results(self, output_path: str):
        """Export analysis results to Excel file."""
        if self.results is None:
            print("No results to export. Run analyze_porosity() first.")
            return
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Summary metrics
            summary_df = pd.DataFrame([self.results['porosity_metrics']])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Individual pore data
            if self.pore_data:
                pore_df = self.get_pore_dataframe()
                pore_df.to_excel(writer, sheet_name='Pore_Data', index=False)
            
            # Size distribution statistics
            if self.results['size_distribution']['statistics']:
                size_stats_df = pd.DataFrame([self.results['size_distribution']['statistics']])
                size_stats_df.to_excel(writer, sheet_name='Size_Statistics', index=False)
            
            # Spatial analysis
            spatial_df = pd.DataFrame([self.results['spatial_analysis']])
            spatial_df.to_excel(writer, sheet_name='Spatial_Analysis', index=False)
        
        print(f"Results exported to {output_path}")


def analyze_fiber_porosity(image: np.ndarray, 
                          fiber_mask: np.ndarray,
                          scale_factor: float = 1.0,
                          config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Convenience function for porosity analysis.
    
    Args:
        image: Input SEM image (grayscale)
        fiber_mask: Binary mask of fiber regions
        scale_factor: Micrometers per pixel conversion factor
        config: Optional configuration dictionary
        
    Returns:
        Dictionary containing porosity analysis results
    """
    analyzer = PorosityAnalyzer(config)
    return analyzer.analyze_porosity(image, fiber_mask, scale_factor)


def quick_porosity_check(image: np.ndarray, 
                        fiber_mask: np.ndarray,
                        scale_factor: float = 1.0) -> float:
    """
    Quick porosity percentage calculation.
    
    Args:
        image: Input SEM image
        fiber_mask: Binary mask of fiber regions
        scale_factor: Micrometers per pixel conversion factor
        
    Returns:
        Porosity percentage
    """
    config = {
        'segmentation': {'method': 'otsu'},
        'pore_detection': {'min_pore_area': 3}
    }
    
    results = analyze_fiber_porosity(image, fiber_mask, scale_factor, config)
    return results['porosity_metrics']['total_porosity_percent']