"""
Enhanced Porosity Analysis Module for SEM Fiber Analysis System
Integrates with the existing system for comprehensive fiber characterization

This module enhances the existing porosity_analysis.py with:
1. Better integration with fiber type detection
2. Scale-aware measurements
3. Enhanced visualization capabilities
4. Improved SEM-specific algorithms

Update your existing modules/porosity_analysis.py with this enhanced version
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


class EnhancedPorosityAnalyzer:
    """
    Enhanced porosity analyzer integrated with the SEM Fiber Analysis System.
    
    This class extends the original PorosityAnalyzer with:
    - Better integration with fiber type detection
    - Scale-aware measurements
    - Enhanced SEM-specific algorithms
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the enhanced porosity analyzer.
        
        Args:
            config: Dictionary containing analysis parameters
        """
        self.default_config = {
            'pore_detection': {
                'min_pore_area': 5,  # minimum pore area in pixels
                'max_pore_area': 50000,  # maximum pore area in pixels
                'contrast_threshold': 0.3,  # contrast threshold for pore detection
                'gaussian_sigma': 1.0,  # Gaussian blur sigma
                'morphology_disk_size': 2,  # morphological operations disk size
                'remove_border_pores': True,  # remove pores touching image border
                'adaptive_thresholds': True,  # use adaptive thresholds based on fiber type
            },
            'segmentation': {
                'method': 'multi_otsu',  # 'otsu', 'multi_otsu', 'adaptive', 'watershed'
                'watershed_markers': 'distance',  # 'distance', 'h_maxima'
                'adaptive_block_size': 51,  # block size for adaptive thresholding
                'adaptive_c': 10,  # constant for adaptive thresholding
                'fiber_type_aware': True,  # adjust parameters based on fiber type
            },
            'analysis': {
                'circularity_weight': 0.3,  # weight for circularity in pore scoring
                'aspect_ratio_weight': 0.2,  # weight for aspect ratio in pore scoring
                'size_bins': 50,  # number of bins for size distribution
                'percentiles': [25, 50, 75, 90, 95],  # percentiles to calculate
                'spatial_analysis': True,  # perform spatial distribution analysis
            },
            'filtering': {
                'denoise_method': 'bilateral',  # 'bilateral', 'gaussian', 'median'
                'enhance_contrast': True,  # apply contrast enhancement
                'clahe_clip_limit': 2.0,  # CLAHE clip limit
                'clahe_tile_size': (8, 8),  # CLAHE tile grid size
            },
            'fiber_integration': {
                'use_individual_fibers': True,  # analyze each fiber separately
                'min_fiber_area_for_analysis': 1000,  # minimum fiber area in pixels
                'hollow_fiber_lumen_exclusion': True,  # exclude lumen from analysis
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
    
    def analyze_fiber_porosity(self, 
                              image: np.ndarray, 
                              fiber_mask: np.ndarray,
                              scale_factor: float = 1.0,
                              fiber_type: str = 'unknown',
                              fiber_analysis_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Enhanced porosity analysis integrated with fiber detection results.
        
        Args:
            image: Input SEM image (grayscale)
            fiber_mask: Binary mask of fiber regions
            scale_factor: Micrometers per pixel conversion factor
            fiber_type: Type of fiber ('hollow_fiber', 'filament', or 'unknown')
            fiber_analysis_data: Results from fiber type detection for enhanced analysis
            
        Returns:
            Dictionary containing comprehensive porosity analysis results
        """
        self.fiber_mask = fiber_mask.astype(bool)
        self.scale_factor = scale_factor
        
        print(f"🕳️  Starting enhanced porosity analysis...")
        print(f"   Fiber type: {fiber_type}")
        print(f"   Scale factor: {scale_factor:.4f} μm/pixel")
        print(f"   Fiber mask area: {np.sum(fiber_mask):,} pixels")
        
        # Step 1: Preprocess image with fiber-type-aware parameters
        processed_image = self._preprocess_image_enhanced(image, fiber_type)
        
        # Step 2: Detect pores with enhanced algorithms
        if self.config['fiber_integration']['use_individual_fibers'] and fiber_analysis_data:
            # Analyze each fiber individually
            pore_results = self._analyze_individual_fibers(
                processed_image, fiber_analysis_data, fiber_type
            )
        else:
            # Analyze the entire fiber mask as one region
            pore_results = self._analyze_fiber_region(
                processed_image, fiber_mask, fiber_type
            )
        
        # Step 3: Calculate comprehensive metrics
        porosity_metrics = self._calculate_enhanced_porosity_metrics(
            pore_results, fiber_mask, fiber_type
        )
        
        # Step 4: Analyze size distribution with enhanced statistics
        size_distribution = self._analyze_enhanced_size_distribution(pore_results)
        
        # Step 5: Spatial analysis
        spatial_analysis = self._analyze_enhanced_spatial_distribution(
            pore_results, fiber_mask
        )
        
        # Step 6: Fiber-type-specific analysis
        specialized_analysis = self._perform_specialized_analysis(
            pore_results, fiber_type, fiber_analysis_data
        )
        
        # Compile enhanced results
        self.results = {
            'porosity_metrics': porosity_metrics,
            'size_distribution': size_distribution,
            'spatial_analysis': spatial_analysis,
            'specialized_analysis': specialized_analysis,
            'pore_results': pore_results,
            'fiber_type': fiber_type,
            'scale_factor': scale_factor,
            'config_used': self.config.copy(),
            'analysis_quality': self._assess_analysis_quality(porosity_metrics, pore_results)
        }
        
        self.pore_data = pore_results
        
        print(f"✅ Enhanced porosity analysis complete!")
        print(f"   Total pores found: {len(pore_results)}")
        print(f"   Total porosity: {porosity_metrics['total_porosity_percent']:.2f}%")
        
        return self.results
    
    def _preprocess_image_enhanced(self, image: np.ndarray, fiber_type: str) -> np.ndarray:
        """
        Enhanced preprocessing with fiber-type-aware parameters.
        """
        processed = image.copy()
        
        # Adjust preprocessing based on fiber type
        if fiber_type == 'hollow_fiber':
            # Hollow fibers often have more complex internal structure
            # Use stronger denoising but preserve fine details
            if self.config['filtering']['denoise_method'] == 'bilateral':
                processed = cv2.bilateralFilter(processed, 9, 80, 80)
            
            # Enhanced contrast for internal structure
            if self.config['filtering']['enhance_contrast']:
                clahe = cv2.createCLAHE(
                    clipLimit=self.config['filtering']['clahe_clip_limit'] * 1.2,
                    tileGridSize=self.config['filtering']['clahe_tile_size']
                )
                processed = clahe.apply(processed.astype(np.uint8))
        
        elif fiber_type == 'filament':
            # Filaments typically have simpler structure
            # Use gentler preprocessing to avoid creating false pores
            if self.config['filtering']['denoise_method'] == 'bilateral':
                processed = cv2.bilateralFilter(processed, 7, 70, 70)
            
            # Standard contrast enhancement
            if self.config['filtering']['enhance_contrast']:
                clahe = cv2.createCLAHE(
                    clipLimit=self.config['filtering']['clahe_clip_limit'],
                    tileGridSize=self.config['filtering']['clahe_tile_size']
                )
                processed = clahe.apply(processed.astype(np.uint8))
        
        else:
            # Unknown fiber type - use default preprocessing
            if self.config['filtering']['denoise_method'] == 'bilateral':
                processed = cv2.bilateralFilter(processed, 8, 75, 75)
            
            if self.config['filtering']['enhance_contrast']:
                clahe = cv2.createCLAHE(
                    clipLimit=self.config['filtering']['clahe_clip_limit'],
                    tileGridSize=self.config['filtering']['clahe_tile_size']
                )
                processed = clahe.apply(processed.astype(np.uint8))
        
        return processed
    
    def _analyze_individual_fibers(self, image: np.ndarray, 
                                 fiber_analysis_data: Dict, 
                                 fiber_type: str) -> List[Dict]:
        """
        Analyze porosity of each individual fiber separately.
        """
        all_pore_results = []
        
        individual_results = fiber_analysis_data.get('individual_results', [])
        
        for i, fiber_result in enumerate(individual_results):
            fiber_props = fiber_result.get('fiber_properties', {})
            fiber_contour = fiber_props.get('contour')
            
            if fiber_contour is None:
                continue
            
            # Create mask for this individual fiber
            individual_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(individual_mask, [fiber_contour], 255)
            individual_mask = individual_mask.astype(bool)
            
            # Check if fiber is large enough for analysis
            fiber_area = np.sum(individual_mask)
            min_area = self.config['fiber_integration']['min_fiber_area_for_analysis']
            
            if fiber_area < min_area:
                continue
            
            # For hollow fibers, exclude the lumen from porosity analysis
            analysis_mask = individual_mask.copy()
            if (fiber_type == 'hollow_fiber' and 
                self.config['fiber_integration']['hollow_fiber_lumen_exclusion'] and
                fiber_result.get('has_lumen', False)):
                
                lumen_props = fiber_result.get('lumen_properties', {})
                lumen_contour = lumen_props.get('contour')
                
                if lumen_contour is not None:
                    # Create lumen mask and subtract from analysis mask
                    lumen_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(lumen_mask, [lumen_contour], 255)
                    analysis_mask = analysis_mask & ~lumen_mask.astype(bool)
            
            # Analyze pores in this fiber
            fiber_pores = self._detect_pores_in_region(image, analysis_mask, fiber_type)
            
            # Add fiber ID to each pore
            for pore in fiber_pores:
                pore['fiber_id'] = i
                pore['fiber_area_pixels'] = fiber_area
                pore['fiber_area_um2'] = fiber_area * (self.scale_factor ** 2)
                pore['has_lumen'] = fiber_result.get('has_lumen', False)
            
            all_pore_results.extend(fiber_pores)
        
        return all_pore_results
    
    def _analyze_fiber_region(self, image: np.ndarray, 
                            fiber_mask: np.ndarray, 
                            fiber_type: str) -> List[Dict]:
        """
        Analyze porosity of the entire fiber mask as one region.
        """
        return self._detect_pores_in_region(image, fiber_mask, fiber_type)
    
    def _detect_pores_in_region(self, image: np.ndarray, 
                              region_mask: np.ndarray, 
                              fiber_type: str) -> List[Dict]:
        """
        Detect pores within a specific region with fiber-type-aware parameters.
        """
        # Restrict analysis to the specified region
        region_image = image * region_mask
        
        # Choose segmentation method based on fiber type and configuration
        method = self.config['segmentation']['method']
        
        if self.config['segmentation']['fiber_type_aware']:
            # Adjust method based on fiber type
            if fiber_type == 'hollow_fiber':
                # Hollow fibers benefit from multi-otsu for complex internal structure
                method = 'multi_otsu'
            elif fiber_type == 'filament':
                # Filaments work well with adaptive thresholding
                method = 'adaptive' if method == 'multi_otsu' else method
        
        # Apply segmentation
        if method == 'multi_otsu':
            pore_mask = self._multi_otsu_segmentation_enhanced(region_image, region_mask, fiber_type)
        elif method == 'otsu':
            pore_mask = self._otsu_segmentation_enhanced(region_image, region_mask, fiber_type)
        elif method == 'adaptive':
            pore_mask = self._adaptive_segmentation_enhanced(region_image, region_mask, fiber_type)
        elif method == 'watershed':
            pore_mask = self._watershed_segmentation_enhanced(region_image, region_mask, fiber_type)
        else:
            raise ValueError(f"Unknown segmentation method: {method}")
        
        # Post-process pore mask
        pore_mask = self._postprocess_pore_mask_enhanced(pore_mask, region_mask, fiber_type)
        
        # Label connected components and extract properties
        pore_labels = measure.label(pore_mask)
        pore_results = self._extract_enhanced_pore_properties(pore_labels, image, region_mask)
        
        return pore_results
    
    def _multi_otsu_segmentation_enhanced(self, image: np.ndarray, 
                                        mask: np.ndarray, 
                                        fiber_type: str) -> np.ndarray:
        """Enhanced multi-Otsu thresholding with fiber-type awareness."""
        
        masked_pixels = image[mask > 0]
        if len(masked_pixels) == 0:
            return np.zeros_like(image, dtype=bool)
        
        try:
            # Adjust number of classes based on fiber type
            if fiber_type == 'hollow_fiber':
                classes = 4  # Background, lumen, fiber material, pores
            else:
                classes = 3  # Background, fiber material, pores
            
            thresholds = filters.threshold_multiotsu(masked_pixels, classes=classes)
            
            # Pores are typically the darkest regions
            pore_threshold = thresholds[0]
            
            # For hollow fibers, might need to adjust threshold selection
            if fiber_type == 'hollow_fiber' and len(thresholds) >= 2:
                # Use a more conservative threshold to avoid false positives
                pore_threshold = np.mean(thresholds[:2])
            
            pore_mask = (image < pore_threshold) & mask
            
        except:
            # Fallback to regular Otsu
            threshold = filters.threshold_otsu(masked_pixels)
            adjustment_factor = 0.7 if fiber_type == 'hollow_fiber' else 0.8
            pore_mask = (image < threshold * adjustment_factor) & mask
        
        return pore_mask
    
    def _otsu_segmentation_enhanced(self, image: np.ndarray, 
                                  mask: np.ndarray, 
                                  fiber_type: str) -> np.ndarray:
        """Enhanced Otsu thresholding with fiber-type awareness."""
        
        masked_pixels = image[mask > 0]
        if len(masked_pixels) == 0:
            return np.zeros_like(image, dtype=bool)
        
        threshold = filters.threshold_otsu(masked_pixels)
        
        # Adjust threshold based on fiber type
        if fiber_type == 'hollow_fiber':
            adjustment_factor = 1 - self.config['pore_detection']['contrast_threshold'] * 1.2
        elif fiber_type == 'filament':
            adjustment_factor = 1 - self.config['pore_detection']['contrast_threshold'] * 0.8
        else:
            adjustment_factor = 1 - self.config['pore_detection']['contrast_threshold']
        
        pore_threshold = threshold * adjustment_factor
        pore_mask = (image < pore_threshold) & mask
        
        return pore_mask
    
    def _adaptive_segmentation_enhanced(self, image: np.ndarray, 
                                      mask: np.ndarray, 
                                      fiber_type: str) -> np.ndarray:
        """Enhanced adaptive thresholding with fiber-type awareness."""
        
        block_size = self.config['segmentation']['adaptive_block_size']
        C = self.config['segmentation']['adaptive_c']
        
        # Adjust parameters based on fiber type
        if fiber_type == 'hollow_fiber':
            # Hollow fibers may need larger block size for internal structure
            block_size = min(block_size * 1.5, 101)
            C = C * 1.2
        elif fiber_type == 'filament':
            # Filaments may work better with smaller block size
            block_size = max(block_size * 0.8, 11)
            C = C * 0.8
        
        # Ensure block_size is odd
        block_size = int(block_size)
        if block_size % 2 == 0:
            block_size += 1
        
        adaptive_thresh = cv2.adaptiveThreshold(
            image.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, C
        )
        
        pore_mask = (adaptive_thresh > 0) & mask
        return pore_mask
    
    def _watershed_segmentation_enhanced(self, image: np.ndarray, 
                                       mask: np.ndarray, 
                                       fiber_type: str) -> np.ndarray:
        """Enhanced watershed segmentation with fiber-type awareness."""
        
        # Start with basic thresholding
        threshold = filters.threshold_otsu(image[mask > 0]) if np.any(mask) else 0
        
        # Adjust threshold based on fiber type
        if fiber_type == 'hollow_fiber':
            threshold_factor = 0.7
        elif fiber_type == 'filament':
            threshold_factor = 0.9
        else:
            threshold_factor = 0.8
        
        binary = (image < threshold * threshold_factor) & mask
        
        if not np.any(binary):
            return binary
        
        # Distance transform for markers
        distance = ndimage.distance_transform_edt(binary)
        
        if self.config['segmentation']['watershed_markers'] == 'distance':
            # Adjust min_distance based on fiber type
            if fiber_type == 'hollow_fiber':
                min_distance = 3  # Smaller distance for more detailed detection
            else:
                min_distance = 5  # Larger distance for cleaner detection
            
            local_maxima = feature.peak_local_maxima(
                distance, min_distance=min_distance, 
                threshold_abs=0.2*distance.max()
            )
            markers = np.zeros_like(binary, dtype=int)
            if len(local_maxima[0]) > 0:
                markers[local_maxima] = np.arange(1, len(local_maxima[0]) + 1)
        else:
            # H-maxima markers
            h_value = 0.2 if fiber_type == 'hollow_fiber' else 0.3
            markers = morphology.h_maxima(distance, h=distance.max()*h_value)
            markers = measure.label(markers)
        
        # Watershed
        if np.max(markers) > 0:
            labels = segmentation.watershed(-distance, markers, mask=binary)
            pore_mask = labels > 0
        else:
            pore_mask = binary
        
        return pore_mask
    
    def _postprocess_pore_mask_enhanced(self, pore_mask: np.ndarray, 
                                      region_mask: np.ndarray, 
                                      fiber_type: str) -> np.ndarray:
        """Enhanced post-processing with fiber-type awareness."""
        
        # Adjust minimum area based on fiber type
        min_area = self.config['pore_detection']['min_pore_area']
        if fiber_type == 'hollow_fiber':
            # Hollow fibers might have smaller genuine pores
            min_area = max(min_area * 0.8, 3)
        elif fiber_type == 'filament':
            # Filaments should have fewer, larger pores
            min_area = min_area * 1.2
        
        # Remove small objects
        pore_mask = remove_small_objects(pore_mask, min_size=int(min_area))
        
        # Morphological operations
        disk_size = self.config['pore_detection']['morphology_disk_size']
        if fiber_type == 'hollow_fiber':
            # More aggressive closing for hollow fibers
            pore_mask = binary_closing(pore_mask, disk(disk_size + 1))
        else:
            pore_mask = binary_closing(pore_mask, disk(disk_size))
        
        # Remove border pores if specified
        if self.config['pore_detection']['remove_border_pores']:
            pore_mask = segmentation.clear_border(pore_mask)
        
        # Ensure pores are within the analysis region
        pore_mask = pore_mask & region_mask
        
        return pore_mask
    
    def _extract_enhanced_pore_properties(self, pore_labels: np.ndarray, 
                                         image: np.ndarray, 
                                         region_mask: np.ndarray) -> List[Dict]:
        """Extract enhanced pore properties with additional metrics."""
        
        properties = measure.regionprops(pore_labels, intensity_image=image)
        pore_data = []
        
        for prop in properties:
            # Basic geometric properties
            area_pixels = prop.area
            area_um2 = area_pixels * (self.scale_factor ** 2)
            
            # Equivalent diameter
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
            eccentricity = prop.eccentricity
            
            # Position
            centroid_y, centroid_x = prop.centroid
            
            # Intensity properties
            mean_intensity = prop.mean_intensity
            max_intensity = prop.max_intensity
            min_intensity = prop.min_intensity
            intensity_std = np.std(image[pore_labels == prop.label])
            
            # Enhanced metrics
            # Compactness (another measure of roundness)
            compactness = area_pixels / (prop.perimeter ** 2) if prop.perimeter > 0 else 0
            
            # Convexity (1 - convex_deficiency)
            convex_area = prop.convex_area
            convexity = area_pixels / convex_area if convex_area > 0 else 0
            
            # Roundness (different from circularity)
            roundness = (4 * area_pixels) / (np.pi * prop.major_axis_length ** 2) if prop.major_axis_length > 0 else 0
            
            pore_info = {
                # Basic properties
                'label': prop.label,
                'area_pixels': area_pixels,
                'area_um2': area_um2,
                'equivalent_diameter_pixels': equiv_diameter_pixels,
                'equivalent_diameter_um': equiv_diameter_um,
                'major_axis_um': major_axis_um,
                'minor_axis_um': minor_axis_um,
                'perimeter_pixels': prop.perimeter,
                'perimeter_um': prop.perimeter * self.scale_factor,
                
                # Shape descriptors
                'aspect_ratio': aspect_ratio,
                'circularity': circularity,
                'solidity': solidity,
                'extent': extent,
                'eccentricity': eccentricity,
                'compactness': compactness,
                'convexity': convexity,
                'roundness': roundness,
                
                # Position
                'centroid_x': centroid_x,
                'centroid_y': centroid_y,
                'bbox': prop.bbox,
                'orientation': prop.orientation,
                
                # Intensity properties
                'mean_intensity': mean_intensity,
                'max_intensity': max_intensity,
                'min_intensity': min_intensity,
                'intensity_std': intensity_std,
                'intensity_range': max_intensity - min_intensity,
                
                # Additional properties (will be set by calling function if available)
                'fiber_id': None,
                'fiber_area_pixels': None,
                'fiber_area_um2': None,
                'has_lumen': None,
            }
            
            pore_data.append(pore_info)
        
        return pore_data
    
    def _calculate_enhanced_porosity_metrics(self, pore_results: List[Dict], 
                                           fiber_mask: np.ndarray, 
                                           fiber_type: str) -> Dict:
        """Calculate enhanced porosity metrics with fiber-type-specific analysis."""
        
        if not pore_results:
            return {
                'total_porosity_percent': 0.0,
                'pore_count': 0,
                'total_pore_area_um2': 0.0,
                'fiber_area_um2': 0.0,
                'average_pore_size_um2': 0.0,
                'pore_density_per_mm2': 0.0,
                'fiber_type': fiber_type,
                'analysis_method': 'enhanced'
            }
        
        # Calculate total areas
        total_pore_area_pixels = sum(pore['area_pixels'] for pore in pore_results)
        total_pore_area_um2 = sum(pore['area_um2'] for pore in pore_results)
        
        fiber_area_pixels = np.sum(fiber_mask)
        fiber_area_um2 = fiber_area_pixels * (self.scale_factor ** 2)
        
        # Calculate porosity percentage
        porosity_percent = (total_pore_area_pixels / fiber_area_pixels) * 100 if fiber_area_pixels > 0 else 0
        
        # Basic metrics
        pore_count = len(pore_results)
        avg_pore_size = total_pore_area_um2 / pore_count if pore_count > 0 else 0
        
        # Pore density (pores per mm²)
        fiber_area_mm2 = fiber_area_um2 / 1e6
        pore_density = pore_count / fiber_area_mm2 if fiber_area_mm2 > 0 else 0
        
        # Enhanced metrics
        pore_areas = [pore['area_um2'] for pore in pore_results]
        pore_diameters = [pore['equivalent_diameter_um'] for pore in pore_results]
        circularities = [pore['circularity'] for pore in pore_results]
        aspect_ratios = [pore['aspect_ratio'] for pore in pore_results]
        
        enhanced_metrics = {
            'total_porosity_percent': porosity_percent,
            'pore_count': pore_count,
            'total_pore_area_um2': total_pore_area_um2,
            'fiber_area_um2': fiber_area_um2,
            'average_pore_size_um2': avg_pore_size,
            'pore_density_per_mm2': pore_density,
            'fiber_type': fiber_type,
            'analysis_method': 'enhanced',
            
            # Size statistics
            'median_pore_size_um2': np.median(pore_areas) if pore_areas else 0,
            'std_pore_size_um2': np.std(pore_areas) if pore_areas else 0,
            'min_pore_size_um2': np.min(pore_areas) if pore_areas else 0,
            'max_pore_size_um2': np.max(pore_areas) if pore_areas else 0,
            
            # Diameter statistics
            'mean_pore_diameter_um': np.mean(pore_diameters) if pore_diameters else 0,
            'median_pore_diameter_um': np.median(pore_diameters) if pore_diameters else 0,
            'std_pore_diameter_um': np.std(pore_diameters) if pore_diameters else 0,
            
            # Shape statistics
            'mean_circularity': np.mean(circularities) if circularities else 0,
            'mean_aspect_ratio': np.mean(aspect_ratios) if aspect_ratios else 0,
            'fraction_circular_pores': sum(1 for c in circularities if c > 0.7) / len(circularities) if circularities else 0,
            'fraction_elongated_pores': sum(1 for ar in aspect_ratios if ar > 2.0) / len(aspect_ratios) if aspect_ratios else 0,
        }
        
        # Fiber-type-specific metrics
        if fiber_type == 'hollow_fiber':
            # For hollow fibers, analyze pores by fiber
            enhanced_metrics.update(self._calculate_hollow_fiber_specific_metrics(pore_results))
        elif fiber_type == 'filament':
            # For filaments, focus on overall distribution
            enhanced_metrics.update(self._calculate_filament_specific_metrics(pore_results))
        
        return enhanced_metrics
    
    def _calculate_hollow_fiber_specific_metrics(self, pore_results: List[Dict]) -> Dict:
        """Calculate metrics specific to hollow fibers."""
        
        # Group pores by fiber if individual fiber analysis was performed
        fiber_pores = {}
        for pore in pore_results:
            fiber_id = pore.get('fiber_id')
            if fiber_id is not None:
                if fiber_id not in fiber_pores:
                    fiber_pores[fiber_id] = []
                fiber_pores[fiber_id].append(pore)
        
        if fiber_pores:
            # Per-fiber porosity statistics
            fiber_porosities = []
            for fiber_id, pores in fiber_pores.items():
                if pores and pores[0]['fiber_area_pixels'] is not None:
                    fiber_pore_area = sum(pore['area_pixels'] for pore in pores)
                    fiber_area = pores[0]['fiber_area_pixels']
                    fiber_porosity = (fiber_pore_area / fiber_area) * 100
                    fiber_porosities.append(fiber_porosity)
            
            return {
                'fibers_analyzed': len(fiber_pores),
                'mean_fiber_porosity_percent': np.mean(fiber_porosities) if fiber_porosities else 0,
                'std_fiber_porosity_percent': np.std(fiber_porosities) if fiber_porosities else 0,
                'min_fiber_porosity_percent': np.min(fiber_porosities) if fiber_porosities else 0,
                'max_fiber_porosity_percent': np.max(fiber_porosities) if fiber_porosities else 0,
            }
        else:
            return {'fibers_analyzed': 0}
    
    def _calculate_filament_specific_metrics(self, pore_results: List[Dict]) -> Dict:
        """Calculate metrics specific to filaments."""
        
        if not pore_results:
            return {}
        
        # For filaments, analyze size distribution characteristics
        pore_areas = [pore['area_um2'] for pore in pore_results]
        
        # Calculate size distribution entropy (measure of uniformity)
        if len(set(pore_areas)) > 1:
            hist, _ = np.histogram(pore_areas, bins=min(10, len(pore_areas)))
            hist = hist[hist > 0]  # Remove empty bins
            probs = hist / np.sum(hist)
            entropy = -np.sum(probs * np.log2(probs))
        else:
            entropy = 0
        
        return {
            'size_distribution_entropy': entropy,
            'size_uniformity': 1 / (1 + np.std(pore_areas) / np.mean(pore_areas)) if pore_areas else 0,
        }
    
    def _analyze_enhanced_size_distribution(self, pore_results: List[Dict]) -> Dict:
        """Analyze pore size distribution with enhanced statistics."""
        
        if not pore_results:
            return {'sizes_um2': [], 'diameters_um': [], 'statistics': {}}
        
        sizes = [pore['area_um2'] for pore in pore_results]
        diameters = [pore['equivalent_diameter_um'] for pore in pore_results]
        
        # Calculate comprehensive statistics
        percentiles = self.config['analysis']['percentiles']
        size_stats = {
            'count': len(sizes),
            'mean_size_um2': np.mean(sizes),
            'median_size_um2': np.median(sizes),
            'std_size_um2': np.std(sizes),
            'min_size_um2': np.min(sizes),
            'max_size_um2': np.max(sizes),
            'mean_diameter_um': np.mean(diameters),
            'median_diameter_um': np.median(diameters),
            'std_diameter_um': np.std(diameters),
            'skewness_size': self._calculate_skewness(sizes),
            'kurtosis_size': self._calculate_kurtosis(sizes),
        }
        
        # Add percentiles
        for p in percentiles:
            size_stats[f'p{p}_size_um2'] = np.percentile(sizes, p)
            size_stats[f'p{p}_diameter_um'] = np.percentile(diameters, p)
        
        # Size distribution characteristics
        cv_size = size_stats['std_size_um2'] / size_stats['mean_size_um2'] if size_stats['mean_size_um2'] > 0 else 0
        size_stats['coefficient_of_variation'] = cv_size
        
        # Classification of distribution
        if cv_size < 0.3:
            size_stats['distribution_type'] = 'uniform'
        elif cv_size < 0.7:
            size_stats['distribution_type'] = 'moderate_variation'
        else:
            size_stats['distribution_type'] = 'high_variation'
        
        return {
            'sizes_um2': sizes,
            'diameters_um': diameters,
            'statistics': size_stats
        }
    
    def _analyze_enhanced_spatial_distribution(self, pore_results: List[Dict], 
                                             fiber_mask: np.ndarray) -> Dict:
        """Analyze spatial distribution with enhanced metrics."""
        
        if len(pore_results) < 2:
            return {
                'spatial_uniformity': 1.0, 
                'clustering_coefficient': 0.0,
                'analysis_possible': False
            }
        
        # Extract centroid coordinates
        centroids = np.array([[pore['centroid_x'], pore['centroid_y']] for pore in pore_results])
        
        # Calculate nearest neighbor distances
        distances = spatial.distance_matrix(centroids, centroids)
        np.fill_diagonal(distances, np.inf)
        nearest_distances = np.min(distances, axis=1)
        
        # Spatial uniformity (coefficient of variation of nearest neighbor distances)
        mean_nn_dist = np.mean(nearest_distances)
        std_nn_dist = np.std(nearest_distances)
        spatial_uniformity = 1 / (1 + std_nn_dist / mean_nn_dist) if mean_nn_dist > 0 else 1.0
        
        # Enhanced clustering analysis
        try:
            eps = np.median(nearest_distances) * 1.5
            clustering = DBSCAN(eps=eps, min_samples=2).fit(centroids)
            n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
            clustering_coefficient = n_clusters / len(pore_results)
            
            # Additional clustering metrics
            cluster_labels = clustering.labels_
            n_noise = list(cluster_labels).count(-1)
            largest_cluster_size = max([list(cluster_labels).count(i) for i in set(cluster_labels) if i != -1]) if n_clusters > 0 else 0
            
        except:
            clustering_coefficient = 0.0
            n_clusters = 0
            n_noise = 0
            largest_cluster_size = 0
        
        # Ripley's K-function approximation (simplified)
        # This measures point pattern regularity
        fiber_area = np.sum(fiber_mask) * (self.scale_factor ** 2)  # in um^2
        density = len(pore_results) / fiber_area if fiber_area > 0 else 0
        
        # Expected distance for random distribution
        expected_nn_dist = 0.5 / np.sqrt(density) if density > 0 else 0
        regularity_index = mean_nn_dist / expected_nn_dist if expected_nn_dist > 0 else 1.0
        
        return {
            'spatial_uniformity': spatial_uniformity,
            'clustering_coefficient': clustering_coefficient,
            'nearest_neighbor_distances_um': nearest_distances * self.scale_factor,
            'mean_nn_distance_um': mean_nn_dist * self.scale_factor,
            'std_nn_distance_um': std_nn_dist * self.scale_factor,
            'n_clusters': n_clusters,
            'n_noise_points': n_noise,
            'largest_cluster_size': largest_cluster_size,
            'pore_density_per_um2': density,
            'regularity_index': regularity_index,
            'analysis_possible': True,
            'spatial_pattern': self._classify_spatial_pattern(spatial_uniformity, clustering_coefficient, regularity_index)
        }
    
    def _classify_spatial_pattern(self, uniformity: float, clustering: float, regularity: float) -> str:
        """Classify the spatial pattern of pores."""
        
        if regularity > 1.5:
            return 'regular'  # More regular than random
        elif regularity < 0.5:
            return 'clustered'  # More clustered than random
        elif uniformity > 0.8:
            return 'uniform_random'  # Well-distributed random
        elif clustering > 0.3:
            return 'clustered_random'  # Random with clusters
        else:
            return 'random'  # Close to completely random
    
    def _perform_specialized_analysis(self, pore_results: List[Dict], 
                                    fiber_type: str, 
                                    fiber_analysis_data: Optional[Dict]) -> Dict:
        """Perform specialized analysis based on fiber type and detection results."""
        
        specialized = {
            'fiber_type': fiber_type,
            'analysis_type': 'specialized'
        }
        
        if fiber_type == 'hollow_fiber':
            specialized.update(self._analyze_hollow_fiber_porosity(pore_results, fiber_analysis_data))
        elif fiber_type == 'filament':
            specialized.update(self._analyze_filament_porosity(pore_results))
        else:
            specialized.update({'analysis': 'generic_fiber_analysis'})
        
        return specialized
    
    def _analyze_hollow_fiber_porosity(self, pore_results: List[Dict], 
                                     fiber_analysis_data: Optional[Dict]) -> Dict:
        """Specialized analysis for hollow fiber porosity."""
        
        analysis = {
            'fiber_specific_analysis': 'hollow_fiber',
            'wall_porosity_analysis': True
        }
        
        if fiber_analysis_data:
            individual_results = fiber_analysis_data.get('individual_results', [])
            
            # Analyze wall porosity (excluding lumen)
            wall_pores = [pore for pore in pore_results if not pore.get('in_lumen', False)]
            
            if wall_pores:
                analysis.update({
                    'wall_pore_count': len(wall_pores),
                    'wall_pore_density': len(wall_pores) / len(pore_results) if pore_results else 0,
                    'mean_wall_pore_size_um2': np.mean([pore['area_um2'] for pore in wall_pores]),
                })
            
            # Analyze pore distribution relative to lumen
            lumens_detected = sum(1 for result in individual_results if result.get('has_lumen', False))
            analysis['lumens_detected'] = lumens_detected
            analysis['lumen_detection_rate'] = lumens_detected / len(individual_results) if individual_results else 0
        
        return analysis
    
    def _analyze_filament_porosity(self, pore_results: List[Dict]) -> Dict:
        """Specialized analysis for filament porosity."""
        
        analysis = {
            'fiber_specific_analysis': 'filament',
            'structural_integrity_focus': True
        }
        
        if pore_results:
            # For filaments, focus on defect analysis
            large_pores = [pore for pore in pore_results if pore['area_um2'] > np.mean([p['area_um2'] for p in pore_results]) * 2]
            elongated_pores = [pore for pore in pore_results if pore['aspect_ratio'] > 3.0]
            
            analysis.update({
                'potential_defects': len(large_pores),
                'elongated_defects': len(elongated_pores),
                'defect_rate': len(large_pores) / len(pore_results),
                'structural_concern_level': self._assess_filament_structural_concern(pore_results)
            })
        
        return analysis
    
    def _assess_filament_structural_concern(self, pore_results: List[Dict]) -> str:
        """Assess structural concern level for filaments based on porosity."""
        
        if not pore_results:
            return 'none'
        
        total_porosity = sum(pore['area_um2'] for pore in pore_results)
        large_pore_count = sum(1 for pore in pore_results if pore['area_um2'] > 10)  # >10 um^2
        
        if total_porosity > 100 or large_pore_count > 5:  # Arbitrary thresholds
            return 'high'
        elif total_porosity > 50 or large_pore_count > 2:
            return 'moderate'
        elif total_porosity > 10 or large_pore_count > 0:
            return 'low'
        else:
            return 'minimal'
    
    def _assess_analysis_quality(self, porosity_metrics: Dict, pore_results: List[Dict]) -> Dict:
        """Assess the quality of the porosity analysis."""
        
        quality = {
            'overall_quality': 'unknown',
            'confidence_score': 0.0,
            'quality_factors': []
        }
        
        score = 0.0
        factors = []
        
        # Pore count quality
        pore_count = len(pore_results)
        if pore_count >= 20:
            score += 0.3
            factors.append('sufficient_pore_count')
        elif pore_count >= 10:
            score += 0.2
            factors.append('moderate_pore_count')
        elif pore_count >= 5:
            score += 0.1
            factors.append('low_pore_count')
        else:
            factors.append('very_low_pore_count')
        
        # Size distribution quality
        if pore_results:
            sizes = [pore['area_um2'] for pore in pore_results]
            cv_size = np.std(sizes) / np.mean(sizes) if np.mean(sizes) > 0 else 0
            
            if 0.3 <= cv_size <= 1.5:  # Reasonable variation
                score += 0.3
                factors.append('good_size_distribution')
            elif cv_size < 2.0:
                score += 0.2
                factors.append('acceptable_size_distribution')
            else:
                factors.append('poor_size_distribution')
        
        # Shape quality
        if pore_results:
            circularities = [pore['circularity'] for pore in pore_results]
            mean_circularity = np.mean(circularities)
            
            if mean_circularity > 0.5:
                score += 0.2
                factors.append('good_pore_shapes')
            elif mean_circularity > 0.3:
                score += 0.1
                factors.append('acceptable_pore_shapes')
            else:
                factors.append('poor_pore_shapes')
        
        # Scale factor quality
        if self.scale_factor > 0 and self.scale_factor != 1.0:
            score += 0.2
            factors.append('scale_calibrated')
        else:
            factors.append('no_scale_calibration')
        
        # Determine overall quality
        if score >= 0.8:
            overall_quality = 'excellent'
        elif score >= 0.6:
            overall_quality = 'good'
        elif score >= 0.4:
            overall_quality = 'moderate'
        elif score >= 0.2:
            overall_quality = 'poor'
        else:
            overall_quality = 'very_poor'
        
        quality.update({
            'overall_quality': overall_quality,
            'confidence_score': score,
            'quality_factors': factors
        })
        
        return quality
    
    def _calculate_skewness(self, data: List[float]) -> float:
        """Calculate skewness of data distribution."""
        if len(data) < 3:
            return 0.0
        
        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        skewness = np.mean(((data - mean) / std) ** 3)
        return skewness
    
    def _calculate_kurtosis(self, data: List[float]) -> float:
        """Calculate kurtosis of data distribution."""
        if len(data) < 4:
            return 0.0
        
        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        kurtosis = np.mean(((data - mean) / std) ** 4) - 3  # Excess kurtosis
        return kurtosis
    
    def get_pore_dataframe(self) -> pd.DataFrame:
        """Convert pore data to pandas DataFrame for easy analysis."""
        if self.pore_data is None:
            return pd.DataFrame()
        
        return pd.DataFrame(self.pore_data)
    
    def export_results(self, output_path: str):
        """Export enhanced analysis results to Excel file."""
        if self.results is None:
            print("No results to export. Run analyze_fiber_porosity() first.")
            return
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Enhanced summary metrics
            summary_data = []
            summary_data.append(self.results['porosity_metrics'])
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Enhanced_Summary', index=False)
            
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
            
            # Specialized analysis
            specialized_df = pd.DataFrame([self.results['specialized_analysis']])
            specialized_df.to_excel(writer, sheet_name='Specialized_Analysis', index=False)
            
            # Analysis quality
            quality_df = pd.DataFrame([self.results['analysis_quality']])
            quality_df.to_excel(writer, sheet_name='Analysis_Quality', index=False)
        
        print(f"Enhanced results exported to {output_path}")


# Convenience functions for integration with existing system

def analyze_fiber_porosity_enhanced(image: np.ndarray, 
                                   fiber_mask: np.ndarray,
                                   scale_factor: float = 1.0,
                                   fiber_type: str = 'unknown',
                                   fiber_analysis_data: Optional[Dict] = None,
                                   config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Enhanced convenience function for porosity analysis.
    
    Args:
        image: Input SEM image (grayscale)
        fiber_mask: Binary mask of fiber regions
        scale_factor: Micrometers per pixel conversion factor
        fiber_type: Type of fiber ('hollow_fiber', 'filament', or 'unknown')
        fiber_analysis_data: Results from fiber type detection
        config: Optional configuration dictionary
        
    Returns:
        Dictionary containing enhanced porosity analysis results
    """
    analyzer = EnhancedPorosityAnalyzer(config)
    return analyzer.analyze_fiber_porosity(
        image, fiber_mask, scale_factor, fiber_type, fiber_analysis_data
    )


def quick_porosity_check_enhanced(image: np.ndarray, 
                                 fiber_mask: np.ndarray,
                                 scale_factor: float = 1.0,
                                 fiber_type: str = 'unknown') -> float:
    """
    Quick enhanced porosity percentage calculation.
    
    Args:
        image: Input SEM image
        fiber_mask: Binary mask of fiber regions
        scale_factor: Micrometers per pixel conversion factor
        fiber_type: Type of fiber
        
    Returns:
        Porosity percentage
    """
    config = {
        'segmentation': {'method': 'otsu', 'fiber_type_aware': True},
        'pore_detection': {'min_pore_area': 3}
    }
    
    results = analyze_fiber_porosity_enhanced(
        image, fiber_mask, scale_factor, fiber_type, None, config
    )
    return results['porosity_metrics']['total_porosity_percent']