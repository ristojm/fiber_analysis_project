"""
Enhanced Porosity Analysis Module for SEM Fiber Analysis System
UPDATED: Integrated with oval fitting data for enhanced fiber characterization

This module is updated to work seamlessly with the enhanced fiber detection
that includes oval fitting capabilities for precise diameter measurements.
"""

import numpy as np
import cv2
from skimage import filters, morphology, measure
from scipy import ndimage, spatial
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional, Any
import time
import warnings
warnings.filterwarnings('ignore')


class PorosityAnalyzer:
    """
    Enhanced porosity analyzer integrated with oval fitting fiber analysis.
    
    This analyzer works with the enhanced fiber detection system that provides
    oval fitting data for more accurate fiber characterization.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the enhanced porosity analyzer.
        
        Args:
            config: Optional configuration dictionary to override defaults
        """
        self.default_config = {
            'pore_detection': {
                'intensity_percentile': 28,
                'min_pore_area_pixels': 3,
                'max_pore_area_ratio': 0.1,
                'fast_filtering': True,
                'early_size_filter': True,
                'vectorized_operations': True,
            },
            'performance': {
                'max_candidates_per_stage': 5000,
                'use_simplified_morphology': True,
                'batch_processing': True,
                'enable_timing': False,
            },
            'quality_control': {
                'circularity_threshold': 0.05,
                'aspect_ratio_threshold': 8,
                'solidity_threshold': 0.25,
                'intensity_validation': True,
                'size_dependent_validation': True,
            },
            'fiber_integration': {
                'use_individual_fibers': True,
                'exclude_lumen': True,
                'lumen_buffer_pixels': 3,
                'min_fiber_area_analysis': 1000,
                'use_oval_fitting_data': True,  # NEW: Use oval fitting for enhanced analysis
            },
            'analysis': {
                'calculate_size_distribution': True,
                'calculate_spatial_metrics': True,
                'detailed_reporting': True,
                'save_individual_pore_data': True,
                'oval_aware_analysis': True,  # NEW: Oval-aware porosity analysis
            }
        }
        
        self.config = self.default_config.copy()
        if config:
            self._update_config(config)
        
        self.results = {}
        self.pore_data = None
        self.processing_stats = {}
    
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
        Enhanced porosity analysis function integrated with oval fitting data.
        
        Args:
            image: Input SEM image (grayscale)
            fiber_mask: Binary mask of fiber regions
            scale_factor: Micrometers per pixel conversion factor
            fiber_type: Type of fiber ('hollow_fiber', 'filament', or 'unknown')
            fiber_analysis_data: Enhanced results from fiber type detection with oval fitting
            
        Returns:
            Dictionary containing comprehensive porosity analysis results
        """
        
        start_time = time.time()
        
        if self.config['performance']['enable_timing']:
            print(f"\n🔬 ENHANCED POROSITY ANALYSIS")
            print(f"   Scale factor: {scale_factor:.4f} μm/pixel")
            print(f"   Fiber type: {fiber_type}")
            print(f"   Oval fitting integration: {self.config['fiber_integration']['use_oval_fitting_data']}")
        
        # Initialize results structure
        self.results = {
            'scale_factor': scale_factor,
            'fiber_type': fiber_type,
            'method': 'enhanced_with_oval_fitting',
            'analysis_timestamp': time.time(),
            'oval_fitting_used': self.config['fiber_integration']['use_oval_fitting_data'],
        }
        
        # Validate inputs
        if image is None or fiber_mask is None:
            return {'error': 'Invalid input: image or fiber_mask is None'}
        
        if np.sum(fiber_mask) < self.config['fiber_integration']['min_fiber_area_analysis']:
            return {'error': f'Insufficient fiber area: {np.sum(fiber_mask)} pixels (min: {self.config["fiber_integration"]["min_fiber_area_analysis"]})'}
        
        # Enhanced analysis using oval fitting data if available
        if (self.config['fiber_integration']['use_individual_fibers'] and 
            self.config['fiber_integration']['use_oval_fitting_data'] and
            fiber_analysis_data and 
            'individual_results' in fiber_analysis_data):
            
            # Individual fiber analysis with oval fitting enhancement
            pore_results = self._analyze_individual_fibers_enhanced(
                image, fiber_analysis_data, scale_factor, fiber_type
            )
        elif (self.config['fiber_integration']['use_individual_fibers'] and 
              fiber_analysis_data and 
              'individual_results' in fiber_analysis_data):
            
            # Individual fiber analysis (standard method)
            pore_results = self._analyze_individual_fibers(
                image, fiber_analysis_data, scale_factor, fiber_type
            )
        else:
            # Fallback: analyze entire fiber mask as one region
            pore_results = self._analyze_fiber_region(
                image, fiber_mask, scale_factor, fiber_type
            )
        
        # Calculate comprehensive metrics with oval fitting awareness
        porosity_metrics = self._calculate_enhanced_porosity_metrics(
            pore_results, scale_factor, fiber_analysis_data
        )
        
        # Additional analysis if requested
        analysis_results = {}
        if self.config['analysis']['calculate_size_distribution']:
            analysis_results['size_distribution'] = self._analyze_size_distribution(pore_results)
        
        if self.config['analysis']['calculate_spatial_metrics']:
            analysis_results['spatial_analysis'] = self._analyze_spatial_distribution(pore_results)
        
        # Enhanced quality assessment with oval fitting considerations
        quality_assessment = self._assess_enhanced_analysis_quality(
            pore_results, porosity_metrics, fiber_analysis_data
        )
        
        # Oval fitting summary for porosity context
        oval_context = self._extract_oval_fitting_context(fiber_analysis_data) if fiber_analysis_data else {}
        
        # Compile final results
        processing_time = time.time() - start_time
        
        self.results.update({
            'porosity_metrics': porosity_metrics,
            'individual_pores': pore_results if self.config['analysis']['save_individual_pore_data'] else [],
            'analysis_results': analysis_results,
            'quality_assessment': quality_assessment,
            'oval_fitting_context': oval_context,  # NEW: Oval fitting context
            'processing_time': processing_time,
            'success': True
        })
        
        self.pore_data = pore_results
        
        if self.config['performance']['enable_timing']:
            print(f"✅ Enhanced analysis complete: {len(pore_results)} pores, {porosity_metrics['total_porosity_percent']:.2f}% porosity")
            print(f"⏱️  Processing time: {processing_time:.3f} seconds")
            if oval_context:
                print(f"🔍 Oval fitting: {oval_context.get('fibers_with_ovals', 0)} fibers analyzed")
        
        return self.results
    
    def _analyze_individual_fibers_enhanced(self, image: np.ndarray, 
                                          fiber_analysis_data: Dict, 
                                          scale_factor: float,
                                          fiber_type: str) -> List[Dict]:
        """Enhanced individual fiber analysis using oval fitting data."""
        
        individual_results = fiber_analysis_data.get('individual_results', [])
        all_pore_results = []
        
        for i, fiber_result in enumerate(individual_results):
            fiber_props = fiber_result.get('fiber_properties', {})
            fiber_contour = fiber_props.get('contour')
            
            if fiber_contour is None:
                continue
            
            # Create fiber mask
            fiber_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(fiber_mask, [fiber_contour], 255)
            
            # Enhanced analysis mask preparation using oval fitting data
            analysis_mask = self._prepare_enhanced_analysis_mask(
                fiber_mask, fiber_result, fiber_type, fiber_props
            )
            
            # Skip if analysis area is too small
            analysis_area = np.sum(analysis_mask > 0)
            if analysis_area < self.config['fiber_integration']['min_fiber_area_analysis']:
                continue
            
            # Detect pores in this fiber with oval-aware parameters
            fiber_pores = self._detect_pores_oval_aware(
                image, analysis_mask, scale_factor, fiber_props
            )
            
            # Add enhanced fiber metadata including oval fitting data
            for pore in fiber_pores:
                pore['fiber_id'] = i
                pore['fiber_area_pixels'] = analysis_area
                pore['fiber_area_um2'] = analysis_area * (scale_factor ** 2)
                pore['has_lumen'] = fiber_result.get('has_lumen', False)
                
                # NEW: Add oval fitting context to pore data
                if fiber_props.get('oval_fitted', False):
                    pore['fiber_oval_fitted'] = True
                    pore['fiber_oval_diameter_um'] = fiber_props.get('oval_mean_diameter', 0) * scale_factor
                    pore['fiber_oval_major_diameter_um'] = fiber_props.get('oval_major_diameter', 0) * scale_factor
                    pore['fiber_oval_minor_diameter_um'] = fiber_props.get('oval_minor_diameter', 0) * scale_factor
                    pore['fiber_oval_eccentricity'] = fiber_props.get('oval_eccentricity', 0)
                    pore['fiber_oval_fit_quality'] = fiber_props.get('oval_fit_quality', 0)
                    
                    # Calculate pore position relative to oval
                    pore['pore_position_in_oval'] = self._calculate_pore_oval_position(
                        pore, fiber_props
                    )
                else:
                    pore['fiber_oval_fitted'] = False
            
            all_pore_results.extend(fiber_pores)
        
        return all_pore_results
    
    def _analyze_individual_fibers(self, image: np.ndarray, 
                                  fiber_analysis_data: Dict, 
                                  scale_factor: float,
                                  fiber_type: str) -> List[Dict]:
        """Standard individual fiber analysis (fallback when oval fitting unavailable)."""
        
        individual_results = fiber_analysis_data.get('individual_results', [])
        all_pore_results = []
        
        for i, fiber_result in enumerate(individual_results):
            fiber_props = fiber_result.get('fiber_properties', {})
            fiber_contour = fiber_props.get('contour')
            
            if fiber_contour is None:
                continue
            
            # Create fiber mask
            fiber_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(fiber_mask, [fiber_contour], 255)
            
            # Standard analysis mask preparation
            analysis_mask = self._prepare_analysis_mask(
                fiber_mask, fiber_result, fiber_type
            )
            
            # Skip if analysis area is too small
            analysis_area = np.sum(analysis_mask > 0)
            if analysis_area < self.config['fiber_integration']['min_fiber_area_analysis']:
                continue
            
            # Standard pore detection
            fiber_pores = self._detect_pores_fast_refined(image, analysis_mask, scale_factor)
            
            # Add standard fiber metadata
            for pore in fiber_pores:
                pore['fiber_id'] = i
                pore['fiber_area_pixels'] = analysis_area
                pore['fiber_area_um2'] = analysis_area * (scale_factor ** 2)
                pore['has_lumen'] = fiber_result.get('has_lumen', False)
                pore['fiber_oval_fitted'] = False  # Mark as not oval-fitted
            
            all_pore_results.extend(fiber_pores)
        
        return all_pore_results
    
    def _analyze_fiber_region(self, image: np.ndarray, 
                             fiber_mask: np.ndarray, 
                             scale_factor: float,
                             fiber_type: str) -> List[Dict]:
        """Analyze porosity of entire fiber mask as one region."""
        
        pore_results = self._detect_pores_fast_refined(image, fiber_mask, scale_factor)
        
        # Mark as not oval-fitted since this is bulk analysis
        for pore in pore_results:
            pore['fiber_oval_fitted'] = False
        
        return pore_results
    
    def _prepare_enhanced_analysis_mask(self, fiber_mask: np.ndarray, 
                                      fiber_result: Dict, 
                                      fiber_type: str,
                                      fiber_props: Dict) -> np.ndarray:
        """Prepare enhanced analysis mask using oval fitting data."""
        
        analysis_mask = fiber_mask.copy()
        
        # Exclude lumen for hollow fibers (standard approach)
        if (self.config['fiber_integration']['exclude_lumen'] and 
            fiber_type == 'hollow_fiber' and 
            fiber_result.get('has_lumen', False)):
            
            lumen_props = fiber_result.get('lumen_properties', {})
            lumen_contour = lumen_props.get('contour')
            
            if lumen_contour is not None:
                # Create lumen mask
                lumen_mask = np.zeros(analysis_mask.shape, dtype=np.uint8)
                cv2.fillPoly(lumen_mask, [lumen_contour], 255)
                
                # Enhanced buffer calculation using oval fitting if available
                if (lumen_props.get('oval_fitted', False) and 
                    fiber_props.get('oval_fitted', False)):
                    
                    # Calculate buffer based on wall thickness from oval fitting
                    fiber_radius = fiber_props.get('oval_mean_diameter', 0) / 2
                    lumen_radius = lumen_props.get('oval_mean_diameter', 0) / 2
                    wall_thickness = fiber_radius - lumen_radius
                    
                    # Use proportional buffer (e.g., 10% of wall thickness)
                    buffer_size = max(self.config['fiber_integration']['lumen_buffer_pixels'],
                                    int(wall_thickness * 0.1))
                else:
                    # Standard buffer
                    buffer_size = self.config['fiber_integration']['lumen_buffer_pixels']
                
                if buffer_size > 0:
                    kernel = np.ones((buffer_size*2+1, buffer_size*2+1), np.uint8)
                    lumen_mask = cv2.dilate(lumen_mask, kernel, iterations=1)
                
                # Exclude lumen from analysis
                analysis_mask = cv2.bitwise_and(analysis_mask, cv2.bitwise_not(lumen_mask))
        
        return analysis_mask
    
    def _prepare_analysis_mask(self, fiber_mask: np.ndarray, 
                              fiber_result: Dict, 
                              fiber_type: str) -> np.ndarray:
        """Standard analysis mask preparation (fallback method)."""
        
        analysis_mask = fiber_mask.copy()
        
        # Standard lumen exclusion
        if (self.config['fiber_integration']['exclude_lumen'] and 
            fiber_type == 'hollow_fiber' and 
            fiber_result.get('has_lumen', False)):
            
            lumen_props = fiber_result.get('lumen_properties', {})
            lumen_contour = lumen_props.get('contour')
            
            if lumen_contour is not None:
                lumen_mask = np.zeros(analysis_mask.shape, dtype=np.uint8)
                cv2.fillPoly(lumen_mask, [lumen_contour], 255)
                
                buffer_size = self.config['fiber_integration']['lumen_buffer_pixels']
                if buffer_size > 0:
                    kernel = np.ones((buffer_size*2+1, buffer_size*2+1), np.uint8)
                    lumen_mask = cv2.dilate(lumen_mask, kernel, iterations=1)
                
                analysis_mask = cv2.bitwise_and(analysis_mask, cv2.bitwise_not(lumen_mask))
        
        return analysis_mask
    
    def _detect_pores_oval_aware(self, image: np.ndarray, 
                               analysis_mask: np.ndarray, 
                               scale_factor: float,
                               fiber_props: Dict) -> List[Dict]:
        """
        Oval-aware pore detection that considers fiber shape and orientation.
        """
        
        # Get oval fitting parameters if available
        if fiber_props.get('oval_fitted', False):
            fiber_eccentricity = fiber_props.get('oval_eccentricity', 0)
            fiber_angle = fiber_props.get('oval_angle', 0)
            fiber_major_axis = fiber_props.get('oval_major_diameter', 0)
            fiber_minor_axis = fiber_props.get('oval_minor_diameter', 0)
            
            # Adjust pore detection parameters based on fiber shape
            if fiber_eccentricity > 0.5:  # Elongated fiber
                # Use slightly different morphological kernels
                kernel_aspect = max(1, int(fiber_major_axis / fiber_minor_axis)) if fiber_minor_axis > 0 else 1
                
                # Create oriented kernel based on fiber orientation
                if kernel_aspect > 1:
                    kernel_size = min(5, kernel_aspect)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, max(2, kernel_size//2)))
                    
                    # Rotate kernel to match fiber orientation
                    if abs(fiber_angle) > 10:  # Only rotate if significant angle
                        M = cv2.getRotationMatrix2D((kernel_size//2, kernel_size//4), fiber_angle, 1)
                        kernel = cv2.warpAffine(kernel.astype(np.float32), M, (kernel_size, max(2, kernel_size//2)))
                        kernel = (kernel > 0.5).astype(np.uint8)
                else:
                    kernel = np.ones((2, 2), np.uint8)  # Default small kernel
            else:
                kernel = np.ones((2, 2), np.uint8)  # Circular fiber - standard kernel
        else:
            # Fallback to standard detection
            return self._detect_pores_fast_refined(image, analysis_mask, scale_factor)
        
        # Perform oval-aware pore detection
        masked_image = cv2.bitwise_and(image, image, mask=analysis_mask)
        region_pixels = masked_image[analysis_mask > 0]
        
        if len(region_pixels) == 0:
            return []
        
        # Standard preprocessing with oval-aware morphology
        if len(region_pixels) > 50000:
            masked_image = cv2.bilateralFilter(masked_image, 3, 15, 15)
            region_pixels = masked_image[analysis_mask > 0]
        
        # Primary detection
        percentile = self.config['pore_detection']['intensity_percentile']
        percentile_threshold = np.percentile(region_pixels, percentile)
        detection_mask = (masked_image < percentile_threshold) & (analysis_mask > 0)
        
        # Apply oval-aware morphological operations
        if self.config['performance']['use_simplified_morphology']:
            detection_mask = cv2.morphologyEx(detection_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        
        # Continue with standard pore analysis
        contours, _ = cv2.findContours(detection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Size filtering
        min_area = self.config['pore_detection']['min_pore_area_pixels']
        max_area = int(np.sum(analysis_mask > 0) * self.config['pore_detection']['max_pore_area_ratio'])
        
        areas = np.array([cv2.contourArea(c) for c in contours])
        size_mask = (areas >= min_area) & (areas <= max_area)
        filtered_contours = [contours[i] for i in np.where(size_mask)[0]]
        filtered_areas = areas[size_mask]
        
        # Performance limiting
        max_candidates = self.config['performance']['max_candidates_per_stage']
        if len(filtered_contours) > max_candidates:
            sorted_indices = np.argsort(filtered_areas)[::-1][:max_candidates]
            filtered_contours = [filtered_contours[i] for i in sorted_indices]
            filtered_areas = filtered_areas[sorted_indices]
        
        # Enhanced property calculation and validation
        validated_pores = []
        for contour, area in zip(filtered_contours, filtered_areas):
            pore_props = self._calculate_pore_properties_fast(contour, area, scale_factor)
            
            # Enhanced validation considering fiber shape
            if self._validate_pore_oval_aware(pore_props, masked_image, analysis_mask, fiber_props):
                validated_pores.append(pore_props)
        
        return validated_pores
    
    def _detect_pores_fast_refined(self, image: np.ndarray, 
                                  analysis_mask: np.ndarray, 
                                  scale_factor: float) -> List[Dict]:
        """
        Standard fast refined pore detection algorithm.
        """
        
        masked_image = cv2.bitwise_and(image, image, mask=analysis_mask)
        region_pixels = masked_image[analysis_mask > 0]
        
        if len(region_pixels) == 0:
            return []
        
        # Fast preprocessing
        if len(region_pixels) > 50000:
            masked_image = cv2.bilateralFilter(masked_image, 3, 15, 15)
            region_pixels = masked_image[analysis_mask > 0]
        
        # Primary detection method
        percentile = self.config['pore_detection']['intensity_percentile']
        percentile_threshold = np.percentile(region_pixels, percentile)
        detection_mask = (masked_image < percentile_threshold) & (analysis_mask > 0)
        
        # Fast morphological cleanup
        if self.config['performance']['use_simplified_morphology']:
            kernel = np.ones((2, 2), np.uint8)
            detection_mask = cv2.morphologyEx(detection_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(detection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Fast size filtering
        min_area = self.config['pore_detection']['min_pore_area_pixels']
        max_area = int(np.sum(analysis_mask > 0) * self.config['pore_detection']['max_pore_area_ratio'])
        
        areas = np.array([cv2.contourArea(c) for c in contours])
        size_mask = (areas >= min_area) & (areas <= max_area)
        filtered_contours = [contours[i] for i in np.where(size_mask)[0]]
        filtered_areas = areas[size_mask]
        
        # Performance limiting
        max_candidates = self.config['performance']['max_candidates_per_stage']
        if len(filtered_contours) > max_candidates:
            sorted_indices = np.argsort(filtered_areas)[::-1][:max_candidates]
            filtered_contours = [filtered_contours[i] for i in sorted_indices]
            filtered_areas = filtered_areas[sorted_indices]
        
        # Fast property calculation and validation
        validated_pores = []
        for contour, area in zip(filtered_contours, filtered_areas):
            pore_props = self._calculate_pore_properties_fast(contour, area, scale_factor)
            
            if self._validate_pore_fast(pore_props, masked_image, analysis_mask):
                validated_pores.append(pore_props)
        
        return validated_pores
    
    def _calculate_pore_properties_fast(self, contour: np.ndarray, 
                                       area_pixels: float, 
                                       scale_factor: float) -> Dict:
        """Fast calculation of essential pore properties."""
        
        # Essential geometric calculations
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        
        # Convert to real units
        area_um2 = area_pixels * (scale_factor ** 2)
        perimeter_um = perimeter * scale_factor
        equivalent_diameter_um = 2 * np.sqrt(area_um2 / np.pi)
        
        # Essential shape descriptors
        circularity = 4 * np.pi * area_pixels / (perimeter ** 2) if perimeter > 0 else 0
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1
        extent = area_pixels / (w * h) if w * h > 0 else 0
        
        # Solidity (only calculate for small pores where it matters)
        if area_um2 < 10:
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area_pixels / hull_area if hull_area > 0 else 0
        else:
            solidity = 0.8  # Reasonable default for larger pores
        
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
            'radius_um': radius * scale_factor,
        }
    
    def _validate_pore_oval_aware(self, pore_props: Dict, 
                                image: np.ndarray, 
                                analysis_mask: np.ndarray,
                                fiber_props: Dict) -> bool:
        """Enhanced pore validation considering fiber oval shape."""
        
        area_um2 = pore_props['area_um2']
        
        # Standard size-dependent shape validation
        if area_um2 < 5:
            if (pore_props['circularity'] < 0.08 or 
                pore_props['aspect_ratio'] > 6 or 
                pore_props['solidity'] < 0.25):
                return False
        elif area_um2 < 25:
            if (pore_props['circularity'] < self.config['quality_control']['circularity_threshold'] or 
                pore_props['aspect_ratio'] > self.config['quality_control']['aspect_ratio_threshold'] or 
                pore_props['solidity'] < self.config['quality_control']['solidity_threshold']):
                return False
        else:
            if (pore_props['circularity'] < 0.03 or 
                pore_props['aspect_ratio'] > 12 or 
                pore_props['solidity'] < 0.2):
                return False
        
        # Enhanced validation using fiber oval properties
        if fiber_props.get('oval_fitted', False):
            fiber_eccentricity = fiber_props.get('oval_eccentricity', 0)
            
            # For highly eccentric (elongated) fibers, allow more elongated pores
            if fiber_eccentricity > 0.6:
                max_pore_aspect_ratio = self.config['quality_control']['aspect_ratio_threshold'] * 1.5
                if pore_props['aspect_ratio'] > max_pore_aspect_ratio:
                    return False
            
            # Check if pore orientation is reasonable relative to fiber
            # (This is a simplified check - could be enhanced further)
            fiber_angle = fiber_props.get('oval_angle', 0)
            # For now, we accept all orientations, but this could be refined
        
        # Fast intensity validation for very small pores
        if (area_um2 < 8 and self.config['quality_control']['intensity_validation']):
            contour = pore_props['contour']
            pore_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(pore_mask, [contour], 255)
            
            pore_pixels = image[pore_mask > 0]
            if len(pore_pixels) > 0:
                pore_mean = np.mean(pore_pixels)
                if pore_mean > 85:
                    return False
        
        return True
    
    def _validate_pore_fast(self, pore_props: Dict, 
                           image: np.ndarray, 
                           analysis_mask: np.ndarray) -> bool:
        """Standard fast pore validation."""
        
        area_um2 = pore_props['area_um2']
        
        # Size-dependent shape validation
        if area_um2 < 5:
            if (pore_props['circularity'] < 0.08 or 
                pore_props['aspect_ratio'] > 6 or 
                pore_props['solidity'] < 0.25):
                return False
        elif area_um2 < 25:
            if (pore_props['circularity'] < self.config['quality_control']['circularity_threshold'] or 
                pore_props['aspect_ratio'] > self.config['quality_control']['aspect_ratio_threshold'] or 
                pore_props['solidity'] < self.config['quality_control']['solidity_threshold']):
                return False
        else:
            if (pore_props['circularity'] < 0.03 or 
                pore_props['aspect_ratio'] > 12 or 
                pore_props['solidity'] < 0.2):
                return False
        
        # Fast intensity validation for very small pores
        if (area_um2 < 8 and self.config['quality_control']['intensity_validation']):
            contour = pore_props['contour']
            pore_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(pore_mask, [contour], 255)
            
            pore_pixels = image[pore_mask > 0]
            if len(pore_pixels) > 0:
                pore_mean = np.mean(pore_pixels)
                if pore_mean > 85:
                    return False
        
        return True
    
    def _calculate_pore_oval_position(self, pore: Dict, fiber_props: Dict) -> Dict:
        """Calculate pore position relative to fiber oval."""
        
        if not fiber_props.get('oval_fitted', False):
            return {'position_calculated': False}
        
        try:
            # Get pore centroid
            pore_x, pore_y = pore['centroid_x'], pore['centroid_y']
            
            # Get fiber oval parameters
            fiber_center = fiber_props.get('oval_center', (0, 0))
            fiber_axes = fiber_props.get('oval_axes', (0, 0))
            fiber_angle = fiber_props.get('oval_angle', 0)
            
            # Calculate normalized position within oval
            # Translate to oval center
            rel_x = pore_x - fiber_center[0]
            rel_y = pore_y - fiber_center[1]
            
            # Rotate to align with oval axes
            angle_rad = np.radians(-fiber_angle)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            
            aligned_x = rel_x * cos_a - rel_y * sin_a
            aligned_y = rel_x * sin_a + rel_y * cos_a
            
            # Normalize by oval axes
            if fiber_axes[0] > 0 and fiber_axes[1] > 0:
                norm_x = aligned_x / (fiber_axes[0] / 2)
                norm_y = aligned_y / (fiber_axes[1] / 2)
                
                # Distance from center in normalized coordinates
                norm_distance = np.sqrt(norm_x**2 + norm_y**2)
                
                return {
                    'position_calculated': True,
                    'normalized_x': norm_x,
                    'normalized_y': norm_y,
                    'normalized_distance_from_center': norm_distance,
                    'angular_position_deg': np.degrees(np.arctan2(norm_y, norm_x)) % 360,
                    'radial_position_ratio': norm_distance  # 0 = center, 1 = edge
                }
            else:
                return {'position_calculated': False, 'error': 'Invalid oval axes'}
        
        except Exception as e:
            return {'position_calculated': False, 'error': str(e)}
    
    def _calculate_enhanced_porosity_metrics(self, pore_results: List[Dict], 
                                           scale_factor: float,
                                           fiber_analysis_data: Optional[Dict] = None) -> Dict:
        """Calculate enhanced porosity metrics with oval fitting awareness."""
        
        if not pore_results:
            return {
                'total_porosity_percent': 0.0,
                'pore_count': 0,
                'total_pore_area_um2': 0.0,
                'average_pore_size_um2': 0.0,
                'median_pore_size_um2': 0.0,
                'pore_density_per_mm2': 0.0,
                'method': 'enhanced_with_oval_fitting',
                'oval_aware_analysis': self.config['analysis']['oval_aware_analysis']
            }
        
        # Calculate total areas
        total_pore_area_pixels = sum(pore['area_pixels'] for pore in pore_results)
        total_pore_area_um2 = sum(pore['area_um2'] for pore in pore_results)
        
        # Enhanced fiber area calculation using oval fitting data
        total_fiber_area_pixels, total_fiber_area_um2 = self._calculate_enhanced_fiber_area(
            pore_results, scale_factor, fiber_analysis_data
        )
        
        # Calculate porosity
        porosity_percent = (total_pore_area_pixels / total_fiber_area_pixels * 100) if total_fiber_area_pixels > 0 else 0
        
        # Pore size statistics
        pore_areas = [pore['area_um2'] for pore in pore_results]
        pore_diameters = [pore['equivalent_diameter_um'] for pore in pore_results]
        
        # Enhanced metrics
        enhanced_metrics = {
            'total_porosity_percent': porosity_percent,
            'pore_count': len(pore_results),
            'total_pore_area_um2': total_pore_area_um2,
            'total_fiber_area_um2': total_fiber_area_um2,
            'average_pore_size_um2': np.mean(pore_areas),
            'median_pore_size_um2': np.median(pore_areas),
            'std_pore_size_um2': np.std(pore_areas),
            'min_pore_size_um2': np.min(pore_areas),
            'max_pore_size_um2': np.max(pore_areas),
            'mean_pore_diameter_um': np.mean(pore_diameters),
            'median_pore_diameter_um': np.median(pore_diameters),
            'pore_density_per_mm2': len(pore_results) / (total_fiber_area_um2 / 1e6) if total_fiber_area_um2 > 0 else 0,
            'method': 'enhanced_with_oval_fitting',
            'oval_aware_analysis': self.config['analysis']['oval_aware_analysis']
        }
        
        # Add oval-specific metrics if oval fitting was used
        oval_fitted_pores = [p for p in pore_results if p.get('fiber_oval_fitted', False)]
        if oval_fitted_pores:
            enhanced_metrics.update(self._calculate_oval_specific_metrics(oval_fitted_pores))
        
        return enhanced_metrics
    
    def _calculate_enhanced_fiber_area(self, pore_results: List[Dict], 
                                     scale_factor: float,
                                     fiber_analysis_data: Optional[Dict]) -> Tuple[float, float]:
        """Calculate fiber area using enhanced methods including oval fitting."""
        
        # Method 1: Use individual fiber areas from oval fitting if available
        if fiber_analysis_data and 'oval_fitting_summary' in fiber_analysis_data:
            oval_summary = fiber_analysis_data['oval_fitting_summary']
            if 'fiber_total_area_um2' in oval_summary:
                total_area_um2 = oval_summary['fiber_total_area_um2']
                total_area_pixels = total_area_um2 / (scale_factor ** 2)
                return total_area_pixels, total_area_um2
        
        # Method 2: Use individual fiber data from pore analysis
        if 'fiber_area_pixels' in pore_results[0]:
            unique_fiber_areas_pixels = list(set(pore['fiber_area_pixels'] for pore in pore_results))
            unique_fiber_areas_um2 = list(set(pore['fiber_area_um2'] for pore in pore_results))
            total_area_pixels = sum(unique_fiber_areas_pixels)
            total_area_um2 = sum(unique_fiber_areas_um2)
            return total_area_pixels, total_area_um2
        
        # Method 3: Estimate from pore locations (fallback)
        if pore_results:
            max_x = max(pore['centroid_x'] for pore in pore_results)
            max_y = max(pore['centroid_y'] for pore in pore_results)
            estimated_area_pixels = max_x * max_y * 0.8  # Rough estimate with 80% fill factor
            estimated_area_um2 = estimated_area_pixels * (scale_factor ** 2)
            return estimated_area_pixels, estimated_area_um2
        
        return 0.0, 0.0
    
    def _calculate_oval_specific_metrics(self, oval_fitted_pores: List[Dict]) -> Dict:
        """Calculate metrics specific to oval-fitted fibers."""
        
        metrics = {
            'oval_fitted_pores_count': len(oval_fitted_pores),
            'oval_fitted_pores_percentage': len(oval_fitted_pores) / len(self.pore_data) * 100 if self.pore_data else 0,
        }
        
        # Analyze pore positions within ovals
        radial_positions = []
        angular_positions = []
        
        for pore in oval_fitted_pores:
            pos_data = pore.get('pore_position_in_oval', {})
            if pos_data.get('position_calculated', False):
                radial_positions.append(pos_data.get('radial_position_ratio', 0))
                angular_positions.append(pos_data.get('angular_position_deg', 0))
        
        if radial_positions:
            metrics.update({
                'mean_pore_radial_position': np.mean(radial_positions),
                'std_pore_radial_position': np.std(radial_positions),
                'pores_near_center_percentage': len([r for r in radial_positions if r < 0.3]) / len(radial_positions) * 100,
                'pores_near_edge_percentage': len([r for r in radial_positions if r > 0.7]) / len(radial_positions) * 100,
            })
        
        if angular_positions and len(angular_positions) > 3:
            # Analyze angular distribution uniformity
            # Convert to radians for circular statistics
            angles_rad = np.array(angular_positions) * np.pi / 180
            
            # Calculate circular mean and variance
            sin_sum = np.sum(np.sin(angles_rad))
            cos_sum = np.sum(np.cos(angles_rad))
            
            circular_mean_rad = np.arctan2(sin_sum, cos_sum)
            circular_mean_deg = circular_mean_rad * 180 / np.pi % 360
            
            # Calculate circular variance (0 = uniform, 1 = all at same angle)
            r_length = np.sqrt(sin_sum**2 + cos_sum**2) / len(angles_rad)
            circular_variance = 1 - r_length
            
            metrics.update({
                'pore_angular_distribution_uniformity': 1 - circular_variance,  # 1 = uniform, 0 = clustered
                'pore_angular_mean_deg': circular_mean_deg,
            })
        
        return metrics
    
    def _analyze_size_distribution(self, pore_results: List[Dict]) -> Dict:
        """Enhanced size distribution analysis."""
        
        if not pore_results:
            return {'error': 'No pores to analyze'}
        
        pore_areas = [pore['area_um2'] for pore in pore_results]
        pore_diameters = [pore['equivalent_diameter_um'] for pore in pore_results]
        
        # Size categories
        categories = {
            'ultra_tiny': [p for p in pore_areas if p < 3],
            'tiny': [p for p in pore_areas if 3 <= p < 10],
            'small': [p for p in pore_areas if 10 <= p < 50],
            'medium': [p for p in pore_areas if 50 <= p < 200],
            'large': [p for p in pore_areas if 200 <= p < 500],
            'very_large': [p for p in pore_areas if p >= 500]
        }
        
        # Calculate statistics for each category
        distribution = {}
        total_pores = len(pore_areas)
        
        for category, pores in categories.items():
            count = len(pores)
            percentage = (count / total_pores * 100) if total_pores > 0 else 0
            total_area = sum(pores)
            
            distribution[category] = {
                'count': count,
                'percentage': percentage,
                'total_area_um2': total_area,
                'range_description': self._get_size_range_description(category)
            }
        
        result = {
            'size_categories': distribution,
            'total_pores': total_pores,
            'area_statistics': {
                'mean': np.mean(pore_areas),
                'median': np.median(pore_areas),
                'std': np.std(pore_areas),
                'min': np.min(pore_areas),
                'max': np.max(pore_areas),
                'percentiles': {
                    'p25': np.percentile(pore_areas, 25),
                    'p75': np.percentile(pore_areas, 75),
                    'p90': np.percentile(pore_areas, 90),
                    'p95': np.percentile(pore_areas, 95),
                }
            },
            'diameter_statistics': {
                'mean': np.mean(pore_diameters),
                'median': np.median(pore_diameters),
                'std': np.std(pore_diameters),
            }
        }
        
        # Add oval-specific analysis if available
        oval_fitted_pores = [p for p in pore_results if p.get('fiber_oval_fitted', False)]
        if oval_fitted_pores:
            oval_areas = [p['area_um2'] for p in oval_fitted_pores]
            result['oval_fitted_pore_statistics'] = {
                'count': len(oval_fitted_pores),
                'percentage_of_total': len(oval_fitted_pores) / total_pores * 100,
                'mean_area': np.mean(oval_areas) if oval_areas else 0,
                'median_area': np.median(oval_areas) if oval_areas else 0,
            }
        
        return result
    
    def _get_size_range_description(self, category: str) -> str:
        """Get human-readable size range description."""
        ranges = {
            'ultra_tiny': '< 3 μm²',
            'tiny': '3-10 μm²',
            'small': '10-50 μm²',
            'medium': '50-200 μm²',
            'large': '200-500 μm²',
            'very_large': '> 500 μm²'
        }
        return ranges.get(category, 'unknown')
    
    def _analyze_spatial_distribution(self, pore_results: List[Dict]) -> Dict:
        """Enhanced spatial distribution analysis."""
        
        if len(pore_results) < 2:
            return {'error': 'Insufficient pores for spatial analysis'}
        
        # Extract centroid coordinates
        centroids = np.array([[pore['centroid_x'], pore['centroid_y']] for pore in pore_results])
        
        try:
            # Calculate nearest neighbor distances
            distances = spatial.distance_matrix(centroids, centroids)
            np.fill_diagonal(distances, np.inf)
            nearest_distances = np.min(distances, axis=1)
            
            result = {
                'nearest_neighbor_distances': {
                    'mean': np.mean(nearest_distances),
                    'median': np.median(nearest_distances),
                    'std': np.std(nearest_distances),
                    'min': np.min(nearest_distances),
                    'max': np.max(nearest_distances),
                },
                'spatial_uniformity': 1.0 / (1.0 + np.std(nearest_distances) / np.mean(nearest_distances)) if np.mean(nearest_distances) > 0 else 0,
                'pore_coordinates': centroids.tolist(),
            }
            
            # Enhanced analysis for oval-fitted pores
            oval_fitted_pores = [p for p in pore_results if p.get('fiber_oval_fitted', False)]
            if len(oval_fitted_pores) > 2:
                # Analyze spatial distribution within oval coordinates
                radial_positions = []
                angular_positions = []
                
                for pore in oval_fitted_pores:
                    pos_data = pore.get('pore_position_in_oval', {})
                    if pos_data.get('position_calculated', False):
                        radial_positions.append(pos_data.get('radial_position_ratio', 0))
                        angular_positions.append(pos_data.get('angular_position_deg', 0))
                
                if len(radial_positions) > 1:
                    result['oval_spatial_analysis'] = {
                        'radial_distribution': {
                            'mean': np.mean(radial_positions),
                            'std': np.std(radial_positions),
                            'uniformity': 1.0 / (1.0 + np.std(radial_positions)) if np.std(radial_positions) > 0 else 1.0
                        }
                    }
                    
                    if len(angular_positions) > 2:
                        # Analyze angular clustering
                        angles_rad = np.array(angular_positions) * np.pi / 180
                        circular_variance = 1 - np.sqrt(np.sum(np.sin(angles_rad))**2 + np.sum(np.cos(angles_rad))**2) / len(angles_rad)
                        
                        result['oval_spatial_analysis']['angular_distribution'] = {
                            'circular_variance': circular_variance,
                            'uniformity': 1 - circular_variance
                        }
            
            return result
            
        except Exception as e:
            return {'error': f'Spatial analysis failed: {str(e)}'}
    
    def _extract_oval_fitting_context(self, fiber_analysis_data: Dict) -> Dict:
        """Extract oval fitting context for porosity analysis."""
        
        if not fiber_analysis_data:
            return {}
        
        oval_summary = fiber_analysis_data.get('oval_fitting_summary', {})
        individual_results = fiber_analysis_data.get('individual_results', [])
        
        context = {
            'total_fibers': len(individual_results),
            'fibers_with_ovals': oval_summary.get('fibers_successfully_fitted', 0),
            'oval_fitting_success_rate': oval_summary.get('fiber_fit_success_rate', 0),
            'average_fiber_diameter_um': oval_summary.get('fiber_avg_mean_diameter', 0),
            'average_fit_quality': oval_summary.get('fiber_avg_fit_quality', 0),
        }
        
        # Add lumen context if available
        if oval_summary.get('lumens_successfully_fitted', 0) > 0:
            context.update({
                'lumens_with_ovals': oval_summary.get('lumens_successfully_fitted', 0),
                'lumen_oval_fitting_success_rate': oval_summary.get('lumen_fitting_success_rate', 0),
                'average_lumen_diameter_um': oval_summary.get('lumen_avg_mean_diameter', 0),
            })
        
        return context
    
    def _assess_enhanced_analysis_quality(self, pore_results: List[Dict], 
                                        porosity_metrics: Dict,
                                        fiber_analysis_data: Optional[Dict]) -> Dict:
        """Enhanced quality assessment considering oval fitting data."""
        
        if not pore_results:
            return {
                'overall_quality': 'poor',
                'confidence': 0.0,
                'issues': ['No pores detected']
            }
        
        quality_score = 0.0
        issues = []
        
        # Pore count assessment
        pore_count = len(pore_results)
        if pore_count >= 100:
            quality_score += 0.35
        elif pore_count >= 50:
            quality_score += 0.25
        elif pore_count >= 20:
            quality_score += 0.15
        else:
            quality_score += 0.05
            issues.append('Low pore count may affect statistical reliability')
        
        # Size distribution assessment
        if pore_results:
            areas = [pore['area_um2'] for pore in pore_results]
            tiny_fraction = len([a for a in areas if a < 10]) / len(areas)
            
            if 0.3 <= tiny_fraction <= 0.8:
                quality_score += 0.2
            elif tiny_fraction < 0.9:
                quality_score += 0.1
            else:
                issues.append('Very high fraction of tiny pores - may include noise')
        
        # Porosity range assessment
        porosity = porosity_metrics.get('total_porosity_percent', 0)
        if 1 <= porosity <= 50:
            quality_score += 0.2
        elif porosity > 50:
            issues.append('Very high porosity - verify detection parameters')
        elif porosity < 0.5:
            issues.append('Very low porosity - may be under-detecting')
        
        # NEW: Oval fitting quality boost
        if fiber_analysis_data and self.config['fiber_integration']['use_oval_fitting_data']:
            oval_summary = fiber_analysis_data.get('oval_fitting_summary', {})
            oval_success_rate = oval_summary.get('fiber_fit_success_rate', 0)
            oval_quality = oval_summary.get('fiber_avg_fit_quality', 0)
            
            if oval_success_rate > 0.8 and oval_quality > 0.7:
                quality_score += 0.15
                quality_score = min(1.0, quality_score)  # Cap at 1.0
            elif oval_success_rate > 0.5:
                quality_score += 0.1
            else:
                issues.append('Low oval fitting success rate may affect accuracy')
        
        # Determine overall quality
        if quality_score >= 0.8:
            overall_quality = 'excellent'
        elif quality_score >= 0.6:
            overall_quality = 'good'
        elif quality_score >= 0.4:
            overall_quality = 'moderate'
        else:
            overall_quality = 'poor'
        
        # Count oval-fitted pores
        oval_fitted_count = len([p for p in pore_results if p.get('fiber_oval_fitted', False)])
        
        return {
            'overall_quality': overall_quality,
            'confidence': quality_score,
            'quality_score': quality_score,
            'issues': issues,
            'pore_count': pore_count,
            'tiny_pore_fraction': tiny_fraction if pore_results else 0,
            'oval_fitted_pores': oval_fitted_count,
            'oval_fitted_percentage': oval_fitted_count / pore_count * 100 if pore_count > 0 else 0,
        }
    
    def get_pore_dataframe(self) -> pd.DataFrame:
        """Convert enhanced pore data to pandas DataFrame for analysis."""
        if self.pore_data is None:
            return pd.DataFrame()
        
        # Flatten pore data for DataFrame including oval fitting data
        flattened_data = []
        for pore in self.pore_data:
            pore_row = pore.copy()
            
            # Remove non-serializable data
            if 'contour' in pore_row:
                del pore_row['contour']
            
            # Flatten oval position data
            if 'pore_position_in_oval' in pore_row:
                pos_data = pore_row.pop('pore_position_in_oval')
                for key, value in pos_data.items():
                    pore_row[f'oval_position_{key}'] = value
            
            flattened_data.append(pore_row)
        
        return pd.DataFrame(flattened_data)
    
    def export_enhanced_results(self, output_path: str):
        """Export enhanced analysis results to Excel file."""
        if not self.results:
            print("No results to export. Run analyze_fiber_porosity() first.")
            return
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Enhanced summary metrics
            summary_data = [self.results['porosity_metrics']]
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Enhanced_Summary', index=False)
            
            # Individual pore data with oval fitting information
            if self.pore_data:
                pore_df = self.get_pore_dataframe()
                pore_df.to_excel(writer, sheet_name='Enhanced_Pore_Data', index=False)
            
            # Size distribution
            if 'size_distribution' in self.results.get('analysis_results', {}):
                size_dist = self.results['analysis_results']['size_distribution']
                if 'size_categories' in size_dist:
                    size_data = []
                    for category, data in size_dist['size_categories'].items():
                        row = {'category': category}
                        row.update(data)
                        size_data.append(row)
                    size_df = pd.DataFrame(size_data)
                    size_df.to_excel(writer, sheet_name='Size_Distribution', index=False)
            
            # Enhanced quality assessment
            quality_df = pd.DataFrame([self.results['quality_assessment']])
            quality_df.to_excel(writer, sheet_name='Enhanced_Quality', index=False)
            
            # Oval fitting context
            if 'oval_fitting_context' in self.results:
                oval_context_df = pd.DataFrame([self.results['oval_fitting_context']])
                oval_context_df.to_excel(writer, sheet_name='Oval_Fitting_Context', index=False)
        
        print(f"Enhanced results exported to {output_path}")


# Enhanced convenience functions

def analyze_fiber_porosity(image: np.ndarray, 
                          fiber_mask: np.ndarray,
                          scale_factor: float = 1.0,
                          fiber_type: str = 'unknown',
                          fiber_analysis_data: Optional[Dict] = None,
                          config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Enhanced convenience function for porosity analysis with oval fitting integration.
    """
    analyzer = PorosityAnalyzer(config)
    return analyzer.analyze_fiber_porosity(
        image, fiber_mask, scale_factor, fiber_type, fiber_analysis_data
    )


def quick_porosity_check(image: np.ndarray, 
                        fiber_mask: np.ndarray,
                        scale_factor: float = 1.0) -> float:
    """
    Quick porosity percentage calculation.
    """
    config = {
        'analysis': {
            'calculate_size_distribution': False,
            'calculate_spatial_metrics': False,
            'detailed_reporting': False,
            'save_individual_pore_data': False,
            'oval_aware_analysis': False,
        },
        'performance': {
            'enable_timing': False,
        },
        'fiber_integration': {
            'use_oval_fitting_data': False,
        }
    }
    
    results = analyze_fiber_porosity(image, fiber_mask, scale_factor, config=config)
    return results.get('porosity_metrics', {}).get('total_porosity_percent', 0.0)


def visualize_enhanced_porosity_results(image: np.ndarray, 
                                       analysis_results: Dict, 
                                       figsize: Tuple[int, int] = (20, 12)):
    """
    Enhanced visualization of porosity analysis results including oval fitting data.
    """
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    axes = axes.flatten()
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original SEM Image', fontweight='bold')
    axes[0].axis('off')
    
    # Pore overlay with oval context
    if 'individual_pores' in analysis_results and analysis_results['individual_pores']:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        for pore in analysis_results['individual_pores']:
            if 'contour' in pore:
                area = pore['area_um2']
                
                # Color code by size and oval fitting status
                if pore.get('fiber_oval_fitted', False):
                    # Oval-fitted pores get special colors
                    if area < 10:
                        color = (0, 255, 128)     # Light green for tiny oval-fitted
                    elif area < 50:
                        color = (0, 255, 255)     # Cyan for small oval-fitted
                    elif area < 200:
                        color = (128, 255, 0)     # Yellow-green for medium oval-fitted
                    else:
                        color = (255, 128, 0)     # Orange for large oval-fitted
                else:
                    # Standard colors for non-oval-fitted
                    if area < 10:
                        color = (0, 255, 0)       # Green for tiny
                    elif area < 50:
                        color = (0, 255, 255)     # Yellow for small
                    elif area < 200:
                        color = (255, 165, 0)     # Orange for medium
                    else:
                        color = (255, 0, 0)       # Red for large
                
                cv2.drawContours(overlay, [pore['contour']], -1, color, 1)
        
        axes[1].imshow(overlay)
        axes[1].set_title('Enhanced Pore Detection\n(Bright=Oval-fitted, Std=Regular)', fontweight='bold')
    else:
        axes[1].imshow(image, cmap='gray')
        axes[1].set_title('No Pores Detected', fontweight='bold')
    axes[1].axis('off')
    
    # Enhanced summary metrics
    metrics = analysis_results.get('porosity_metrics', {})
    oval_context = analysis_results.get('oval_fitting_context', {})
    quality = analysis_results.get('quality_assessment', {})
    
    summary_text = f"ENHANCED POROSITY ANALYSIS\n\n"
    summary_text += f"Method: {metrics.get('method', 'unknown')}\n"
    summary_text += f"Total Porosity: {metrics.get('total_porosity_percent', 0):.2f}%\n"
    summary_text += f"Pore Count: {metrics.get('pore_count', 0)}\n"
    summary_text += f"Average Pore Size: {metrics.get('average_pore_size_um2', 0):.1f} μm²\n"
    summary_text += f"Pore Density: {metrics.get('pore_density_per_mm2', 0):.0f} pores/mm²\n\n"
    
    if oval_context:
        summary_text += f"Oval Fitting Context:\n"
        summary_text += f"Fibers with Ovals: {oval_context.get('fibers_with_ovals', 0)}\n"
        summary_text += f"Oval Success Rate: {oval_context.get('oval_fitting_success_rate', 0):.1%}\n"
        summary_text += f"Avg Fiber Diameter: {oval_context.get('average_fiber_diameter_um', 0):.1f} μm\n\n"
    
    summary_text += f"Quality: {quality.get('overall_quality', 'unknown')}\n"
    summary_text += f"Confidence: {quality.get('confidence', 0):.2f}\n"
    if 'oval_fitted_pores' in quality:
        summary_text += f"Oval-fitted Pores: {quality['oval_fitted_pores']} ({quality.get('oval_fitted_percentage', 0):.1f}%)"
    
    axes[2].text(0.05, 0.95, summary_text, transform=axes[2].transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue'))
    axes[2].set_title('Enhanced Analysis Summary', fontweight='bold')
    axes[2].axis('off')
    
    # Size distribution histogram
    if 'analysis_results' in analysis_results and 'size_distribution' in analysis_results['analysis_results']:
        size_dist = analysis_results['analysis_results']['size_distribution']
        if 'size_categories' in size_dist:
            categories = []
            counts = []
            for category, data in size_dist['size_categories'].items():
                if data['count'] > 0:
                    categories.append(category.replace('_', ' ').title())
                    counts.append(data['count'])
            
            if categories:
                bars = axes[3].bar(categories, counts, alpha=0.7, color='skyblue', edgecolor='navy')
                axes[3].set_title('Pore Size Distribution', fontweight='bold')
                axes[3].set_xlabel('Size Category')
                axes[3].set_ylabel('Count')
                axes[3].tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, count in zip(bars, counts):
                    axes[3].text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(counts)*0.01,
                               f'{count}', ha='center', va='bottom', fontsize=8)
            else:
                axes[3].text(0.5, 0.5, 'No size distribution data', 
                           transform=axes[3].transAxes, ha='center', va='center')
                axes[3].set_title('Size Distribution', fontweight='bold')
        axes[3].set_title('Size Distribution', fontweight='bold')
    else:
        axes[3].axis('off')
    
    # Oval fitting statistics
    if oval_context and oval_context.get('fibers_with_ovals', 0) > 0:
        oval_data = [
            ['Total Fibers', oval_context.get('total_fibers', 0)],
            ['Oval-fitted Fibers', oval_context.get('fibers_with_ovals', 0)],
            ['Success Rate', f"{oval_context.get('oval_fitting_success_rate', 0):.1%}"],
            ['Avg Fit Quality', f"{oval_context.get('average_fit_quality', 0):.2f}"],
            ['Avg Diameter', f"{oval_context.get('average_fiber_diameter_um', 0):.1f} μm"],
        ]
        
        table_text = "OVAL FITTING STATISTICS\n\n"
        for label, value in oval_data:
            table_text += f"{label}: {value}\n"
        
        axes[4].text(0.05, 0.95, table_text, transform=axes[4].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen'))
        axes[4].set_title('Oval Fitting Stats', fontweight='bold')
        axes[4].axis('off')
    else:
        axes[4].text(0.5, 0.5, 'No oval fitting\ndata available', 
                    ha='center', va='center', transform=axes[4].transAxes)
        axes[4].set_title('Oval Fitting Stats', fontweight='bold')
        axes[4].axis('off')
    
    # Processing information
    processing_text = f"ENHANCED PROCESSING INFO\n\n"
    processing_text += f"Scale Factor: {analysis_results.get('scale_factor', 0):.4f} μm/pixel\n"
    processing_text += f"Fiber Type: {analysis_results.get('fiber_type', 'unknown')}\n"
    processing_text += f"Oval Integration: {analysis_results.get('oval_fitting_used', False)}\n"
    processing_text += f"Processing Time: {analysis_results.get('processing_time', 0):.3f}s\n\n"
    
    if 'quality_assessment' in analysis_results:
        qa = analysis_results['quality_assessment']
        processing_text += f"Quality Issues:\n"
        issues = qa.get('issues', [])
        if issues:
            for issue in issues[:2]:  # Show first 2 issues
                processing_text += f"• {issue[:40]}...\n" if len(issue) > 40 else f"• {issue}\n"
        else:
            processing_text += "• No issues detected\n"
    
    axes[5].text(0.05, 0.95, processing_text, transform=axes[5].transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow'))
    axes[5].set_title('Processing Info', fontweight='bold')
    axes[5].axis('off')
    
    # Spatial analysis if available
    if 'analysis_results' in analysis_results and 'spatial_analysis' in analysis_results['analysis_results']:
        spatial = analysis_results['analysis_results']['spatial_analysis']
        if 'pore_coordinates' in spatial:
            coords = np.array(spatial['pore_coordinates'])
            scatter = axes[6].scatter(coords[:, 0], coords[:, 1], 
                                    c=range(len(coords)), cmap='viridis', 
                                    alpha=0.6, s=30)
            axes[6].set_title('Pore Spatial Distribution', fontweight='bold')
            axes[6].set_xlabel('X (pixels)')
            axes[6].set_ylabel('Y (pixels)')
            axes[6].set_aspect('equal')
        else:
            axes[6].text(0.5, 0.5, 'Spatial analysis\nnot available', 
                        ha='center', va='center', transform=axes[6].transAxes)
            axes[6].set_title('Spatial Distribution', fontweight='bold')
            axes[6].axis('off')
    else:
        axes[6].axis('off')
    
    # Quality factors
    if quality and 'issues' in quality:
        quality_text = f"QUALITY ASSESSMENT\n\n"
        quality_text += f"Overall: {quality.get('overall_quality', 'unknown').title()}\n"
        quality_text += f"Score: {quality.get('quality_score', 0):.2f}/1.0\n\n"
        
        issues = quality.get('issues', [])
        if issues:
            quality_text += "Issues:\n"
            for issue in issues[:3]:  # Show max 3 issues
                quality_text += f"• {issue[:35]}...\n" if len(issue) > 35 else f"• {issue}\n"
        else:
            quality_text += "✅ No quality issues detected"
        
        axes[7].text(0.05, 0.95, quality_text, transform=axes[7].transAxes,
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcoral'))
        axes[7].set_title('Quality Assessment', fontweight='bold')
        axes[7].axis('off')
    else:
        axes[7].axis('off')
    
    plt.suptitle(f"Enhanced Porosity Analysis with Oval Fitting Integration", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()