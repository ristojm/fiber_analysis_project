#!/usr/bin/env python3
"""
Enhanced Multi-Processing Comprehensive SEM Fiber Analyzer
UPDATED: Added comprehensive oval fitting analysis for fiber diameter measurements
FIXED: Centralized results output using results_config.py

Features:
- Parallel batch processing with multiprocessing
- Enhanced fiber diameter measurements via oval fitting
- Detailed pore analysis (size distribution, shape, spatial)
- Comprehensive fiber measurements (diameter, wall thickness, lumen)
- Real-time progress tracking
- Comprehensive Excel reporting with 100+ measurements including oval fitting data
- Memory-efficient processing
- FIXED: All outputs now managed by centralized results_config.py
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import time
from datetime import datetime
import json
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import gc
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Setup paths for module imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "modules"))

# Import centralized results configuration
try:
    from results_config import (
        get_multiprocessing_path, get_excel_report_path, get_json_results_path,
        MULTIPROCESSING_DIR, get_results_info, print_results_structure
    )
    RESULTS_CONFIGURED = True
    print("✅ Centralized results configuration loaded")
except ImportError as e:
    # Fallback if results_config.py doesn't exist
    RESULTS_CONFIGURED = False
    MULTIPROCESSING_DIR = Path("results") / "multiprocessing_results"
    MULTIPROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    
    def get_multiprocessing_path(filename: str) -> Path:
        return MULTIPROCESSING_DIR / filename
    
    def get_excel_report_path(report_type: str = "COMPREHENSIVE_ANALYSIS") -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return MULTIPROCESSING_DIR / f"{report_type}_{timestamp}.xlsx"
    
    def get_json_results_path(result_type: str = "batch_results") -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return MULTIPROCESSING_DIR / f"{result_type}_{timestamp}.json"
    
    print("⚠️ Using fallback results configuration")

# Import analysis modules
try:
    from modules.scale_detection import ScaleBarDetector
    from modules.fiber_type_detection import FiberTypeDetector
    from modules.image_preprocessing import load_image
    from modules.porosity_analysis import PorosityAnalyzer
    print("✅ All analysis modules loaded successfully")
except ImportError as e:
    print(f"❌ Could not import modules: {e}")
    sys.exit(1)

def process_single_image_worker(image_info: Dict) -> Dict:
    """
    Enhanced worker function for processing a single image with oval fitting analysis.
    """
    image_path = image_info['image_path']
    config = image_info.get('config', {})
    
    start_time = time.time()
    process_id = os.getpid()
    
    result = {
        'image_path': str(image_path),
        'image_name': Path(image_path).name,
        'process_id': process_id,
        'analysis_timestamp': datetime.now().isoformat(),
        'success': False,
        'total_processing_time': 0.0
    }
    
    try:
        # Monitor memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Step 1: Load image
        image = load_image(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Basic preprocessing
        preprocessed = _preprocess_for_analysis(image)
        
        result.update({
            'image_shape': image.shape,
            'image_size_mb': os.path.getsize(image_path) / (1024 * 1024)
        })
        
        # Step 2: Scale detection
        scale_detector = ScaleBarDetector(use_enhanced_detection=True)
        scale_result = scale_detector.detect_scale_bar(
            image, debug=False, save_debug_image=False, output_dir=None
        )
        
        scale_factor = scale_result['micrometers_per_pixel'] if scale_result['scale_detected'] else 1.0
        result['scale_detection'] = scale_result
        
        # Step 3: Enhanced fiber type detection with oval fitting
        fiber_detector = FiberTypeDetector()
        fiber_type, fiber_confidence, fiber_analysis_data = fiber_detector.classify_fiber_type(preprocessed, scale_factor)
        
        result['fiber_detection'] = {
            'fiber_type': fiber_type,
            'confidence': fiber_confidence,
            'total_fibers': fiber_analysis_data.get('total_fibers', 0),
            'hollow_fibers': fiber_analysis_data.get('hollow_fibers', 0),
            'filaments': fiber_analysis_data.get('filaments', 0),
            'oval_fitting_summary': fiber_analysis_data.get('oval_fitting_summary', {})
        }
        
        # Step 4: Porosity analysis
        fiber_mask = fiber_analysis_data.get('fiber_mask', np.zeros_like(image, dtype=np.uint8))
        
        if np.sum(fiber_mask > 0) > 1000:
            try:
                porosity_config = {
                    'analysis': {
                        'calculate_size_distribution': True,
                        'calculate_spatial_metrics': True,
                        'detailed_reporting': True,
                        'save_individual_pore_data': True
                    }
                }
                
                porosity_analyzer = PorosityAnalyzer(config=porosity_config)
                porosity_result = porosity_analyzer.analyze_fiber_porosity(
                    preprocessed, fiber_mask, scale_factor, fiber_type, fiber_analysis_data
                )
                
                result['porosity_analysis'] = porosity_result
                
            except Exception as e:
                result['porosity_analysis'] = {'error': str(e)}
        else:
            result['porosity_analysis'] = {'error': 'Insufficient fiber area'}
        
        # Step 5: Extract enhanced detailed measurements including oval fitting
        detailed_analysis = extract_enhanced_detailed_measurements(
            result.get('porosity_analysis', {}),
            fiber_analysis_data,
            scale_factor
        )
        
        result['detailed_measurements'] = detailed_analysis
        
        # Step 6: Calculate enhanced quality metrics
        quality_metrics = calculate_enhanced_quality_metrics(result, scale_factor)
        result['quality_metrics'] = quality_metrics
        
        # Final memory check
        final_memory = process.memory_info().rss / 1024 / 1024
        result['memory_usage_mb'] = final_memory - initial_memory
        
        result['success'] = True
        result['total_processing_time'] = time.time() - start_time
        
        # Cleanup
        del image, preprocessed, fiber_mask
        gc.collect()
        
    except Exception as e:
        result['error'] = str(e)
        result['total_processing_time'] = time.time() - start_time
    
    return result

def _preprocess_for_analysis(image: np.ndarray) -> np.ndarray:
    """Minimal preprocessing for analysis."""
    denoised = cv2.bilateralFilter(image, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    return enhanced

def extract_enhanced_detailed_measurements(porosity_result: Dict, fiber_analysis_data: Dict, scale_factor: float) -> Dict:
    """
    Extract comprehensive measurements including oval fitting data.
    """
    
    measurements = {
        'pore_analysis': {},
        'fiber_analysis': {},
        'lumen_analysis': {},
        'oval_fitting_analysis': {}  # NEW: Dedicated oval fitting section
    }
    
    # === PORE ANALYSIS ===
    if porosity_result and 'individual_pores' in porosity_result:
        pores = porosity_result['individual_pores']
        
        if pores:
            # Extract pore measurements
            pore_areas = [pore.get('area_um2', 0) for pore in pores]
            pore_diameters = [pore.get('equivalent_diameter_um', 0) for pore in pores]
            pore_circularities = [pore.get('circularity', 0) for pore in pores]
            pore_aspect_ratios = [pore.get('aspect_ratio', 1) for pore in pores]
            
            # Size categories
            size_categories = {
                'nano_pores': len([a for a in pore_areas if a < 1]),
                'micro_pores': len([a for a in pore_areas if 1 <= a < 10]),
                'small_pores': len([a for a in pore_areas if 10 <= a < 50]),
                'medium_pores': len([a for a in pore_areas if 50 <= a < 200]),
                'large_pores': len([a for a in pore_areas if 200 <= a < 500]),
                'macro_pores': len([a for a in pore_areas if a >= 500])
            }
            
            # Calculate percentages
            total_pores = len(pores)
            size_percentages = {k: (v/total_pores*100) if total_pores > 0 else 0 
                              for k, v in size_categories.items()}
            
            measurements['pore_analysis'] = {
                'total_count': total_pores,
                'mean_diameter_um': np.mean(pore_diameters) if pore_diameters else 0,
                'median_diameter_um': np.median(pore_diameters) if pore_diameters else 0,
                'std_diameter_um': np.std(pore_diameters) if pore_diameters else 0,
                'min_diameter_um': np.min(pore_diameters) if pore_diameters else 0,
                'max_diameter_um': np.max(pore_diameters) if pore_diameters else 0,
                'mean_area_um2': np.mean(pore_areas) if pore_areas else 0,
                'total_area_um2': np.sum(pore_areas) if pore_areas else 0,
                'mean_circularity': np.mean(pore_circularities) if pore_circularities else 0,
                'mean_aspect_ratio': np.mean(pore_aspect_ratios) if pore_aspect_ratios else 0,
                'elongated_pores': len([ar for ar in pore_aspect_ratios if ar > 2.0]),
                'round_pores': len([ar for ar in pore_aspect_ratios if ar <= 1.5]),
                **size_categories,
                **{f"{k}_percent": v for k, v in size_percentages.items()}
            }
            
            # Spatial analysis if enough pores
            if len(pores) >= 2:
                try:
                    centroids = np.array([[pore.get('centroid_x', 0), pore.get('centroid_y', 0)] for pore in pores])
                    if len(centroids) > 1:
                        distances = []
                        for i, cent1 in enumerate(centroids):
                            min_dist = float('inf')
                            for j, cent2 in enumerate(centroids):
                                if i != j:
                                    dist = np.sqrt(np.sum((cent1 - cent2)**2)) * scale_factor
                                    min_dist = min(min_dist, dist)
                            if min_dist != float('inf'):
                                distances.append(min_dist)
                        
                        if distances:
                            measurements['pore_analysis']['spatial_analysis'] = {
                                'mean_nearest_neighbor_um': np.mean(distances),
                                'std_nearest_neighbor_um': np.std(distances),
                                'distribution_uniformity': 'uniform' if np.std(distances)/np.mean(distances) < 0.5 else 'clustered'
                            }
                except:
                    pass
    
    # === ENHANCED FIBER ANALYSIS WITH OVAL FITTING ===
    if fiber_analysis_data and 'individual_results' in fiber_analysis_data:
        individual_results = fiber_analysis_data['individual_results']
        
        if individual_results:
            # Extract traditional fiber measurements
            fiber_areas_um2 = []
            fiber_diameters_um = []
            fiber_circularities = []
            fiber_aspect_ratios = []
            lumen_diameters_um = []
            wall_thicknesses_um = []
            
            # NEW: Extract oval fitting measurements
            oval_fitted_count = 0
            oval_mean_diameters = []
            oval_major_diameters = []
            oval_minor_diameters = []
            oval_eccentricities = []
            oval_fit_qualities = []
            oval_areas = []
            
            lumen_oval_fitted_count = 0
            lumen_oval_diameters = []
            lumen_oval_eccentricities = []
            lumen_oval_qualities = []
            
            for result in individual_results:
                fiber_props = result.get('fiber_properties', {})
                area_pixels = fiber_props.get('area', 0)
                
                if area_pixels > 0:
                    area_um2 = area_pixels * (scale_factor ** 2)
                    diameter_um = 2 * np.sqrt(area_um2 / np.pi)
                    
                    fiber_areas_um2.append(area_um2)
                    fiber_diameters_um.append(diameter_um)
                    fiber_circularities.append(fiber_props.get('circularity', 0))
                    fiber_aspect_ratios.append(fiber_props.get('aspect_ratio', 1))
                    
                    # NEW: Extract oval fitting data for fibers
                    if fiber_props.get('oval_fitted', False):
                        oval_fitted_count += 1
                        
                        # Use micrometers measurements if available, otherwise convert
                        if 'oval_mean_diameter_um' in fiber_props:
                            oval_mean_diameters.append(fiber_props['oval_mean_diameter_um'])
                            oval_major_diameters.append(fiber_props.get('oval_major_diameter_um', 0))
                            oval_minor_diameters.append(fiber_props.get('oval_minor_diameter_um', 0))
                            oval_areas.append(fiber_props.get('oval_area_um2', 0))
                        else:
                            # Fallback: convert from pixels
                            oval_mean_diameters.append(fiber_props.get('oval_mean_diameter', 0) * scale_factor)
                            oval_major_diameters.append(fiber_props.get('oval_major_diameter', 0) * scale_factor)
                            oval_minor_diameters.append(fiber_props.get('oval_minor_diameter', 0) * scale_factor)
                            oval_areas.append(fiber_props.get('oval_area', 0) * (scale_factor ** 2))
                        
                        oval_eccentricities.append(fiber_props.get('oval_eccentricity', 0))
                        oval_fit_qualities.append(fiber_props.get('oval_fit_quality', 0))
                
                # Enhanced lumen measurements with oval fitting
                if result.get('has_lumen', False):
                    lumen_props = result.get('lumen_properties', {})
                    lumen_area_pixels = lumen_props.get('area', 0)
                    
                    if lumen_area_pixels > 0:
                        lumen_area_um2 = lumen_area_pixels * (scale_factor ** 2)
                        lumen_diameter_um = 2 * np.sqrt(lumen_area_um2 / np.pi)
                        lumen_diameters_um.append(lumen_diameter_um)
                        
                        # NEW: Extract lumen oval fitting data
                        if lumen_props.get('oval_fitted', False):
                            lumen_oval_fitted_count += 1
                            
                            # Use micrometers measurements if available
                            if 'oval_mean_diameter_um' in lumen_props:
                                lumen_oval_diameters.append(lumen_props['oval_mean_diameter_um'])
                            else:
                                lumen_oval_diameters.append(lumen_props.get('oval_mean_diameter', 0) * scale_factor)
                            
                            lumen_oval_eccentricities.append(lumen_props.get('oval_eccentricity', 0))
                            lumen_oval_qualities.append(lumen_props.get('oval_fit_quality', 0))
                        
                        # Calculate wall thickness (enhanced with oval fitting if available)
                        if area_pixels > 0:
                            if (fiber_props.get('oval_fitted', False) and 
                                lumen_props.get('oval_fitted', False)):
                                # Use oval fitting for more accurate wall thickness
                                if 'oval_mean_diameter_um' in fiber_props:
                                    fiber_radius_um = fiber_props['oval_mean_diameter_um'] / 2
                                else:
                                    fiber_radius_um = fiber_props.get('oval_mean_diameter', 0) * scale_factor / 2
                                
                                if 'oval_mean_diameter_um' in lumen_props:
                                    lumen_radius_um = lumen_props['oval_mean_diameter_um'] / 2
                                else:
                                    lumen_radius_um = lumen_props.get('oval_mean_diameter', 0) * scale_factor / 2
                            else:
                                # Fallback to circular approximation
                                fiber_radius_um = diameter_um / 2
                                lumen_radius_um = lumen_diameter_um / 2
                            
                            wall_thickness = fiber_radius_um - lumen_radius_um
                            if wall_thickness > 0:
                                wall_thicknesses_um.append(wall_thickness)
            
            # Fiber size categories
            fiber_categories = {
                'ultra_fine_fibers': len([d for d in fiber_diameters_um if d < 10]),
                'fine_fibers': len([d for d in fiber_diameters_um if 10 <= d < 50]),
                'medium_fibers': len([d for d in fiber_diameters_um if 50 <= d < 100]),
                'coarse_fibers': len([d for d in fiber_diameters_um if 100 <= d < 200]),
                'very_coarse_fibers': len([d for d in fiber_diameters_um if d >= 200])
            }
            
            measurements['fiber_analysis'] = {
                'total_count': len(individual_results),
                'mean_diameter_um': np.mean(fiber_diameters_um) if fiber_diameters_um else 0,
                'median_diameter_um': np.median(fiber_diameters_um) if fiber_diameters_um else 0,
                'std_diameter_um': np.std(fiber_diameters_um) if fiber_diameters_um else 0,
                'min_diameter_um': np.min(fiber_diameters_um) if fiber_diameters_um else 0,
                'max_diameter_um': np.max(fiber_diameters_um) if fiber_diameters_um else 0,
                'diameter_cv': np.std(fiber_diameters_um)/np.mean(fiber_diameters_um) if fiber_diameters_um and np.mean(fiber_diameters_um) > 0 else 0,
                'mean_area_um2': np.mean(fiber_areas_um2) if fiber_areas_um2 else 0,
                'total_area_um2': np.sum(fiber_areas_um2) if fiber_areas_um2 else 0,
                'mean_circularity': np.mean(fiber_circularities) if fiber_circularities else 0,
                'mean_aspect_ratio': np.mean(fiber_aspect_ratios) if fiber_aspect_ratios else 0,
                'elongated_fibers': len([ar for ar in fiber_aspect_ratios if ar > 2.0]),
                **fiber_categories
            }
            
            # NEW: Dedicated oval fitting analysis
            measurements['oval_fitting_analysis'] = {
                'fibers_total_analyzed': len(individual_results),
                'fibers_successfully_fitted': oval_fitted_count,
                'fiber_oval_success_rate': oval_fitted_count / len(individual_results) if individual_results else 0,
                'lumens_total_analyzed': len([r for r in individual_results if r.get('has_lumen', False)]),
                'lumens_successfully_fitted': lumen_oval_fitted_count,
                'lumen_oval_success_rate': lumen_oval_fitted_count / len([r for r in individual_results if r.get('has_lumen', False)]) if len([r for r in individual_results if r.get('has_lumen', False)]) > 0 else 0,
            }
            
            # Fiber oval fitting statistics
            if oval_mean_diameters:
                measurements['oval_fitting_analysis'].update({
                    'fiber_oval_mean_diameter_um': np.mean(oval_mean_diameters),
                    'fiber_oval_median_diameter_um': np.median(oval_mean_diameters),
                    'fiber_oval_std_diameter_um': np.std(oval_mean_diameters),
                    'fiber_oval_min_diameter_um': np.min(oval_mean_diameters),
                    'fiber_oval_max_diameter_um': np.max(oval_mean_diameters),
                    'fiber_oval_major_mean_um': np.mean(oval_major_diameters),
                    'fiber_oval_minor_mean_um': np.mean(oval_minor_diameters),
                    'fiber_oval_mean_eccentricity': np.mean(oval_eccentricities),
                    'fiber_oval_mean_fit_quality': np.mean(oval_fit_qualities),
                    'fiber_oval_mean_area_um2': np.mean(oval_areas),
                    'fiber_oval_diameter_cv': np.std(oval_mean_diameters) / np.mean(oval_mean_diameters) if np.mean(oval_mean_diameters) > 0 else 0,
                })
                
                # Oval-based fiber size categories
                oval_categories = {
                    'oval_ultra_fine_fibers': len([d for d in oval_mean_diameters if d < 10]),
                    'oval_fine_fibers': len([d for d in oval_mean_diameters if 10 <= d < 50]),
                    'oval_medium_fibers': len([d for d in oval_mean_diameters if 50 <= d < 100]),
                    'oval_coarse_fibers': len([d for d in oval_mean_diameters if 100 <= d < 200]),
                    'oval_very_coarse_fibers': len([d for d in oval_mean_diameters if d >= 200])
                }
                measurements['oval_fitting_analysis'].update(oval_categories)
            
            # Lumen oval fitting statistics
            if lumen_oval_diameters:
                measurements['oval_fitting_analysis'].update({
                    'lumen_oval_mean_diameter_um': np.mean(lumen_oval_diameters),
                    'lumen_oval_std_diameter_um': np.std(lumen_oval_diameters),
                    'lumen_oval_mean_eccentricity': np.mean(lumen_oval_eccentricities),
                    'lumen_oval_mean_fit_quality': np.mean(lumen_oval_qualities),
                })
            
            # Enhanced lumen analysis
            if lumen_diameters_um:
                measurements['lumen_analysis'] = {
                    'has_lumen_data': True,
                    'mean_lumen_diameter_um': np.mean(lumen_diameters_um),
                    'median_lumen_diameter_um': np.median(lumen_diameters_um),
                    'std_lumen_diameter_um': np.std(lumen_diameters_um),
                    'mean_wall_thickness_um': np.mean(wall_thicknesses_um) if wall_thicknesses_um else 0,
                    'median_wall_thickness_um': np.median(wall_thicknesses_um) if wall_thicknesses_um else 0,
                    'std_wall_thickness_um': np.std(wall_thicknesses_um) if wall_thicknesses_um else 0,
                    'lumen_count': len(lumen_diameters_um),
                    # NEW: Wall thickness to fiber diameter ratios
                    'mean_wall_to_fiber_ratio': np.mean([wt / fd for wt, fd in zip(wall_thicknesses_um, fiber_diameters_um) if fd > 0]) if wall_thicknesses_um and fiber_diameters_um else 0,
                    'mean_lumen_to_fiber_ratio': np.mean([ld / fd for ld, fd in zip(lumen_diameters_um, fiber_diameters_um) if fd > 0]) if lumen_diameters_um and fiber_diameters_um else 0,
                }
            else:
                measurements['lumen_analysis'] = {'has_lumen_data': False}
    
    return measurements

def calculate_enhanced_quality_metrics(result: Dict, scale_factor: float) -> Dict:
    """Calculate enhanced quality metrics including oval fitting assessment."""
    
    quality_score = 0.0
    quality_factors = []
    
    # Scale detection quality (20%)
    scale_result = result.get('scale_detection', {})
    if scale_result.get('scale_detected', False):
        scale_conf = scale_result.get('confidence', 0.0)
        quality_score += scale_conf * 0.20
        quality_factors.append(f"Scale: {scale_conf:.2f}")
    else:
        quality_factors.append("Scale: failed")
    
    # Fiber detection quality (25%)
    fiber_result = result.get('fiber_detection', {})
    fiber_conf = fiber_result.get('confidence', 0.0)
    quality_score += fiber_conf * 0.25
    quality_factors.append(f"Fiber: {fiber_conf:.2f}")
    
    # NEW: Oval fitting quality (15%)
    oval_summary = fiber_result.get('oval_fitting_summary', {})
    oval_success_rate = oval_summary.get('fiber_fit_success_rate', 0.0)
    oval_quality = oval_summary.get('fiber_avg_fit_quality', 0.0)
    oval_combined = (oval_success_rate + oval_quality) / 2
    quality_score += oval_combined * 0.15
    quality_factors.append(f"Oval Fitting: {oval_combined:.2f}")
    
    # Porosity analysis quality (40%)
    porosity_result = result.get('porosity_analysis', {})
    if porosity_result and 'porosity_metrics' in porosity_result:
        pm = porosity_result['porosity_metrics']
        pore_count = pm.get('pore_count', 0)
        
        if pore_count >= 100:
            porosity_quality = 1.0
        elif pore_count >= 50:
            porosity_quality = 0.9
        elif pore_count >= 20:
            porosity_quality = 0.8
        elif pore_count >= 10:
            porosity_quality = 0.6
        elif pore_count > 0:
            porosity_quality = 0.4
        else:
            porosity_quality = 0.0
        
        quality_score += porosity_quality * 0.40
        quality_factors.append(f"Porosity: {porosity_quality:.2f}")
    else:
        quality_factors.append("Porosity: unavailable")
    
    # Determine overall quality level
    if quality_score >= 0.85:
        quality_level = "excellent"
    elif quality_score >= 0.70:
        quality_level = "good"
    elif quality_score >= 0.50:
        quality_level = "moderate"
    elif quality_score >= 0.30:
        quality_level = "poor"
    else:
        quality_level = "very_poor"
    
    return {
        'overall_quality': quality_level,
        'quality_score': quality_score,
        'quality_factors': quality_factors,
        'oval_fitting_quality': oval_combined,
        'oval_success_rate': oval_success_rate,
        'oval_fit_quality': oval_quality
    }

class MultiProcessingFiberAnalyzer:
    """Enhanced multi-processing SEM fiber analyzer with oval fitting capabilities."""
    
    def __init__(self, num_processes: Optional[int] = None):
        """Initialize the enhanced multi-processing analyzer."""
        
        if num_processes is None:
            self.num_processes = max(1, mp.cpu_count() - 1)
        else:
            self.num_processes = max(1, min(num_processes, mp.cpu_count()))
        
        print(f"🚀 Enhanced Multi-Processing Analyzer initialized")
        print(f"   CPU cores available: {mp.cpu_count()}")
        print(f"   Processes to use: {self.num_processes}")
        print(f"   Features: Oval fitting, enhanced measurements")
        if RESULTS_CONFIGURED:
            print(f"   Results will be saved to: {MULTIPROCESSING_DIR}")
        else:
            print(f"   Results will be saved to: {MULTIPROCESSING_DIR} (fallback)")
    
    def analyze_batch_parallel(self, image_directory: str, 
                              output_dir: Optional[str] = None,
                              max_images: Optional[int] = None) -> Dict:
        """Perform parallel batch analysis with oval fitting."""
        
        print(f"\n🚀 PARALLEL BATCH ANALYSIS WITH OVAL FITTING")
        print("=" * 70)
        
        # Setup directories - FIXED: Use centralized results management
        image_dir = Path(image_directory)
        if not image_dir.exists():
            return {'error': f'Directory not found: {image_dir}'}
        
        if output_dir is None:
            output_dir = MULTIPROCESSING_DIR  # ✅ Use centralized results folder
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(image_dir.glob(f'*{ext}'))
            image_files.extend(image_dir.glob(f'*{ext.upper()}'))
        
        image_files = sorted(set(image_files))
        
        if max_images:
            image_files = image_files[:max_images]
        
        if not image_files:
            return {'error': f'No images found in {image_dir}'}
        
        print(f"📁 Processing {len(image_files)} images")
        print(f"🔧 Using {self.num_processes} processes")
        print(f"📊 Enhanced with oval fitting analysis")
        print(f"💾 Results will be saved to: {output_dir}")
        
        # Prepare worker arguments
        worker_args = [{'image_path': str(img_path), 'config': {}} for img_path in image_files]
        
        # Process images in parallel
        results = []
        start_time = time.time()
        
        print(f"\n⚡ Starting parallel processing...")
        
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            future_to_image = {
                executor.submit(process_single_image_worker, arg): arg['image_path'] 
                for arg in worker_args
            }
            
            completed = 0
            successful = 0
            
            for future in as_completed(future_to_image):
                image_path = future_to_image[future]
                completed += 1
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['success']:
                        successful += 1
                        status = "✅"
                        
                        # Show oval fitting info
                        oval_summary = result.get('fiber_detection', {}).get('oval_fitting_summary', {})
                        oval_rate = oval_summary.get('fiber_fit_success_rate', 0)
                        oval_info = f"Oval: {oval_rate:.1%}"
                    else:
                        status = "❌"
                        oval_info = "Oval: N/A"
                    
                    # Progress update
                    progress = completed / len(image_files) * 100
                    elapsed = time.time() - start_time
                    eta = elapsed * (len(image_files) - completed) / completed if completed > 0 else 0
                    
                    print(f"{status} [{completed:3d}/{len(image_files)}] {progress:5.1f}% | "
                          f"{Path(image_path).name:<25} | "
                          f"Time: {result.get('total_processing_time', 0):5.2f}s | "
                          f"{oval_info} | "
                          f"ETA: {eta:5.0f}s")
                    
                except Exception as e:
                    print(f"❌ [{completed:3d}/{len(image_files)}] Failed: {Path(image_path).name} - {e}")
                    results.append({
                        'image_path': image_path,
                        'image_name': Path(image_path).name,
                        'success': False,
                        'error': str(e)
                    })
        
        total_time = time.time() - start_time
        
        # Generate enhanced summary
        summary = {
            'batch_info': {
                'timestamp': datetime.now().isoformat(),
                'input_directory': str(image_dir),
                'output_directory': str(output_dir),
                'total_images': len(image_files),
                'successful_analyses': successful,
                'success_rate': successful / len(image_files) * 100 if image_files else 0,
                'total_processing_time': total_time,
                'average_time_per_image': total_time / len(image_files) if image_files else 0,
                'num_processes_used': self.num_processes,
                'images_per_second': len(image_files) / total_time if total_time > 0 else 0,
                'analysis_features': ['oval_fitting', 'enhanced_measurements', 'diameter_analysis'],
                'results_system': 'centralized' if RESULTS_CONFIGURED else 'fallback'
            },
            'individual_results': results
        }
        
        # Calculate oval fitting batch statistics
        oval_stats = self._calculate_batch_oval_statistics(results)
        summary['batch_oval_statistics'] = oval_stats
        
        # Save results - FIXED: Use centralized path functions
        try:
            # JSON results
            json_path = get_json_results_path('enhanced_batch_results')
            with open(json_path, 'w') as f:
                json.dump(self._prepare_for_json(summary), f, indent=2, default=str)
            
            # Enhanced Excel report
            excel_path = get_excel_report_path('ENHANCED_OVAL_ANALYSIS')
            self._create_enhanced_excel_report(summary, excel_path)
            print(f"\n📊 ENHANCED EXCEL REPORT: {excel_path}")
            print(f"📄 JSON RESULTS: {json_path}")
            
        except Exception as e:
            print(f"⚠️ Could not save results using centralized paths: {e}")
            # Fallback to output_dir
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_path = output_dir / f'enhanced_batch_results_{timestamp}.json'
            excel_path = output_dir / f'ENHANCED_OVAL_ANALYSIS_{timestamp}.xlsx'
            
            with open(json_path, 'w') as f:
                json.dump(self._prepare_for_json(summary), f, indent=2, default=str)
            self._create_enhanced_excel_report(summary, excel_path)
            print(f"\n📊 ENHANCED EXCEL REPORT (fallback): {excel_path}")
        
        # Performance summary
        print(f"\n🎯 ENHANCED ANALYSIS COMPLETE!")
        print("=" * 50)
        print(f"📊 Success Rate: {successful}/{len(image_files)} ({successful/len(image_files)*100:.1f}%)")
        print(f"⏱️  Total Time: {total_time:.2f} seconds")
        print(f"🚀 Processes: {self.num_processes}")
        print(f"⚡ Speed: {len(image_files)/total_time:.2f} images/sec")
        print(f"🔍 Oval Fitting: {oval_stats.get('overall_success_rate', 0):.1%} success rate")
        print(f"📏 Avg Diameter: {oval_stats.get('avg_fiber_diameter_um', 0):.1f} μm")
        print(f"💾 Results saved to: {output_dir}")
        
        return summary
    
    def _calculate_batch_oval_statistics(self, results: List[Dict]) -> Dict:
        """Calculate batch-level oval fitting statistics."""
        
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            return {'error': 'No successful analyses for oval statistics'}
        
        # Collect oval fitting data across all samples
        total_fibers = 0
        total_fitted_fibers = 0
        total_lumens = 0
        total_fitted_lumens = 0
        all_fiber_diameters = []
        all_lumen_diameters = []
        all_fit_qualities = []
        all_eccentricities = []
        
        for result in successful_results:
            detailed = result.get('detailed_measurements', {})
            oval_analysis = detailed.get('oval_fitting_analysis', {})
            
            total_fibers += oval_analysis.get('fibers_total_analyzed', 0)
            total_fitted_fibers += oval_analysis.get('fibers_successfully_fitted', 0)
            total_lumens += oval_analysis.get('lumens_total_analyzed', 0)
            total_fitted_lumens += oval_analysis.get('lumens_successfully_fitted', 0)
            
            # Collect diameter data (if available)
            if oval_analysis.get('fiber_oval_mean_diameter_um', 0) > 0:
                all_fiber_diameters.append(oval_analysis['fiber_oval_mean_diameter_um'])
            
            if oval_analysis.get('lumen_oval_mean_diameter_um', 0) > 0:
                all_lumen_diameters.append(oval_analysis['lumen_oval_mean_diameter_um'])
            
            if oval_analysis.get('fiber_oval_mean_fit_quality', 0) > 0:
                all_fit_qualities.append(oval_analysis['fiber_oval_mean_fit_quality'])
            
            if oval_analysis.get('fiber_oval_mean_eccentricity', 0) >= 0:
                all_eccentricities.append(oval_analysis['fiber_oval_mean_eccentricity'])
        
        stats = {
            'total_samples_analyzed': len(successful_results),
            'total_fibers_analyzed': total_fibers,
            'total_fibers_fitted': total_fitted_fibers,
            'overall_success_rate': total_fitted_fibers / total_fibers if total_fibers > 0 else 0,
            'total_lumens_analyzed': total_lumens,
            'total_lumens_fitted': total_fitted_lumens,
            'lumen_fitting_success_rate': total_fitted_lumens / total_lumens if total_lumens > 0 else 0,
        }
        
        # Diameter statistics
        if all_fiber_diameters:
            stats.update({
                'avg_fiber_diameter_um': np.mean(all_fiber_diameters),
                'std_fiber_diameter_um': np.std(all_fiber_diameters),
                'min_fiber_diameter_um': np.min(all_fiber_diameters),
                'max_fiber_diameter_um': np.max(all_fiber_diameters),
                'median_fiber_diameter_um': np.median(all_fiber_diameters),
            })
        
        if all_lumen_diameters:
            stats.update({
                'avg_lumen_diameter_um': np.mean(all_lumen_diameters),
                'std_lumen_diameter_um': np.std(all_lumen_diameters),
            })
        
        # Quality statistics
        if all_fit_qualities:
            stats.update({
                'avg_fit_quality': np.mean(all_fit_qualities),
                'std_fit_quality': np.std(all_fit_qualities),
            })
        
        if all_eccentricities:
            stats.update({
                'avg_eccentricity': np.mean(all_eccentricities),
                'std_eccentricity': np.std(all_eccentricities),
            })
        
        return stats
    
    def _prepare_for_json(self, obj):
        """Prepare object for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._prepare_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        else:
            return obj
    
    def _create_enhanced_excel_report(self, batch_summary: Dict, excel_path: Path):
        """
        Create enhanced Excel report with comprehensive oval fitting data.
        """
        
        results = batch_summary['individual_results']
        batch_info = batch_summary['batch_info']
        oval_stats = batch_summary.get('batch_oval_statistics', {})
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            
            # 1. ENHANCED OVERVIEW SHEET
            overview_data = {
                'Metric': [
                    'Analysis Date', 'Total Images', 'Successful', 'Success Rate (%)',
                    'Total Time (s)', 'Avg Time/Image (s)', 'Images/Second',
                    'Processes Used', 'Analysis Features', 'Results System',
                    # NEW: Oval fitting overview
                    'Total Fibers Analyzed', 'Fibers Successfully Fitted', 'Oval Fitting Success Rate (%)',
                    'Total Lumens Analyzed', 'Lumens Successfully Fitted', 'Lumen Fitting Success Rate (%)',
                    'Avg Fiber Diameter (μm)', 'Avg Lumen Diameter (μm)', 'Avg Fit Quality'
                ],
                'Value': [
                    batch_info['timestamp'][:19].replace('T', ' '),
                    batch_info['total_images'],
                    batch_info['successful_analyses'],
                    f"{batch_info['success_rate']:.1f}%",
                    f"{batch_info['total_processing_time']:.2f}",
                    f"{batch_info['average_time_per_image']:.2f}",
                    f"{batch_info['images_per_second']:.2f}",
                    batch_info['num_processes_used'],
                    ', '.join(batch_info.get('analysis_features', [])),
                    batch_info.get('results_system', 'unknown'),
                    # NEW: Oval fitting data
                    oval_stats.get('total_fibers_analyzed', 0),
                    oval_stats.get('total_fibers_fitted', 0),
                    f"{oval_stats.get('overall_success_rate', 0)*100:.1f}%",
                    oval_stats.get('total_lumens_analyzed', 0),
                    oval_stats.get('total_lumens_fitted', 0),
                    f"{oval_stats.get('lumen_fitting_success_rate', 0)*100:.1f}%",
                    f"{oval_stats.get('avg_fiber_diameter_um', 0):.2f}",
                    f"{oval_stats.get('avg_lumen_diameter_um', 0):.2f}",
                    f"{oval_stats.get('avg_fit_quality', 0):.3f}"
                ]
            }
            overview_df = pd.DataFrame(overview_data)
            overview_df.to_excel(writer, sheet_name='Enhanced_Overview', index=False)
            
            # 2. COMPREHENSIVE RESULTS WITH OVAL FITTING
            comprehensive_results = []
            for result in results:
                if result.get('success', False):
                    # Basic info
                    row = {
                        'Image_Name': result['image_name'],
                        'Success': result['success'],
                        'Processing_Time_s': result.get('total_processing_time', 0),
                        'Memory_Usage_MB': result.get('memory_usage_mb', 0),
                        'Process_ID': result.get('process_id', 0)
                    }
                    
                    # Scale detection
                    scale_data = result.get('scale_detection', {})
                    row.update({
                        'Scale_Detected': scale_data.get('scale_detected', False),
                        'Scale_Factor_um_per_pixel': scale_data.get('micrometers_per_pixel', 0),
                        'Scale_Confidence': scale_data.get('confidence', 0)
                    })
                    
                    # Fiber detection
                    fiber_data = result.get('fiber_detection', {})
                    row.update({
                        'Fiber_Type': fiber_data.get('fiber_type', 'unknown'),
                        'Fiber_Confidence': fiber_data.get('confidence', 0),
                        'Total_Fibers': fiber_data.get('total_fibers', 0),
                        'Hollow_Fibers': fiber_data.get('hollow_fibers', 0),
                        'Filaments': fiber_data.get('filaments', 0)
                    })
                    
                    # Porosity analysis
                    porosity_data = result.get('porosity_analysis', {})
                    pm = porosity_data.get('porosity_metrics', {}) if porosity_data else {}
                    row.update({
                        'Porosity_Success': 'porosity_metrics' in porosity_data if porosity_data else False,
                        'Total_Porosity_Percent': pm.get('total_porosity_percent', 0),
                        'Pore_Count': pm.get('pore_count', 0),
                        'Average_Pore_Size_um2': pm.get('average_pore_size_um2', 0),
                        'Pore_Density_per_mm2': pm.get('pore_density_per_mm2', 0)
                    })
                    
                    # Quality metrics
                    quality_data = result.get('quality_metrics', {})
                    row.update({
                        'Analysis_Quality': quality_data.get('overall_quality', 'unknown'),
                        'Quality_Score': quality_data.get('quality_score', 0),
                        'Oval_Fitting_Quality': quality_data.get('oval_fitting_quality', 0)
                    })
                    
                    # NEW: Enhanced measurements with oval fitting
                    detailed = result.get('detailed_measurements', {})
                    
                    # Pore details
                    pore_analysis = detailed.get('pore_analysis', {})
                    row.update({
                        'Pore_Mean_Diameter_um': pore_analysis.get('mean_diameter_um', 0),
                        'Pore_Std_Diameter_um': pore_analysis.get('std_diameter_um', 0),
                        'Nano_Pores_Count': pore_analysis.get('nano_pores', 0),
                        'Micro_Pores_Count': pore_analysis.get('micro_pores', 0),
                        'Small_Pores_Count': pore_analysis.get('small_pores', 0),
                        'Medium_Pores_Count': pore_analysis.get('medium_pores', 0),
                        'Large_Pores_Count': pore_analysis.get('large_pores', 0),
                        'Macro_Pores_Count': pore_analysis.get('macro_pores', 0),
                    })
                    
                    # Traditional fiber details
                    fiber_analysis = detailed.get('fiber_analysis', {})
                    row.update({
                        'Traditional_Fiber_Mean_Diameter_um': fiber_analysis.get('mean_diameter_um', 0),
                        'Traditional_Fiber_Std_Diameter_um': fiber_analysis.get('std_diameter_um', 0),
                        'Fiber_Mean_Area_um2': fiber_analysis.get('mean_area_um2', 0),
                        'Fiber_Total_Area_um2': fiber_analysis.get('total_area_um2', 0),
                        'Fiber_Mean_Circularity': fiber_analysis.get('mean_circularity', 0),
                        'Fiber_Elongated_Count': fiber_analysis.get('elongated_fibers', 0),
                    })
                    
                    # NEW: Oval fitting details
                    oval_analysis = detailed.get('oval_fitting_analysis', {})
                    row.update({
                        'Oval_Fibers_Analyzed': oval_analysis.get('fibers_total_analyzed', 0),
                        'Oval_Fibers_Successfully_Fitted': oval_analysis.get('fibers_successfully_fitted', 0),
                        'Oval_Fiber_Success_Rate': oval_analysis.get('fiber_oval_success_rate', 0),
                        'Oval_Fiber_Mean_Diameter_um': oval_analysis.get('fiber_oval_mean_diameter_um', 0),
                        'Oval_Fiber_Median_Diameter_um': oval_analysis.get('fiber_oval_median_diameter_um', 0),
                        'Oval_Fiber_Std_Diameter_um': oval_analysis.get('fiber_oval_std_diameter_um', 0),
                        'Oval_Fiber_Min_Diameter_um': oval_analysis.get('fiber_oval_min_diameter_um', 0),
                        'Oval_Fiber_Max_Diameter_um': oval_analysis.get('fiber_oval_max_diameter_um', 0),
                        'Oval_Fiber_Major_Mean_um': oval_analysis.get('fiber_oval_major_mean_um', 0),
                        'Oval_Fiber_Minor_Mean_um': oval_analysis.get('fiber_oval_minor_mean_um', 0),
                        'Oval_Fiber_Mean_Eccentricity': oval_analysis.get('fiber_oval_mean_eccentricity', 0),
                        'Oval_Fiber_Mean_Fit_Quality': oval_analysis.get('fiber_oval_mean_fit_quality', 0),
                        'Oval_Fiber_Mean_Area_um2': oval_analysis.get('fiber_oval_mean_area_um2', 0),
                        'Oval_Fiber_Diameter_CV': oval_analysis.get('fiber_oval_diameter_cv', 0),
                        
                        # Oval-based size categories
                        'Oval_Ultra_Fine_Fibers': oval_analysis.get('oval_ultra_fine_fibers', 0),
                        'Oval_Fine_Fibers': oval_analysis.get('oval_fine_fibers', 0),
                        'Oval_Medium_Fibers': oval_analysis.get('oval_medium_fibers', 0),
                        'Oval_Coarse_Fibers': oval_analysis.get('oval_coarse_fibers', 0),
                        'Oval_Very_Coarse_Fibers': oval_analysis.get('oval_very_coarse_fibers', 0),
                        
                        # Lumen oval fitting
                        'Oval_Lumens_Analyzed': oval_analysis.get('lumens_total_analyzed', 0),
                        'Oval_Lumens_Successfully_Fitted': oval_analysis.get('lumens_successfully_fitted', 0),
                        'Oval_Lumen_Success_Rate': oval_analysis.get('lumen_oval_success_rate', 0),
                        'Oval_Lumen_Mean_Diameter_um': oval_analysis.get('lumen_oval_mean_diameter_um', 0),
                        'Oval_Lumen_Std_Diameter_um': oval_analysis.get('lumen_oval_std_diameter_um', 0),
                        'Oval_Lumen_Mean_Eccentricity': oval_analysis.get('lumen_oval_mean_eccentricity', 0),
                        'Oval_Lumen_Mean_Fit_Quality': oval_analysis.get('lumen_oval_mean_fit_quality', 0),
                    })
                    
                    # Enhanced lumen details
                    lumen_analysis = detailed.get('lumen_analysis', {})
                    row.update({
                        'Has_Lumen_Data': lumen_analysis.get('has_lumen_data', False),
                        'Lumen_Mean_Diameter_um': lumen_analysis.get('mean_lumen_diameter_um', 0),
                        'Lumen_Std_Diameter_um': lumen_analysis.get('std_lumen_diameter_um', 0),
                        'Wall_Mean_Thickness_um': lumen_analysis.get('mean_wall_thickness_um', 0),
                        'Wall_Median_Thickness_um': lumen_analysis.get('median_wall_thickness_um', 0),
                        'Wall_Std_Thickness_um': lumen_analysis.get('std_wall_thickness_um', 0),
                        'Wall_to_Fiber_Ratio': lumen_analysis.get('mean_wall_to_fiber_ratio', 0),
                        'Lumen_to_Fiber_Ratio': lumen_analysis.get('mean_lumen_to_fiber_ratio', 0),
                        'Lumen_Count': lumen_analysis.get('lumen_count', 0),
                    })
                    
                else:
                    # Failed analysis
                    row = {
                        'Image_Name': result['image_name'],
                        'Success': False,
                        'Error': result.get('error', 'Unknown error'),
                        'Processing_Time_s': result.get('total_processing_time', 0)
                    }
                
                comprehensive_results.append(row)
            
            # Save comprehensive results
            main_df = pd.DataFrame(comprehensive_results)
            main_df.to_excel(writer, sheet_name='Comprehensive_Results', index=False)
            
            # 3. OVAL FITTING DEDICATED SHEET
            oval_details = []
            for result in results:
                if result.get('success', False):
                    detailed = result.get('detailed_measurements', {})
                    oval_analysis = detailed.get('oval_fitting_analysis', {})
                    
                    oval_details.append({
                        'Image_Name': result['image_name'],
                        'Fibers_Total_Analyzed': oval_analysis.get('fibers_total_analyzed', 0),
                        'Fibers_Successfully_Fitted': oval_analysis.get('fibers_successfully_fitted', 0),
                        'Fiber_Success_Rate': oval_analysis.get('fiber_oval_success_rate', 0),
                        'Fiber_Mean_Diameter_um': oval_analysis.get('fiber_oval_mean_diameter_um', 0),
                        'Fiber_Std_Diameter_um': oval_analysis.get('fiber_oval_std_diameter_um', 0),
                        'Fiber_Major_Mean_um': oval_analysis.get('fiber_oval_major_mean_um', 0),
                        'Fiber_Minor_Mean_um': oval_analysis.get('fiber_oval_minor_mean_um', 0),
                        'Fiber_Mean_Eccentricity': oval_analysis.get('fiber_oval_mean_eccentricity', 0),
                        'Fiber_Mean_Fit_Quality': oval_analysis.get('fiber_oval_mean_fit_quality', 0),
                        'Lumens_Total_Analyzed': oval_analysis.get('lumens_total_analyzed', 0),
                        'Lumens_Successfully_Fitted': oval_analysis.get('lumens_successfully_fitted', 0),
                        'Lumen_Success_Rate': oval_analysis.get('lumen_oval_success_rate', 0),
                        'Lumen_Mean_Diameter_um': oval_analysis.get('lumen_oval_mean_diameter_um', 0),
                        'Lumen_Mean_Eccentricity': oval_analysis.get('lumen_oval_mean_eccentricity', 0),
                        'Lumen_Mean_Fit_Quality': oval_analysis.get('lumen_oval_mean_fit_quality', 0),
                    })
            
            oval_df = pd.DataFrame(oval_details)
            oval_df.to_excel(writer, sheet_name='Oval_Fitting_Details', index=False)
            
            # 4. BATCH OVAL STATISTICS SHEET
            if oval_stats and 'error' not in oval_stats:
                batch_oval_data = []
                for key, value in oval_stats.items():
                    if isinstance(value, (int, float)):
                        batch_oval_data.append({
                            'Statistic': key.replace('_', ' ').title(),
                            'Value': value
                        })
                
                batch_oval_df = pd.DataFrame(batch_oval_data)
                batch_oval_df.to_excel(writer, sheet_name='Batch_Oval_Statistics', index=False)
            
            # 5. PERFORMANCE ANALYSIS
            performance_data = []
            for result in results:
                if result.get('success', False):
                    quality_data = result.get('quality_metrics', {})
                    performance_data.append({
                        'Image_Name': result['image_name'],
                        'Total_Time_s': result.get('total_processing_time', 0),
                        'Memory_Usage_MB': result.get('memory_usage_mb', 0),
                        'Process_ID': result.get('process_id', 0),
                        'Image_Size_MB': result.get('image_size_mb', 0),
                        'Overall_Quality_Score': quality_data.get('quality_score', 0),
                        'Oval_Fitting_Quality': quality_data.get('oval_fitting_quality', 0),
                        'Oval_Success_Rate': quality_data.get('oval_success_rate', 0),
                    })
            
            performance_df = pd.DataFrame(performance_data)
            performance_df.to_excel(writer, sheet_name='Performance_Analysis', index=False)
        
        print(f"📊 Enhanced Excel report created with {len(results)} results and oval fitting analysis")
        print(f"💾 Saved to: {excel_path}")

def main():
    """Main function with enhanced command line interface."""
    
    parser = argparse.ArgumentParser(
        description='Enhanced Multi-Processing SEM Fiber Analysis with Oval Fitting and Centralized Results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python multiprocessing_analyzer.py --batch sample_images/
  python multiprocessing_analyzer.py --batch sample_images/ --output custom_output/
  python multiprocessing_analyzer.py --batch sample_images/ --processes 4 --max-images 10

Results Management:
  - All results are automatically saved to the centralized results/ folder structure
  - Use --output to override the default location
  - Excel reports include 100+ measurements with oval fitting data
        """
    )
    
    parser.add_argument('--batch', '-b', required=True, help='Directory with images to analyze')
    parser.add_argument('--output', '-o', help='Output directory for results (default: uses centralized results/)')
    parser.add_argument('--processes', '-p', type=str, default='auto', 
                       help='Number of processes (auto, or specific number)')
    parser.add_argument('--max-images', type=int, help='Maximum images to process (for testing)')
    parser.add_argument('--show-results-info', action='store_true', 
                       help='Show information about results directory structure')
    
    args = parser.parse_args()
    
    # Show results info if requested
    if args.show_results_info:
        if RESULTS_CONFIGURED:
            print_results_structure()
            info = get_results_info()
            print(f"\nResults Configuration Details:")
            for key, value in info.items():
                if key != 'subdirectories':
                    print(f"  {key}: {value}")
        else:
            print("Using fallback results configuration")
        return
    
    # Determine number of processes
    if args.processes == 'auto':
        num_processes = None
    else:
        try:
            num_processes = int(args.processes)
        except ValueError:
            print(f"❌ Invalid processes value: {args.processes}")
            return
    
    # Initialize enhanced analyzer
    analyzer = MultiProcessingFiberAnalyzer(num_processes=num_processes)
    
    # Run analysis
    batch_dir = Path(args.batch)
    if not batch_dir.exists():
        print(f"❌ Directory not found: {batch_dir}")
        return
    
    summary = analyzer.analyze_batch_parallel(
        str(batch_dir), 
        args.output,
        max_images=args.max_images
    )
    
    if 'error' not in summary:
        print(f"\n🎉 Enhanced batch analysis completed successfully!")
        print(f"📁 Results saved to: {summary['batch_info']['output_directory']}")
        print(f"📊 Excel report includes oval fitting data for all samples")
        print(f"🔧 Results system: {summary['batch_info'].get('results_system', 'unknown')}")
        
        if RESULTS_CONFIGURED:
            print(f"\n✅ All outputs are now centralized in the results/ folder structure!")
            print(f"   Run with --show-results-info to see the directory layout")

if __name__ == "__main__":
    mp.freeze_support()  # Required for Windows
    main()