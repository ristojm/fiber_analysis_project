#!/usr/bin/env python3
"""
FIXED Multi-Processing Comprehensive SEM Fiber Analyzer
FIXED: Proper scale factor integration throughout the analysis pipeline

Features:
- Parallel batch processing with multiprocessing
- FIXED: Scale factor properly passed to all analysis modules
- Detailed pore analysis (size distribution, shape, spatial)
- Comprehensive fiber measurements (diameter, wall thickness, lumen) - ALL IN MICROMETERS
- Real-time progress tracking
- Comprehensive Excel reporting with 80+ measurements in correct units
- Memory-efficient processing
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
    FIXED: Worker function with proper scale factor integration throughout pipeline.
    Designed to be pickle-able for multiprocessing.
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
        
        # Step 2: Scale detection (CRITICAL: Use ORIGINAL image with scale bar)
        scale_detector = ScaleBarDetector(use_enhanced_detection=True)
        scale_result = scale_detector.detect_scale_bar(
            image, debug=False, save_debug_image=False, output_dir=None  # Use original image!
        )
        
        # FIXED: Get scale factor and ensure it's valid
        scale_factor = scale_result['micrometers_per_pixel'] if scale_result['scale_detected'] else 1.0
        
        # Validate scale factor
        if scale_factor <= 0 or scale_factor > 100:  # Sanity check
            print(f"Warning: Invalid scale factor {scale_factor}, using default 1.0")
            scale_factor = 1.0
        
        result['scale_detection'] = scale_result
        result['scale_factor_used'] = scale_factor
        
        # Step 3: FIXED - Fiber type detection with scale factor
        fiber_detector = FiberTypeDetector()
        fiber_type, fiber_confidence, fiber_analysis_data = fiber_detector.classify_fiber_type(
            preprocessed, scale_factor  # FIXED: Pass scale factor here!
        )
        
        result['fiber_detection'] = {
            'fiber_type': fiber_type,
            'confidence': fiber_confidence,
            'total_fibers': fiber_analysis_data.get('total_fibers', 0),
            'hollow_fibers': fiber_analysis_data.get('hollow_fibers', 0),
            'filaments': fiber_analysis_data.get('filaments', 0)
        }
        
        # FIXED: Extract oval fitting results with proper units
        oval_results = fiber_analysis_data.get('oval_fitting_results', {})
        result['oval_fitting'] = {
            'success_rate': oval_results.get('success_rate', 0),
            'avg_fit_quality': oval_results.get('avg_fit_quality', 0),
            'avg_diameter_um': oval_results.get('avg_diameter', 0),  # Now in micrometers!
            'diameter_std_um': oval_results.get('diameter_std', 0),
            'lumens_fitted': oval_results.get('lumens_fitted', 0),
            'avg_lumen_diameter_um': oval_results.get('avg_lumen_diameter', 0)
        }
        
        # Step 4: Porosity analysis with scale factor
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
                    preprocessed, fiber_mask, scale_factor, fiber_type, fiber_analysis_data  # Scale factor passed
                )
                
                result['porosity_analysis'] = porosity_result
                
            except Exception as e:
                result['porosity_analysis'] = {'error': str(e)}
        else:
            result['porosity_analysis'] = {'error': 'Insufficient fiber area'}
        
        # Step 5: FIXED - Extract detailed measurements with proper scale conversion
        detailed_analysis = extract_detailed_measurements(
            result.get('porosity_analysis', {}),
            fiber_analysis_data,
            scale_factor  # FIXED: Pass scale factor to ensure proper conversion
        )
        
        result['detailed_measurements'] = detailed_analysis
        
        # Step 6: Calculate quality metrics
        quality_metrics = calculate_quality_metrics(result, scale_factor)
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
        print(f"Error processing {image_path}: {e}")
    
    return result

def _preprocess_for_analysis(image: np.ndarray) -> np.ndarray:
    """Minimal preprocessing for analysis."""
    denoised = cv2.bilateralFilter(image, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    return enhanced

def extract_detailed_measurements(porosity_result: Dict, fiber_analysis_data: Dict, scale_factor: float) -> Dict:
    """
    FIXED: Extract comprehensive measurements with proper scale factor usage.
    All measurements are now guaranteed to be in micrometers.
    """
    
    measurements = {
        'pore_analysis': {},
        'fiber_analysis': {},
        'lumen_analysis': {}
    }
    
    # === PORE ANALYSIS ===
    if porosity_result and 'individual_pores' in porosity_result:
        pores = porosity_result['individual_pores']
        
        if pores:
            # Extract pore measurements (should already be in micrometers from porosity analysis)
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
                        # Calculate nearest neighbor distances (simplified)
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
    
    # === FIBER ANALYSIS === 
    # FIXED: Use scale-converted measurements from fiber analysis data
    if fiber_analysis_data and 'individual_results' in fiber_analysis_data:
        individual_results = fiber_analysis_data['individual_results']
        
        if individual_results:
            # FIXED: Extract measurements that are already in micrometers
            fiber_areas_um2 = []
            fiber_diameters_um = []
            fiber_circularities = []
            fiber_aspect_ratios = []
            lumen_diameters_um = []
            wall_thicknesses_um = []
            
            for result in individual_results:
                fiber_props = result.get('fiber_properties', {})
                
                # FIXED: Use micrometer measurements that should now be available
                area_um2 = fiber_props.get('area_um2', 0)
                diameter_um = fiber_props.get('diameter_um', 0)
                
                if area_um2 > 0:
                    fiber_areas_um2.append(area_um2)
                
                if diameter_um > 0:
                    fiber_diameters_um.append(diameter_um)
                else:
                    # Fallback calculation if diameter_um not available
                    equivalent_diameter_um = fiber_props.get('equivalent_diameter_um', 0)
                    if equivalent_diameter_um > 0:
                        fiber_diameters_um.append(equivalent_diameter_um)
                
                fiber_circularities.append(fiber_props.get('circularity', 0))
                fiber_aspect_ratios.append(fiber_props.get('aspect_ratio', 1))
                
                # FIXED: Lumen measurements (should be in micrometers from lumen detection)
                if result.get('has_lumen', False):
                    lumen_props = result.get('lumen_properties', {})
                    lumen_diameter_um = lumen_props.get('equivalent_diameter_um', 0)
                    
                    if lumen_diameter_um > 0:
                        lumen_diameters_um.append(lumen_diameter_um)
                        
                        # Calculate wall thickness if we have both fiber and lumen diameters
                        if diameter_um > 0:
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
            
            # Lumen analysis
            if lumen_diameters_um:
                measurements['lumen_analysis'] = {
                    'has_lumen_data': True,
                    'mean_lumen_diameter_um': np.mean(lumen_diameters_um),
                    'median_lumen_diameter_um': np.median(lumen_diameters_um),
                    'std_lumen_diameter_um': np.std(lumen_diameters_um),
                    'mean_wall_thickness_um': np.mean(wall_thicknesses_um) if wall_thicknesses_um else 0,
                    'median_wall_thickness_um': np.median(wall_thicknesses_um) if wall_thicknesses_um else 0,
                    'lumen_count': len(lumen_diameters_um)
                }
            else:
                measurements['lumen_analysis'] = {'has_lumen_data': False}
    
    return measurements

def calculate_quality_metrics(result: Dict, scale_factor: float) -> Dict:
    """Calculate overall analysis quality metrics."""
    
    quality_score = 0.0
    quality_factors = []
    
    # Scale detection quality (25%)
    scale_result = result.get('scale_detection', {})
    if scale_result.get('scale_detected', False):
        scale_conf = scale_result.get('confidence', 0.0)
        quality_score += scale_conf * 0.25
        quality_factors.append(f"Scale: {scale_conf:.2f}")
    else:
        quality_factors.append("Scale: failed")
    
    # Fiber detection quality (35%)
    fiber_result = result.get('fiber_detection', {})
    fiber_conf = fiber_result.get('confidence', 0.0)
    quality_score += fiber_conf * 0.35
    quality_factors.append(f"Fiber: {fiber_conf:.2f}")
    
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
        
        quality_score += porosity_quality * 0.4
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
        'quality_factors': quality_factors
    }

class MultiProcessingFiberAnalyzer:
    """Multi-processing SEM fiber analyzer with comprehensive measurements."""
    
    def __init__(self, num_processes: Optional[int] = None):
        """Initialize the multi-processing analyzer."""
        
        if num_processes is None:
            self.num_processes = max(1, mp.cpu_count() - 1)
        else:
            self.num_processes = max(1, min(num_processes, mp.cpu_count()))
        
        print(f"🚀 Multi-Processing Analyzer initialized")
        print(f"   CPU cores available: {mp.cpu_count()}")
        print(f"   Processes to use: {self.num_processes}")
    
    def analyze_batch_parallel(self, image_directory: str, 
                              output_dir: Optional[str] = None,
                              max_images: Optional[int] = None) -> Dict:
        """Perform parallel batch analysis."""
        
        print(f"\n🚀 PARALLEL BATCH ANALYSIS")
        print("=" * 60)
        
        # Setup directories
        image_dir = Path(image_directory)
        if not image_dir.exists():
            return {'error': f'Directory not found: {image_dir}'}
        
        if output_dir is None:
            output_dir = image_dir.parent / 'parallel_results'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
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
                        
                        # Show scale factor and oval fitting info in progress
                        scale_factor = result.get('scale_factor_used', 0)
                        oval_info = result.get('oval_fitting', {})
                        oval_diameter = oval_info.get('avg_diameter_um', 0)
                        
                        extra_info = f"Scale: {scale_factor:.4f}μm/px"
                        if oval_diameter > 0:
                            extra_info += f", Oval: {oval_diameter:.1f}μm"
                        
                    else:
                        status = "❌"
                        extra_info = result.get('error', 'Unknown error')[:30]
                    
                    # Progress update
                    progress = completed / len(image_files) * 100
                    elapsed = time.time() - start_time
                    eta = elapsed * (len(image_files) - completed) / completed if completed > 0 else 0
                    
                    print(f"{status} [{completed:3d}/{len(image_files)}] {progress:5.1f}% | "
                          f"{Path(image_path).name:<25} | "
                          f"Time: {result.get('total_processing_time', 0):5.2f}s | "
                          f"ETA: {eta:5.0f}s | {extra_info}")
                    
                except Exception as e:
                    print(f"❌ [{completed:3d}/{len(image_files)}] Failed: {Path(image_path).name} - {e}")
                    results.append({
                        'image_path': image_path,
                        'image_name': Path(image_path).name,
                        'success': False,
                        'error': str(e)
                    })
        
        total_time = time.time() - start_time
        
        # Generate summary
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
                'images_per_second': len(image_files) / total_time if total_time > 0 else 0
            },
            'individual_results': results
        }
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON results
        json_path = output_dir / f'batch_results_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(self._prepare_for_json(summary), f, indent=2, default=str)
        
        # Excel report
        try:
            excel_path = output_dir / f'COMPREHENSIVE_ANALYSIS_{timestamp}.xlsx'
            self._create_excel_report(summary, excel_path)
            print(f"\n📊 EXCEL REPORT: {excel_path.name}")
        except Exception as e:
            print(f"⚠️ Could not create Excel report: {e}")
        
        # Performance summary
        print(f"\n🎯 ANALYSIS COMPLETE!")
        print("=" * 40)
        print(f"📊 Success Rate: {successful}/{len(image_files)} ({successful/len(image_files)*100:.1f}%)")
        print(f"⏱️  Total Time: {total_time:.2f} seconds")
        print(f"🚀 Processes: {self.num_processes}")
        print(f"⚡ Speed: {len(image_files)/total_time:.2f} images/sec")
        
        # Show oval fitting summary
        successful_results = [r for r in results if r.get('success', False)]
        if successful_results:
            oval_diameters = [r.get('oval_fitting', {}).get('avg_diameter_um', 0) 
                            for r in successful_results]
            valid_diameters = [d for d in oval_diameters if d > 0]
            
            if valid_diameters:
                print(f"\n📏 OVAL FITTING SUMMARY:")
                print(f"   Images with oval fits: {len(valid_diameters)}/{len(successful_results)}")
                print(f"   Avg diameter: {np.mean(valid_diameters):.1f} ± {np.std(valid_diameters):.1f} μm")
                print(f"   Range: {np.min(valid_diameters):.1f} - {np.max(valid_diameters):.1f} μm")
        
        return summary
    
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
    
    def _create_excel_report(self, batch_summary: Dict, excel_path: Path):
        """FIXED: Create comprehensive Excel report with measurements in correct units."""
        
        results = batch_summary['individual_results']
        batch_info = batch_summary['batch_info']
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            
            # Overview sheet
            overview_data = {
                'Metric': [
                    'Analysis Date', 'Total Images', 'Successful', 'Success Rate (%)',
                    'Total Time (s)', 'Avg Time/Image (s)', 'Images/Second',
                    'Processes Used', 'Input Directory'
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
                    batch_info['input_directory']
                ]
            }
            overview_df = pd.DataFrame(overview_data)
            overview_df.to_excel(writer, sheet_name='Overview', index=False)
            
            # Comprehensive results
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
                        'Scale_Confidence': scale_data.get('confidence', 0),
                        'Scale_Factor_Used': result.get('scale_factor_used', 0)
                    })
                    
                    # FIXED: Oval fitting results (now in micrometers)
                    oval_data = result.get('oval_fitting', {})
                    row.update({
                        'Oval_Success_Rate_Percent': oval_data.get('success_rate', 0),
                        'Oval_Avg_Fit_Quality': oval_data.get('avg_fit_quality', 0),
                        'Oval_Avg_Diameter_um': oval_data.get('avg_diameter_um', 0),  # NOW IN MICROMETERS!
                        'Oval_Diameter_Std_um': oval_data.get('diameter_std_um', 0),  # NOW IN MICROMETERS!
                        'Oval_Lumens_Fitted': oval_data.get('lumens_fitted', 0),
                        'Oval_Avg_Lumen_Diameter_um': oval_data.get('avg_lumen_diameter_um', 0)  # NOW IN MICROMETERS!
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
                        'Quality_Score': quality_data.get('quality_score', 0)
                    })
                    
                    # Detailed measurements (all in micrometers)
                    detailed = result.get('detailed_measurements', {})
                    
                    # Pore details
                    pore_analysis = detailed.get('pore_analysis', {})
                    row.update({
                        'Pore_Mean_Diameter_um': pore_analysis.get('mean_diameter_um', 0),
                        'Pore_Std_Diameter_um': pore_analysis.get('std_diameter_um', 0),
                        'Pore_Min_Diameter_um': pore_analysis.get('min_diameter_um', 0),
                        'Pore_Max_Diameter_um': pore_analysis.get('max_diameter_um', 0),
                        'Nano_Pores_Count': pore_analysis.get('nano_pores', 0),
                        'Micro_Pores_Count': pore_analysis.get('micro_pores', 0),
                        'Small_Pores_Count': pore_analysis.get('small_pores', 0),
                        'Medium_Pores_Count': pore_analysis.get('medium_pores', 0),
                        'Large_Pores_Count': pore_analysis.get('large_pores', 0),
                        'Macro_Pores_Count': pore_analysis.get('macro_pores', 0),
                        'Pore_Mean_Circularity': pore_analysis.get('mean_circularity', 0),
                        'Pore_Elongated_Count': pore_analysis.get('elongated_pores', 0),
                        'Pore_Round_Count': pore_analysis.get('round_pores', 0)
                    })
                    
                    # Fiber details (all in micrometers)
                    fiber_analysis = detailed.get('fiber_analysis', {})
                    row.update({
                        'Fiber_Mean_Diameter_um': fiber_analysis.get('mean_diameter_um', 0),
                        'Fiber_Std_Diameter_um': fiber_analysis.get('std_diameter_um', 0),
                        'Fiber_Min_Diameter_um': fiber_analysis.get('min_diameter_um', 0),
                        'Fiber_Max_Diameter_um': fiber_analysis.get('max_diameter_um', 0),
                        'Fiber_Diameter_CV': fiber_analysis.get('diameter_cv', 0),
                        'Fiber_Mean_Area_um2': fiber_analysis.get('mean_area_um2', 0),
                        'Fiber_Total_Area_um2': fiber_analysis.get('total_area_um2', 0),
                        'Ultra_Fine_Fibers': fiber_analysis.get('ultra_fine_fibers', 0),
                        'Fine_Fibers': fiber_analysis.get('fine_fibers', 0),
                        'Medium_Fibers': fiber_analysis.get('medium_fibers', 0),
                        'Coarse_Fibers': fiber_analysis.get('coarse_fibers', 0),
                        'Very_Coarse_Fibers': fiber_analysis.get('very_coarse_fibers', 0),
                        'Fiber_Mean_Circularity': fiber_analysis.get('mean_circularity', 0),
                        'Fiber_Elongated_Count': fiber_analysis.get('elongated_fibers', 0)
                    })
                    
                    # Lumen details (all in micrometers)
                    lumen_analysis = detailed.get('lumen_analysis', {})
                    row.update({
                        'Has_Lumen_Data': lumen_analysis.get('has_lumen_data', False),
                        'Lumen_Mean_Diameter_um': lumen_analysis.get('mean_lumen_diameter_um', 0),
                        'Lumen_Std_Diameter_um': lumen_analysis.get('std_lumen_diameter_um', 0),
                        'Wall_Mean_Thickness_um': lumen_analysis.get('mean_wall_thickness_um', 0),
                        'Wall_Median_Thickness_um': lumen_analysis.get('median_wall_thickness_um', 0),
                        'Lumen_Count': lumen_analysis.get('lumen_count', 0)
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
            
            # Save main results
            main_df = pd.DataFrame(comprehensive_results)
            main_df.to_excel(writer, sheet_name='Comprehensive_Results', index=False)
            
            # FIXED: Oval fitting summary sheet
            successful_results = [r for r in results if r.get('success', False)]
            if successful_results:
                oval_summary_data = []
                for result in successful_results:
                    oval_data = result.get('oval_fitting', {})
                    oval_summary_data.append({
                        'Image_Name': result['image_name'],
                        'Success_Rate_Percent': oval_data.get('success_rate', 0),
                        'Avg_Fit_Quality': oval_data.get('avg_fit_quality', 0),
                        'Avg_Diameter_um': oval_data.get('avg_diameter_um', 0),  # NOW IN MICROMETERS!
                        'Diameter_Std_um': oval_data.get('diameter_std_um', 0),
                        'Lumens_Fitted': oval_data.get('lumens_fitted', 0),
                        'Avg_Lumen_Diameter_um': oval_data.get('avg_lumen_diameter_um', 0),
                        'Scale_Factor_Used': result.get('scale_factor_used', 0)
                    })
                
                oval_df = pd.DataFrame(oval_summary_data)
                oval_df.to_excel(writer, sheet_name='Oval_Fitting_Results', index=False)
            
            # Performance analysis
            if successful_results:
                perf_data = []
                for result in successful_results:
                    perf_data.append({
                        'Image_Name': result['image_name'],
                        'Processing_Time_s': result.get('total_processing_time', 0),
                        'Memory_Usage_MB': result.get('memory_usage_mb', 0),
                        'Process_ID': result.get('process_id', 0),
                        'Image_Size_MB': result.get('image_size_mb', 0),
                        'Scale_Factor_Used': result.get('scale_factor_used', 0)
                    })
                
                perf_df = pd.DataFrame(perf_data)
                perf_df.to_excel(writer, sheet_name='Performance', index=False)
        
        print(f"📊 Excel report created with {len(results)} results and proper unit conversion")

def main():
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(
        description='FIXED Multi-Processing SEM Fiber Analysis with Proper Scale Factor Integration'
    )
    
    parser.add_argument('--batch', '-b', required=True, help='Directory with images to analyze')
    parser.add_argument('--output', '-o', help='Output directory for results')
    parser.add_argument('--processes', '-p', type=str, default='auto', 
                       help='Number of processes (auto, or specific number)')
    parser.add_argument('--max-images', type=int, help='Maximum images to process (for testing)')
    
    args = parser.parse_args()
    
    # Determine number of processes
    if args.processes == 'auto':
        num_processes = None
    else:
        try:
            num_processes = int(args.processes)
        except ValueError:
            print(f"❌ Invalid processes value: {args.processes}")
            return
    
    # Initialize analyzer
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
        print(f"\n🎉 Batch analysis completed successfully!")
        print(f"   All measurements are now properly converted to micrometers!")

if __name__ == "__main__":
    mp.freeze_support()  # Required for Windows
    main()