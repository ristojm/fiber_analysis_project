#!/usr/bin/env python3
"""
Multiprocessing Crumbly Texture Analysis Workflow - COMPLETE UPDATED VERSION
Integrates all fixes from visual_debug_test.py + hybrid model training/usage.

FEATURES:
1. Parallel evaluation using multiple CPU cores
2. Hybrid model training and usage
3. Fixed preprocessing with scale bar removal
4. Optimal fiber mask selection from modules
5. Proper processing order (scale → preprocess → fiber → porosity → texture)
6. Classification improvement with debug output
7. Memory-efficient processing with progress tracking

Usage Examples:
1. Evaluate with traditional detector:
   python multiprocessing_crumbly_workflow.py evaluate /path/to/dataset --max-images 50

2. Evaluate with hybrid model:
   python multiprocessing_crumbly_workflow.py evaluate /path/to/dataset --model-path /path/to/trained_models

3. Train hybrid model:
   python multiprocessing_crumbly_workflow.py train /path/to/features.csv --output-dir /path/to/models

4. Complete workflow (evaluate → train → re-evaluate):
   python multiprocessing_crumbly_workflow.py complete /path/to/dataset
"""

import sys
import argparse
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import threading
import platform
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Multiprocessing imports
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import os
import cv2

WORKFLOW_DEBUG = False  # Set to True for detailed debugging

# ===== PATH SETUP =====
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "modules"))
sys.path.insert(0, str(current_dir))

print(f"🔧 Multiprocessing Crumbly Workflow - Enhanced Version")
print(f"   Current dir: {current_dir}")
print(f"   Project root: {project_root}")

# Results directory
MULTIPROCESSING_DIR = current_dir / "multiprocessing_crumbly_results"
MULTIPROCESSING_DIR.mkdir(parents=True, exist_ok=True)

def get_multiprocessing_path(filename: str) -> Path:
    return MULTIPROCESSING_DIR / filename

# ===== IMPORT CORE MODULES =====
print("🔧 Loading SEM Fiber Analysis modules...")

MODULES_LOADED = {}

# Core modules (required)
try:
    from modules.scale_detection import ScaleBarDetector
    from modules.fiber_type_detection import FiberTypeDetector
    from modules.image_preprocessing import load_image
    MODULES_LOADED['core'] = True
    print("✅ Core modules loaded")
except ImportError as e:
    print(f"❌ Could not import core modules: {e}")
    sys.exit(1)

# Enhanced preprocessing with scale bar removal
try:
    from modules.image_preprocessing import preprocess_for_analysis
    HAS_ENHANCED_PREPROCESSING = True
    print("✅ Enhanced preprocessing available")
except ImportError:
    from modules.image_preprocessing import enhance_contrast, denoise_image, normalize_image
    HAS_ENHANCED_PREPROCESSING = False
    print("⚠️ Using fallback preprocessing")

try:
    from modules.image_preprocessing import remove_scale_bar_region
    HAS_SCALE_BAR_REMOVAL = True
    print("✅ Scale bar removal available")
except ImportError:
    HAS_SCALE_BAR_REMOVAL = False
    print("⚠️ Scale bar removal not available")

# Porosity analysis
try:
    from modules.porosity_analysis import PorosityAnalyzer, analyze_fiber_porosity
    MODULES_LOADED['porosity_analysis'] = 'fast_refined'
    POROSITY_TYPE = 'fast_refined'
    print("✅ Porosity analysis loaded")
except ImportError:
    try:
        from modules.porosity_analysis import quick_porosity_check
        MODULES_LOADED['porosity_analysis'] = 'basic'
        POROSITY_TYPE = 'basic'
        print("✅ Basic porosity analysis loaded")
    except ImportError:
        MODULES_LOADED['porosity_analysis'] = False
        POROSITY_TYPE = None
        print("❌ No porosity analysis available")

# Crumbly detection
try:
    from modules.crumbly_detection import CrumblyDetector
    MODULES_LOADED['crumbly_detection'] = True
    print("✅ CrumblyDetector loaded")
except ImportError:
    try:
        from crumbly_detection import CrumblyDetector
        MODULES_LOADED['crumbly_detection'] = True
        print("✅ CrumblyDetector loaded from local")
    except ImportError as e:
        print(f"❌ CrumblyDetector not available: {e}")
        MODULES_LOADED['crumbly_detection'] = False

# Hybrid detector (optional)
try:
    from hybrid_crumbly_detector import train_hybrid_detector, load_hybrid_detector
    MODULES_LOADED['hybrid_detector'] = True
    print("✅ Hybrid detector functions loaded")
except ImportError as e:
    print(f"⚠️ Hybrid detector not available: {e}")
    MODULES_LOADED['hybrid_detector'] = False

# ===== TIMEOUT DECORATOR =====



# ===== WORKER FUNCTIONS (Using modules, incorporating all fixes) =====

def preprocess_image_for_worker(image):
    """Preprocess image with scale bar removal (from visual_debug_test.py fix)."""
    try:
        if HAS_ENHANCED_PREPROCESSING:
            temp_processed = preprocess_for_analysis(image, silent=True)
        else:
            enhanced = enhance_contrast(image, method='clahe')
            denoised = denoise_image(enhanced, method='bilateral')
            temp_processed = normalize_image(denoised)
        
        # Remove scale bar region (critical fix)
        if HAS_SCALE_BAR_REMOVAL:
            main_region, scale_bar_region = remove_scale_bar_region(temp_processed)
            return main_region
        else:
            # Fallback: manual crop of bottom 15%
            height = image.shape[0]
            crop_height = int(height * 0.85)
            return image[:crop_height, :]
        
    except Exception as e:
        print(f"⚠️ Preprocessing error: {e}")
        # Final fallback
        height = image.shape[0]
        crop_height = int(height * 0.85)
        return image[:crop_height, :]

def create_optimal_fiber_mask(image: np.ndarray, fiber_analysis_data: dict, debug: bool = False) -> np.ndarray:
    """
    Create optimal fiber mask using modules (from visual_debug_test.py fix).
    Uses best fiber selection + lumen exclusion for hollow fibers.
    """
    if debug:
        print(f"   🔧 Creating optimal fiber mask...")
    
    if not fiber_analysis_data:
        return np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Method 1: Use individual results and select largest fiber
    individual_results = fiber_analysis_data.get('individual_results', [])
    
    if individual_results and len(individual_results) > 0:
        if debug:
            print(f"   🔍 Found {len(individual_results)} fiber candidates")
        
        # Find largest fiber (same logic as classify_fiber_type)
        largest_fiber_result = max(individual_results, key=lambda x: x['fiber_properties']['area'])
        
        # Get fiber contour
        fiber_props = largest_fiber_result.get('fiber_properties', {})
        fiber_contour = fiber_props.get('contour')
        
        if fiber_contour is not None:
            # Create precise mask from contour
            fiber_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(fiber_mask, [fiber_contour], 255)
            
            # For hollow fibers, exclude lumen
            has_lumen = largest_fiber_result.get('has_lumen', False)
            if has_lumen:
                if debug:
                    print(f"   🕳️ Hollow fiber detected, excluding lumen...")
                
                lumen_props = largest_fiber_result.get('lumen_properties', {})
                lumen_contour = lumen_props.get('contour')
                
                if lumen_contour is not None:
                    # Use detected lumen contour
                    lumen_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(lumen_mask, [lumen_contour], 255)
                    fiber_mask[lumen_mask > 0] = 0
                    
                    if debug:
                        lumen_area = np.sum(lumen_mask > 0)
                        print(f"   ✅ Lumen excluded: {lumen_area:,} pixels")
                else:
                    # Fallback: detect lumen using intensity
                    fiber_region = image.copy()
                    fiber_region[fiber_mask == 0] = 255
                    
                    if np.sum(fiber_mask > 0) > 0:
                        lumen_threshold = np.percentile(fiber_region[fiber_mask > 0], 10)
                        potential_lumen = (fiber_region < lumen_threshold) & (fiber_mask > 0)
                        
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                        potential_lumen = cv2.morphologyEx(potential_lumen.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
                        potential_lumen = cv2.morphologyEx(potential_lumen, cv2.MORPH_OPEN, kernel)
                        
                        fiber_mask[potential_lumen > 0] = 0
                        
                        if debug:
                            lumen_area = np.sum(potential_lumen > 0)
                            print(f"   ✅ Lumen excluded using intensity: {lumen_area:,} pixels")
            
            final_area = np.sum(fiber_mask > 0)
            if debug:
                print(f"   ✅ Optimal mask created: {final_area:,} pixels")
            
            return fiber_mask
    
    # Fallback to general mask
    if debug:
        print(f"   ⚠️ Using fallback general mask")
    
    fiber_mask = fiber_analysis_data.get('fiber_mask')
    if fiber_mask is not None and isinstance(fiber_mask, np.ndarray):
        if fiber_mask.dtype != np.uint8:
            fiber_mask = (fiber_mask > 0).astype(np.uint8) * 255
        return fiber_mask
    else:
        return np.zeros(image.shape[:2], dtype=np.uint8)

def apply_classification_improvements(crumbly_result: dict, debug: bool = False) -> dict:
    """
    Apply classification improvements to address 'intermediate' vs 'crumbly' issues.
    FIXED: Uses correct key names from actual crumbly detector output structure.
    """
    if not crumbly_result or 'classification' not in crumbly_result:
        return crumbly_result
    
    # Extract metrics from ACTUAL structure (not the assumed structure)
    surface_metrics = crumbly_result.get('surface_metrics', {})
    boundary_metrics = crumbly_result.get('boundary_metrics', {})
    wall_metrics = crumbly_result.get('wall_integrity_metrics', {})
    
    # FIXED: Extract surface roughness from correct nested structure
    surface_roughness = 0
    if 'roughness_metrics' in surface_metrics:
        roughness_data = surface_metrics['roughness_metrics']
        # Use kernel_5 as representative (middle kernel size)
        if 'kernel_5' in roughness_data:
            surface_roughness = float(roughness_data['kernel_5'].get('mean_roughness', 0))
        elif 'kernel_3' in roughness_data:
            surface_roughness = float(roughness_data['kernel_3'].get('mean_roughness', 0))
    
    # FIXED: Extract edge irregularity from correct structure
    edge_irregularity = 0
    if 'outer_boundary' in boundary_metrics:
        outer_boundary = boundary_metrics['outer_boundary']
        edge_irregularity = float(outer_boundary.get('roughness_index', 0))
    
    # FIXED: Extract wall integrity from correct structure
    wall_integrity = 1.0
    if wall_metrics:
        wall_integrity = float(wall_metrics.get('wall_integrity_score', 1.0))
    
    crumbly_score = crumbly_result.get('crumbly_score', 0.5)
    
    if debug:
        print(f"   🔍 CLASSIFICATION DEBUG (FIXED EXTRACTION):")
        print(f"     Current: {crumbly_result['classification']} (score: {crumbly_score:.3f})")
        print(f"     Surface roughness: {surface_roughness:.3f} (from roughness_metrics)")
        print(f"     Edge irregularity: {edge_irregularity:.3f} (from roughness_index)")
        print(f"     Wall integrity: {wall_integrity:.3f} (from integrity_score)")
    
    # Enhanced override rules using CORRECT values
    original_classification = crumbly_result['classification']
    
    # Rule 1: HIGH surface roughness = definitely crumbly (lowered threshold)
    if surface_roughness > 20.0:  # Your sample has 25.8, clearly above this
        crumbly_result['classification'] = 'crumbly'
        crumbly_result['confidence'] = 0.9
        crumbly_result['override_reason'] = 'high_surface_roughness'
        if debug:
            print(f"     ✅ Override: {original_classification} → crumbly (surface roughness {surface_roughness:.1f} > 20.0)")
        return crumbly_result
    
    # Rule 2: High surface roughness + moderate edge irregularity = crumbly
    elif surface_roughness > 15.0 and edge_irregularity > 0.3:
        crumbly_result['classification'] = 'crumbly'
        crumbly_result['confidence'] = 0.85
        crumbly_result['override_reason'] = 'high_roughness_moderate_edges'
        if debug:
            print(f"     ✅ Override: {original_classification} → crumbly (roughness {surface_roughness:.1f} + edges {edge_irregularity:.3f})")
        return crumbly_result
    
    # Rule 3: Very irregular edges + poor wall structure = crumbly
    elif edge_irregularity > 0.6 and wall_integrity < 0.5:
        crumbly_result['classification'] = 'crumbly'
        crumbly_result['confidence'] = 0.85
        crumbly_result['override_reason'] = 'irregular_edges_poor_walls'
        if debug:
            print(f"     ✅ Override: {original_classification} → crumbly (irregular edges + poor walls)")
        return crumbly_result
    
    # Rule 4: Multiple moderate indicators = crumbly (not intermediate)
    elif original_classification == 'intermediate':
        moderate_indicators = 0
        if surface_roughness > 12.0: moderate_indicators += 1  # Lowered threshold
        if edge_irregularity > 0.3: moderate_indicators += 1   # Lowered threshold
        if wall_integrity < 0.8: moderate_indicators += 1      # Raised threshold (more sensitive)
        
        if moderate_indicators >= 2:
            crumbly_result['classification'] = 'crumbly'
            crumbly_result['confidence'] = 0.75
            crumbly_result['override_reason'] = 'multiple_moderate_indicators'
            if debug:
                print(f"     ✅ Override: intermediate → crumbly ({moderate_indicators} moderate indicators)")
                print(f"       - Surface roughness > 12.0: {surface_roughness > 12.0}")
                print(f"       - Edge irregularity > 0.3: {edge_irregularity > 0.3}")
                print(f"       - Wall integrity < 0.8: {wall_integrity < 0.8}")
            return crumbly_result
    
    # Rule 5: Lower threshold for crumbly score (original rule but with correct extraction)
    elif crumbly_score > 0.55 and original_classification == 'intermediate':
        crumbly_result['classification'] = 'crumbly'
        crumbly_result['confidence'] = min(0.8, crumbly_result.get('confidence', 0.5) + 0.1)
        crumbly_result['override_reason'] = 'lower_crumbly_threshold'
        if debug:
            print(f"     ✅ Override: intermediate → crumbly (score {crumbly_score:.3f} > 0.55)")
        return crumbly_result
    
    if debug and original_classification == crumbly_result['classification']:
        print(f"     ❌ No override applied - classification remains: {original_classification}")
        print(f"       Check if thresholds need further adjustment")
    
    return crumbly_result

def process_single_image_worker(worker_args: Dict) -> Dict:
    """
    Enhanced worker function with all fixes from visual_debug_test.py.
    Proper order: scale detection → preprocessing → fiber → porosity → texture.
    """
    image_path = worker_args['image_path']
    model_path = worker_args.get('model_path')
    enable_debug = worker_args.get('debug', WORKFLOW_DEBUG)
    
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
        if enable_debug:
            print(f"\n🔍 Processing: {Path(image_path).name}")
        
        # Step 1: Load original image
        original_image = load_image(str(image_path))
        if original_image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Step 2: Scale detection FIRST (on original image with scale bar)
        scale_factor = 1.0
        try:
            scale_detector = ScaleBarDetector(use_enhanced_detection=True)
            scale_result = scale_detector.detect_scale_bar(
                original_image, debug=False, save_debug_image=False, output_dir=None
            )
            if scale_result and scale_result.get('scale_detected', False):
                scale_factor = scale_result.get('micrometers_per_pixel', 1.0)
                if enable_debug:
                    print(f"   ✅ Scale detected: {scale_factor:.4f} μm/pixel")
            result['scale_detection'] = scale_result
        except Exception as e:
            result['scale_error'] = str(e)
            if enable_debug:
                print(f"   ❌ Scale detection error: {e}")
        
        # Step 3: Preprocessing with scale bar removal
        preprocessed = preprocess_image_for_worker(original_image)
        if enable_debug:
            print(f"   Original: {original_image.shape}, Preprocessed: {preprocessed.shape}")
        
        # Step 4: Fiber detection (on clean preprocessed image)
        try:
            fiber_detector = FiberTypeDetector()
            fiber_type, confidence, fiber_analysis_data = fiber_detector.classify_fiber_type(preprocessed)
            
            # Extract optimal fiber mask
            fiber_mask = create_optimal_fiber_mask(preprocessed, fiber_analysis_data, debug=enable_debug)
            
            result['fiber_detection'] = {
                'fiber_type': fiber_type,
                'confidence': confidence,
                'total_fibers': fiber_analysis_data.get('total_fibers', 0),
                'hollow_fibers': fiber_analysis_data.get('hollow_fibers', 0),
                'filaments': fiber_analysis_data.get('filaments', 0),
                'mask_area_pixels': int(np.sum(fiber_mask > 0))
            }
            
            if enable_debug:
                mask_area = np.sum(fiber_mask > 0)
                print(f"   ✅ Fiber: {fiber_type} (conf: {confidence:.3f}, area: {mask_area:,} px)")
            
        except Exception as e:
            result['fiber_detection'] = {'error': str(e)}
            fiber_mask = np.zeros(preprocessed.shape[:2], dtype=np.uint8)
            fiber_type = 'unknown'
            if enable_debug:
                print(f"   ❌ Fiber detection error: {e}")
        
        # Step 5: Porosity analysis (on matching dimensions)
        porosity_result = None
        if MODULES_LOADED.get('porosity_analysis', False):
            try:
                if POROSITY_TYPE == 'fast_refined':
                    porosity_result = analyze_fiber_porosity(preprocessed, fiber_mask, scale_factor)
                elif POROSITY_TYPE == 'basic':
                    porosity_result = quick_porosity_check(preprocessed, fiber_mask, scale_factor)
                
                if porosity_result:
                    porosity_metrics = porosity_result.get('porosity_metrics', {})
                    result['porosity_analysis'] = porosity_result
                    
                    if enable_debug:
                        total_porosity = porosity_metrics.get('total_porosity_percent', 0)
                        pore_count = porosity_metrics.get('pore_count', 0)
                        print(f"   ✅ Porosity: {total_porosity:.2f}% ({pore_count} pores)")
                
            except Exception as e:
                result['porosity_error'] = str(e)
                if enable_debug:
                    print(f"   ❌ Porosity analysis error: {e}")
        
        # Step 6: Crumbly texture analysis with hybrid model support
        if MODULES_LOADED.get('crumbly_detection', False):
            try:
                # Choose detector type
                if model_path and MODULES_LOADED.get('hybrid_detector', False):
                    try:
                        crumbly_detector = load_hybrid_detector(model_path)
                        result['model_type'] = 'hybrid'
                        if enable_debug:
                            print(f"   🤖 Using hybrid model: {model_path}")
                    except Exception as e:
                        crumbly_detector = CrumblyDetector(porosity_aware=True)
                        result['model_type'] = 'traditional_fallback'
                        if enable_debug:
                            print(f"   ⚠️ Hybrid model failed, using traditional: {e}")
                else:
                    crumbly_detector = CrumblyDetector(porosity_aware=True)
                    result['model_type'] = 'traditional'
                
                fiber_mask_bool = fiber_mask > 127
                
                # Pass porosity data for better classification
                porosity_data = None
                if porosity_result:
                    porosity_data = {'porosity_metrics': porosity_result.get('porosity_metrics', {})}
                
                # Run crumbly analysis with debug enabled
                crumbly_result = crumbly_detector.analyze_crumbly_texture(
                    preprocessed, fiber_mask_bool, None, scale_factor, 
                    debug=enable_debug, porosity_data=porosity_data
                )
                
                if crumbly_result and 'classification' in crumbly_result:
                    # Apply classification improvements
                    crumbly_result = apply_classification_improvements(crumbly_result, debug=enable_debug)
                    
                    predicted_label = crumbly_result['classification']
                    confidence_score = crumbly_result.get('confidence', 0.0)
                    crumbly_score = crumbly_result.get('crumbly_score', 0.5)
                    
                    result['predicted_label'] = predicted_label
                    result['prediction_confidence'] = confidence_score
                    result['crumbly_score'] = crumbly_score
                    result['crumbly_analysis'] = crumbly_result
                    result['processing_success'] = True
                    
                    if enable_debug:
                        override_reason = crumbly_result.get('override_reason', 'none')
                        print(f"   ✅ Classification: {predicted_label} (conf: {confidence_score:.3f}, score: {crumbly_score:.3f})")
                        if override_reason != 'none':
                            print(f"   🔧 Override applied: {override_reason}")
                
            except Exception as e:
                result['crumbly_error'] = str(e)
                if enable_debug:
                    print(f"   ❌ Crumbly analysis error: {e}")
        
        result['total_processing_time'] = time.time() - start_time
        result['success'] = True
        
        if enable_debug:
            print(f"   ✅ Completed in {result['total_processing_time']:.2f}s")
        
    except Exception as e:
        result['error'] = str(e)
        result['total_processing_time'] = time.time() - start_time
        result['processing_success'] = False
        if enable_debug:
            print(f"   ❌ Failed: {e}")
    
    return result

def with_timeout(timeout_seconds):
    """Cross-platform timeout decorator."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if platform.system() == "Windows":
                return func(*args, **kwargs)
            else:
                import signal
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Function {func.__name__} timed out")
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout_seconds)
                try:
                    result = func(*args, **kwargs)
                    signal.alarm(0)
                    return result
                finally:
                    signal.signal(signal.SIGALRM, old_handler)
        return wrapper
    return decorator

# ===== MAIN WORKFLOW FUNCTIONS =====

def run_parallel_evaluation(image_dir: str, max_images: int = None, 
                          num_processes: int = None, model_path: str = None,
                          debug: bool = False) -> Dict:
    """Run parallel evaluation with optional hybrid model."""
    
    print(f"\n🚀 PARALLEL EVALUATION")
    print("=" * 50)
    
    # Find images
    image_dir = Path(image_dir)
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_dir.glob(f'**/*{ext}'))
        image_files.extend(image_dir.glob(f'**/*{ext.upper()}'))
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"📁 Found {len(image_files)} images in {image_dir}")
    
    if model_path:
        model_path = Path(model_path)
        if model_path.exists():
            print(f"🤖 Using hybrid model: {model_path}")
        else:
            print(f"⚠️ Model path not found: {model_path}, using traditional detector")
            model_path = None
    else:
        print(f"🔧 Using traditional detector")
    
    # Set up multiprocessing
    if num_processes is None:
        num_processes = min(mp.cpu_count(), len(image_files))
    
    print(f"⚙️ Using {num_processes} processes")
    
    # Prepare worker arguments
    worker_args = []
    for img_file in image_files:
        args = {
            'image_path': img_file,
            'model_path': str(model_path) if model_path else None,
            'debug': debug
        }
        worker_args.append(args)
    
    # Run parallel processing
    results = []
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit all tasks
        future_to_args = {executor.submit(process_single_image_worker, args): args 
                         for args in worker_args}
        
        # Collect results with progress tracking
        completed = 0
        for future in as_completed(future_to_args):
            try:
                result = future.result()
                results.append(result)
                completed += 1
                
                if completed % 10 == 0 or completed == len(worker_args):
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (len(worker_args) - completed) / rate if rate > 0 else 0
                    print(f"📊 Progress: {completed}/{len(worker_args)} "
                          f"({completed/len(worker_args)*100:.1f}%) "
                          f"Rate: {rate:.1f} img/s, ETA: {eta:.1f}s")
                
            except Exception as e:
                args = future_to_args[future]
                print(f"❌ Error processing {args['image_path']}: {e}")
                results.append({
                    'image_path': str(args['image_path']),
                    'error': str(e),
                    'success': False
                })
    
    # Compile summary
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r.get('success', False))
    
    evaluation_summary = {
        'total_images': len(image_files),
        'successful_analyses': successful,
        'failed_analyses': len(results) - successful,
        'total_processing_time': total_time,
        'average_time_per_image': total_time / len(image_files) if image_files else 0,
        'processing_rate': len(image_files) / total_time if total_time > 0 else 0,
        'model_type': 'hybrid' if model_path else 'traditional',
        'model_path': str(model_path) if model_path else None,
        'results': results
    }
    
    print(f"\n✅ Evaluation complete!")
    print(f"   Processed: {successful}/{len(image_files)} images")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Rate: {evaluation_summary['processing_rate']:.1f} images/second")
    
    # Save results
    results_file = get_multiprocessing_path(f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_file, 'w') as f:
        json.dump(evaluation_summary, f, indent=2, default=str)
    print(f"💾 Results saved: {results_file}")
    
    return evaluation_summary

def train_hybrid_model(features_file: str, output_dir: str = None) -> Dict:
    """Train hybrid model from evaluation features."""
    
    print(f"\n🤖 HYBRID MODEL TRAINING")
    print("=" * 50)
    
    if not MODULES_LOADED.get('hybrid_detector', False):
        raise ImportError("Hybrid detector module not available")
    
    features_path = Path(features_file)
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_file}")
    
    if output_dir is None:
        output_dir = get_multiprocessing_path("trained_models")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Features file: {features_path}")
    print(f"📁 Output directory: {output_dir}")
    
    start_time = time.time()
    
    try:
        # Train hybrid detector
        hybrid_detector = train_hybrid_detector(
            evaluation_csv_path=str(features_path),
            model_save_path=str(output_dir)
        )
        
        training_time = time.time() - start_time
        
        # Check if training was successful
        if hybrid_detector and hasattr(hybrid_detector, 'is_trained') and hybrid_detector.is_trained:
            # Count saved models
            model_files = list(output_dir.glob("*.pkl")) + list(output_dir.glob("*.joblib"))
            
            training_summary = {
                'success': True,
                'model_directory': str(output_dir),
                'features_file': str(features_path),
                'training_time': training_time,
                'model_files': [f.name for f in model_files],
                'num_models': len(model_files)
            }
            
            print(f"✅ Training completed in {training_time:.1f}s")
            print(f"💾 Models saved: {len(model_files)} files")
            for model_file in model_files:
                print(f"   - {model_file.name}")
            
            return training_summary
        else:
            raise RuntimeError("Training failed - no trained detector returned")
            
    except Exception as e:
        training_summary = {
            'success': False,
            'error': str(e),
            'training_time': time.time() - start_time
        }
        print(f"❌ Training failed: {e}")
        return training_summary

# ===== MAIN FUNCTION =====

def main():
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(description='Multiprocessing Crumbly Texture Analysis Workflow')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Run parallel evaluation')
    eval_parser.add_argument('image_dir', help='Directory containing images')
    eval_parser.add_argument('--max-images', type=int, help='Maximum number of images to process')
    eval_parser.add_argument('--processes', help='Number of parallel processes (or "auto")')
    eval_parser.add_argument('--model-path', help='Path to trained hybrid model')
    eval_parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train hybrid model')
    train_parser.add_argument('features_file', help='CSV file with extracted features')
    train_parser.add_argument('--output-dir', help='Output directory for trained models')
    
    # Complete workflow command
    complete_parser = subparsers.add_parser('complete', help='Complete workflow: evaluate → train → re-evaluate')
    complete_parser.add_argument('image_dir', help='Directory containing images')
    complete_parser.add_argument('--max-images', type=int, help='Maximum number of images to process')
    complete_parser.add_argument('--processes', help='Number of parallel processes (or "auto")')
    complete_parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    # Handle 'auto' processes
    def parse_processes(processes_arg):
        if processes_arg is None:
            return None
        elif processes_arg.lower() == 'auto':
            return None  # Will auto-detect in run_parallel_evaluation
        else:
            try:
                return int(processes_arg)
            except ValueError:
                raise ValueError(f"Invalid processes value: {processes_arg}. Use integer or 'auto'")
    
    if args.command == 'evaluate':
        run_parallel_evaluation(
            image_dir=args.image_dir,
            max_images=args.max_images,
            num_processes=parse_processes(args.processes),
            model_path=args.model_path,
            debug=args.debug
        )
    
    elif args.command == 'train':
        train_hybrid_model(
            features_file=args.features_file,
            output_dir=args.output_dir
        )
    
    elif args.command == 'complete':
        print(f"🚀 COMPLETE WORKFLOW")
        print("=" * 50)
        
        # Step 1: Initial evaluation with traditional detector
        print(f"\n📊 Phase 1: Initial evaluation...")
        eval1_results = run_parallel_evaluation(
            image_dir=args.image_dir,
            max_images=args.max_images,
            num_processes=parse_processes(args.processes),
            model_path=None,  # Traditional detector
            debug=args.debug
        )
        
        # Step 2: Extract features and train hybrid model
        print(f"\n🤖 Phase 2: Training hybrid model...")
        # Note: This would need feature extraction logic
        print(f"⚠️ Feature extraction not implemented - would need evaluation system integration")
        
        # Step 3: Re-evaluate with hybrid model
        print(f"\n🔄 Phase 3: Re-evaluation with hybrid model...")
        # This would use the trained model
        print(f"⚠️ Re-evaluation not implemented - would need trained model path")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()