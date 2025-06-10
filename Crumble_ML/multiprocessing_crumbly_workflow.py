#!/usr/bin/env python3
"""
Multiprocessing Crumbly Texture Analysis Workflow
Integrates multiprocessing for fast parallel evaluation and training.

FEATURES:
1. Parallel evaluation using multiple CPU cores
2. Memory-efficient processing
3. Progress tracking with ETA
4. Same analysis quality as single-threaded version
5. Automatic CPU core detection and optimization

Usage Examples:
1. Fast parallel evaluation:
   python multiprocessing_crumbly_workflow.py evaluate /path/to/dataset --max-images 50 --processes auto

2. Train hybrid model:
   python multiprocessing_crumbly_workflow.py train /path/to/features.csv

3. Complete parallel workflow:
   python multiprocessing_crumbly_workflow.py complete /path/to/dataset --max-images 100 --processes 4
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

WORKFLOW_DEBUG = False  # Set to True for workflow debugging

# ===== CROSS-PLATFORM TIMEOUT PROTECTION =====
class TimeoutException(Exception):
    pass

def with_timeout(timeout_seconds):
    """Cross-platform timeout decorator that works on Windows and Unix."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout_seconds)
            
            if thread.is_alive():
                print(f"⏰ Operation timed out after {timeout_seconds} seconds")
                raise TimeoutException(f"Operation timed out after {timeout_seconds} seconds")
            
            if exception[0]:
                raise exception[0]
            
            return result[0]
        return wrapper
    return decorator

# ===== PATH SETUP (MATCHES comprehensive_analyzer_main.py EXACTLY) =====
project_root = Path(__file__).parent.parent  # Crumble_ML/ -> project_root/
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "modules"))

print(f"🔧 Multiprocessing crumbly workflow setup:")
print(f"   Project root: {project_root}")
print(f"   Modules path: {project_root / 'modules'}")
print(f"   CPU cores available: {mp.cpu_count()}")

# ===== IMPORT RESULTS CONFIGURATION =====
try:
    from results_config import (
        get_multiprocessing_path, get_excel_report_path, get_json_results_path,
        MULTIPROCESSING_DIR, get_results_info, print_results_structure,
        ensure_directory_exists
    )
    RESULTS_CONFIGURED = True
    print("✅ Centralized results configuration loaded")
except ImportError as e:
    RESULTS_CONFIGURED = False
    print(f"⚠️ Results config not found: {e}")
    MULTIPROCESSING_DIR = Path("multiprocessing_crumbly_results")
    MULTIPROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    
    def get_multiprocessing_path(filename: str) -> Path:
        return MULTIPROCESSING_DIR / filename
    
    def ensure_directory_exists(path: str) -> Path:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return p

# ===== IMPORT CORE MODULES =====
print("🔧 Loading SEM Fiber Analysis modules...")

# Track available modules
MODULES_LOADED = {}
POROSITY_AVAILABLE = False
POROSITY_TYPE = None

try:
    from modules.scale_detection import ScaleBarDetector, detect_scale_bar
    from modules.fiber_type_detection import FiberTypeDetector, detect_fiber_type
    from modules.image_preprocessing import load_image, preprocess_pipeline
    MODULES_LOADED['core'] = True
    print("✅ Core modules loaded successfully")
except ImportError as e:
    print(f"❌ Could not import core modules: {e}")
    sys.exit(1)

# Try to import the new preprocessing function
try:
    from modules.image_preprocessing import preprocess_for_analysis
    print("✅ Enhanced preprocessing function available")
    HAS_ENHANCED_PREPROCESSING = True
except ImportError:
    print("⚠️ Enhanced preprocessing not available, using fallback")
    from modules.image_preprocessing import enhance_contrast, denoise_image, normalize_image
    HAS_ENHANCED_PREPROCESSING = False

# Porosity analysis with fallback logic
try:
    from modules.porosity_analysis import PorosityAnalyzer, analyze_fiber_porosity, quick_porosity_check
    print("✅ Fast refined porosity module loaded")
    POROSITY_AVAILABLE = True
    POROSITY_TYPE = "fast_refined"
    MODULES_LOADED['porosity_analysis'] = 'fast_refined'
except ImportError:
    print("⚠️ Fast refined porosity module not found, trying legacy versions...")
    try:
        from modules.porosity_analysis import EnhancedPorosityAnalyzer, analyze_fiber_porosity_enhanced
        print("✅ Enhanced porosity module loaded")
        POROSITY_AVAILABLE = True
        POROSITY_TYPE = "enhanced"
        MODULES_LOADED['porosity_analysis'] = 'enhanced'
    except ImportError:
        try:
            from modules.porosity_analysis import PorosityAnalyzer
            print("✅ Basic porosity module loaded")
            POROSITY_AVAILABLE = True
            POROSITY_TYPE = "basic"
            MODULES_LOADED['porosity_analysis'] = 'basic'
        except ImportError:
            print("❌ No porosity analysis available")
            POROSITY_AVAILABLE = False
            POROSITY_TYPE = None
            MODULES_LOADED['porosity_analysis'] = False

# Crumbly detection module
try:
    from modules.crumbly_detection import CrumblyDetector
    MODULES_LOADED['crumbly_detection'] = True
    print("✅ CrumblyDetector loaded from modules/")
except ImportError as e:
    try:
        from crumbly_detection import CrumblyDetector
        MODULES_LOADED['crumbly_detection'] = True
        print("✅ CrumblyDetector loaded from Crumble_ML/")
    except ImportError as e2:
        print(f"❌ CrumblyDetector not available: {e}")
        MODULES_LOADED['crumbly_detection'] = False

# Import training functions
try:
    from hybrid_crumbly_detector import train_hybrid_detector, load_hybrid_detector
    MODULES_LOADED['hybrid_detector'] = True
    print("✅ Hybrid detector training functions loaded")
except ImportError as e:
    print(f"❌ Hybrid detector functions not available: {e}")
    MODULES_LOADED['hybrid_detector'] = False

# ===== WORKER FUNCTIONS =====
def preprocess_image_for_worker(image):
    """Preprocess image for worker processes."""
    try:
        if HAS_ENHANCED_PREPROCESSING:
            return preprocess_for_analysis(image, silent=True)
        else:
            # Fallback preprocessing
            enhanced = enhance_contrast(image, method='clahe')
            denoised = denoise_image(enhanced, method='bilateral')
            normalized = normalize_image(denoised)
            return normalized
    except Exception as e:
        return image

def fix_fiber_mask_extraction_worker(image, fiber_analysis_data):
    """Worker version of fiber mask extraction - FIXED to match comprehensive_analyzer_main.py"""
    
    if WORKFLOW_DEBUG == True:
        print(f"   🔧 Extracting fiber mask from analysis data...")
        print(f"   📊 Analysis data keys: {list(fiber_analysis_data.keys()) if fiber_analysis_data else 'None'}")
    
    # Get fiber mask from analysis data (direct format from classify_fiber_type)
    fiber_mask = fiber_analysis_data.get('fiber_mask') if fiber_analysis_data else None
    
    if fiber_mask is not None and isinstance(fiber_mask, np.ndarray):
        # Ensure proper format
        if fiber_mask.dtype != np.uint8:
            fiber_mask = (fiber_mask > 0).astype(np.uint8) * 255
        
        # Check if mask has sufficient content
        mask_area = np.sum(fiber_mask > 0)
        print(f"   ✅ Fiber mask extracted: {mask_area:,} pixels")
        
        if mask_area > 1000:  # Reasonable minimum
            return fiber_mask
        else:
            print(f"   ⚠️ Mask too small: {mask_area} pixels")
    else:
        print(f"   ❌ No valid fiber mask found in analysis data")
        print(f"   📊 Fiber mask type: {type(fiber_mask)}")
    
    # Return empty mask as fallback
    return np.zeros(image.shape[:2], dtype=np.uint8)

def process_single_image_worker(worker_args: Dict) -> Dict:
    """
    Worker function for processing a single image in parallel.
    This runs in a separate process.
    """
    
    image_info = worker_args['image_info']
    config = worker_args.get('config', {})
    model_path = config.get('model_path', None)  # NEW: Get model path from config
    
    start_time = time.time()
    process_id = os.getpid()
    
    result = {
        'image_path': str(image_info['path']),
        'image_name': Path(image_info['path']).name,
        'true_label': image_info['true_label'],
        'process_id': process_id,
        'processing_success': False,
        'predicted_label': 'unknown',
        'prediction_confidence': 0.0,
        'crumbly_score': 0.5,
        'ml_features': {},
        'total_processing_time': 0.0,
        'model_type': 'traditional'  # NEW: Track which model type was used
    }
    
    try:
        # Monitor memory usage
        try:
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        except:
            initial_memory = 0
        
        # Load image
        original_image = load_image(str(image_info['path']))
        if original_image is None:
            result['error'] = "Failed to load image"
            return result
        
        # Preprocess image
        preprocessed = preprocess_image_for_worker(original_image)
        
        result.update({
            'image_shape': original_image.shape,
            'image_size_mb': os.path.getsize(image_info['path']) / (1024 * 1024)
        })
        
        # Scale detection
        scale_factor = 1.0
        try:
            scale_detector = ScaleBarDetector(use_enhanced_detection=True)
            scale_result = scale_detector.detect_scale_bar(
                original_image, debug=False, save_debug_image=False, output_dir=None
            )
            if scale_result and scale_result.get('scale_detected', False):
                scale_factor = scale_result.get('micrometers_per_pixel', 1.0)
            result['scale_detection'] = scale_result
        except Exception as e:
            result['scale_detection'] = {'error': str(e)}
            scale_factor = 1.0
        
         # Fiber type detection (FIXED - Use same API as comprehensive_analyzer_main.py)
        try:
            fiber_detector = FiberTypeDetector()
            fiber_type, confidence, fiber_analysis_data = fiber_detector.classify_fiber_type(preprocessed)
            
            # Extract fiber mask using the working method
            fiber_mask = fix_fiber_mask_extraction_worker(preprocessed, fiber_analysis_data)
            
            # Create result structure similar to comprehensive_analyzer_main.py
            result['fiber_detection'] = {
                'fiber_type': fiber_type,
                'confidence': confidence,
                'total_fibers': fiber_analysis_data.get('total_fibers', 0),
                'hollow_fibers': fiber_analysis_data.get('hollow_fibers', 0),
                'filaments': fiber_analysis_data.get('filaments', 0),
                'method': 'classify_fiber_type'
            }
            
        except Exception as e:
            result['fiber_detection'] = {'error': str(e)}
            # Create empty mask as fallback
            fiber_mask = np.zeros(preprocessed.shape[:2], dtype=np.uint8)
        
        # Porosity analysis
        porosity_features = {}
        if MODULES_LOADED['porosity_analysis']:
            try:
                if POROSITY_TYPE == 'fast_refined':
                    porosity_result = analyze_fiber_porosity(preprocessed, fiber_mask, scale_factor)
                elif POROSITY_TYPE == 'enhanced':
                    porosity_analyzer = EnhancedPorosityAnalyzer()
                    porosity_result = porosity_analyzer.analyze_fiber_porosity(
                        preprocessed, fiber_mask, scale_factor
                    )
                else:
                    porosity_result = quick_porosity_check(preprocessed, fiber_mask, scale_factor)
                
                porosity_metrics = porosity_result.get('porosity_metrics', {})
                porosity_features = {
                    'total_porosity_percent': porosity_metrics.get('total_porosity_percent', 0),
                    'pore_count': porosity_metrics.get('pore_count', 0),
                    'average_pore_size': porosity_metrics.get('average_pore_size_um2', 0),
                    'pore_density': porosity_metrics.get('pore_density_per_mm2', 0)
                }
                result['porosity_analysis'] = porosity_result
            except Exception as e:
                result['porosity_error'] = str(e)
        
        # NEW: Choose detector based on model_path
        if model_path and MODULES_LOADED['hybrid_detector']:
            try:
                from hybrid_crumbly_detector import load_hybrid_detector
                crumbly_detector = load_hybrid_detector(model_path)
                result['model_type'] = 'hybrid'
                # Uncomment for debugging: print(f"     🤖 Using hybrid model")
            except Exception as e:
                print(f"     ⚠️ Failed to load hybrid model: {e}, falling back to traditional")
                crumbly_detector = CrumblyDetector(porosity_aware=True)
                result['model_type'] = 'traditional_fallback'
        else:
            # Use traditional detector
            crumbly_detector = CrumblyDetector(porosity_aware=True)
            result['model_type'] = 'traditional'
        
                # Crumbly texture analysis
        # REPLACE the crumbly analysis section in process_single_image_worker() function
# Around line 180-220 in multiprocessing_crumbly_workflow.py

        # Crumbly texture analysis
        if MODULES_LOADED['crumbly_detection']:
            try:
                fiber_mask_bool = fiber_mask > 127
                
                crumbly_result = crumbly_detector.analyze_crumbly_texture(
                    preprocessed, fiber_mask_bool, None, scale_factor, debug=False
                )
                
                if crumbly_result and 'classification' in crumbly_result:
                    # FIXED: Handle both traditional and hybrid detector results
                    if result.get('model_type', '') == 'hybrid' and 'traditional_result' in crumbly_result:
                        # For hybrid detector, extract from traditional_result
                        traditional_data = crumbly_result['traditional_result']
                        predicted_label = crumbly_result['classification']  # Use final hybrid prediction
                        confidence = crumbly_result.get('confidence', 0.0)
                        crumbly_score = crumbly_result.get('crumbly_score', traditional_data.get('crumbly_score', 0.5))
                        
                        # Extract evidence from traditional_result
                        porous_evidence = traditional_data.get('porous_evidence', 0)
                        crumbly_evidence = traditional_data.get('crumbly_evidence', 0)
                        intermediate_evidence = traditional_data.get('intermediate_evidence', 0)
                        confidence_factors = traditional_data.get('confidence_factors', [])
                        
                        # Debug info shows hybrid was used
                        print(f"     🤖 Using hybrid model for {Path(image_info['path']).name}")
                    else:
                        # For traditional detector, use direct results
                        predicted_label = crumbly_result['classification']
                        confidence = crumbly_result.get('confidence', 0.0)
                        crumbly_score = crumbly_result.get('crumbly_score', 0.5)
                        
                        # Extract evidence directly
                        porous_evidence = crumbly_result.get('porous_evidence', 0)
                        crumbly_evidence = crumbly_result.get('crumbly_evidence', 0)
                        intermediate_evidence = crumbly_result.get('intermediate_evidence', 0)
                        confidence_factors = crumbly_result.get('confidence_factors', [])

                    # DEBUG SECTION (same for both):
                    print(f"     🔍 DEBUG for {Path(image_info['path']).name}:")
                    print(f"         Model type: {result.get('model_type', 'unknown')}")
                    print(f"         Porous evidence: {porous_evidence:.3f}")
                    print(f"         Crumbly evidence: {crumbly_evidence:.3f}")
                    print(f"         Intermediate evidence: {intermediate_evidence:.3f}")
                    print(f"         Confidence factors: {confidence_factors[:3]}")  # Show first 3
                    
                    result['predicted_label'] = predicted_label
                    result['prediction_confidence'] = confidence
                    result['crumbly_score'] = crumbly_score
                    result['crumbly_analysis'] = crumbly_result
                    result['processing_success'] = True
                    
                    # Extract ML features - COMPATIBLE WITH TRAINED MODELS (9 features only)
                    ml_features = {}

                    # Original 4 porosity features (from training data)
                    ml_features['total_porosity_percent'] = porosity_features.get('total_porosity_percent', 0)
                    ml_features['pore_count'] = porosity_features.get('pore_count', 0)
                    ml_features['average_pore_size'] = porosity_features.get('average_pore_size', 0)
                    ml_features['pore_density'] = porosity_features.get('pore_density', 0)

                    # Original 2 crumbly features (from training data)
                    ml_features['crumbly_score'] = crumbly_score
                    ml_features['traditional_classification_confidence'] = confidence

                    # Original 3 additional features (from training data structure)
                    # Extract these from the appropriate result structure
                    if result.get('model_type', '') == 'hybrid' and 'traditional_result' in crumbly_result:
                        # Extract from nested traditional_result
                        trad_result = crumbly_result['traditional_result']
                        
                        # Try to get wall integrity score
                        if 'wall_integrity_metrics' in trad_result:
                            wall_metrics = trad_result['wall_integrity_metrics']
                            ml_features['wall_integrity_score'] = wall_metrics.get('wall_integrity_score', 0.5)
                        else:
                            ml_features['wall_integrity_score'] = 0.5
                        
                        # Try to get organized porosity score
                        if 'pore_metrics' in trad_result:
                            pore_metrics = trad_result['pore_metrics']
                            ml_features['organized_porosity_score'] = pore_metrics.get('organized_porosity_score', 0.5)
                        else:
                            ml_features['organized_porosity_score'] = 0.5
                        
                        # Mean pore circularity
                        if 'pore_metrics' in trad_result:
                            pore_metrics = trad_result['pore_metrics']
                            ml_features['mean_pore_circularity'] = pore_metrics.get('mean_pore_circularity', 0.5)
                        else:
                            ml_features['mean_pore_circularity'] = 0.5
                            
                    else:
                        # Extract from direct traditional result
                        if 'wall_integrity_metrics' in crumbly_result:
                            wall_metrics = crumbly_result['wall_integrity_metrics']
                            ml_features['wall_integrity_score'] = wall_metrics.get('wall_integrity_score', 0.5)
                        else:
                            ml_features['wall_integrity_score'] = 0.5
                        
                        if 'pore_metrics' in crumbly_result:
                            pore_metrics = crumbly_result['pore_metrics']
                            ml_features['organized_porosity_score'] = pore_metrics.get('organized_porosity_score', 0.5)
                            ml_features['mean_pore_circularity'] = pore_metrics.get('mean_pore_circularity', 0.5)
                        else:
                            ml_features['organized_porosity_score'] = 0.5
                            ml_features['mean_pore_circularity'] = 0.5
                    
                    result['ml_features'] = ml_features
                else:
                    result['error'] = "Crumbly analysis returned invalid result"
            except Exception as e:
                result['error'] = f"Crumbly analysis failed: {e}"
                import traceback
                print(f"     ❌ Crumbly analysis error: {e}")
                print(f"     Traceback: {traceback.format_exc()}")
        else:
            result['error'] = "Crumbly detector not available"
        
        # Final timing and memory
        result['total_processing_time'] = time.time() - start_time
        
        try:
            final_memory = process.memory_info().rss / 1024 / 1024
            result['memory_usage_mb'] = final_memory - initial_memory
        except:
            result['memory_usage_mb'] = 0
    
    except Exception as e:
        result['error'] = f"Processing failed: {str(e)}"
        result['total_processing_time'] = time.time() - start_time
    
    return result

class MultiprocessingCrumblyWorkflowManager:
    """
    Multiprocessing-enabled crumbly workflow manager for fast parallel processing.
    """
    
    def __init__(self, output_dir: str = "multiprocessing_crumbly_results", num_processes: Optional[int] = None):
        # Use results_config if available
        if RESULTS_CONFIGURED:
            self.output_dir = ensure_directory_exists(output_dir)
        else:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine number of processes
        if num_processes is None:
            self.num_processes = max(1, mp.cpu_count() - 1)  # Leave one core free
        else:
            self.num_processes = max(1, min(num_processes, mp.cpu_count()))
        
        self.workflow_results = {}
        
        print(f"🚀 Multiprocessing Crumbly Workflow Manager Initialized")
        print(f"   Output directory: {self.output_dir}")
        print(f"   CPU cores available: {mp.cpu_count()}")
        print(f"   Processes to use: {self.num_processes}")
    
    def get_image_files(self, dataset_path: str) -> List[Dict]:
        """Get all image files with flexible folder-to-label mapping."""
        dataset_path = Path(dataset_path)
        image_files = []
        
        # More comprehensive mapping including common folder naming conventions
        label_map = {
            # Crumbly variations
            'crumbly': {'label': 'crumbly', 'numeric': 2},
            'crumbly_texture': {'label': 'crumbly', 'numeric': 2},
            'rough': {'label': 'crumbly', 'numeric': 2},
            
            # Intermediate variations  
            'intermediate': {'label': 'intermediate', 'numeric': 1},
            'medium': {'label': 'intermediate', 'numeric': 1},
            'semi_crumbly': {'label': 'intermediate', 'numeric': 1},
            
            # Not crumbly variations (smooth/porous)
            'not': {'label': 'porous', 'numeric': 0},
            'not_crumbly': {'label': 'porous', 'numeric': 0},
            'smooth': {'label': 'porous', 'numeric': 0},
            'porous': {'label': 'porous', 'numeric': 0},  # Key addition for your case
            'organized': {'label': 'porous', 'numeric': 0},
            'clean': {'label': 'porous', 'numeric': 0}
        }
        
        # Log which folders are found
        found_folders = []
        for folder_name, label_info in label_map.items():
            folder_path = dataset_path / folder_name
            if folder_path.exists():
                found_folders.append(f"{folder_name} -> {label_info['label']}")
                extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp']
                for ext in extensions:
                    for img_file in folder_path.glob(ext):
                        image_files.append({
                            'path': img_file,
                            'true_label': label_info['label'],
                            'label_numeric': label_info['numeric'],
                            'source_folder': folder_name  # Track original folder
                        })
        
        print(f"   Found folders: {', '.join(found_folders)}")
        print(f"   Found {len(image_files)} images total")
        return image_files
    
    @with_timeout(1800)  # 30 minute timeout for evaluation
    def run_parallel_evaluation(self, dataset_path: str, max_images: Optional[int] = None) -> Dict:
        """
        Run parallel evaluation of the crumbly detector.
        """
        print(f"\n🚀 STARTING PARALLEL EVALUATION")
        print("=" * 60)
        print(f"   Dataset: {dataset_path}")
        print(f"   Max images: {max_images or 'all'}")
        print(f"   Processes: {self.num_processes}")
        
        start_time = time.time()
        
        try:
            # Validate dataset
            dataset_path_obj = Path(dataset_path)
            if not dataset_path_obj.exists():
                raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
            
            # Get image files
            image_files = self.get_image_files(dataset_path)
            
            if max_images:
                image_files = image_files[:max_images]
                print(f"   Limiting to {max_images} images for testing")
            
            if not image_files:
                raise ValueError("No image files found in dataset")
            
            print(f"   Processing {len(image_files)} images...")
            
            # Prepare worker arguments
            worker_args = [
                {'image_info': image_info, 'config': {}} 
                for image_info in image_files
            ]
            
            # Process images in parallel
            results = []
            successful_processes = 0
            
            print(f"\n⚡ Starting parallel processing...")
            
            with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                # Submit all jobs
                future_to_image = {
                    executor.submit(process_single_image_worker, arg): arg['image_info']['path']
                    for arg in worker_args
                }
                
                completed = 0
                
                # Collect results as they complete
                for future in as_completed(future_to_image):
                    image_path = future_to_image[future]
                    completed += 1
                    
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result['processing_success']:
                            successful_processes += 1
                            status = "✅"
                            
                            # Show prediction results
                            predicted = result['predicted_label']
                            confidence = result['prediction_confidence']
                            true_label = result.get('true_label', 'unknown')
                            match = "✓" if predicted == true_label else "✗"
                            
                            prediction_info = f"{predicted} (conf: {confidence:.2f}) {match}"
                        else:
                            status = "❌"
                            prediction_info = f"Failed: {result.get('error', 'Unknown error')[:30]}..."
                        
                        # Progress update with ETA
                        progress = completed / len(image_files) * 100
                        elapsed = time.time() - start_time
                        eta = elapsed * (len(image_files) - completed) / completed if completed > 0 else 0
                        
                        print(f"{status} [{completed:3d}/{len(image_files)}] {progress:5.1f}% | "
                              f"{Path(image_path).name:<25} | "
                              f"Time: {result.get('total_processing_time', 0):5.2f}s | "
                              f"{prediction_info} | "
                              f"ETA: {eta:5.0f}s")
                        
                    except Exception as e:
                        print(f"❌ [{completed:3d}/{len(image_files)}] Failed: {Path(image_path).name} - {e}")
                        results.append({
                            'image_path': str(image_path),
                            'image_name': Path(image_path).name,
                            'processing_success': False,
                            'error': str(e)
                        })
            
            total_time = time.time() - start_time
            success_rate = successful_processes / len(image_files) * 100
            
            print(f"\n📊 PARALLEL EVALUATION COMPLETE")
            print(f"   Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
            print(f"   Success rate: {success_rate:.1f}% ({successful_processes}/{len(image_files)})")
            print(f"   Images per second: {len(image_files)/total_time:.1f}")
            print(f"   Average time per image: {total_time/len(image_files):.2f}s")
            
            # Analyze results
            evaluation_results = self.analyze_parallel_results(results)
            evaluation_results.update({
                'success': True,
                'total_images': len(image_files),
                'successful_images': successful_processes,
                'processing_time': total_time,
                'dataset_path': dataset_path,
                'num_processes_used': self.num_processes,
                'images_per_second': len(image_files) / total_time if total_time > 0 else 0
            })
            
            # Save results
            self.save_parallel_results(results, evaluation_results)
            
            self.workflow_results['evaluation'] = evaluation_results
            return evaluation_results
            
        except TimeoutException:
            print(f"   ⏰ Parallel evaluation timed out after 30 minutes")
            evaluation_results = {
                'success': False,
                'error': 'Parallel evaluation timed out',
                'processing_time': time.time() - start_time
            }
            self.workflow_results['evaluation'] = evaluation_results
            return evaluation_results
            
        except Exception as e:
            print(f"   ❌ Parallel evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            evaluation_results = {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
            self.workflow_results['evaluation'] = evaluation_results
            return evaluation_results
    
    def analyze_parallel_results(self, results: List[Dict]) -> Dict:
        """Analyze parallel evaluation results."""
        print(f"\n📈 ANALYZING PARALLEL RESULTS")
        print("=" * 40)
        
        successful_results = [r for r in results if r.get('processing_success', False)]
        
        if not successful_results:
            print("❌ No successful results to analyze!")
            return {'overall_accuracy': 0.0}
        
        # Calculate accuracy
        correct_predictions = sum(1 for r in successful_results 
                                if r.get('predicted_label') == r.get('true_label'))
        accuracy = correct_predictions / len(successful_results)
        
        # Generate detailed metrics
        from sklearn.metrics import classification_report
        
        true_labels = [r.get('true_label', 'unknown') for r in successful_results]
        predicted_labels = [r.get('predicted_label', 'unknown') for r in successful_results]
        
        print(f"📊 Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"📊 Correct predictions: {correct_predictions}/{len(successful_results)}")
        
        try:
            classification_rep = classification_report(true_labels, predicted_labels, output_dict=True)
            print(f"📊 Classification Report:")
            print(classification_report(true_labels, predicted_labels))
        except Exception as e:
            print(f"⚠️ Could not generate classification report: {e}")
            classification_rep = {}
        
        return {
            'overall_accuracy': accuracy,
            'total_images': len(results),
            'successful_images': len(successful_results),
            'correct_predictions': correct_predictions,
            'classification_report': classification_rep
        }
    
    def save_parallel_results(self, results: List[Dict], analysis_results: Dict):
        """Save parallel processing results."""
        print(f"\n💾 SAVING PARALLEL RESULTS")
        print("=" * 30)
        
        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_dir / f"parallel_evaluation_run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_df = pd.DataFrame(results)
        results_file = run_dir / "detailed_results.csv"
        results_df.to_csv(results_file, index=False)
        print(f"   📊 Detailed results: {results_file.name}")
        
        # Save ML features for training
        successful_results = [r for r in results if r.get('processing_success', False) and r.get('ml_features')]
        if successful_results:
            feature_data = []
            labels = []
            
            for result in successful_results:
                if result.get('ml_features'):
                    feature_row = result['ml_features'].copy()
                    feature_data.append(feature_row)
                    
                    # Get numeric label
                    true_label = result.get('true_label', 'unknown')
                    label_map = {'crumbly': 2, 'intermediate': 1, 'not': 0}
                    labels.append(label_map.get(true_label, 1))
            
            if feature_data:
                features_df = pd.DataFrame(feature_data)
                features_df['true_label'] = labels
                # Add text labels for better compatibility
                label_map = {0: 'not', 1: 'intermediate', 2: 'crumbly'}
                features_df['true_label_name'] = [label_map[label] for label in labels]
                features_file = run_dir / "ml_features_dataset.csv"
                features_df.to_csv(features_file, index=False)
                print(f"   🤖 ML features: {features_file.name}")
        
        # Save analysis summary
        analysis_file = run_dir / "analysis_summary.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        print(f"   📈 Analysis summary: {analysis_file.name}")
        
        print(f"   📁 All results saved to: {run_dir}")
    
    def train_hybrid_model(self, features_file: str) -> Dict:
        """Train hybrid model (same as single-threaded version)."""
        print(f"\n🤖 Starting Training Phase...")
        print(f"   Features file: {features_file}")
        
        start_time = time.time()
        
        try:
            if not MODULES_LOADED['hybrid_detector']:
                raise ImportError("Hybrid detector training functions not available")
            
            features_path = Path(features_file)
            if not features_path.exists():
                raise FileNotFoundError(f"Features file not found: {features_file}")
            
            # Use results_config for output paths if available
            if RESULTS_CONFIGURED:
                model_dir = get_multiprocessing_path("trained_models")
            else:
                model_dir = self.output_dir / "trained_models"
            
            model_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"   Model output directory: {model_dir}")
            print(f"   Training hybrid model...")
            
            # Train the hybrid model
            try:
                hybrid_detector = train_hybrid_detector(
                    evaluation_csv_path=str(features_path),
                    model_save_path=str(model_dir)
                )
                
                # Create success dictionary since train_hybrid_detector returns a detector object, not a dict
                if hybrid_detector and hybrid_detector.is_trained:
                    training_results = {'success': True, 'detector': hybrid_detector}
                else:
                    training_results = {'success': False, 'error': 'Training failed'}
                    
            except Exception as e:
                training_results = {'success': False, 'error': str(e)}

            if training_results and training_results.get('success', False):
                train_summary = {
                    'success': True,
                    'model_dir': str(model_dir),
                    'features_file': str(features_path),
                    'processing_time': time.time() - start_time,
                    'training_results': training_results
                }
                
                # Count models trained
                try:
                    model_files = list(model_dir.glob("*.pkl")) + list(model_dir.glob("*.joblib"))
                    train_summary['num_ml_models'] = len(model_files)
                    train_summary['has_ensemble'] = any('ensemble' in f.name.lower() for f in model_files)
                    print(f"   ✅ Training completed: {len(model_files)} models saved")
                except Exception as e:
                    print(f"   ⚠️ Could not count model files: {e}")
                
                self.workflow_results['training'] = train_summary
                return train_summary
            else:
                error_msg = training_results.get('error', 'Unknown training error')
                train_summary = {
                    'success': False,
                    'error': error_msg,
                    'processing_time': time.time() - start_time
                }
                self.workflow_results['training'] = train_summary
                return train_summary
                
        except Exception as e:
            print(f"   ❌ Training failed: {e}")
            train_summary = {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
            self.workflow_results['training'] = train_summary
            return train_summary
    
    def run_parallel_evaluation_hybrid(self, dataset_path: str, model_path: str, max_images: Optional[int] = None) -> Dict:
        """
        Run parallel evaluation using trained hybrid models.
        
        Args:
            dataset_path: Path to dataset with labeled folders
            model_path: Path to trained hybrid models
            max_images: Maximum number of images to process
            
        Returns:
            Dictionary with evaluation results
        """
        
        print(f"\n🤖 PARALLEL HYBRID EVALUATION")
        print("=" * 60)
        print(f"   Dataset: {dataset_path}")
        print(f"   Model path: {model_path}")
        print(f"   Max images: {max_images or 'all'}")
        print(f"   Processes: {self.num_processes}")
        
        start_time = time.time()
        
        try:
            # Verify model path exists
            model_path_obj = Path(model_path)
            if not model_path_obj.exists():
                raise FileNotFoundError(f"Model path does not exist: {model_path}")
            
            # Test hybrid model loading
            print(f"🔍 Verifying hybrid models...")
            if not MODULES_LOADED['hybrid_detector']:
                raise ImportError("Hybrid detector not available")
            
            from hybrid_crumbly_detector import load_hybrid_detector
            test_detector = load_hybrid_detector(model_path)
            
            if not test_detector.is_trained:
                raise ValueError("Models not properly trained")
            
            print(f"✅ Hybrid models verified:")
            print(f"   Models available: {list(test_detector.ml_models.keys())}")
            print(f"   Ensemble available: {hasattr(test_detector, 'ensemble_model')}")
            print(f"   Feature count: {len(test_detector.feature_names)}")
            
            # Validate dataset
            dataset_path_obj = Path(dataset_path)
            if not dataset_path_obj.exists():
                raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
            
            # Get image files
            image_files = self.get_image_files(dataset_path)
            
            if max_images:
                image_files = image_files[:max_images]
                print(f"   Limiting to {max_images} images for testing")
            
            if not image_files:
                raise ValueError("No image files found in dataset")
            
            print(f"   Processing {len(image_files)} images with hybrid models...")
            
            # Prepare worker arguments WITH model path in config
            worker_args = [
                {
                    'image_info': image_info, 
                    'config': {'model_path': str(model_path)}  # NEW: Pass model path in config
                } 
                for image_info in image_files
            ]
            
            # Process images in parallel
            results = []
            successful_processes = 0
            
            print(f"\n⚡ Starting hybrid parallel processing...")
            
            with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                # Submit all jobs
                future_to_image = {
                    executor.submit(process_single_image_worker, arg): arg['image_info']['path']
                    for arg in worker_args
                }
                
                completed = 0
                
                # Collect results as they complete
                for future in as_completed(future_to_image):
                    image_path = future_to_image[future]
                    completed += 1
                    
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result['processing_success']:
                            successful_processes += 1
                            status = "✅"
                            
                            # Show prediction results
                            predicted = result['predicted_label']
                            confidence = result['prediction_confidence']
                            true_label = result.get('true_label', 'unknown')
                            match = "✓" if predicted == true_label else "✗"
                            model_type = result.get('model_type', 'unknown')
                            
                            prediction_info = f"{predicted} (conf: {confidence:.2f}) {match} [{model_type}]"
                        else:
                            status = "❌"
                            prediction_info = f"Failed: {result.get('error', 'Unknown error')[:30]}..."
                        
                        # Progress update with ETA
                        progress = completed / len(image_files) * 100
                        elapsed = time.time() - start_time
                        eta = elapsed * (len(image_files) - completed) / completed if completed > 0 else 0
                        
                        print(f"{status} [{completed:3d}/{len(image_files)}] {progress:5.1f}% | "
                            f"{Path(image_path).name:30s} | Time: {elapsed:5.1f}s | "
                            f"{prediction_info:50s} | ETA: {eta:5.0f}s")
                        
                    except Exception as e:
                        print(f"❌ [{completed:3d}/{len(image_files)}] Error processing {Path(image_path).name}: {e}")
                        results.append({
                            'image_path': str(image_path),
                            'processing_success': False,
                            'error': str(e)
                        })
            
            total_time = time.time() - start_time
            
            # Analyze results
            analysis_results = self.analyze_parallel_results(results)
            
            # Save results
            self.save_parallel_results(results, analysis_results)
            
            print(f"\n🎉 HYBRID EVALUATION COMPLETE!")
            print(f"   Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
            print(f"   Success rate: {successful_processes}/{len(image_files)} ({successful_processes/len(image_files)*100:.1f}%)")
            print(f"   Overall accuracy: {analysis_results.get('overall_accuracy', 0):.3f}")
            print(f"   Hybrid models used: ✅")
            
            return {
                'success': True,
                'total_images': len(image_files),
                'successful_images': successful_processes,
                'overall_accuracy': analysis_results.get('overall_accuracy', 0),
                'processing_time': total_time,
                'analysis_results': analysis_results,
                'model_type': 'hybrid'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'total_time': time.time() - start_time
            }


    def run_complete_parallel_workflow(self, dataset_path: str, max_images: Optional[int] = None,
                                     test_split: float = 0.3) -> Dict:
        """Run complete workflow with parallel evaluation."""
        print(f"\n🚀 Starting Complete Parallel Workflow...")
        print(f"   Dataset: {dataset_path}")
        print(f"   Max images: {max_images or 'all'}")
        print(f"   Processes: {self.num_processes}")
        print(f"   Test split: {test_split}")
        
        workflow_start = time.time()
        
        # Step 1: Parallel Evaluation
        print("\n" + "="*60)
        print("STEP 1: PARALLEL EVALUATION")
        print("="*60)
        
        eval_results = self.run_parallel_evaluation(dataset_path, max_images)
        if not eval_results.get('success', False):
            return {
                'success': False,
                'error': f"Parallel evaluation failed: {eval_results.get('error', 'Unknown error')}",
                'step': 'evaluation',
                'total_time': time.time() - workflow_start
            }
        
        # Step 2: Training
        print("\n" + "="*60)
        print("STEP 2: TRAINING")
        print("="*60)
        
        # Find the features file from evaluation results
        features_file = None
        try:
            # Look for the features file in the most recent run
            run_dirs = sorted([d for d in self.output_dir.glob("parallel_evaluation_run_*") if d.is_dir()], 
                            key=lambda x: x.stat().st_mtime, reverse=True)
            if run_dirs:
                latest_run = run_dirs[0]
                features_file = latest_run / "ml_features_dataset.csv"
                if features_file.exists():
                    print(f"   Found features file: {features_file}")
                else:
                    features_file = None
        except Exception as e:
            print(f"   ⚠️ Could not locate features file: {e}")
        
        if not features_file:
            return {
                'success': False,
                'error': 'No features file from evaluation',
                'step': 'evaluation->training',
                'total_time': time.time() - workflow_start
            }
        
        train_results = self.train_hybrid_model(str(features_file))
        
        # Generate workflow summary
        total_time = time.time() - workflow_start
        all_successful = all([
            eval_results.get('success', False),
            train_results.get('success', False)
        ])
        
        workflow_summary = {
            'success': True,
            'total_time': total_time,
            'evaluation': eval_results,
            'training': train_results,
            'workflow_complete': True,
            'all_steps_successful': all_successful,
            'parallel_processing': True,
            'num_processes_used': self.num_processes,
            'platform': platform.system(),
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\n🎉 COMPLETE PARALLEL WORKFLOW FINISHED!")
        print(f"   Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"   Parallel processing: {self.num_processes} processes")
        print(f"   All steps successful: {all_successful}")
        print(f"   Speed improvement: ~{self.num_processes:.1f}x faster than single-threaded")
        
        return workflow_summary

def main():
    """Command line interface for the multiprocessing crumbly workflow."""
    
    parser = argparse.ArgumentParser(
        description='Multiprocessing Crumbly Texture Analysis Workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s evaluate /path/to/dataset --max-images 50 --processes auto
  %(prog)s train /path/to/features.csv
  %(prog)s complete /path/to/dataset --max-images 100 --processes 4

Performance Tips:
  - Use --processes auto for optimal CPU utilization
  - Start with --max-images 20-50 for testing
  - Each process uses ~200-500MB RAM per image
        """
    )
    
    parser.add_argument('command', choices=['evaluate', 'evaluate-hybrid', 'train', 'complete'],
                   help='Workflow command to execute')
    parser.add_argument('path', help='Path to dataset or features file')
    parser.add_argument('--output', '-o', default='multiprocessing_crumbly_results',
                       help='Output directory')
    parser.add_argument('--max-images', type=int,
                       help='Maximum images to process (recommended: 20-100 for testing)')
    parser.add_argument('--processes', '-p', type=str, default='auto',
                       help='Number of processes (auto, or specific number)')
    parser.add_argument('--test-split', type=float, default=0.3,
                       help='Test split fraction for complete workflow')
    parser.add_argument('--model-path', type=str,  # NEW LINE
                    help='Path to trained hybrid models (for evaluate-hybrid command)')
    
    args = parser.parse_args()
    
    # Determine number of processes
    if args.processes == 'auto':
        num_processes = None
    else:
        try:
            num_processes = int(args.processes)
        except ValueError:
            print(f"❌ Invalid processes value: {args.processes}")
            return 1
    
    print(f"🚀 Starting Multiprocessing Crumbly Workflow")
    print(f"   Command: {args.command}")
    print(f"   Path: {args.path}")
    print(f"   Max images: {args.max_images or 'unlimited'}")
    print(f"   Processes: {args.processes}")
    print(f"   Platform: {platform.system()}")
    
    # Initialize workflow manager
    workflow = MultiprocessingCrumblyWorkflowManager(
        output_dir=args.output,
        num_processes=num_processes
    )
    
    try:
        if args.command == 'evaluate':
            print(f"🔍 Running parallel evaluation on: {args.path}")
            results = workflow.run_parallel_evaluation(args.path, args.max_images)
        
        elif args.command == 'evaluate-hybrid':
            if not args.model_path:
                print("❌ --model-path is required for evaluate-hybrid command")
                return 1
            print(f"🤖 Running hybrid evaluation on: {args.path}")
            print(f"   Using models from: {args.model_path}")
            results = workflow.run_parallel_evaluation_hybrid(args.path, args.model_path, args.max_images)
            
        elif args.command == 'train':
            print(f"🤖 Training hybrid model with: {args.path}")
            results = workflow.train_hybrid_model(args.path)
            
        elif args.command == 'complete':
            print(f"🚀 Running complete parallel workflow on: {args.path}")
            results = workflow.run_complete_parallel_workflow(
                args.path, args.max_images, args.test_split
            )
        
        # Check if workflow was successful
        if isinstance(results, dict) and results.get('success', True):
            print(f"\n✅ Command '{args.command}' completed successfully!")
            if RESULTS_CONFIGURED:
                print(f"📁 Results saved to centralized results/ folder structure")
            else:
                print(f"📁 Results saved to: {workflow.output_dir}")
            
            # Show performance summary
            if args.command in ['evaluate', 'complete']:
                total_time = results.get('processing_time', 0) or results.get('total_time', 0)
                num_images = results.get('total_images', 0)
                num_processes = results.get('num_processes_used', workflow.num_processes)
                
                if total_time > 0 and num_images > 0:
                    print(f"\n📊 Performance Summary:")
                    print(f"   Images processed: {num_images}")
                    print(f"   Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
                    print(f"   Images per second: {num_images/total_time:.1f}")
                    print(f"   Processes used: {num_processes}")
                    print(f"   Estimated speedup: ~{num_processes:.1f}x vs single-threaded")
            
            return 0
        else:
            print(f"\n❌ Command '{args.command}' failed!")
            error = results.get('error', 'Unknown error') if isinstance(results, dict) else 'Unknown error'
            print(f"   Error: {error}")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n🛑 Workflow interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n💥 Workflow error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())