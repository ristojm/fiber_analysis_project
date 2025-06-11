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

# ===== IMPORT REORGANIZED MODULES =====
print("🔧 Loading reorganized SEM Fiber Analysis modules...")

# Import all reorganized functions from modules
from modules import (
    load_image, preprocess_for_analysis, 
    detect_fiber_type, create_optimal_fiber_mask, extract_fiber_mask_from_analysis,
    detect_scale_bar_from_crop, CrumblyDetector, improve_crumbly_classification,
    enable_global_debug, disable_global_debug, is_debug_enabled,
    HAS_ENHANCED_PREPROCESSING, HAS_ENHANCED_FIBER_DETECTION,
    HAS_ENHANCED_SCALE_DETECTION, HAS_ENHANCED_CRUMBLY_DETECTION,
    POROSITY_AVAILABLE, POROSITY_TYPE
)

# Track available modules (updated)
MODULES_LOADED = {
    'core': True,  # Always true since we import from modules
    'image_preprocessing': HAS_ENHANCED_PREPROCESSING,
    'fiber_type_detection': HAS_ENHANCED_FIBER_DETECTION,
    'scale_detection': HAS_ENHANCED_SCALE_DETECTION,
    'crumbly_detection': HAS_ENHANCED_CRUMBLY_DETECTION,
    'porosity_analysis': POROSITY_AVAILABLE
}

print("✅ Reorganized modules loaded successfully!")
print(f"   Enhanced preprocessing: {HAS_ENHANCED_PREPROCESSING}")
print(f"   Enhanced fiber detection: {HAS_ENHANCED_FIBER_DETECTION}")
print(f"   Enhanced scale detection: {HAS_ENHANCED_SCALE_DETECTION}")
print(f"   Enhanced crumbly detection: {HAS_ENHANCED_CRUMBLY_DETECTION}")
print(f"   Porosity analysis: {POROSITY_AVAILABLE} ({POROSITY_TYPE})")

# ===== WORKER FUNCTIONS =====
# REPLACE the process_single_image_worker function in multiprocessing_crumbly_workflow.py with this:

def process_single_image_orchestrator(worker_args: Dict) -> Dict:
    """
    Orchestrates single image processing using modular functions.
    Contains NO processing logic - only workflow coordination.
    
    This replaces the old process_single_image_worker function with clean
    orchestration that calls the reorganized module functions.
    """
    image_path = worker_args['image_path']
    debug = worker_args.get('debug', False)
    model_path = worker_args.get('model_path')
    
    result = {
        'image_path': str(image_path),
        'success': False,
        'processing_steps': []
    }
    
    try:
        # Enable debug if requested
        if debug:
            from modules import enable_global_debug
            enable_global_debug(save_images=True, show_plots=False)
            print(f"\n🔄 Processing: {Path(image_path).name}")
        
        # Import all the reorganized functions
        from modules.image_preprocessing import load_image, preprocess_for_analysis
        from modules.scale_detection import detect_scale_bar_from_crop
        from modules.fiber_type_detection import FiberTypeDetector, create_optimal_fiber_mask
        
        # Optional imports based on availability
        porosity_data = None
        if MODULES_LOADED.get('porosity_analysis'):
            if POROSITY_TYPE == 'fast_refined':
                from modules.porosity_analysis import analyze_fiber_porosity
            else:
                from modules.porosity_analysis import quick_porosity_check
        
        if MODULES_LOADED.get('crumbly_detection'):
            from modules.crumbly_detection import CrumblyDetector, improve_crumbly_classification
        
        # STEP 1: Load original image
        if debug:
            print(f"📂 Step 1: Loading image...")
        
        original_image = load_image(str(image_path))
        if original_image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        result['processing_steps'].append('image_loaded')
        if debug:
            print(f"   ✅ Image loaded: {original_image.shape}")
        
        # STEP 2: Scale detection (on bottom crop) 
        if debug:
            print(f"📏 Step 2: Scale bar detection...")
        
        scale_result = detect_scale_bar_from_crop(original_image, 
                                         crop_bottom_percent=20,  # Changed from 15 to 20
                                         debug=debug)
        
        scale_factor = scale_result.get('micrometers_per_pixel', 1.0)
        result['scale_detection'] = {
            'scale_detected': scale_result.get('scale_detected', False),
            'micrometers_per_pixel': scale_factor,
            'confidence': scale_result.get('confidence', 0.0),
            'method': scale_result.get('detection_method', 'unknown')
        }
        result['processing_steps'].append('scale_detection')
        
        if debug:
            detected = scale_result.get('scale_detected', False)
            print(f"   ✅ Scale detection: {detected}, factor: {scale_factor:.4f}")
        
        # STEP 3: Preprocessing (removes scale bar)
        if debug:
            print(f"🔧 Step 3: Image preprocessing...")
        
        preprocess_result = preprocess_for_analysis(original_image, 
                                                   remove_scale_bar=True, 
                                                   debug=debug)
        processed_image = preprocess_result['processed_image']
        result['preprocessing'] = {
            'scale_bar_removed': preprocess_result['scale_bar_removed'],
            'processing_steps': preprocess_result['processing_steps'],
            'success': True
        }
        result['processing_steps'].append('preprocessing')
        
        if debug:
            print(f"   ✅ Preprocessing complete: {processed_image.shape}")
        
        # STEP 4: Fiber detection (on clean image)
        if debug:
            print(f"🔍 Step 4: Fiber type detection...")
        
        # Use FiberTypeDetector class for full analysis data
        fiber_detector = FiberTypeDetector()
        fiber_type, confidence, fiber_analysis_data = fiber_detector.classify_fiber_type(processed_image)
        
        result['fiber_detection'] = {
            'fiber_type': fiber_type,
            'confidence': confidence,
            'total_fibers': fiber_analysis_data.get('total_fibers', 0),
            'hollow_fibers': fiber_analysis_data.get('hollow_fibers', 0),
            'filaments': fiber_analysis_data.get('filaments', 0),
            'method': 'classify_fiber_type'
        }
        result['processing_steps'].append('fiber_detection')
        
        if debug:
            print(f"   ✅ Fiber detection: {fiber_type} (conf: {confidence:.2f})")
        
        # STEP 5: Optimal mask creation
        if debug:
            print(f"🎯 Step 5: Creating optimal fiber mask...")
        
        fiber_mask = create_optimal_fiber_mask(processed_image, 
                                              fiber_analysis_data, 
                                              method='best_available', 
                                              debug=debug)
        
        mask_area = np.sum(fiber_mask > 0)
        result['mask_creation'] = {
            'mask_area_pixels': int(mask_area),
            'coverage_percent': float(mask_area / fiber_mask.size * 100),
            'method': 'create_optimal_fiber_mask'
        }
        result['processing_steps'].append('mask_creation')
        
        if debug:
            coverage = mask_area / fiber_mask.size * 100
            print(f"   ✅ Mask created: {mask_area:,} pixels ({coverage:.1f}%)")
        
        # STEP 6: Porosity analysis (if available)
        if MODULES_LOADED.get('porosity_analysis'):
            if debug:
                print(f"🔬 Step 6: Porosity analysis...")
            
            try:
                if POROSITY_TYPE == 'fast_refined':
                    porosity_result = analyze_fiber_porosity(processed_image, fiber_mask, scale_factor)
                else:
                    porosity_result = quick_porosity_check(processed_image, fiber_mask, scale_factor)
                
                # FIXED: Ensure porosity_data is always a dict
                if isinstance(porosity_result, dict):
                    porosity_data = porosity_result
                elif isinstance(porosity_result, (int, float, np.number)):
                    # Convert single number to dict format
                    porosity_data = {
                        'porosity_percentage': float(porosity_result),
                        'total_pores': 0,
                        'average_pore_size': 0.0,
                        'pore_areas': [],
                        'detection_method': 'single_value_conversion'
                    }
                    if debug:
                        print(f"   🔄 Converted single porosity value to dict: {float(porosity_result):.1f}%")
                else:
                    # Unknown format, create default dict
                    porosity_data = {
                        'porosity_percentage': 0.0,
                        'total_pores': 0,
                        'average_pore_size': 0.0,
                        'pore_areas': [],
                        'detection_method': 'fallback_unknown_format',
                        'original_type': str(type(porosity_result))
                    }
                    if debug:
                        print(f"   ⚠️ Unknown porosity format {type(porosity_result)}, using fallback")
                
                result['porosity'] = porosity_data
                result['processing_steps'].append('porosity_analysis')
                
                if debug and porosity_data:
                    porosity_pct = porosity_data.get('porosity_percentage', 0)
                    pore_count = porosity_data.get('total_pores', 0)
                    print(f"   ✅ Porosity: {porosity_pct:.1f}%, {pore_count} pores")
                    
            except Exception as e:
                if debug:
                    print(f"   ⚠️ Porosity analysis failed: {e}")
                result['porosity'] = {'error': str(e)}
        else:
            if debug:
                print(f"   ⚠️ Porosity analysis not available")
        
        # STEP 7: Crumbly detection (if available)   
        if MODULES_LOADED.get('crumbly_detection'):
            if debug:
                print(f"🧩 Step 7: Crumbly texture analysis...")
            
            try:
                crumbly_detector = CrumblyDetector()
                
                # FIXED: Ensure all inputs are proper format
                # Ensure fiber_mask is proper uint8 format
                if not isinstance(fiber_mask, np.ndarray):
                    fiber_mask_clean = np.array(fiber_mask, dtype=np.uint8)
                elif fiber_mask.dtype != np.uint8:
                    fiber_mask_clean = (fiber_mask > 0).astype(np.uint8) * 255
                else:
                    fiber_mask_clean = fiber_mask.copy()
                
                # Ensure 2D mask
                if len(fiber_mask_clean.shape) > 2:
                    fiber_mask_clean = fiber_mask_clean[:, :, 0] if fiber_mask_clean.shape[2] > 0 else fiber_mask_clean.squeeze()
                
                # Ensure scale_factor is a number
                if not isinstance(scale_factor, (int, float)):
                    scale_factor_clean = 1.0
                    if debug:
                        print(f"   🔄 Invalid scale factor type, using 1.0")
                else:
                    scale_factor_clean = float(scale_factor)
                
                initial_classification = crumbly_detector.analyze_crumbly_texture(
                    processed_image, fiber_mask_clean, scale_factor_clean
                )
                
                # STEP 8: Classification improvement
                if debug:
                    print(f"⚡ Step 8: Classification improvement...")
                
                # Ensure porosity data is in correct format for improvement
                porosity_for_improvement = result.get('porosity')
                if porosity_for_improvement and 'error' in porosity_for_improvement:
                    porosity_for_improvement = None  # Don't use error data
                
                final_classification = improve_crumbly_classification(
                    processed_image, 
                    fiber_mask_clean, 
                    initial_classification, 
                    porosity_data=porosity_for_improvement, 
                    debug=debug
                )
                
                result['crumbly_analysis'] = {
                    'initial_classification': initial_classification,
                    'final_classification': final_classification,
                    'improvement_applied': final_classification.get('improvement_applied', False)
                }
                result['processing_steps'].append('crumbly_analysis')
                
                if debug:
                    initial_class = initial_classification.get('classification', 'unknown') if isinstance(initial_classification, dict) else str(initial_classification)
                    final_class = final_classification.get('final_classification', 'unknown')
                    improved = final_classification.get('improvement_applied', False)
                    print(f"   ✅ Crumbly analysis: {initial_class} → {final_class} (improved: {improved})")
                    
            except Exception as e:
                if debug:
                    print(f"   ⚠️ Crumbly analysis failed: {e}")
                    import traceback
                    traceback.print_exc()
                result['crumbly_analysis'] = {'error': str(e)}
        else:
            if debug:
                print(f"   ⚠️ Crumbly detection not available")
        
        # Mark as successful
        result['success'] = True
        result['processing_steps'].append('workflow_complete')
        
        if debug:
            print(f"✅ Processing complete: {len(result['processing_steps'])} steps")
            from modules import disable_global_debug
            disable_global_debug()
        
    except Exception as e:
        result['error'] = str(e)
        result['processing_steps'].append('workflow_error')
        
        if debug:
            print(f"❌ Processing failed for {image_path}: {e}")
            import traceback
            result['traceback'] = traceback.format_exc()
            from modules import disable_global_debug
            disable_global_debug()
    
    return result


class MultiprocessingCrumblyWorkflow:
    """
    Multiprocessing-enabled crumbly workflow manager for fast parallel processing.
    """
    def __init__(self, output_dir: str = "multiprocessing_crumbly_results", num_processes: Optional[int] = None):
        """
        Initialize the enhanced multiprocessing workflow.
        
        Args:
            output_dir: Directory for saving results
            num_processes: Number of processes to use (auto-detect if None)
        """
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
        
        print(f"🚀 Enhanced Multiprocessing Crumbly Workflow Manager Initialized")
        print(f"   Output directory: {self.output_dir}")
        print(f"   CPU cores available: {mp.cpu_count()}")
        print(f"   Processes to use: {self.num_processes}")
        print(f"   Using reorganized modular architecture v2.0")
    
    def get_image_files(self, dataset_path: str) -> List[Dict]:
        """
        Get all image files with their labels from dataset.
        
        Args:
            dataset_path: Path to dataset with labeled folders
            
        Returns:
            List of dictionaries with image info and labels
        """
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
            'porous': {'label': 'porous', 'numeric': 0},
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
                            'source_folder': folder_name
                        })
        
        print(f"   Found folders: {', '.join(found_folders)}")
        print(f"   Found {len(image_files)} images total")
        return image_files
    
    @with_timeout(1800)  # 30 minute timeout for evaluation
    def run_parallel_evaluation(self, dataset_path: str, max_images: Optional[int] = None, 
                               debug_mode: bool = False) -> Dict:
        """
        Run parallel evaluation using the new orchestrator approach.
        
        Args:
            dataset_path: Path to dataset with labeled folders
            max_images: Maximum number of images to process (None for all)
            debug_mode: Enable debug output for detailed processing info
        
        Returns:
            Dict with evaluation results and statistics
        """
        print(f"\n🔄 Starting Enhanced Parallel Evaluation")
        print(f"   Dataset: {dataset_path}")
        print(f"   Max images: {max_images or 'all'}")
        print(f"   Debug mode: {debug_mode}")
        print(f"   Processes: {self.num_processes}")
        
        start_time = time.time()
        
        # Get image files with labels
        image_files = self.get_image_files(dataset_path)
        if max_images:
            image_files = image_files[:max_images]
        
        if not image_files:
            return {
                'success': False,
                'error': 'No images found in dataset',
                'total_images': 0
            }
        
        print(f"📊 Processing {len(image_files)} images...")
        
        # Prepare worker arguments using new orchestrator
        worker_args_list = []
        for img_info in image_files:
            worker_args = {
                'image_path': img_info['path'],
                'true_label': img_info['true_label'],
                'label_numeric': img_info['label_numeric'],
                'source_folder': img_info['source_folder'],
                'debug': debug_mode  # Pass debug mode to workers
            }
            worker_args_list.append(worker_args)
        
        # Process images in parallel using new orchestrator
        results = []
        successful_results = []
        failed_results = []
        
        print(f"⚡ Starting parallel processing...")
        
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            # Submit jobs using new orchestrator function
            future_to_args = {
                executor.submit(process_single_image_orchestrator, args): args 
                for args in worker_args_list
            }
            
            # Collect results with progress tracking
            completed = 0
            for future in as_completed(future_to_args):
                args = future_to_args[future]
                completed += 1
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result.get('success', False):
                        # Add label information for evaluation
                        result['true_label'] = args['true_label']
                        result['label_numeric'] = args['label_numeric']
                        result['source_folder'] = args['source_folder']
                        successful_results.append(result)
                    else:
                        failed_results.append(result)
                        
                except Exception as e:
                    error_result = {
                        'image_path': str(args['image_path']),
                        'success': False,
                        'error': str(e),
                        'true_label': args['true_label'],
                        'source_folder': args['source_folder']
                    }
                    results.append(error_result)
                    failed_results.append(error_result)
                
                # Progress update
                progress = completed / len(worker_args_list) * 100
                if completed % 10 == 0 or completed == len(worker_args_list):
                    print(f"   Progress: {completed}/{len(worker_args_list)} ({progress:.1f}%)")
        
        processing_time = time.time() - start_time
        
        # Calculate evaluation metrics
        evaluation_results = self._calculate_evaluation_metrics(successful_results)
        
        # Compile final results
        final_results = {
            'success': True,
            'processing_time': processing_time,
            'total_images': len(image_files),
            'successful_predictions': len(successful_results),
            'failed_predictions': len(failed_results),
            'success_rate': len(successful_results) / len(image_files) * 100,
            'evaluation_metrics': evaluation_results,
            'detailed_results': results,
            'workflow_info': {
                'version': '2.0.0',
                'architecture': 'reorganized_modular',
                'processes_used': self.num_processes,
                'modules_loaded': MODULES_LOADED,
                'enhanced_functions': {
                    'preprocessing': HAS_ENHANCED_PREPROCESSING,
                    'fiber_detection': HAS_ENHANCED_FIBER_DETECTION,
                    'scale_detection': HAS_ENHANCED_SCALE_DETECTION,
                    'crumbly_detection': HAS_ENHANCED_CRUMBLY_DETECTION
                }
            }
        }
        
        print(f"\n✅ Enhanced Parallel Evaluation Complete!")
        print(f"   Total time: {processing_time:.1f} seconds ({processing_time/60:.1f} minutes)")
        print(f"   Success rate: {final_results['success_rate']:.1f}%")
        print(f"   Successful: {len(successful_results)}/{len(image_files)}")
        
        if evaluation_results:
            accuracy = evaluation_results.get('accuracy', 0)
            print(f"   Accuracy: {accuracy:.3f}")
        
        # Save results
        self._save_evaluation_results(final_results)
        
        return final_results

    def _calculate_evaluation_metrics(self, successful_results: List[Dict]) -> Dict:
        """Calculate evaluation metrics from successful results."""
        if not successful_results:
            return {}
        
        try:
            # Extract predictions and true labels
            predictions = []
            true_labels = []
            confidences = []
            
            for result in successful_results:
                # Get final classification from crumbly analysis
                crumbly_analysis = result.get('crumbly_analysis', {})
                if isinstance(crumbly_analysis, dict):
                    final_classification = crumbly_analysis.get('final_classification', {})
                    if isinstance(final_classification, dict):
                        predicted_class = final_classification.get('final_classification', 'unknown')
                        confidence = final_classification.get('confidence', 0.0)
                    else:
                        predicted_class = str(final_classification)
                        confidence = 0.5
                else:
                    predicted_class = 'unknown'
                    confidence = 0.0
                
                true_label = result.get('true_label', 'unknown')
                
                predictions.append(predicted_class)
                true_labels.append(true_label)
                confidences.append(confidence)
            
            # Calculate accuracy
            correct = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
            accuracy = correct / len(predictions) if predictions else 0
            
            # Calculate per-class metrics
            unique_labels = sorted(set(true_labels + predictions))
            per_class_metrics = {}
            
            for label in unique_labels:
                true_positives = sum(1 for t, p in zip(true_labels, predictions) 
                                   if t == label and p == label)
                false_positives = sum(1 for t, p in zip(true_labels, predictions) 
                                    if t != label and p == label)
                false_negatives = sum(1 for t, p in zip(true_labels, predictions) 
                                    if t == label and p != label)
                
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                per_class_metrics[label] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score,
                    'true_positives': true_positives,
                    'false_positives': false_positives,
                    'false_negatives': false_negatives
                }
            
            # Create confusion matrix data
            confusion_data = {}
            for true_label in unique_labels:
                confusion_data[true_label] = {}
                for pred_label in unique_labels:
                    count = sum(1 for t, p in zip(true_labels, predictions) 
                              if t == true_label and p == pred_label)
                    confusion_data[true_label][pred_label] = count
            
            return {
                'accuracy': accuracy,
                'total_samples': len(predictions),
                'correct_predictions': correct,
                'unique_labels': unique_labels,
                'confusion_matrix': confusion_data,
                'per_class_metrics': per_class_metrics,
                'average_confidence': np.mean(confidences) if confidences else 0,
                'predictions': predictions,
                'true_labels': true_labels,
                'confidences': confidences
            }
            
        except Exception as e:
            print(f"⚠️ Error calculating evaluation metrics: {e}")
            return {'error': str(e)}
    
    def _save_evaluation_results(self, results: Dict):
        """Save evaluation results to files."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save JSON results
            json_file = self.output_dir / f"evaluation_results_{timestamp}.json"
            
            # Convert any Path objects to strings for JSON serialization
            json_results = self._prepare_for_json(results)
            
            with open(json_file, 'w') as f:
                json.dump(json_results, f, indent=2, default=str)
            
            print(f"📄 Results saved: {json_file}")
            
            # Save summary report
            report_file = self.output_dir / f"evaluation_report_{timestamp}.txt"
            self._save_text_report(results, report_file)
            
            # Save CSV with detailed results
            csv_file = self.output_dir / f"detailed_results_{timestamp}.csv"
            self._save_csv_results(results, csv_file)
            
        except Exception as e:
            print(f"⚠️ Error saving results: {e}")
    
    def _prepare_for_json(self, obj):
        """Recursively prepare object for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._prepare_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    def _save_text_report(self, results: Dict, report_file: Path):
        """Save human-readable text report."""
        with open(report_file, 'w') as f:
            f.write("ENHANCED SEM FIBER ANALYSIS - EVALUATION REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary
            f.write("SUMMARY:\n")
            f.write(f"  Total images: {results.get('total_images', 0)}\n")
            f.write(f"  Successful: {results.get('successful_predictions', 0)}\n")
            f.write(f"  Failed: {results.get('failed_predictions', 0)}\n")
            f.write(f"  Success rate: {results.get('success_rate', 0):.1f}%\n")
            f.write(f"  Processing time: {results.get('processing_time', 0):.1f}s\n\n")
            
            # Evaluation metrics
            eval_metrics = results.get('evaluation_metrics', {})
            if eval_metrics and 'accuracy' in eval_metrics:
                f.write("EVALUATION METRICS:\n")
                f.write(f"  Accuracy: {eval_metrics['accuracy']:.3f}\n")
                f.write(f"  Correct predictions: {eval_metrics['correct_predictions']}/{eval_metrics['total_samples']}\n")
                f.write(f"  Average confidence: {eval_metrics.get('average_confidence', 0):.3f}\n\n")
                
                # Per-class metrics
                per_class = eval_metrics.get('per_class_metrics', {})
                if per_class:
                    f.write("PER-CLASS METRICS:\n")
                    for label, metrics in per_class.items():
                        f.write(f"  {label}:\n")
                        f.write(f"    Precision: {metrics['precision']:.3f}\n")
                        f.write(f"    Recall: {metrics['recall']:.3f}\n")
                        f.write(f"    F1-Score: {metrics['f1_score']:.3f}\n")
                    f.write("\n")
                
                # Confusion matrix
                if 'confusion_matrix' in eval_metrics:
                    f.write("CONFUSION MATRIX:\n")
                    confusion = eval_metrics['confusion_matrix']
                    labels = eval_metrics['unique_labels']
                    
                    # Header
                    f.write("True\\Predicted".ljust(15))
                    for label in labels:
                        f.write(f"{label}".ljust(12))
                    f.write("\n")
                    
                    # Rows
                    for true_label in labels:
                        f.write(f"{true_label}".ljust(15))
                        for pred_label in labels:
                            count = confusion.get(true_label, {}).get(pred_label, 0)
                            f.write(f"{count}".ljust(12))
                        f.write("\n")
                    f.write("\n")
            
            # Workflow info
            workflow_info = results.get('workflow_info', {})
            if workflow_info:
                f.write(f"WORKFLOW INFO:\n")
                f.write(f"  Version: {workflow_info.get('version', 'unknown')}\n")
                f.write(f"  Architecture: {workflow_info.get('architecture', 'unknown')}\n")
                f.write(f"  Processes used: {workflow_info.get('processes_used', 'unknown')}\n")
                
                enhanced_funcs = workflow_info.get('enhanced_functions', {})
                if enhanced_funcs:
                    f.write(f"  Enhanced functions:\n")
                    for func_name, available in enhanced_funcs.items():
                        f.write(f"    {func_name}: {'✅' if available else '❌'}\n")
        
        print(f"📄 Report saved: {report_file}")
    
    def _save_csv_results(self, results: Dict, csv_file: Path):
        """Save detailed results as CSV."""
        try:
            import pandas as pd
            
            detailed_results = results.get('detailed_results', [])
            if not detailed_results:
                return
            
            # Flatten results for CSV
            csv_data = []
            for result in detailed_results:
                if not result.get('success', False):
                    continue
                
                row = {
                    'image_path': result.get('image_path', ''),
                    'true_label': result.get('true_label', ''),
                    'source_folder': result.get('source_folder', ''),
                }
                
                # Add crumbly analysis results
                crumbly_analysis = result.get('crumbly_analysis', {})
                if isinstance(crumbly_analysis, dict):
                    initial = crumbly_analysis.get('initial_classification', {})
                    final = crumbly_analysis.get('final_classification', {})
                    
                    if isinstance(initial, dict):
                        row['initial_classification'] = initial.get('classification', 'unknown')
                        row['initial_confidence'] = initial.get('confidence', 0.0)
                    
                    if isinstance(final, dict):
                        row['final_classification'] = final.get('final_classification', 'unknown')
                        row['final_confidence'] = final.get('confidence', 0.0)
                        row['improvement_applied'] = final.get('improvement_applied', False)
                        row['override_reason'] = final.get('override_reason', 'none')
                
                # Add other analysis results
                fiber_detection = result.get('fiber_detection', {})
                row['fiber_type'] = fiber_detection.get('fiber_type', 'unknown')
                row['fiber_confidence'] = fiber_detection.get('confidence', 0.0)
                
                porosity = result.get('porosity', {})
                if isinstance(porosity, dict):
                    row['porosity_percentage'] = porosity.get('porosity_percentage', 0.0)
                    row['total_pores'] = porosity.get('total_pores', 0)
                
                scale_detection = result.get('scale_detection', {})
                row['scale_detected'] = scale_detection.get('scale_detected', False)
                row['scale_factor'] = scale_detection.get('micrometers_per_pixel', 1.0)
                
                csv_data.append(row)
            
            if csv_data:
                df = pd.DataFrame(csv_data)
                df.to_csv(csv_file, index=False)
                print(f"📊 CSV results saved: {csv_file}")
            
        except Exception as e:
            print(f"⚠️ Error saving CSV results: {e}")

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
                    executor.submit(process_single_image_orchestrator, arg): arg['image_info']['path']
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
    
    def run_complete_parallel_workflow(self, dataset_path: str, max_images: Optional[int] = None) -> Dict:
        """Run complete workflow with parallel evaluation and training."""
        print(f"\n🚀 Starting Complete Parallel Workflow...")
        print(f"   Dataset: {dataset_path}")
        print(f"   Max images: {max_images or 'all'}")
        print(f"   Processes: {self.num_processes}")
        # REMOVED: print(f"   Test split: {test_split}")  # This variable wasn't defined
        
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