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

# ===== IMPORT REORGANIZED MODULES =====
print("🔧 Loading reorganized SEM Fiber Analysis modules...")

# Import all reorganized functions from modules
from modules import (
    # Core functions
    load_image, preprocess_for_analysis, 
    detect_fiber_type, create_optimal_fiber_mask, extract_fiber_mask_from_analysis,
    detect_scale_bar_from_crop, CrumblyDetector, improve_crumbly_classification,
    
    # Debug control
    enable_global_debug, disable_global_debug, is_debug_enabled,
    
    # Availability flags
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
        from modules.fiber_type_detection import detect_fiber_type, create_optimal_fiber_mask
        
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
                                                 crop_bottom_percent=15, 
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
        
        fiber_type, confidence, fiber_analysis_data = detect_fiber_type(processed_image)
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
                    porosity_data = analyze_fiber_porosity(processed_image, fiber_mask, scale_factor)
                else:
                    porosity_data = quick_porosity_check(processed_image, fiber_mask, scale_factor)
                
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
                initial_classification = crumbly_detector.analyze_crumbly_texture(
                    processed_image, fiber_mask, scale_factor
                )
                
                # STEP 8: Classification improvement
                if debug:
                    print(f"⚡ Step 8: Classification improvement...")
                
                final_classification = improve_crumbly_classification(
                    processed_image, 
                    fiber_mask, 
                    initial_classification, 
                    porosity_data=porosity_data, 
                    debug=debug
                )
                
                result['crumbly_analysis'] = {
                    'initial_classification': initial_classification,
                    'final_classification': final_classification,
                    'improvement_applied': final_classification.get('improvement_applied', False)
                }
                result['processing_steps'].append('crumbly_analysis')
                
                if debug:
                    initial_class = initial_classification.get('classification', 'unknown')
                    final_class = final_classification.get('final_classification', 'unknown')
                    improved = final_classification.get('improvement_applied', False)
                    print(f"   ✅ Crumbly analysis: {initial_class} → {final_class} (improved: {improved})")
                    
            except Exception as e:
                if debug:
                    print(f"   ⚠️ Crumbly analysis failed: {e}")
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