#!/usr/bin/env python3
"""
Complete Crumbly Texture Analysis Workflow - FIXED VERSION
Integrates evaluation, training, and hybrid detection into a single workflow.

FIXES:
1. Proper path setup for modules/ directory 
2. Import of results_config.py for output management
3. Corrected module imports to match comprehensive_analyzer_main.py pattern
4. Fixed OCR initialization by properly setting up paths

Usage Examples:
1. Evaluate current detector:
   python crumbly_workflow.py evaluate /path/to/dataset

2. Train hybrid model:
   python crumbly_workflow.py train /path/to/evaluation_results.csv

3. Run hybrid analysis on new images:
   python crumbly_workflow.py analyze /path/to/images --model /path/to/trained_model

4. Complete workflow (evaluate + train + test):
   python crumbly_workflow.py complete /path/to/dataset
"""

import sys
import argparse
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

# ===== PATH SETUP (CRITICAL FIX) =====
# Set up paths to match comprehensive_analyzer_main.py
project_root = Path(__file__).parent.parent  # Go up one level from Crumble_ML/
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "modules"))

print(f"🔧 Fixed path setup:")
print(f"   Project root: {project_root}")
print(f"   Modules path: {project_root / 'modules'}")
print(f"   Crumble_ML path: {project_root / 'Crumble_ML'}")

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
    # Fallback if results_config.py doesn't exist
    RESULTS_CONFIGURED = False
    print(f"⚠️ Results config not found: {e}")
    # Create basic fallback
    MULTIPROCESSING_DIR = Path("crumbly_workflow_results")
    MULTIPROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    
    def get_multiprocessing_path(filename: str) -> Path:
        return MULTIPROCESSING_DIR / filename

# ===== IMPORT CORE MODULES (FIXED PATHS) =====
try:
    # Import from modules/ directory (like comprehensive_analyzer_main.py does)
    from modules.image_preprocessing import load_image, preprocess_pipeline
    from modules.scale_detection import ScaleBarDetector, detect_scale_bar
    from modules.fiber_type_detection import FiberTypeDetector
    print("✅ Core modules imported successfully from modules/")
except ImportError as e:
    print(f"❌ Core modules import error: {e}")
    print("Make sure modules/ directory is accessible")
    sys.exit(1)

# ===== IMPORT CRUMBLY-SPECIFIC MODULES =====
try:
    # Import the fully corrected evaluation system and hybrid detector
    # Note: Use the fully corrected version with all proper method integrations
    from crumbly_evaluation_system import CrumblyEvaluationSystem
    from hybrid_crumbly_detector import HybridCrumblyDetector, train_hybrid_detector, load_hybrid_detector
    print("✅ Crumbly-specific modules imported successfully")
except ImportError as e:
    print(f"❌ Crumbly modules import error: {e}")
    print("Make sure crumbly modules are in Crumble_ML/ directory")
    print("SOLUTION: Replace crumbly_evaluation_system.py with the fully corrected version")
    sys.exit(1)

class CrumblyWorkflowManager:
    """
    Manages the complete workflow for crumbly texture analysis improvement.
    FIXED: Now properly handles paths and imports like comprehensive_analyzer_main.py
    """
    
    def __init__(self, output_dir: str = "crumbly_workflow_results"):
        # Use results_config if available, otherwise fallback
        if RESULTS_CONFIGURED:
            self.output_dir = ensure_directory_exists(output_dir)
        else:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(exist_ok=True)
        
        self.evaluation_system = None
        self.hybrid_detector = None
        self.workflow_results = {}
        
        # Initialize components with proper error handling
        try:
            # Initialize scale detector like comprehensive_analyzer_main.py does
            self.scale_detector = ScaleBarDetector(
                ocr_backend=None,  # Will auto-detect available backend
                use_enhanced_detection=True
            )
            print(f"✅ Scale detector initialized: {self.scale_detector.ocr_backend}")
        except Exception as e:
            print(f"⚠️ Scale detector initialization issue: {e}")
            self.scale_detector = None
        
        print(f"🔬 Crumbly Workflow Manager Initialized")
        print(f"   Output directory: {self.output_dir}")
    
    def run_evaluation(self, dataset_path: str, max_images: Optional[int] = None) -> Dict:
        """
        Step 1: Evaluate current detector performance.
        FIXED: Now uses correct method name and proper error handling
        """
        print(f"\n🔍 Starting Evaluation Phase...")
        print(f"   Dataset: {dataset_path}")
        print(f"   Max images: {max_images or 'all'}")
        
        start_time = time.time()
        
        try:
            # Initialize evaluation system
            if self.evaluation_system is None:
                self.evaluation_system = CrumblyEvaluationSystem(
                    dataset_path=dataset_path,
                    output_dir=str(self.output_dir / "evaluation")
                )
            
            # Run evaluation using the correct method name
            success = self.evaluation_system.run_evaluation(
                max_images=max_images,
                debug_images=False
            )
            
            if success:
                # Store evaluation results
                evaluation_results = {
                    'success': True,
                    'total_images': len(self.evaluation_system.evaluation_results),
                    'successful_images': len([r for r in self.evaluation_system.evaluation_results 
                                            if r.get('processing_success', False)]),
                    'processing_time': time.time() - start_time,
                    'evaluation_dir': str(self.evaluation_system.run_dir),
                    'feature_csv': None,
                    'analysis_results': getattr(self.evaluation_system, 'analysis_results', {})
                }
                
                # Find the generated feature CSV
                feature_files = list(self.evaluation_system.run_dir.glob("ml_features_*.csv"))
                if feature_files:
                    evaluation_results['feature_csv'] = str(feature_files[0])
                    evaluation_results['output_file'] = str(feature_files[0])
                    print(f"   📊 ML features saved to: {feature_files[0].name}")
                
                # Print key metrics
                if hasattr(self.evaluation_system, 'analysis_results'):
                    accuracy = self.evaluation_system.analysis_results.get('overall_accuracy', 0)
                    print(f"   📈 Current detector accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
                
                print(f"   ✅ Evaluation complete: {self.evaluation_system.run_dir}")
                
                self.workflow_results['evaluation'] = evaluation_results
                return evaluation_results
            else:
                eval_summary = {
                    'success': False,
                    'error': 'Evaluation process returned False',
                    'processing_time': time.time() - start_time
                }
                self.workflow_results['evaluation'] = eval_summary
                return eval_summary
            
        except Exception as e:
            print(f"   ❌ Evaluation failed: {e}")
            eval_summary = {
                'success': False, 
                'error': str(e),
                'processing_time': time.time() - start_time
            }
            self.workflow_results['evaluation'] = eval_summary
            return eval_summary
    
    def train_hybrid_model(self, features_csv_path: str) -> Dict:
        """
        Step 2: Train hybrid ML model using evaluation features.
        """
        print(f"\n🤖 Starting Training Phase...")
        print(f"   Features file: {features_csv_path}")
        
        start_time = time.time()
        
        try:
            # Load features from evaluation
            features_df = pd.read_csv(features_csv_path)
            
            # Extract features and labels
            feature_columns = [col for col in features_df.columns 
                             if col not in ['image_name', 'true_label', 'predicted_label', 'image_path']]
            
            X = features_df[feature_columns].values
            y = features_df['true_label'].values
            
            # Train hybrid detector
            model_results = train_hybrid_detector(
                X, y, 
                feature_names=feature_columns,
                output_dir=str(self.output_dir / "models")
            )
            
            train_summary = {
                'success': True,
                'model_dir': model_results['model_dir'],
                'num_ml_models': len(model_results['models']),
                'has_ensemble': 'ensemble' in model_results,
                'training_accuracy': model_results.get('training_accuracy', 0),
                'cross_val_score': model_results.get('cross_val_score', 0),
                'processing_time': time.time() - start_time,
                'feature_importance': model_results.get('feature_importance', {})
            }
            
            print(f"   ✅ Training complete: {model_results['model_dir']}")
            
            self.workflow_results['training'] = train_summary
            return train_summary
            
        except Exception as e:
            print(f"   ❌ Training failed: {e}")
            train_summary = {'success': False, 'error': str(e), 'processing_time': time.time() - start_time}
            self.workflow_results['training'] = train_summary
            return train_summary
    
    def test_hybrid_model(self, test_path: str, model_path: str, max_images: Optional[int] = None) -> Dict:
        """
        Step 3: Test trained hybrid model on new dataset.
        """
        print(f"\n🧪 Starting Testing Phase...")
        print(f"   Test dataset: {test_path}")
        print(f"   Model: {model_path}")
        
        start_time = time.time()
        
        try:
            # Load hybrid detector
            self.hybrid_detector = load_hybrid_detector(model_path)
            
            # Initialize evaluation system for testing
            test_evaluation = CrumblyEvaluationSystem(
                dataset_path=test_path,
                output_dir=str(self.output_dir / "testing")
            )
            
            # Run testing with hybrid detector
            test_results = test_evaluation.test_hybrid_detector(
                self.hybrid_detector,
                max_images=max_images
            )
            
            # Save test results
            output_file = "test_results.csv"
            output_path = self.output_dir / output_file
            
            if isinstance(test_results, dict) and 'detailed_results' in test_results:
                results_df = pd.DataFrame(test_results['detailed_results'])
                results_df.to_csv(output_path, index=False)
                
                test_summary = {
                    'success': True,
                    'total_images': test_results.get('total_images', 0),
                    'test_accuracy': test_results.get('accuracy', 0),
                    'confusion_matrix': test_results.get('confusion_matrix', []),
                    'processing_time': time.time() - start_time,
                    'output_file': str(output_path)
                }
            else:
                test_summary = {
                    'success': False,
                    'error': 'Invalid test results format',
                    'processing_time': time.time() - start_time
                }
            
            print(f"   ✅ Testing complete: {output_path}")
            
            self.workflow_results['testing'] = test_summary
            return test_summary
            
        except Exception as e:
            print(f"   ❌ Testing failed: {e}")
            test_summary = {'success': False, 'error': str(e), 'processing_time': time.time() - start_time}
            self.workflow_results['testing'] = test_summary
            return test_summary
    
    def analyze_new_images(self, images_path: str, model_path: str, output_file: str = "hybrid_analysis_results.csv") -> Dict:
        """
        Step 4: Analyze new images using trained hybrid model.
        FIXED: Now properly handles scale detection like comprehensive_analyzer_main.py
        """
        print(f"\n📊 Starting Analysis Phase...")
        print(f"   Images: {images_path}")
        print(f"   Model: {model_path}")
        
        start_time = time.time()
        
        try:
            # Load hybrid detector
            if self.hybrid_detector is None:
                self.hybrid_detector = load_hybrid_detector(model_path)
            
            # Get image files
            images_path = Path(images_path)
            if images_path.is_file():
                image_files = [images_path]
            else:
                image_files = list(images_path.glob("*.tif")) + list(images_path.glob("*.png")) + list(images_path.glob("*.jpg"))
            
            print(f"   Found {len(image_files)} images to analyze")
            
            analysis_results = []
            
            for i, image_path in enumerate(image_files):
                print(f"   Processing {i+1}/{len(image_files)}: {image_path.name}")
                
                try:
                    # Load and preprocess image (like comprehensive_analyzer_main.py)
                    image = load_image(str(image_path))
                    if image is None:
                        raise ValueError("Could not load image")
                    
                    # Scale detection (like comprehensive_analyzer_main.py)
                    scale_result = None
                    if self.scale_detector:
                        try:
                            scale_result = self.scale_detector.detect_scale_bar(image)
                        except Exception as e:
                            print(f"     ⚠️ Scale detection issue: {e}")
                    
                    # Run hybrid analysis
                    hybrid_result = self.hybrid_detector.analyze_image(image)
                    
                    # Compile result
                    result = {
                        'image_name': image_path.name,
                        'image_path': str(image_path),
                        'status': 'success',
                        'classification': hybrid_result.get('classification', 'unknown'),
                        'confidence': hybrid_result.get('confidence', 0.0),
                        'method_used': 'hybrid',
                        'processing_time': hybrid_result.get('processing_time', 0.0),
                        'scale_detected': scale_result.get('scale_detected', False) if scale_result else False,
                        'scale_factor': scale_result.get('micrometers_per_pixel', 0) if scale_result else 0
                    }
                    
                    # Add detailed features if available
                    if 'features' in hybrid_result:
                        result.update(hybrid_result['features'])
                    
                    analysis_results.append(result)
                    
                except Exception as e:
                    print(f"     ❌ Error: {e}")
                    analysis_results.append({
                        'image_name': image_path.name,
                        'image_path': str(image_path),
                        'status': 'error',
                        'error': str(e)
                    })
            
            # Save results using results_config if available
            if RESULTS_CONFIGURED:
                output_path = get_multiprocessing_path(output_file)
            else:
                output_path = self.output_dir / output_file
                
            results_df = pd.DataFrame(analysis_results)
            results_df.to_csv(output_path, index=False)
            
            analysis_summary = {
                'success': True,
                'total_images': len(image_files),
                'processed_images': len([r for r in analysis_results if r.get('status') != 'error']),
                'output_file': str(output_path),
                'processing_time': time.time() - start_time,
                'results': analysis_results
            }
            
            print(f"   ✅ Analysis complete: {output_path}")
            
            self.workflow_results['analysis'] = analysis_summary
            return analysis_summary
            
        except Exception as e:
            print(f"   ❌ Analysis failed: {e}")
            analysis_summary = {'success': False, 'error': str(e), 'processing_time': time.time() - start_time}
            self.workflow_results['analysis'] = analysis_summary
            return analysis_summary
    
    def run_complete_workflow(self, dataset_path: str, max_images: Optional[int] = None,
                            test_split: float = 0.3) -> Dict:
        """
        Run the complete workflow: evaluate -> train -> test.
        """
        print(f"\n🚀 Starting Complete Workflow...")
        print(f"   Dataset: {dataset_path}")
        print(f"   Test split: {test_split}")
        
        workflow_start = time.time()
        
        # Step 1: Evaluation
        eval_results = self.run_evaluation(dataset_path, max_images)
        if not eval_results.get('success', False):
            return {'success': False, 'error': 'Evaluation failed', 'step': 'evaluation'}
        
        # Step 2: Training
        features_file = eval_results.get('output_file')
        if not features_file:
            return {'success': False, 'error': 'No features file from evaluation', 'step': 'evaluation'}
        
        train_results = self.train_hybrid_model(features_file)
        if not train_results.get('success', False):
            return {'success': False, 'error': 'Training failed', 'step': 'training'}
        
        # Step 3: Testing (use same dataset with different samples if needed)
        model_dir = train_results.get('model_dir')
        if not model_dir:
            return {'success': False, 'error': 'No model directory from training', 'step': 'training'}
        
        test_results = self.test_hybrid_model(dataset_path, model_dir, max_images)
        
        # Generate workflow summary
        workflow_summary = {
            'success': True,
            'total_time': time.time() - workflow_start,
            'evaluation': eval_results,
            'training': train_results,
            'testing': test_results,
            'workflow_complete': True,
            'all_steps_successful': all([
                eval_results.get('success', False),
                train_results.get('success', False),
                test_results.get('success', False)
            ])
        }
        
        # Save complete workflow report
        self._generate_workflow_report(workflow_summary)
        
        return workflow_summary
    
    def _generate_workflow_report(self, summary: Dict):
        """Generate comprehensive workflow report."""
        print(f"\n📋 WORKFLOW SUMMARY REPORT")
        print("=" * 50)
        
        # Use results_config if available
        if RESULTS_CONFIGURED:
            report_file = get_multiprocessing_path("workflow_report.txt")
            json_file = get_multiprocessing_path("workflow_results.json")
        else:
            report_file = self.output_dir / "workflow_report.txt"
            json_file = self.output_dir / "workflow_results.json"
        
        with open(report_file, 'w') as f:
            f.write("CRUMBLY TEXTURE ANALYSIS WORKFLOW REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Evaluation results
            if 'evaluation' in self.workflow_results:
                eval_results = self.workflow_results['evaluation']
                f.write("1. EVALUATION PHASE:\n")
                f.write(f"   Status: {'SUCCESS' if eval_results['success'] else 'FAILED'}\n")
                
                if eval_results['success']:
                    f.write(f"   Total images: {eval_results['total_images']}\n")
                    f.write(f"   Successful: {eval_results['successful_images']}\n")
                    
                    analysis = eval_results.get('analysis_results', {})
                    if analysis:
                        accuracy = analysis.get('overall_accuracy', 0)
                        f.write(f"   Current detector accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)\n")
                f.write("\n")
            
            # Training results
            if 'training' in self.workflow_results:
                train_results = self.workflow_results['training']
                f.write("2. TRAINING PHASE:\n")
                f.write(f"   Status: {'SUCCESS' if train_results['success'] else 'FAILED'}\n")
                
                if train_results['success']:
                    f.write(f"   Model directory: {train_results['model_dir']}\n")
                    f.write(f"   ML models trained: {train_results['num_ml_models']}\n")
                    f.write(f"   Ensemble model: {'Yes' if train_results['has_ensemble'] else 'No'}\n")
                f.write("\n")
            
            # Testing results
            if 'testing' in self.workflow_results:
                test_results = self.workflow_results['testing']
                f.write("3. TESTING PHASE:\n")
                f.write(f"   Status: {'SUCCESS' if test_results['success'] else 'FAILED'}\n")
                
                if test_results['success']:
                    f.write(f"   Test accuracy: {test_results['test_accuracy']:.3f}\n")
                f.write("\n")
            
            # Summary
            f.write("WORKFLOW SUMMARY:\n")
            f.write(f"   Total time: {summary['total_time']:.2f} seconds\n")
            f.write(f"   Complete: {'Yes' if summary['workflow_complete'] else 'No'}\n")
            f.write(f"   All steps successful: {'Yes' if summary['all_steps_successful'] else 'No'}\n")
        
        print(f"📄 Workflow report saved: {report_file}")
        
        # Save workflow results as JSON
        with open(json_file, 'w') as f:
            json.dump(self.workflow_results, f, indent=2, default=str)
        
        print(f"📄 Workflow data saved: {json_file}")

def main():
    """Command line interface for the crumbly workflow."""
    parser = argparse.ArgumentParser(
        description='Crumbly Texture Analysis Workflow - FIXED VERSION',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s evaluate /path/to/dataset
  %(prog)s train /path/to/features.csv
  %(prog)s complete /path/to/dataset --max-images 50
  %(prog)s analyze /path/to/images --model /path/to/model
        """
    )
    
    parser.add_argument('command', choices=['evaluate', 'train', 'test', 'analyze', 'complete'],
                       help='Workflow command to execute')
    parser.add_argument('path', help='Path to dataset, features, or images')
    parser.add_argument('--output', '-o', default='crumbly_workflow_results',
                       help='Output directory')
    parser.add_argument('--max-images', type=int, help='Maximum images to process')
    parser.add_argument('--model', help='Path to trained model (for test/analyze)')
    parser.add_argument('--test-split', type=float, default=0.3,
                       help='Test split fraction for complete workflow')
    
    args = parser.parse_args()
    
    # Initialize workflow manager
    workflow = CrumblyWorkflowManager(output_dir=args.output)
    
    try:
        if args.command == 'evaluate':
            print(f"🔍 Running evaluation on: {args.path}")
            results = workflow.run_evaluation(args.path, args.max_images)
            
        elif args.command == 'train':
            print(f"🤖 Training hybrid model with: {args.path}")
            results = workflow.train_hybrid_model(args.path)
            
        elif args.command == 'test':
            if not args.model:
                print("❌ --model required for test command")
                return 1
            print(f"🧪 Testing model {args.model} on: {args.path}")
            results = workflow.test_hybrid_model(args.path, args.model, args.max_images)
            
        elif args.command == 'analyze':
            if not args.model:
                print("❌ --model required for analyze command")
                return 1
            print(f"📊 Analyzing images in {args.path} with model {args.model}")
            results = workflow.analyze_new_images(args.path, args.model)
            
        elif args.command == 'complete':
            print(f"🚀 Running complete workflow on: {args.path}")
            results = workflow.run_complete_workflow(
                args.path, args.max_images, args.test_split
            )
        
        # Check if workflow was successful
        if isinstance(results, dict) and results.get('success', True):
            print(f"\n✅ Command '{args.command}' completed successfully!")
            if RESULTS_CONFIGURED:
                print(f"📁 Results saved to centralized results/ folder structure")
            else:
                print(f"📁 Results saved to: {workflow.output_dir}")
            return 0
        else:
            print(f"\n❌ Command '{args.command}' failed!")
            return 1
            
    except Exception as e:
        print(f"\n💥 Workflow error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())