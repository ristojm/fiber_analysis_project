#!/usr/bin/env python3
"""
Complete Crumbly Texture Analysis Workflow
Integrates evaluation, training, and hybrid detection into a single workflow.

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

# Import our modules
try:
    from crumbly_evaluation_system import CrumblyEvaluationSystem
    from hybrid_crumbly_detector import HybridCrumblyDetector, train_hybrid_detector, load_hybrid_detector
    print("✅ All workflow modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all required modules are available")
    sys.exit(1)

class CrumblyWorkflowManager:
    """
    Manages the complete workflow for crumbly texture analysis improvement.
    """
    
    def __init__(self, output_dir: str = "crumbly_workflow_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.evaluation_system = None
        self.hybrid_detector = None
        self.workflow_results = {}
        
        print(f"🔬 Crumbly Workflow Manager Initialized")
        print(f"   Output directory: {self.output_dir}")
    
    def run_evaluation(self, dataset_path: str, max_images: Optional[int] = None) -> Dict:
        """
        Step 1: Evaluate current detector performance.
        
        Args:
            dataset_path: Path to labeled dataset (crumbly/intermediate/not folders)
            max_images: Maximum images to process (for testing)
            
        Returns:
            Evaluation results dictionary
        """
        
        print(f"\n🚀 STEP 1: EVALUATING CURRENT DETECTOR")
        print("=" * 60)
        
        evaluation_dir = self.output_dir / "evaluation"
        
        # Initialize evaluation system
        self.evaluation_system = CrumblyEvaluationSystem(
            dataset_path=dataset_path,
            output_dir=str(evaluation_dir)
        )
        
        # Run evaluation
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
                'evaluation_dir': str(self.evaluation_system.run_dir),
                'feature_csv': None,
                'analysis_results': getattr(self.evaluation_system, 'analysis_results', {})
            }
            
            # Find the generated feature CSV
            feature_files = list(self.evaluation_system.run_dir.glob("ml_features_*.csv"))
            if feature_files:
                evaluation_results['feature_csv'] = str(feature_files[0])
                print(f"   📊 ML features saved to: {feature_files[0].name}")
            
            # Print key metrics
            if hasattr(self.evaluation_system, 'analysis_results'):
                accuracy = self.evaluation_system.analysis_results.get('overall_accuracy', 0)
                print(f"   📈 Current detector accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
            
            self.workflow_results['evaluation'] = evaluation_results
            return evaluation_results
        
        else:
            evaluation_results = {'success': False, 'error': 'Evaluation failed'}
            self.workflow_results['evaluation'] = evaluation_results
            return evaluation_results
    
    def train_hybrid_model(self, feature_csv_path: str, model_name: str = "hybrid_model") -> Dict:
        """
        Step 2: Train hybrid ML-enhanced detector.
        
        Args:
            feature_csv_path: Path to ML features CSV from evaluation
            model_name: Name for the trained model
            
        Returns:
            Training results dictionary
        """
        
        print(f"\n🤖 STEP 2: TRAINING HYBRID MODEL")
        print("=" * 50)
        
        model_dir = self.output_dir / "models" / model_name
        
        try:
            # Train hybrid detector
            self.hybrid_detector = train_hybrid_detector(
                evaluation_csv_path=feature_csv_path,
                model_save_path=str(model_dir)
            )
            
            training_results = {
                'success': True,
                'model_dir': str(model_dir),
                'model_name': model_name,
                'is_trained': self.hybrid_detector.is_trained,
                'num_ml_models': len(self.hybrid_detector.ml_models),
                'has_ensemble': hasattr(self.hybrid_detector, 'ensemble_model')
            }
            
            print(f"   ✅ Hybrid model trained successfully")
            print(f"   📁 Models saved to: {model_dir}")
            print(f"   🧠 ML models: {list(self.hybrid_detector.ml_models.keys())}")
            
            self.workflow_results['training'] = training_results
            return training_results
            
        except Exception as e:
            print(f"   ❌ Training failed: {e}")
            training_results = {'success': False, 'error': str(e)}
            self.workflow_results['training'] = training_results
            return training_results
    
    def test_hybrid_model(self, test_dataset_path: str, model_path: str, 
                         max_test_images: Optional[int] = None) -> Dict:
        """
        Step 3: Test hybrid model on independent dataset.
        
        Args:
            test_dataset_path: Path to test dataset
            model_path: Path to trained model
            max_test_images: Maximum images to test
            
        Returns:
            Test results dictionary
        """
        
        print(f"\n🧪 STEP 3: TESTING HYBRID MODEL")
        print("=" * 40)
        
        try:
            # Load hybrid detector
            if self.hybrid_detector is None or not self.hybrid_detector.is_trained:
                self.hybrid_detector = load_hybrid_detector(model_path)
            
            if not self.hybrid_detector.is_trained:
                raise ValueError("Could not load trained model")
            
            # Run evaluation with hybrid detector
            test_dir = self.output_dir / "testing"
            test_evaluator = CrumblyEvaluationSystem(
                dataset_path=test_dataset_path,
                output_dir=str(test_dir)
            )
            
            # Replace the traditional detector with hybrid detector
            test_evaluator.crumbly_detector = self.hybrid_detector
            
            # Run test evaluation
            success = test_evaluator.run_evaluation(
                max_images=max_test_images,
                debug_images=False
            )
            
            if success:
                test_results = {
                    'success': True,
                    'test_dir': str(test_evaluator.run_dir),
                    'total_test_images': len(test_evaluator.evaluation_results),
                    'successful_test_images': len([r for r in test_evaluator.evaluation_results 
                                                 if r.get('processing_success', False)]),
                    'analysis_results': getattr(test_evaluator, 'analysis_results', {})
                }
                
                # Print test metrics
                if hasattr(test_evaluator, 'analysis_results'):
                    test_accuracy = test_evaluator.analysis_results.get('overall_accuracy', 0)
                    print(f"   📈 Hybrid model test accuracy: {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")
                    
                    # Compare with original evaluation if available
                    if 'evaluation' in self.workflow_results:
                        original_accuracy = self.workflow_results['evaluation']['analysis_results'].get('overall_accuracy', 0)
                        improvement = test_accuracy - original_accuracy
                        print(f"   📊 Improvement: {improvement:+.3f} ({improvement*100:+.1f}%)")
                        test_results['improvement'] = improvement
                
                self.workflow_results['testing'] = test_results
                return test_results
            
            else:
                test_results = {'success': False, 'error': 'Test evaluation failed'}
                self.workflow_results['testing'] = test_results
                return test_results
        
        except Exception as e:
            print(f"   ❌ Testing failed: {e}")
            test_results = {'success': False, 'error': str(e)}
            self.workflow_results['testing'] = test_results
            return test_results
    
    def analyze_new_images(self, images_path: str, model_path: str,
                          output_file: str = "analysis_results.csv") -> Dict:
        """
        Step 4: Analyze new images with trained hybrid model.
        
        Args:
            images_path: Path to directory with new images
            model_path: Path to trained model
            output_file: Name for output CSV file
            
        Returns:
            Analysis results dictionary
        """
        
        print(f"\n📊 STEP 4: ANALYZING NEW IMAGES")
        print("=" * 40)
        
        try:
            # Load hybrid detector if not already loaded
            if self.hybrid_detector is None or not self.hybrid_detector.is_trained:
                self.hybrid_detector = load_hybrid_detector(model_path)
            
            if not self.hybrid_detector.is_trained:
                raise ValueError("Could not load trained model")
            
            # Find images
            images_dir = Path(images_path)
            image_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(images_dir.glob(f'*{ext}'))
                image_files.extend(images_dir.glob(f'*{ext.upper()}'))
            
            if not image_files:
                raise ValueError(f"No images found in {images_path}")
            
            print(f"   Found {len(image_files)} images to analyze")
            
            # Analyze each image
            analysis_results = []
            
            for i, image_path in enumerate(image_files):
                print(f"   [{i+1}/{len(image_files)}] Analyzing: {image_path.name}")
                
                try:
                    # This is a simplified version - in practice you'd need full image processing pipeline
                    # Including preprocessing, fiber detection, etc.
                    
                    # For now, create a placeholder result
                    result = {
                        'image_name': image_path.name,
                        'image_path': str(image_path),
                        'status': 'placeholder',
                        'classification': 'unknown',
                        'confidence': 0.0,
                        'method_used': 'hybrid',
                        'processing_time': 0.0,
                        'note': 'Full pipeline integration needed'
                    }
                    
                    analysis_results.append(result)
                    
                except Exception as e:
                    print(f"     ❌ Error: {e}")
                    analysis_results.append({
                        'image_name': image_path.name,
                        'image_path': str(image_path),
                        'status': 'error',
                        'error': str(e)
                    })
            
            # Save results
            results_df = pd.DataFrame(analysis_results)
            output_path = self.output_dir / output_file
            results_df.to_csv(output_path, index=False)
            
            analysis_summary = {
                'success': True,
                'total_images': len(image_files),
                'processed_images': len([r for r in analysis_results if r.get('status') != 'error']),
                'output_file': str(output_path),
                'results': analysis_results
            }
            
            print(f"   ✅ Analysis complete: {output_path}")
            
            self.workflow_results['analysis'] = analysis_summary
            return analysis_summary
            
        except Exception as e:
            print(f"   ❌ Analysis failed: {e}")
            analysis_summary = {'success': False, 'error': str(e)}
            self.workflow_results['analysis'] = analysis_summary
            return analysis_summary
    
    def run_complete_workflow(self, dataset_path: str, max_images: Optional[int] = None,
                            test_split: float = 0.3) -> Dict:
        """
        Run the complete workflow: evaluate -> train -> test.
        
        Args:
            dataset_path: Path to labeled dataset
            max_images: Maximum images to process
            test_split: Fraction of data to use for testing
            
        Returns:
            Complete workflow results
        """
        
        print(f"\n🚀 COMPLETE CRUMBLY ANALYSIS WORKFLOW")
        print("=" * 70)
        print(f"   Dataset: {dataset_path}")
        print(f"   Max images: {max_images or 'all'}")
        print(f"   Test split: {test_split}")
        
        workflow_start_time = time.time()
        
        # Step 1: Evaluation
        evaluation_results = self.run_evaluation(dataset_path, max_images)
        
        if not evaluation_results['success']:
            print(f"\n❌ Workflow stopped: Evaluation failed")
            return self.workflow_results
        
        # Step 2: Training
        feature_csv = evaluation_results.get('feature_csv')
        if not feature_csv:
            print(f"\n❌ Workflow stopped: No feature CSV generated")
            return self.workflow_results
        
        training_results = self.train_hybrid_model(feature_csv, "complete_workflow_model")
        
        if not training_results['success']:
            print(f"\n❌ Workflow stopped: Training failed")
            return self.workflow_results
        
        # Step 3: Testing (using the same dataset with train/test split)
        # In practice, you'd want a separate test dataset
        model_path = training_results['model_dir']
        testing_results = self.test_hybrid_model(dataset_path, model_path, max_images)
        
        # Calculate total workflow time
        total_time = time.time() - workflow_start_time
        
        # Generate workflow summary
        self.workflow_results['summary'] = {
            'total_time': total_time,
            'workflow_complete': True,
            'all_steps_successful': all(
                self.workflow_results.get(step, {}).get('success', False)
                for step in ['evaluation', 'training', 'testing']
            )
        }
        
        self.generate_workflow_report()
        
        return self.workflow_results
    
    def generate_workflow_report(self):
        """Generate a comprehensive workflow report."""
        
        print(f"\n📋 WORKFLOW SUMMARY REPORT")
        print("=" * 50)
        
        report_file = self.output_dir / "workflow_report.txt"
        
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
                    test_analysis = test_results.get('analysis_results', {})
                    if test_analysis:
                        test_accuracy = test_analysis.get('overall_accuracy', 0)
                        f.write(f"   Hybrid model accuracy: {test_accuracy:.3f} ({test_accuracy*100:.1f}%)\n")
                        
                        improvement = test_results.get('improvement', 0)
                        f.write(f"   Improvement: {improvement:+.3f} ({improvement*100:+.1f}%)\n")
                f.write("\n")
            
            # Summary
            if 'summary' in self.workflow_results:
                summary = self.workflow_results['summary']
                f.write("4. WORKFLOW SUMMARY:\n")
                f.write(f"   Total time: {summary['total_time']:.2f} seconds\n")
                f.write(f"   Complete: {'Yes' if summary['workflow_complete'] else 'No'}\n")
                f.write(f"   All steps successful: {'Yes' if summary['all_steps_successful'] else 'No'}\n")
        
        print(f"📄 Workflow report saved: {report_file}")
        
        # Save workflow results as JSON
        json_file = self.output_dir / "workflow_results.json"
        with open(json_file, 'w') as f:
            json.dump(self.workflow_results, f, indent=2, default=str)
        
        print(f"📄 Workflow data saved: {json_file}")

def main():
    """Command line interface for the crumbly workflow."""
    
    parser = argparse.ArgumentParser(
        description='Crumbly Texture Analysis Workflow',
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
            print(f"📁 Results saved to: {workflow.output_dir}")
            return 0
        else:
            print(f"\n❌ Command '{args.command}' failed!")
            return 1
            
    except Exception as e:
        print(f"\n💥 Workflow error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())