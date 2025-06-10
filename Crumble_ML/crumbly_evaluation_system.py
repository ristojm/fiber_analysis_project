#!/usr/bin/env python3
"""
Crumbly Texture Evaluation System - FIXED WITH COMPREHENSIVE_ANALYZER APPROACH
Uses the exact same mask handling approach as comprehensive_analyzer_main.py

CRITICAL FIX: 
- Uses the same fix_fiber_mask_extraction method from comprehensive_analyzer_main.py
- Uses preprocessed images consistently like comprehensive_analyzer_main.py
- Proper mask format handling exactly matching the working implementation
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ===== PATH SETUP (MATCHING comprehensive_analyzer_main.py) =====
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "modules"))
sys.path.insert(0, str(current_dir))

print(f"🔧 Path setup for evaluation system:")
print(f"   Current dir: {current_dir}")
print(f"   Project root: {project_root}")
print(f"   Modules path: {project_root / 'modules'}")

# ===== IMPORT MODULES (MATCHING comprehensive_analyzer_main.py) =====
MODULES_LOADED = {}

try:
    from modules.image_preprocessing import load_image, preprocess_pipeline, load_and_preprocess
    MODULES_LOADED['image_preprocessing'] = True
    print("✅ Successfully imported image preprocessing")
except ImportError as e:
    print(f"❌ Image preprocessing import error: {e}")
    MODULES_LOADED['image_preprocessing'] = False

try:
    from modules.scale_detection import detect_scale_bar, ScaleBarDetector
    MODULES_LOADED['scale_detection'] = True
    print("✅ Successfully imported scale detection")
except ImportError as e:
    print(f"❌ Scale detection import error: {e}")
    MODULES_LOADED['scale_detection'] = False

try:
    from modules.fiber_type_detection import FiberTypeDetector, detect_fiber_type
    MODULES_LOADED['fiber_type_detection'] = True
    print("✅ Successfully imported fiber type detection")
except ImportError as e:
    print(f"❌ Fiber type detection import error: {e}")
    MODULES_LOADED['fiber_type_detection'] = False

# Porosity analysis with same fallback logic as comprehensive_analyzer_main.py
try:
    from modules.porosity_analysis import PorosityAnalyzer, analyze_fiber_porosity, quick_porosity_check
    MODULES_LOADED['porosity_analysis'] = 'fast_refined'
    print("✅ Successfully imported fast refined porosity analysis")
except ImportError:
    try:
        from modules.porosity_analysis import EnhancedPorosityAnalyzer, analyze_fiber_porosity_enhanced
        MODULES_LOADED['porosity_analysis'] = 'enhanced'
        print("✅ Successfully imported enhanced porosity analysis")
    except ImportError:
        try:
            from modules.porosity_analysis import PorosityAnalyzer
            MODULES_LOADED['porosity_analysis'] = 'basic'
            print("✅ Successfully imported basic porosity analysis")
        except ImportError:
            MODULES_LOADED['porosity_analysis'] = False
            print("❌ No porosity analysis available")

# Crumbly detector
try:
    from modules.crumbly_detection import CrumblyDetector
    MODULES_LOADED['crumbly_detection'] = True
    print("✅ Successfully imported CrumblyDetector from modules/")
except ImportError as e:
    try:
        from crumbly_detection import CrumblyDetector
        MODULES_LOADED['crumbly_detection'] = True
        print("✅ Successfully imported CrumblyDetector from current directory")
    except ImportError as e2:
        print(f"❌ CrumblyDetector import error: {e}")
        print(f"❌ Fallback import also failed: {e2}")
        MODULES_LOADED['crumbly_detection'] = False

class CrumblyEvaluationSystem:
    """
    Crumbly evaluation system using the exact same approach as comprehensive_analyzer_main.py
    """
    
    def __init__(self, dataset_path: str, output_dir: str = "crumbly_evaluation_results"):
        """Initialize evaluation system."""
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create timestamp for this evaluation run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"evaluation_run_{self.timestamp}"
        self.run_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize components using the same pattern as comprehensive_analyzer_main.py
        self.init_components()
        
        # Results storage
        self.evaluation_results = []
        self.feature_matrix = []
        self.labels = []
        self.label_names = []
        self.analysis_results = {}
        
        print(f"🔬 Crumbly Evaluation System Initialized")
        print(f"   Dataset: {self.dataset_path}")
        print(f"   Output: {self.run_dir}")
    
    def init_components(self):
        """Initialize components using the same pattern as comprehensive_analyzer_main.py"""
        
        # Initialize CrumblyDetector
        if MODULES_LOADED['crumbly_detection']:
            try:
                self.crumbly_detector = CrumblyDetector(porosity_aware=True)
                print("✅ CrumblyDetector initialized")
            except Exception as e:
                print(f"❌ Error initializing CrumblyDetector: {e}")
                self.crumbly_detector = None
        else:
            self.crumbly_detector = None
        
        # Initialize FiberTypeDetector (same as comprehensive_analyzer_main.py)
        if MODULES_LOADED['fiber_type_detection']:
            try:
                self.fiber_detector = FiberTypeDetector()
                print("✅ FiberTypeDetector initialized")
            except Exception as e:
                print(f"❌ Error initializing FiberTypeDetector: {e}")
                self.fiber_detector = None
        else:
            self.fiber_detector = None
        
        # Initialize ScaleBarDetector (same as comprehensive_analyzer_main.py)
        if MODULES_LOADED['scale_detection']:
            try:
                self.scale_detector = ScaleBarDetector(
                    ocr_backend=None,  # Auto-detect
                    use_enhanced_detection=True
                )
                print("✅ ScaleBarDetector initialized")
            except Exception as e:
                print(f"❌ Error initializing ScaleBarDetector: {e}")
                self.scale_detector = None
        else:
            self.scale_detector = None
        
        # Initialize PorosityAnalyzer (same fallback logic as comprehensive_analyzer_main.py)
        if MODULES_LOADED['porosity_analysis']:
            try:
                porosity_type = MODULES_LOADED['porosity_analysis']
                if porosity_type == 'fast_refined':
                    self.porosity_analyzer = PorosityAnalyzer()
                elif porosity_type == 'enhanced':
                    self.porosity_analyzer = EnhancedPorosityAnalyzer()
                else:
                    self.porosity_analyzer = PorosityAnalyzer()
                print(f"✅ PorosityAnalyzer initialized: {porosity_type}")
            except Exception as e:
                print(f"❌ Error initializing PorosityAnalyzer: {e}")
                self.porosity_analyzer = None
        else:
            self.porosity_analyzer = None
    
    def fix_fiber_mask_extraction(self, image: np.ndarray, fiber_analysis_data: Dict, debug: bool = False) -> np.ndarray:
        """
        COPIED EXACTLY FROM comprehensive_analyzer_main.py
        Simple and clean fiber mask extraction from analysis data.
        
        Args:
            image: Original image for shape reference
            fiber_analysis_data: Results from fiber type detection
            debug: Enable debug output
            
        Returns:
            Binary fiber mask (uint8, 0 or 255)
        """
        
        if debug:
            print(f"   🔧 Extracting fiber mask from analysis data...")
        
        # Get fiber mask from analysis data
        fiber_mask = fiber_analysis_data.get('fiber_mask')
        
        if fiber_mask is not None and isinstance(fiber_mask, np.ndarray):
            # Ensure proper format
            if fiber_mask.dtype != np.uint8:
                fiber_mask = (fiber_mask > 0).astype(np.uint8) * 255
            
            # Check if mask has sufficient content
            mask_area = np.sum(fiber_mask > 0)
            
            if debug:
                print(f"   ✅ Fiber mask extracted: {mask_area:,} pixels")
                print(f"   Coverage: {mask_area / fiber_mask.size * 100:.1f}%")
            
            return fiber_mask
        else:
            if debug:
                print(f"   ❌ No valid fiber mask found in analysis data")
            # Return empty mask as fallback
            return np.zeros(image.shape[:2], dtype=np.uint8)
    
    def validate_dataset_structure(self) -> bool:
        """Validate that dataset has the required structure."""
        print(f"\n🔍 Validating dataset structure...")
        
        if not self.dataset_path.exists():
            print(f"❌ Dataset path does not exist: {self.dataset_path}")
            return False
        
        required_folders = ['crumbly', 'intermediate', 'not']
        missing_folders = []
        
        for folder in required_folders:
            folder_path = self.dataset_path / folder
            if not folder_path.exists():
                missing_folders.append(folder)
        
        if missing_folders:
            print(f"❌ Missing required folders: {missing_folders}")
            return False
        
        print(f"✅ Dataset structure validated")
        return True
    
    def get_image_files(self) -> List[Dict]:
        """Get all image files with their labels."""
        image_files = []
        
        # Define label mapping
        label_map = {
            'crumbly': {'label': 'crumbly', 'numeric': 2},
            'intermediate': {'label': 'intermediate', 'numeric': 1},
            'not': {'label': 'not', 'numeric': 0}
        }
        
        for folder_name, label_info in label_map.items():
            folder_path = self.dataset_path / folder_name
            if folder_path.exists():
                extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp']
                for ext in extensions:
                    for img_file in folder_path.glob(ext):
                        image_files.append({
                            'path': img_file,
                            'true_label': label_info['label'],
                            'label_numeric': label_info['numeric']
                        })
        
        print(f"   Found {len(image_files)} images total")
        return image_files
    
    def process_single_image(self, image_info, debug=False):
        """
        Process a single image using the EXACT SAME approach as comprehensive_analyzer_main.py
        """
        
        result = {
            'image_path': str(image_info['path']),
            'true_label': image_info['true_label'],
            'processing_success': False,
            'predicted_label': 'unknown',
            'prediction_confidence': 0.0,
            'crumbly_score': 0.5,
            'ml_features': {},
            'error_details': {}
        }
        
        try:
            # Load and preprocess image (SAME AS comprehensive_analyzer_main.py)
            print(f"     📸 Loading image...")
            image_path = str(image_info['path'])
            
            # Use the same loading approach as comprehensive_analyzer_main.py
            original_image = load_image(image_path)
            if original_image is None:
                result['error'] = "Failed to load image"
                return result
            
            # Preprocess image (CORRECT: Use individual parameters, not config dict)
            preprocessed = preprocess_pipeline(
                str(image_info['path']),
                enhance_contrast_method='clahe',
                denoise_method='bilateral',
                remove_scale_bar=True,
                normalize=True
            )
            
            # Extract the processed image from the result dictionary
            if isinstance(preprocessed, dict):
                preprocessed = preprocessed['processed']
            
            print(f"     ✅ Image loaded: {original_image.shape}")
            
            # Scale detection (SAME AS comprehensive_analyzer_main.py)
            scale_factor = 1.0
            print(f"     🔍 Scale detection...")
            if self.scale_detector and MODULES_LOADED['scale_detection']:
                try:
                    scale_result = self.scale_detector.detect_scale_bar(original_image)
                    if scale_result and scale_result.get('scale_detected', False):
                        scale_factor = scale_result.get('micrometers_per_pixel', 1.0)
                        print(f"     ✅ Scale detected: {scale_factor:.4f} μm/pixel")
                    else:
                        print(f"     ⚠️ Scale detection failed")
                except Exception as e:
                    print(f"     ⚠️ Scale detection error: {e}")
            
            # Fiber type detection (SAME AS comprehensive_analyzer_main.py)
            fiber_type = 'unknown'
            fiber_analysis_data = None
            print(f"     🔬 Fiber detection...")
            if self.fiber_detector and MODULES_LOADED['fiber_type_detection']:
                try:
                    # CRITICAL: Use classify_fiber_type with PREPROCESSED image (same as comprehensive_analyzer_main.py)
                    fiber_type_result, confidence, fiber_analysis_data = self.fiber_detector.classify_fiber_type(
                        preprocessed, scale_factor  # Use preprocessed image!
                    )
                    
                    if fiber_analysis_data:
                        fiber_type = fiber_type_result
                        result['fiber_type'] = fiber_type
                        result['fiber_confidence'] = confidence
                        print(f"     ✅ Fiber type: {fiber_type} (conf: {confidence:.3f})")
                    else:
                        print(f"     ⚠️ No fiber_analysis_data returned")
                        
                except Exception as e:
                    print(f"     ⚠️ Fiber detection issue: {e}")
                    result['fiber_type'] = 'unknown'
            else:
                print(f"     ⚠️ Fiber detector not available")
            
            # Extract fiber mask using the SAME method as comprehensive_analyzer_main.py
            fiber_mask = self.fix_fiber_mask_extraction(preprocessed, fiber_analysis_data or {}, debug=debug)
            
            # Porosity analysis (SAME approach as comprehensive_analyzer_main.py)
            porosity_features = {}
            print(f"     📊 Porosity analysis...")
            
            if self.porosity_analyzer and MODULES_LOADED['porosity_analysis'] and np.sum(fiber_mask) > 1000:
                try:
                    porosity_type = MODULES_LOADED['porosity_analysis']
                    
                    # Call porosity analysis using the SAME pattern as comprehensive_analyzer_main.py
                    if porosity_type == 'fast_refined':
                        porosity_result = self.porosity_analyzer.analyze_fiber_porosity(
                            preprocessed, 
                            fiber_mask.astype(np.uint8), 
                            scale_factor, 
                            fiber_type,
                            fiber_analysis_data
                        )
                    elif porosity_type == 'enhanced':
                        porosity_result = self.porosity_analyzer.analyze_fiber_porosity(
                            preprocessed, 
                            fiber_mask.astype(np.uint8), 
                            scale_factor
                        )
                    else:
                        porosity_result = self.porosity_analyzer.analyze_fiber_porosity(
                            preprocessed, 
                            fiber_mask.astype(np.uint8), 
                            scale_factor
                        )
                    
                    if porosity_result and porosity_result.get('success', False):
                        porosity_metrics = porosity_result.get('porosity_metrics', {})
                        porosity_features = {
                            'total_porosity_percent': porosity_metrics.get('total_porosity_percent', 0),
                            'pore_count': porosity_metrics.get('pore_count', 0),
                            'average_pore_size': porosity_metrics.get('average_pore_size_um2', 0),
                            'pore_density': porosity_metrics.get('pore_density_per_mm2', 0)
                        }
                        result['porosity_analysis'] = porosity_result
                        print(f"     ✅ Porosity: {porosity_features['total_porosity_percent']:.1f}%")
                    else:
                        print(f"     ⚠️ Porosity analysis failed")
                        
                except Exception as e:
                    print(f"     ⚠️ Porosity analysis issue: {e}")
            else:
                print(f"     ⚠️ Porosity analyzer not available or insufficient mask area")
            
            # Crumbly texture analysis 
            print(f"     🧩 Crumbly texture analysis...")
            
            if self.crumbly_detector and MODULES_LOADED['crumbly_detection']:
                try:
                    # Convert fiber mask to boolean for crumbly detector
                    fiber_mask_bool = fiber_mask > 127
                    
                    # Call crumbly analysis with preprocessed image (same pattern as comprehensive_analyzer_main.py)
                    crumbly_result = self.crumbly_detector.analyze_crumbly_texture(
                        preprocessed,  # Use preprocessed image
                        fiber_mask_bool,
                        None,  # No lumen mask for now
                        scale_factor,
                        debug=False
                    )
                    
                    if crumbly_result and 'classification' in crumbly_result:
                        predicted_label = crumbly_result['classification']
                        confidence = crumbly_result.get('confidence', 0.0)
                        crumbly_score = crumbly_result.get('crumbly_score', 0.5)
                        
                        result['predicted_label'] = predicted_label
                        result['prediction_confidence'] = confidence
                        result['crumbly_score'] = crumbly_score
                        result['crumbly_analysis'] = crumbly_result
                        result['processing_success'] = True
                        
                        print(f"     ✅ Crumbly prediction: {predicted_label} (conf: {confidence:.3f}, score: {crumbly_score:.3f})")
                        
                        # Extract ML features
                        ml_features = {}
                        ml_features.update(porosity_features)
                        
                        # Add crumbly features
                        ml_features['crumbly_score'] = crumbly_score
                        ml_features['traditional_confidence'] = confidence
                        
                        # Add other analysis features if available
                        if 'pore_metrics' in crumbly_result:
                            pore_metrics = crumbly_result['pore_metrics']
                            ml_features['organized_porosity_score'] = pore_metrics.get('organized_porosity_score', 0.5)
                            ml_features['mean_pore_circularity'] = pore_metrics.get('mean_pore_circularity', 0.5)
                        
                        if 'wall_integrity_metrics' in crumbly_result:
                            wall_metrics = crumbly_result['wall_integrity_metrics']
                            ml_features['wall_integrity_score'] = wall_metrics.get('wall_integrity_score', 0.5)
                        
                        result['ml_features'] = ml_features
                        
                    else:
                        print(f"     ❌ Crumbly analysis failed: Invalid result format")
                        result['error'] = "Crumbly analysis returned invalid result"
                        
                except Exception as e:
                    print(f"     ❌ Crumbly analysis failed: {e}")
                    result['error'] = f"Crumbly analysis failed: {e}"
            else:
                print(f"     ⚠️ Crumbly detector not available")
                result['error'] = "Crumbly detector not available"
                
        except Exception as e:
            result['error'] = str(e)
            print(f"     ❌ Processing error: {e}")
            import traceback
            traceback.print_exc()
        
        return result
    
    def run_evaluation(self, max_images: Optional[int] = None, debug_images: bool = False) -> bool:
        """Run comprehensive evaluation of the crumbly detector."""
        print(f"\n🚀 STARTING COMPREHENSIVE EVALUATION")
        print("=" * 60)
        
        # Validate dataset
        if not self.validate_dataset_structure():
            return False
        
        # Check if required modules are loaded
        if not MODULES_LOADED['crumbly_detection']:
            print("❌ CrumblyDetector not available - cannot run evaluation")
            return False
        
        # Get all image files
        image_files = self.get_image_files()
        
        if max_images:
            image_files = image_files[:max_images]
            print(f"   Limiting to {max_images} images for testing")
        
        print(f"   Processing {len(image_files)} images...")
        
        # Process each image
        start_time = time.time()
        successful_processes = 0
        
        for i, image_info in enumerate(image_files):
            print(f"\n[{i+1:3d}/{len(image_files)}] Processing: {image_info['path'].name}")
            print(f"   True label: {image_info['true_label']}")
            
            # Process image
            result = self.process_single_image(image_info, debug=debug_images)
            
            if result['processing_success']:
                predicted_label = result['predicted_label']
                confidence = result['prediction_confidence']
                crumbly_score = result['crumbly_score']
                
                print(f"   Predicted: {predicted_label} (conf: {confidence:.3f}, score: {crumbly_score:.3f})")
                
                # Check if prediction matches ground truth
                match_status = "✅" if predicted_label == image_info['true_label'] else "❌"
                print(f"   {match_status} Match: {predicted_label == image_info['true_label']}")
                
                # Store for ML training
                if 'ml_features' in result and result['ml_features']:
                    feature_vector = list(result['ml_features'].values())
                    self.feature_matrix.append(feature_vector)
                    self.labels.append(image_info['label_numeric'])
                
                successful_processes += 1
            else:
                print(f"   ❌ Processing failed: {result.get('error', 'Unknown error')}")
            
            self.evaluation_results.append(result)
        
        total_time = time.time() - start_time
        success_rate = successful_processes / len(image_files) * 100
        
        print(f"\n📊 EVALUATION COMPLETE")
        print(f"   Total time: {total_time:.2f} seconds")
        print(f"   Success rate: {success_rate:.1f}% ({successful_processes}/{len(image_files)})")
        
        # Generate analysis
        self.analyze_results()
        self.save_results()
        
        return True
    
    def analyze_results(self):
        """Analyze evaluation results and generate metrics."""
        print(f"\n📈 ANALYZING RESULTS")
        print("=" * 40)
        
        successful_results = [r for r in self.evaluation_results if r['processing_success']]
        
        if not successful_results:
            print("❌ No successful results to analyze!")
            self.analysis_results = {'overall_accuracy': 0.0}
            return
        
        # Calculate accuracy
        correct_predictions = sum(1 for r in successful_results 
                                if r['predicted_label'] == r['true_label'])
        accuracy = correct_predictions / len(successful_results)
        
        # Generate detailed classification report
        true_labels = [r['true_label'] for r in successful_results]
        predicted_labels = [r['predicted_label'] for r in successful_results]
        
        print(f"📊 Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"📊 Classification Report:")
        print(classification_report(true_labels, predicted_labels))
        
        # Store analysis results
        self.analysis_results = {
            'overall_accuracy': accuracy,
            'total_images': len(self.evaluation_results),
            'successful_images': len(successful_results),
            'classification_report': classification_report(true_labels, predicted_labels, output_dict=True)
        }
        
        print(f"✅ Analysis complete!")
    
    def save_results(self):
        """Save evaluation results to files."""
        print(f"\n💾 SAVING RESULTS")
        print("=" * 30)
        
        # Save detailed results
        results_df = pd.DataFrame(self.evaluation_results)
        results_file = self.run_dir / "detailed_results.csv"
        results_df.to_csv(results_file, index=False)
        print(f"   📊 Detailed results: {results_file.name}")
        
        # Save ML features for training
        if self.feature_matrix and self.labels:
            features_df = pd.DataFrame(self.feature_matrix)
            features_df['true_label'] = self.labels
            # Add text labels for better compatibility  
            label_map = {0: 'not', 1: 'intermediate', 2: 'crumbly'}
            features_df['true_label_name'] = [label_map[label] for label in self.labels]
            features_file = self.run_dir / "ml_features_dataset.csv"
            features_df.to_csv(features_file, index=False)
            print(f"   🤖 ML features: {features_file.name}")
        
        # Save analysis summary
        analysis_file = self.run_dir / "analysis_summary.json"
        with open(analysis_file, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        print(f"   📈 Analysis summary: {analysis_file.name}")
        
        print(f"   📁 All results saved to: {self.run_dir}")
    
    def generate_ml_training_recommendations(self):
        """Generate recommendations for ML training."""
        print(f"\n🎯 ML TRAINING RECOMMENDATIONS")
        print("=" * 35)
        
        if not self.feature_matrix:
            print("❌ No feature data available for ML training")
            return
        
        accuracy = self.analysis_results.get('overall_accuracy', 0)
        
        print(f"   Current detector accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        if accuracy < 0.7:
            print(f"   ⚠️ Low accuracy detected")
            print(f"   💡 Recommendations:")
            print(f"   1. Train hybrid ML model using the features CSV")
            print(f"   2. Analyze misclassified samples")
            print(f"   3. Consider implementing hybrid detector")
            print(f"   4. Use feature analysis for threshold tuning")
        else:
            print(f"   ✅ Good baseline performance")
            print(f"   💡 Consider ML enhancement for marginal improvements")
        
        print(f"\n🔧 MODULE STATUS:")
        for module, status in MODULES_LOADED.items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {module}: {status}")

def main():
    """Main function to run the evaluation."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Crumbly Texture Detection System')
    parser.add_argument('dataset_path', help='Path to dataset with crumbly/intermediate/not folders')
    parser.add_argument('--output', '-o', default='crumbly_evaluation_results', 
                       help='Output directory for results')
    parser.add_argument('--max-images', type=int, help='Maximum images to process (for testing)')
    parser.add_argument('--debug', action='store_true', help='Enable debug visualizations')
    
    args = parser.parse_args()
    
    # Initialize evaluation system
    evaluator = CrumblyEvaluationSystem(args.dataset_path, args.output)
    
    # Run evaluation
    success = evaluator.run_evaluation(max_images=args.max_images, debug_images=args.debug)
    
    if success:
        # Generate ML recommendations
        evaluator.generate_ml_training_recommendations()
        
        print(f"\n🎉 EVALUATION COMPLETE!")
        print(f"   Check results in: {evaluator.run_dir}")
        print(f"   Use the ML features CSV for hybrid model training")
    else:
        print(f"\n❌ Evaluation failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())