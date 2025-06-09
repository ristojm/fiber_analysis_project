#!/usr/bin/env python3
"""
Crumbly Texture Evaluation System - FULLY CORRECTED VERSION
Evaluates current detector against labeled dataset and prepares ML training data.

COMPREHENSIVE FIXES:
1. Proper path setup for modules/ directory imports
2. Corrected ALL method names to match actual implementations
3. Fixed module initialization and integration issues
4. Proper error handling and fallbacks
5. Correct usage of all detector classes and their methods
6. Fixed to use preprocessed images like comprehensive_analyzer_main.py

Folder structure expected:
your_dataset/
├── crumbly/
│   ├── image1.tif
│   ├── image2.png
│   └── ...
├── intermediate/
│   ├── image3.tif
│   └── ...
└── not/
    ├── image4.tif
    └── ...
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

# ===== PATH SETUP (CRITICAL FIX) =====
# Set up paths to find both modules/ and Crumble_ML/
current_dir = Path(__file__).parent
project_root = current_dir.parent  # Go up to main project directory
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "modules"))
sys.path.insert(0, str(current_dir))  # Current Crumble_ML directory

print(f"🔧 Path setup for evaluation system:")
print(f"   Current dir: {current_dir}")
print(f"   Project root: {project_root}")
print(f"   Modules path: {project_root / 'modules'}")

# ===== IMPORT MODULES WITH PROPER ERROR HANDLING =====
MODULES_LOADED = {}

try:
    # Import from modules/ directory
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

try:
    # Import porosity analysis (multiple versions supported)
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

# Import crumbly detector from modules/ directory (like comprehensive_analyzer_main.py does)
try:
    from modules.crumbly_detection import CrumblyDetector
    MODULES_LOADED['crumbly_detection'] = True
    print("✅ Successfully imported CrumblyDetector from modules/")
except ImportError as e:
    try:
        # Fallback: try current directory
        from crumbly_detection import CrumblyDetector
        MODULES_LOADED['crumbly_detection'] = True
        print("✅ Successfully imported CrumblyDetector from current directory")
    except ImportError as e2:
        print(f"❌ CrumblyDetector import error: {e}")
        print(f"❌ Fallback import also failed: {e2}")
        print("Make sure crumbly_detection.py is in modules/ directory")
        MODULES_LOADED['crumbly_detection'] = False

class CrumblyEvaluationSystem:
    """
    Comprehensive evaluation system for crumbly texture detection.
    FULLY CORRECTED VERSION: All method names and integrations fixed.
    """
    
    def __init__(self, dataset_path: str, output_dir: str = "crumbly_evaluation_results"):
        """
        Initialize evaluation system.
        
        Args:
            dataset_path: Path to dataset with crumbly/intermediate/not folders
            output_dir: Directory to save evaluation results
        """
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create timestamp for this evaluation run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"evaluation_run_{self.timestamp}"
        self.run_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize detectors with proper error handling and correct method usage
        self.crumbly_detector = None
        self.fiber_detector = None
        self.scale_detector = None
        self.porosity_analyzer = None
        
        # Initialize CrumblyDetector (using the working version from modules/)
        if MODULES_LOADED['crumbly_detection']:
            try:
                self.crumbly_detector = CrumblyDetector(porosity_aware=True)
                print("✅ CrumblyDetector initialized (from modules/)")
            except Exception as e:
                print(f"❌ Error initializing CrumblyDetector: {e}")
                self.crumbly_detector = None
        else:
            print("❌ CrumblyDetector not available")
            self.crumbly_detector = None
        
        # Initialize FiberTypeDetector
        if MODULES_LOADED['fiber_type_detection']:
            try:
                self.fiber_detector = FiberTypeDetector()
                print("✅ FiberTypeDetector initialized")
            except Exception as e:
                print(f"❌ Error initializing FiberTypeDetector: {e}")
        
        # Initialize ScaleBarDetector
        if MODULES_LOADED['scale_detection']:
            try:
                self.scale_detector = ScaleBarDetector(
                    ocr_backend=None,  # Auto-detect
                    use_enhanced_detection=True
                )
                print("✅ ScaleBarDetector initialized")
            except Exception as e:
                print(f"❌ Error initializing ScaleBarDetector: {e}")
        
        # Initialize PorosityAnalyzer
        if MODULES_LOADED['porosity_analysis']:
            try:
                porosity_type = MODULES_LOADED['porosity_analysis']
                if porosity_type == 'fast_refined':
                    self.porosity_analyzer = PorosityAnalyzer()
                elif porosity_type == 'enhanced':
                    self.porosity_analyzer = EnhancedPorosityAnalyzer()
                elif porosity_type == 'basic':
                    self.porosity_analyzer = PorosityAnalyzer()
                print(f"✅ PorosityAnalyzer initialized ({porosity_type})")
            except Exception as e:
                print(f"❌ Error initializing PorosityAnalyzer: {e}")
        
        # Results storage
        self.evaluation_results = []
        self.feature_matrix = []
        self.labels = []
        self.label_mapping = {'not': 0, 'intermediate': 1, 'crumbly': 2}
        self.reverse_label_mapping = {0: 'not', 1: 'intermediate', 2: 'crumbly'}
        self.analysis_results = {}
        
        print(f"🔬 Crumbly Evaluation System Initialized")
        print(f"   Dataset: {self.dataset_path}")
        print(f"   Output: {self.run_dir}")
    
    def validate_dataset_structure(self) -> bool:
        """Validate that the dataset has the expected structure."""
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
            print(f"   Expected structure: dataset/{{crumbly, intermediate, not}}/")
            return False
        
        # Check for images in each folder
        total_images = 0
        for folder in required_folders:
            folder_path = self.dataset_path / folder
            image_count = len(list(folder_path.glob("*.tif")) + 
                            list(folder_path.glob("*.png")) + 
                            list(folder_path.glob("*.jpg")))
            total_images += image_count
            print(f"   📁 {folder}: {image_count} images")
        
        if total_images == 0:
            print("❌ No images found in dataset!")
            return False
        
        print(f"✅ Dataset structure valid: {total_images} total images")
        return True
    
    def get_image_files(self) -> List[Dict]:
        """Get all image files with their labels."""
        image_files = []
        
        for label in ['crumbly', 'intermediate', 'not']:
            folder_path = self.dataset_path / label
            
            # Get all image files
            extensions = ['*.tif', '*.png', '*.jpg', '*.jpeg', '*.bmp']
            for ext in extensions:
                for image_path in folder_path.glob(ext):
                    image_files.append({
                        'path': image_path,
                        'true_label': label,
                        'label_numeric': self.label_mapping[label]
                    })
        
        print(f"📊 Found {len(image_files)} images total")
        return image_files
    
    def process_single_image(self, image_info: Dict, debug: bool = False) -> Dict:
        """
        Process a single image and extract features for evaluation.
        FULLY CORRECTED: All method names and integrations fixed + preprocessed images.
        """
        image_path = image_info['path']
        true_label = image_info['true_label']
        
        result = {
            'image_name': image_path.name,
            'image_path': str(image_path),
            'true_label': true_label,
            'processing_success': False,
            'error': None
        }
        
        try:
            print(f"     📸 Loading image...")
            
            # Load image using correct method
            if MODULES_LOADED['image_preprocessing']:
                image = load_image(str(image_path))
                # CRITICAL FIX: Preprocess the image like comprehensive_analyzer_main.py does
                preprocessed_result = preprocess_pipeline(str(image_path))
                preprocessed = preprocessed_result['processed']
            else:
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                preprocessed = image.copy()
            
            if image is None:
                raise ValueError("Failed to load image")
            
            result['image_shape'] = image.shape
            print(f"     ✅ Image loaded: {image.shape}")

            # Scale detection with Windows-compatible timeout protection
            scale_factor = 1.0
            print(f"     🔍 Scale detection...")
            
            if self.scale_detector and MODULES_LOADED['scale_detection']:
                try:
                    import threading
                    import time
                    
                    scale_result = None
                    error_occurred = None
                    
                    def scale_detection_worker():
                        nonlocal scale_result, error_occurred
                        try:
                            # CORRECTED: Use detect_scale_bar method on ORIGINAL image (not preprocessed)
                            scale_result = self.scale_detector.detect_scale_bar(image)
                        except Exception as e:
                            error_occurred = e
                    
                    # Run scale detection in a separate thread with timeout
                    thread = threading.Thread(target=scale_detection_worker)
                    thread.daemon = True
                    thread.start()
                    thread.join(timeout=30)  # 30 second timeout
                    
                    if thread.is_alive():
                        print(f"     ⏰ Scale detection timeout - continuing without scale")
                        result['scale_detected'] = False
                        result['scale_factor'] = 1.0
                        scale_factor = 1.0
                    elif error_occurred:
                        raise error_occurred
                    elif scale_result and scale_result.get('scale_detected', False):
                        scale_factor = scale_result.get('micrometers_per_pixel', 1.0)
                        result['scale_detected'] = True
                        result['scale_factor'] = scale_factor
                        print(f"     ✅ Scale detected: {scale_factor:.4f} μm/pixel")
                    else:
                        result['scale_detected'] = False
                        result['scale_factor'] = 1.0
                        print(f"     ⚠️ No scale detected")
                        
                except Exception as e:
                    print(f"     ⚠️ Scale detection issue: {e}")
                    result['scale_detected'] = False
                    result['scale_factor'] = 1.0
                    scale_factor = 1.0
            else:
                result['scale_detected'] = False
                result['scale_factor'] = 1.0
                print(f"     ⚠️ Scale detector not available")
            
            # Fiber detection using correct method (FIXED: Use preprocessed image)
            fiber_mask = None
            lumen_mask = None
            fiber_type = 'unknown'
            
            print(f"     🔬 Fiber detection...")
            
            if self.fiber_detector and MODULES_LOADED['fiber_type_detection']:
                try:
                    # CRITICAL FIX: Use classify_fiber_type method with PREPROCESSED image ONLY
                    # This matches comprehensive_analyzer_main.py pattern exactly
                    fiber_type_result, confidence, fiber_analysis_data = self.fiber_detector.classify_fiber_type(preprocessed)
                    
                    # FIXED: Extract fiber_mask directly from fiber_analysis_data (no success check needed)
                    if fiber_analysis_data:
                        fiber_mask = fiber_analysis_data.get('fiber_mask')
                        lumen_mask = fiber_analysis_data.get('lumen_mask')
                        fiber_type = fiber_type_result
                        result['fiber_type'] = fiber_type
                        result['fiber_confidence'] = confidence
                        print(f"     ✅ Fiber type: {fiber_type} (conf: {confidence:.3f})")
                        
                        # Log what we got from fiber_analysis_data
                        if fiber_mask is not None:
                            print(f"     ✅ Fiber mask extracted: {fiber_mask.shape}")
                        else:
                            print(f"     ⚠️ No fiber mask in analysis data")
                    else:
                        result['fiber_type'] = 'unknown'
                        print(f"     ⚠️ No fiber_analysis_data returned")
                        
                except Exception as e:
                    print(f"     ⚠️ Fiber detection issue: {e}")
                    import traceback
                    traceback.print_exc()
                    result['fiber_type'] = 'unknown'
            else:
                print(f"     ⚠️ Fiber detector not available")
            
            # Create basic masks if fiber detection failed
            if fiber_mask is None:
                print(f"     🔧 Creating fallback mask...")
                # Create a basic mask using thresholding
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
                fiber_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                fiber_mask = fiber_mask.astype(bool)
                print(f"     ✅ Fallback mask created")
            
            # Porosity analysis using correct method
            porosity_features = {}
            print(f"     📊 Porosity analysis...")
            
            if self.porosity_analyzer and MODULES_LOADED['porosity_analysis']:
                try:
                    porosity_type = MODULES_LOADED['porosity_analysis']
                    if porosity_type == 'fast_refined':
                        # CORRECTED: Use analyze_fiber_porosity method
                        porosity_result = self.porosity_analyzer.analyze_fiber_porosity(
                            image, fiber_mask.astype(np.uint8), scale_factor, fiber_type
                        )
                    elif porosity_type == 'enhanced':
                        # CORRECTED: Use enhanced method
                        porosity_result = analyze_fiber_porosity_enhanced(
                            image, fiber_mask.astype(np.uint8), scale_factor
                        )
                    else:
                        # Basic version
                        porosity_result = self.porosity_analyzer.analyze_fiber_porosity(
                            image, fiber_mask.astype(np.uint8), scale_factor
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
                print(f"     ⚠️ Porosity analyzer not available")
            
            # Crumbly texture analysis with timeout protection
            print(f"     🧩 Crumbly texture analysis...")
            
            if self.crumbly_detector and MODULES_LOADED['crumbly_detection']:
                try:
                    import threading
                    
                    crumbly_result = None
                    error_occurred = None
                    
                    def crumbly_analysis_worker():
                        nonlocal crumbly_result, error_occurred
                        try:
                            # CORRECTED: Use analyze_crumbly_texture method
                            crumbly_result = self.crumbly_detector.analyze_crumbly_texture(
                                image, fiber_mask.astype(np.uint8), lumen_mask, scale_factor, debug=debug
                            )
                        except Exception as e:
                            error_occurred = e
                    
                    # Run crumbly analysis in a separate thread with timeout
                    thread = threading.Thread(target=crumbly_analysis_worker)
                    thread.daemon = True
                    thread.start()
                    thread.join(timeout=60)  # 60 second timeout for crumbly analysis
                    
                    if thread.is_alive():
                        print(f"     ⏰ Crumbly analysis timeout - skipping this image")
                        result['error'] = "Crumbly analysis timeout"
                        return result
                    elif error_occurred:
                        raise error_occurred
                    elif crumbly_result:
                        # Extract prediction results
                        result['predicted_label'] = crumbly_result.get('classification', 'unknown')
                        result['prediction_confidence'] = crumbly_result.get('confidence', 0.0)
                        result['crumbly_score'] = crumbly_result.get('crumbly_score', 0.0)
                        result['processing_success'] = True
                        
                        print(f"     ✅ Crumbly analysis: {result['predicted_label']} (score: {result['crumbly_score']:.3f})")
                        
                        # Extract ML features for training
                        ml_features = {}
                        
                        # Add detailed metrics from crumbly analysis
                        if 'detailed_metrics' in crumbly_result:
                            ml_features.update(crumbly_result['detailed_metrics'])
                        
                        # Add basic crumbly features
                        ml_features.update({
                            'crumbly_score': crumbly_result.get('crumbly_score', 0.0),
                            'confidence': crumbly_result.get('confidence', 0.0),
                            'num_crumbly_indicators': crumbly_result.get('num_crumbly_indicators', 0),
                            'num_porous_indicators': crumbly_result.get('num_porous_indicators', 0),
                            'crumbly_evidence': crumbly_result.get('crumbly_evidence', 0.0),
                            'porous_evidence': crumbly_result.get('porous_evidence', 0.0),
                        })
                        
                        # Add porosity features
                        ml_features.update(porosity_features)
                        
                        # Add fiber features
                        ml_features.update({
                            'fiber_type_is_hollow': 1 if fiber_type == 'hollow_fiber' else 0,
                            'fiber_type_is_filament': 1 if fiber_type == 'filament' else 0,
                            'scale_detected': 1 if result['scale_detected'] else 0,
                            'scale_factor': scale_factor
                        })
                        
                        result['ml_features'] = ml_features
                        
                        # Add scale-aware physical features
                        if scale_factor > 1.0:
                            physical_features = {}
                            for key, value in ml_features.items():
                                if isinstance(value, (int, float)) and value > 0:
                                    if 'area' in key.lower():
                                        physical_features[f'{key}_um2'] = value * (scale_factor ** 2)
                                    elif any(x in key.lower() for x in ['length', 'diameter', 'size']):
                                        physical_features[f'{key}_um'] = value * scale_factor
                            result['physical_features'] = physical_features
                    else:
                        print(f"     ❌ Crumbly analysis returned no result")
                        result['error'] = "Crumbly analysis returned no result"
                        return result
                        
                except Exception as e:
                    print(f"     ❌ Crumbly analysis failed: {e}")
                    result['error'] = f"Crumbly analysis failed: {e}"
                    import traceback
                    traceback.print_exc()
                    return result
            else:
                result['error'] = "CrumblyDetector not available"
                return result
            
            print(f"     🎉 Processing complete!")
            
        except Exception as e:
            result['error'] = str(e)
            print(f"     ❌ Processing error: {e}")
            import traceback
            traceback.print_exc()
            return result
        
        return result
    
    def run_evaluation(self, max_images: Optional[int] = None, debug_images: bool = False) -> bool:
        """
        Run comprehensive evaluation of the crumbly detector.
        CORRECTED: Proper method implementation with all fixes.
        """
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
        
        # Get successful results only
        successful_results = [r for r in self.evaluation_results if r['processing_success']]
        
        if not successful_results:
            print("❌ No successful results to analyze!")
            return
        
        # Calculate accuracy metrics
        true_labels = [r['true_label'] for r in successful_results]
        predicted_labels = [r['predicted_label'] for r in successful_results]
        
        # Overall accuracy
        correct_predictions = sum(1 for t, p in zip(true_labels, predicted_labels) if t == p)
        overall_accuracy = correct_predictions / len(successful_results)
        
        print(f"   Overall accuracy: {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)")
        
        # Per-class accuracy
        class_accuracies = {}
        for label in ['not', 'intermediate', 'crumbly']:
            class_true = [i for i, t in enumerate(true_labels) if t == label]
            if class_true:
                class_correct = sum(1 for i in class_true if predicted_labels[i] == label)
                class_accuracies[label] = class_correct / len(class_true)
                print(f"   {label.capitalize()} accuracy: {class_accuracies[label]:.3f}")
        
        # Store analysis results
        self.analysis_results = {
            'overall_accuracy': overall_accuracy,
            'class_accuracies': class_accuracies,
            'total_successful': len(successful_results),
            'total_images': len(self.evaluation_results),
            'confusion_matrix': confusion_matrix(true_labels, predicted_labels, 
                                               labels=['not', 'intermediate', 'crumbly']).tolist(),
            'classification_report': classification_report(true_labels, predicted_labels, 
                                                         labels=['not', 'intermediate', 'crumbly'],
                                                         output_dict=True)
        }
    
    def save_results(self):
        """Save evaluation results to files."""
        print(f"\n💾 SAVING RESULTS")
        print("=" * 30)
        
        # Save detailed results
        results_df = pd.DataFrame(self.evaluation_results)
        detailed_file = self.run_dir / "detailed_results.csv"
        results_df.to_csv(detailed_file, index=False)
        print(f"   📊 Detailed results: {detailed_file.name}")
        
        # Save ML features for training
        if self.feature_matrix and self.labels:
            # Get feature names from first successful result
            feature_names = []
            for result in self.evaluation_results:
                if result.get('processing_success', False) and 'ml_features' in result:
                    feature_names = list(result['ml_features'].keys())
                    break
            
            if feature_names:
                features_df = pd.DataFrame(self.feature_matrix, columns=feature_names)
                features_df['true_label'] = [self.reverse_label_mapping[label] for label in self.labels]
                features_df['true_label_numeric'] = self.labels
                
                ml_features_file = self.run_dir / f"ml_features_{self.timestamp}.csv"
                features_df.to_csv(ml_features_file, index=False)
                print(f"   🤖 ML features: {ml_features_file.name}")
        
        # Save analysis summary
        if self.analysis_results:
            summary_file = self.run_dir / "analysis_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(self.analysis_results, f, indent=2)
            print(f"   📈 Analysis summary: {summary_file.name}")
        
        print(f"   📁 All results saved to: {self.run_dir}")
    
    def generate_ml_training_recommendations(self):
        """Generate recommendations for ML model training."""
        print(f"\n🤖 ML TRAINING RECOMMENDATIONS")
        print("=" * 40)
        
        if not self.analysis_results:
            print("❌ No analysis results available")
            return
        
        accuracy = self.analysis_results.get('overall_accuracy', 0)
        
        if accuracy < 0.5:
            print("❌ Current detector accuracy is very low (<50%)")
            print("   Recommendation: Focus on feature engineering and data collection")
        elif accuracy < 0.7:
            print("⚠️ Current detector has moderate accuracy (<70%)")
            print("   Recommendation: Hybrid ML approach could provide significant improvement")
        else:
            print("✅ Current detector has good accuracy (>70%)")
            print("   Recommendation: ML can fine-tune and provide confidence estimates")
        
        print(f"\n📋 NEXT STEPS:")
        print(f"   1. Review the detailed results in {self.run_dir}")
        print(f"   2. Analyze misclassified samples")
        print(f"   3. Consider implementing hybrid detector")
        print(f"   4. Use feature analysis for threshold tuning")
        
        # Print module status
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