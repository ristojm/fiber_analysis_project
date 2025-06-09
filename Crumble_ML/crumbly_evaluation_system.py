#!/usr/bin/env python3
"""
Crumbly Texture Evaluation System
Evaluates current detector against labeled dataset and prepares ML training data.

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
import warnings
warnings.filterwarnings('ignore')

# Import your existing modules
try:
    from modules.image_preprocessing import load_image, preprocess_pipeline
    from modules.scale_detection import detect_scale_bar
    from modules.fiber_type_detection import FiberTypeDetector
    # Import your crumbly detector (assuming it's in the same directory or path)
    from crumbly_detection import CrumblyDetector  # Replace with actual import path
    print("✅ Successfully imported all modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all modules are in the correct path")
    sys.exit(1)

class CrumblyEvaluationSystem:
    """
    Comprehensive evaluation system for crumbly texture detection.
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
        self.output_dir.mkdir(exist_ok=True)
        
        # Create timestamp for this evaluation run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"evaluation_run_{self.timestamp}"
        self.run_dir.mkdir(exist_ok=True)
        
        # Initialize detectors
        self.crumbly_detector = CrumblyDetector(porosity_aware=True)
        self.fiber_detector = FiberTypeDetector()
        
        # Results storage
        self.evaluation_results = []
        self.feature_matrix = []
        self.labels = []
        self.label_mapping = {'not': 0, 'intermediate': 1, 'crumbly': 2}
        self.reverse_label_mapping = {0: 'not', 1: 'intermediate', 2: 'crumbly'}
        
        print(f"🔬 Crumbly Evaluation System Initialized")
        print(f"   Dataset: {self.dataset_path}")
        print(f"   Output: {self.run_dir}")
    
    def validate_dataset_structure(self):
        """Validate that the dataset has the expected structure."""
        
        required_folders = ['crumbly', 'intermediate', 'not']
        missing_folders = []
        
        for folder in required_folders:
            folder_path = self.dataset_path / folder
            if not folder_path.exists():
                missing_folders.append(folder)
        
        if missing_folders:
            print(f"❌ Missing folders: {missing_folders}")
            print(f"Expected structure: {self.dataset_path}/{{crumbly,intermediate,not}}/")
            return False
        
        # Count images in each folder
        image_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp']
        folder_counts = {}
        
        for folder in required_folders:
            folder_path = self.dataset_path / folder
            count = 0
            for ext in image_extensions:
                count += len(list(folder_path.glob(f'*{ext}')))
                count += len(list(folder_path.glob(f'*{ext.upper()}')))
            folder_counts[folder] = count
        
        print(f"📁 Dataset Structure Validated:")
        for folder, count in folder_counts.items():
            print(f"   {folder}: {count} images")
        
        total_images = sum(folder_counts.values())
        if total_images == 0:
            print("❌ No images found in dataset!")
            return False
        
        print(f"   Total: {total_images} images")
        return True
    
    def get_image_files(self):
        """Get all image files from the dataset."""
        
        image_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp']
        image_files = []
        
        for label in ['crumbly', 'intermediate', 'not']:
            folder_path = self.dataset_path / label
            
            for ext in image_extensions:
                # Case insensitive search
                files = list(folder_path.glob(f'*{ext}'))
                files.extend(list(folder_path.glob(f'*{ext.upper()}')))
                
                for file_path in files:
                    image_files.append({
                        'path': file_path,
                        'true_label': label,
                        'label_numeric': self.label_mapping[label]
                    })
        
        return image_files
    
    def process_single_image(self, image_info: dict, debug: bool = False):
        """Process a single image and extract all features."""
        
        image_path = image_info['path']
        true_label = image_info['true_label']
        
        result = {
            'image_path': str(image_path),
            'image_name': image_path.name,
            'true_label': true_label,
            'true_label_numeric': image_info['label_numeric'],
            'processing_success': False,
            'error': None
        }
        
        try:
            # 1. Load and preprocess image
            image = load_image(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Basic preprocessing
            processed_result = preprocess_pipeline(str(image_path))
            processed_image = processed_result.get('processed', image)
            
            # 2. Scale detection (use original image)
            scale_result = detect_scale_bar(image)
            scale_factor = scale_result if isinstance(scale_result, float) else 1.0
            
            # 3. Fiber type detection and segmentation
            fiber_type, fiber_confidence, fiber_analysis = self.fiber_detector.classify_fiber_type(
                processed_image, scale_factor
            )
            
            fiber_mask = fiber_analysis.get('fiber_mask', np.zeros_like(processed_image, dtype=bool))
            
            # Get lumen mask if it's a hollow fiber
            lumen_mask = None
            if fiber_type == 'hollow_fiber' and fiber_analysis.get('individual_results'):
                for fiber_result in fiber_analysis['individual_results']:
                    if fiber_result.get('has_lumen', False):
                        lumen_props = fiber_result.get('lumen_properties', {})
                        lumen_contour = lumen_props.get('contour')
                        if lumen_contour is not None:
                            lumen_mask = np.zeros_like(processed_image, dtype=bool)
                            cv2.fillPoly(lumen_mask.astype(np.uint8), [lumen_contour], 1)
                            break
            
            # 4. Crumbly texture analysis
            crumbly_result = self.crumbly_detector.analyze_crumbly_texture(
                processed_image, fiber_mask, lumen_mask, scale_factor, debug=debug
            )
            
            # 5. Extract comprehensive features for ML
            features = self.extract_ml_features(crumbly_result, fiber_analysis, scale_factor)
            
            # 6. Store results
            result.update({
                'processing_success': True,
                'scale_factor': scale_factor,
                'fiber_type': fiber_type,
                'fiber_confidence': fiber_confidence,
                'predicted_label': crumbly_result.get('classification', 'unknown'),
                'prediction_confidence': crumbly_result.get('confidence', 0.0),
                'crumbly_score': crumbly_result.get('crumbly_score', 0.5),
                'crumbly_result': crumbly_result,
                'fiber_analysis': fiber_analysis,
                'ml_features': features,
                'image_shape': image.shape
            })
            
        except Exception as e:
            result['error'] = str(e)
            print(f"❌ Error processing {image_path.name}: {e}")
        
        return result
    
    def extract_ml_features(self, crumbly_result: dict, fiber_analysis: dict, scale_factor: float):
        """Extract numerical features suitable for machine learning."""
        
        features = {}
        
        try:
            # 1. Crumbly detection features
            features['crumbly_score'] = crumbly_result.get('crumbly_score', 0.5)
            features['crumbly_confidence'] = crumbly_result.get('confidence', 0.5)
            features['crumbly_evidence'] = crumbly_result.get('crumbly_evidence', 0.5)
            features['porous_intact_evidence'] = crumbly_result.get('porous_intact_evidence', 0.5)
            features['num_crumbly_indicators'] = crumbly_result.get('num_crumbly_indicators', 0)
            features['num_intact_indicators'] = crumbly_result.get('num_intact_indicators', 0)
            
            # 2. Pore characteristics
            pore_metrics = crumbly_result.get('pore_metrics', {})
            features['pore_count'] = pore_metrics.get('pore_count', 0)
            features['organized_porosity_score'] = pore_metrics.get('organized_porosity_score', 0.5)
            features['mean_pore_circularity'] = pore_metrics.get('mean_pore_circularity', 0.5)
            features['pore_circularity_consistency'] = pore_metrics.get('pore_circularity_consistency', 0.5)
            features['mean_pore_edge_smoothness'] = pore_metrics.get('mean_pore_edge_smoothness', 0.5)
            features['pore_size_variation'] = pore_metrics.get('pore_size_variation', 0.5)
            features['spatial_organization'] = pore_metrics.get('spatial_organization', 0.5)
            features['total_pore_area_fraction'] = pore_metrics.get('total_pore_area_fraction', 0.0)
            
            # 3. Wall integrity features
            wall_metrics = crumbly_result.get('wall_integrity_metrics', {})
            features['wall_integrity_score'] = wall_metrics.get('wall_integrity_score', 0.5)
            
            thickness_metrics = wall_metrics.get('thickness_metrics', {})
            features['wall_thickness_consistency'] = thickness_metrics.get('thickness_consistency', 0.5)
            features['wall_thickness_variation'] = thickness_metrics.get('thickness_variation', 0.5)
            
            continuity_metrics = wall_metrics.get('continuity_metrics', {})
            features['wall_continuity_score'] = continuity_metrics.get('continuity_score', 0.5)
            features['num_wall_components'] = continuity_metrics.get('num_wall_components', 1)
            
            fragmentation_metrics = wall_metrics.get('fragmentation_metrics', {})
            features['fragmentation_ratio'] = fragmentation_metrics.get('fragmentation_ratio', 0.0)
            features['edge_roughness'] = fragmentation_metrics.get('edge_roughness', 0.5)
            
            # 4. Boundary features
            boundary_metrics = crumbly_result.get('boundary_metrics', {})
            outer_boundary = boundary_metrics.get('outer_boundary', {})
            features['boundary_circularity'] = outer_boundary.get('circularity', 0.5)
            features['boundary_solidity'] = outer_boundary.get('solidity', 0.5)
            features['boundary_roughness'] = outer_boundary.get('roughness_index', 0.5)
            features['fractal_dimension'] = outer_boundary.get('fractal_dimension', 1.0)
            
            curvature_stats = outer_boundary.get('curvature_stats', {})
            features['mean_curvature'] = curvature_stats.get('mean_curvature', 0.0)
            features['curvature_variation'] = curvature_stats.get('curvature_variation', 0.0)
            
            # 5. Texture features
            texture_metrics = crumbly_result.get('texture_metrics', {})
            
            lbp_metrics = texture_metrics.get('lbp', {})
            features['lbp_uniformity'] = lbp_metrics.get('lbp_uniformity', 0.3)
            features['lbp_entropy'] = lbp_metrics.get('lbp_entropy', 0.5)
            features['non_uniform_ratio'] = lbp_metrics.get('non_uniform_ratio', 0.3)
            
            glcm_metrics = texture_metrics.get('glcm', {})
            features['glcm_contrast'] = glcm_metrics.get('contrast', 0.0)
            features['glcm_homogeneity'] = glcm_metrics.get('homogeneity', 0.5)
            features['glcm_energy'] = glcm_metrics.get('energy', 0.1)
            features['glcm_correlation'] = glcm_metrics.get('correlation', 0.0)
            
            edge_metrics = texture_metrics.get('edges', {})
            features['edge_density'] = edge_metrics.get('edge_density', 0.3)
            features['mean_gradient_magnitude'] = edge_metrics.get('mean_gradient_magnitude', 0.0)
            features['direction_uniformity'] = edge_metrics.get('direction_uniformity', 0.5)
            
            # 6. Fiber analysis features
            features['fiber_count'] = fiber_analysis.get('total_fibers', 0)
            features['hollow_fiber_ratio'] = (fiber_analysis.get('hollow_fibers', 0) / 
                                            max(1, fiber_analysis.get('total_fibers', 1)))
            
            # 7. Scale and size features
            features['scale_factor'] = scale_factor
            features['fiber_mask_coverage'] = fiber_analysis.get('mask_coverage_percent', 0.0)
            
        except Exception as e:
            print(f"Warning: Feature extraction error: {e}")
            # Return default features if extraction fails
            features = {f'feature_{i}': 0.0 for i in range(30)}
        
        return features
    
    def run_evaluation(self, max_images: int = None, debug_images: bool = False):
        """Run comprehensive evaluation on the dataset."""
        
        print(f"\n🚀 STARTING COMPREHENSIVE EVALUATION")
        print("=" * 60)
        
        # Validate dataset
        if not self.validate_dataset_structure():
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
                feature_vector = list(result['ml_features'].values())
                self.feature_matrix.append(feature_vector)
                self.labels.append(image_info['label_numeric'])
                
                successful_processes += 1
            else:
                print(f"   ❌ Processing failed: {result['error']}")
            
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
        
        # Extract predictions and ground truth
        y_true = [r['true_label'] for r in successful_results]
        y_pred = [r['predicted_label'] for r in successful_results]
        y_scores = [r['crumbly_score'] for r in successful_results]
        y_confidence = [r['prediction_confidence'] for r in successful_results]
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_true, y_pred, digits=3))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=['not', 'intermediate', 'crumbly'])
        print(f"\nConfusion Matrix:")
        print("Predicted →")
        print("True ↓    not  inter crumbly")
        for i, true_label in enumerate(['not', 'intermediate', 'crumbly']):
            row_str = f"{true_label:8} "
            for j in range(3):
                row_str += f"{cm[i,j]:4d}  "
            print(row_str)
        
        # Per-class analysis
        print(f"\nPer-Class Analysis:")
        for true_label in ['not', 'intermediate', 'crumbly']:
            class_results = [r for r in successful_results if r['true_label'] == true_label]
            if class_results:
                correct_predictions = [r for r in class_results if r['predicted_label'] == true_label]
                class_accuracy = len(correct_predictions) / len(class_results)
                avg_confidence = np.mean([r['prediction_confidence'] for r in class_results])
                avg_score = np.mean([r['crumbly_score'] for r in class_results])
                
                print(f"  {true_label:12}: {class_accuracy:.3f} accuracy, "
                      f"avg conf: {avg_confidence:.3f}, avg score: {avg_score:.3f}")
        
        # Feature analysis for ML
        if len(self.feature_matrix) > 10:
            self.analyze_features_for_ml()
        
        # Store analysis results
        self.analysis_results = {
            'overall_accuracy': accuracy,
            'total_images': len(successful_results),
            'y_true': y_true,
            'y_pred': y_pred,
            'y_scores': y_scores,
            'y_confidence': y_confidence,
            'confusion_matrix': cm.tolist(),
            'per_class_stats': {}
        }
    
    def analyze_features_for_ml(self):
        """Analyze features to understand which ones are most predictive."""
        
        print(f"\n🤖 MACHINE LEARNING FEATURE ANALYSIS")
        print("=" * 50)
        
        if len(self.feature_matrix) < 10:
            print("❌ Not enough samples for ML analysis")
            return
        
        # Convert to numpy arrays
        X = np.array(self.feature_matrix)
        y = np.array(self.labels)
        
        print(f"   Feature matrix shape: {X.shape}")
        print(f"   Label distribution: {np.bincount(y)}")
        
        # Check for missing values and inf
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Split data
        if len(X) > 20:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = X, X, y, y
            print("   Using all data for both training and testing (small dataset)")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Test different ML models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True)
        }
        
        ml_results = {}
        
        for model_name, model in models.items():
            try:
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Test predictions
                y_pred_ml = model.predict(X_test_scaled)
                y_prob_ml = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
                
                # Calculate accuracy
                ml_accuracy = accuracy_score(y_test, y_pred_ml)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=min(5, len(X_train)//2))
                cv_mean = np.mean(cv_scores)
                cv_std = np.std(cv_scores)
                
                ml_results[model_name] = {
                    'accuracy': ml_accuracy,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'model': model,
                    'scaler': scaler
                }
                
                print(f"   {model_name}:")
                print(f"     Test accuracy: {ml_accuracy:.3f}")
                print(f"     CV score: {cv_mean:.3f} ± {cv_std:.3f}")
                
                # Feature importance (if available)
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    top_features = np.argsort(importances)[-5:][::-1]  # Top 5 features
                    print(f"     Top features: {top_features} (importance: {importances[top_features]})")
                
            except Exception as e:
                print(f"   ❌ Error with {model_name}: {e}")
                ml_results[model_name] = {'error': str(e)}
        
        self.ml_analysis = ml_results
        
        # Save ML models for hybrid approach
        best_model_name = max(ml_results.keys(), 
                             key=lambda k: ml_results[k].get('cv_mean', 0))
        if 'model' in ml_results[best_model_name]:
            self.best_ml_model = ml_results[best_model_name]['model']
            self.best_scaler = ml_results[best_model_name]['scaler']
            print(f"   🏆 Best model: {best_model_name} (CV: {ml_results[best_model_name]['cv_mean']:.3f})")
    
    def save_results(self):
        """Save all evaluation results to files."""
        
        print(f"\n💾 SAVING RESULTS")
        print("=" * 30)
        
        # 1. Save detailed results as JSON
        json_file = self.run_dir / f"detailed_results_{self.timestamp}.json"
        
        # Prepare results for JSON serialization
        json_results = []
        for result in self.evaluation_results:
            json_result = result.copy()
            # Remove non-serializable items
            if 'crumbly_result' in json_result:
                del json_result['crumbly_result']
            if 'fiber_analysis' in json_result:
                del json_result['fiber_analysis']
            json_results.append(json_result)
        
        with open(json_file, 'w') as f:
            json.dump({
                'evaluation_metadata': {
                    'timestamp': self.timestamp,
                    'dataset_path': str(self.dataset_path),
                    'total_images': len(self.evaluation_results),
                    'successful_images': len([r for r in self.evaluation_results if r['processing_success']])
                },
                'results': json_results,
                'analysis': getattr(self, 'analysis_results', {}),
                'ml_analysis': getattr(self, 'ml_analysis', {})
            }, f, indent=2, default=str)
        
        print(f"   📄 Detailed results: {json_file.name}")
        
        # 2. Save feature matrix for ML
        if self.feature_matrix:
            features_file = self.run_dir / f"ml_features_{self.timestamp}.csv"
            
            # Get feature names from first successful result
            feature_names = []
            for result in self.evaluation_results:
                if result.get('processing_success') and 'ml_features' in result:
                    feature_names = list(result['ml_features'].keys())
                    break
            
            # Create DataFrame
            feature_df = pd.DataFrame(self.feature_matrix, columns=feature_names)
            feature_df['true_label'] = self.labels
            feature_df['true_label_name'] = [self.reverse_label_mapping[label] for label in self.labels]
            
            feature_df.to_csv(features_file, index=False)
            print(f"   📊 ML features: {features_file.name}")
        
        # 3. Save summary report
        report_file = self.run_dir / f"evaluation_summary_{self.timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("CRUMBLY TEXTURE DETECTION EVALUATION REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {self.dataset_path}\n\n")
            
            if hasattr(self, 'analysis_results'):
                f.write(f"PERFORMANCE SUMMARY:\n")
                f.write(f"Overall Accuracy: {self.analysis_results['overall_accuracy']:.3f}\n")
                f.write(f"Total Images: {self.analysis_results['total_images']}\n\n")
                
                f.write("Classification Performance:\n")
                for i, (true_labels, pred_labels) in enumerate(zip(self.analysis_results['y_true'], 
                                                                   self.analysis_results['y_pred'])):
                    f.write(f"  {true_labels} → {pred_labels}\n")
            
            if hasattr(self, 'ml_analysis'):
                f.write(f"\nMACHINE LEARNING ANALYSIS:\n")
                for model_name, results in self.ml_analysis.items():
                    if 'accuracy' in results:
                        f.write(f"  {model_name}: {results['cv_mean']:.3f} ± {results['cv_std']:.3f}\n")
        
        print(f"   📋 Summary report: {report_file.name}")
        
        # 4. Create visualizations
        try:
            self.create_visualizations()
            print(f"   📈 Visualizations: {self.run_dir}/plots/")
        except Exception as e:
            print(f"   ⚠️ Visualization error: {e}")
        
        print(f"\n✅ All results saved to: {self.run_dir}")
    
    def create_visualizations(self):
        """Create analysis visualizations."""
        
        plots_dir = self.run_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        if not hasattr(self, 'analysis_results'):
            return
        
        # 1. Confusion Matrix Heatmap
        plt.figure(figsize=(8, 6))
        cm = np.array(self.analysis_results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['not', 'intermediate', 'crumbly'],
                   yticklabels=['not', 'intermediate', 'crumbly'])
        plt.title('Confusion Matrix - Current Detector')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(plots_dir / 'confusion_matrix.png', dpi=150)
        plt.close()
        
        # 2. Score Distribution by Class
        successful_results = [r for r in self.evaluation_results if r['processing_success']]
        
        plt.figure(figsize=(12, 4))
        
        for i, true_label in enumerate(['not', 'intermediate', 'crumbly']):
            plt.subplot(1, 3, i+1)
            class_results = [r for r in successful_results if r['true_label'] == true_label]
            scores = [r['crumbly_score'] for r in class_results]
            
            plt.hist(scores, bins=20, alpha=0.7, edgecolor='black')
            plt.title(f'Crumbly Scores\n{true_label} (n={len(scores)})')
            plt.xlabel('Crumbly Score')
            plt.ylabel('Count')
            plt.axvline(0.5, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'score_distributions.png', dpi=150)
        plt.close()
        
        # 3. Feature Correlation Matrix (if ML features available)
        if self.feature_matrix and len(self.feature_matrix) > 10:
            # Get feature names
            feature_names = []
            for result in self.evaluation_results:
                if result.get('processing_success') and 'ml_features' in result:
                    feature_names = list(result['ml_features'].keys())
                    break
            
            if feature_names:
                X = np.array(self.feature_matrix)
                X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
                
                # Calculate correlation matrix
                corr_matrix = np.corrcoef(X.T)
                
                plt.figure(figsize=(12, 10))
                sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                           xticklabels=feature_names, yticklabels=feature_names)
                plt.title('Feature Correlation Matrix')
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                plt.savefig(plots_dir / 'feature_correlation.png', dpi=150)
                plt.close()
    
    def generate_ml_training_recommendations(self):
        """Generate recommendations for ML model training."""
        
        print(f"\n🎯 ML TRAINING RECOMMENDATIONS")
        print("=" * 50)
        
        if not hasattr(self, 'analysis_results'):
            print("❌ No analysis results available")
            return
        
        accuracy = self.analysis_results['overall_accuracy']
        total_samples = self.analysis_results['total_images']
        
        print(f"Current Detector Performance: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"Total Samples Available: {total_samples}")
        
        # Recommendations based on performance
        if accuracy >= 0.85:
            print("\n✅ EXCELLENT: Current detector performs very well!")
            print("   Recommendations:")
            print("   • Your current approach is working great")
            print("   • Consider minor threshold tuning only")
            print("   • ML may provide marginal improvement")
            
        elif accuracy >= 0.70:
            print("\n🟡 GOOD: Current detector is decent, ML could help")
            print("   Recommendations:")
            print("   • Hybrid ML approach recommended")
            print("   • Focus on ensemble methods")
            print("   • Use ML to refine borderline cases")
            
        else:
            print("\n🔴 NEEDS IMPROVEMENT: ML integration strongly recommended")
            print("   Recommendations:")
            print("   • Current approach needs significant enhancement")
            print("   • Deep learning or ensemble methods needed")
            print("   • Consider feature engineering improvements")
        
        # Sample size recommendations
        if total_samples < 50:
            print(f"\n⚠️  SMALL DATASET WARNING:")
            print(f"   • {total_samples} samples may not be enough for robust ML")
            print(f"   • Consider collecting more labeled data")
            print(f"   • Use simple models (Random Forest, SVM)")
            
        elif total_samples < 200:
            print(f"\n🟡 MODERATE DATASET:")
            print(f"   • {total_samples} samples suitable for classical ML")
            print(f"   • Avoid deep learning (overfitting risk)")
            print(f"   • Focus on feature engineering and ensembles")
            
        else:
            print(f"\n✅ GOOD DATASET SIZE:")
            print(f"   • {total_samples} samples support various ML approaches")
            print(f"   • Deep learning feasible with proper validation")
            print(f"   • Consider transfer learning from texture datasets")
        
        # ML model recommendations
        if hasattr(self, 'ml_analysis'):
            print(f"\n🤖 ML MODEL PERFORMANCE:")
            for model_name, results in self.ml_analysis.items():
                if 'cv_mean' in results:
                    improvement = results['cv_mean'] - accuracy
                    print(f"   {model_name}: {results['cv_mean']:.3f} "
                          f"({improvement:+.3f} vs current)")
        
        print(f"\n📝 NEXT STEPS:")
        print(f"   1. Review the detailed results in {self.run_dir}")
        print(f"   2. Analyze misclassified samples")
        print(f"   3. Consider implementing hybrid detector")
        print(f"   4. Use feature analysis for threshold tuning")

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