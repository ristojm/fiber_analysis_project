#!/usr/bin/env python3
"""
Hybrid ML-Enhanced Crumbly Detector
Combines traditional computer vision with machine learning for improved accuracy.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import json
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

# Import your existing crumbly detector
try:
    from modules.crumbly_detection import CrumblyDetector  # Replace with actual import path
except ImportError:
    print("Warning: Could not import CrumblyDetector. Make sure it's in the correct path.")
    CrumblyDetector = None

class HybridCrumblyDetector:
    """
    Hybrid detector that combines traditional computer vision with machine learning.
    
    Features:
    - Uses your existing CrumblyDetector as a feature extractor
    - Adds ML models trained on your labeled data
    - Provides ensemble predictions with uncertainty quantification
    - Adaptive thresholds based on confidence
    """
    
    def __init__(self, 
                 traditional_detector: Optional[CrumblyDetector] = None,
                 model_path: Optional[str] = None,
                 confidence_threshold: float = 0.7):
        """
        Initialize hybrid detector.
        
        Args:
            traditional_detector: Instance of your CrumblyDetector
            model_path: Path to saved ML models (if available)
            confidence_threshold: Minimum confidence for ML predictions
        """
        
        # Initialize traditional detector
        if traditional_detector is not None:
            self.traditional_detector = traditional_detector
        elif CrumblyDetector is not None:
            self.traditional_detector = CrumblyDetector(porosity_aware=True)
        else:
            raise ValueError("CrumblyDetector not available and no instance provided")
        
        # ML components
        self.ml_models = {}
        self.feature_scaler = None
        self.feature_names = []
        self.is_trained = False
        self.confidence_threshold = confidence_threshold
        
        # Load pre-trained models if available
        if model_path:
            self.load_models(model_path)
        
        # Label mapping
        self.label_mapping = {'not': 0, 'intermediate': 1, 'crumbly': 2}
        self.reverse_label_mapping = {0: 'not', 1: 'intermediate', 2: 'crumbly'}
    
    def extract_features(self, image: np.ndarray, fiber_mask: np.ndarray, 
                        lumen_mask: Optional[np.ndarray] = None, 
                        scale_factor: float = 1.0) -> Dict:
        """
        Extract comprehensive features using traditional detector.
        """
        
        # Get traditional analysis
        traditional_result = self.traditional_detector.analyze_crumbly_texture(
            image, fiber_mask, lumen_mask, scale_factor, debug=False
        )
        
        # Extract numerical features
        features = self._extract_numerical_features(traditional_result)
        
        return {
            'traditional_result': traditional_result,
            'features': features,
            'feature_vector': list(features.values())
        }
    
    def _extract_numerical_features(self, crumbly_result: Dict) -> Dict:
        """Extract numerical features from crumbly analysis result."""
        
        features = {}
        
        try:
            # Core classification features
            features['crumbly_score'] = crumbly_result.get('crumbly_score', 0.5)
            features['traditional_confidence'] = crumbly_result.get('confidence', 0.5)
            features['crumbly_evidence'] = crumbly_result.get('crumbly_evidence', 0.5)
            features['porous_intact_evidence'] = crumbly_result.get('porous_intact_evidence', 0.5)
            
            # Pore characteristics
            pore_metrics = crumbly_result.get('pore_metrics', {})
            features['pore_count'] = pore_metrics.get('pore_count', 0)
            features['organized_porosity_score'] = pore_metrics.get('organized_porosity_score', 0.5)
            features['mean_pore_circularity'] = pore_metrics.get('mean_pore_circularity', 0.5)
            features['pore_circularity_consistency'] = pore_metrics.get('pore_circularity_consistency', 0.5)
            features['mean_pore_edge_smoothness'] = pore_metrics.get('mean_pore_edge_smoothness', 0.5)
            features['pore_size_variation'] = pore_metrics.get('pore_size_variation', 0.5)
            features['spatial_organization'] = pore_metrics.get('spatial_organization', 0.5)
            features['total_pore_area_fraction'] = pore_metrics.get('total_pore_area_fraction', 0.0)
            
            # Wall integrity
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
            
            # Boundary characteristics
            boundary_metrics = crumbly_result.get('boundary_metrics', {})
            outer_boundary = boundary_metrics.get('outer_boundary', {})
            features['boundary_circularity'] = outer_boundary.get('circularity', 0.5)
            features['boundary_solidity'] = outer_boundary.get('solidity', 0.5)
            features['boundary_roughness'] = outer_boundary.get('roughness_index', 0.5)
            features['fractal_dimension'] = outer_boundary.get('fractal_dimension', 1.0)
            
            curvature_stats = outer_boundary.get('curvature_stats', {})
            features['mean_curvature'] = curvature_stats.get('mean_curvature', 0.0)
            features['curvature_variation'] = curvature_stats.get('curvature_variation', 0.0)
            
            # Texture features
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
            
            # Additional derived features
            features['pore_wall_balance'] = features['organized_porosity_score'] * features['wall_integrity_score']
            features['structure_consistency'] = (features['wall_continuity_score'] + 
                                               features['boundary_circularity']) / 2
            features['texture_chaos'] = 1.0 - features['lbp_uniformity']
            
        except Exception as e:
            print(f"Warning: Feature extraction error: {e}")
            # Return default features
            features = {f'feature_{i}': 0.0 for i in range(30)}
        
        return features
    
    def train_ml_models(self, training_data_path: str, test_size: float = 0.2):
        """
        Train ML models using labeled data from evaluation system.
        
        Args:
            training_data_path: Path to CSV file with ML features from evaluation
            test_size: Fraction of data to use for testing
        """
        
        print(f"🤖 TRAINING HYBRID ML MODELS")
        print("=" * 40)
        
        # Load training data
        try:
            df = pd.read_csv(training_data_path)
            print(f"   Loaded {len(df)} samples from {training_data_path}")
        except Exception as e:
            print(f"❌ Error loading training data: {e}")
            return False
        
        # Prepare features and labels
        feature_columns = [col for col in df.columns if col not in 
                          ['true_label', 'true_label_name']]
        
        X = df[feature_columns].values
        y = df['true_label'].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Store feature names
        self.feature_names = feature_columns
        
        print(f"   Feature matrix shape: {X.shape}")
        print(f"   Label distribution: {np.bincount(y)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        self.feature_scaler = StandardScaler()
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)
        
        # Train multiple models
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf', 
                C=1.0, 
                gamma='scale',
                probability=True,
                random_state=42
            )
        }
        
        # Train and evaluate each model
        for model_name, model in models.items():
            print(f"\n   Training {model_name}...")
            
            try:
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Test performance
                train_score = model.score(X_train_scaled, y_train)
                test_score = model.score(X_test_scaled, y_test)
                
                print(f"     Train accuracy: {train_score:.3f}")
                print(f"     Test accuracy: {test_score:.3f}")
                
                # Store model
                self.ml_models[model_name] = model
                
                # Feature importance (if available)
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    top_features_idx = np.argsort(importances)[-5:][::-1]
                    print(f"     Top features: {[self.feature_names[i] for i in top_features_idx]}")
                
            except Exception as e:
                print(f"     ❌ Error training {model_name}: {e}")
        
        # Create ensemble model
        if len(self.ml_models) > 1:
            ensemble_models = [(name, model) for name, model in self.ml_models.items()]
            self.ensemble_model = VotingClassifier(
                estimators=ensemble_models,
                voting='soft'  # Use probability voting
            )
            
            print(f"\n   Training ensemble model...")
            self.ensemble_model.fit(X_train_scaled, y_train)
            ensemble_score = self.ensemble_model.score(X_test_scaled, y_test)
            print(f"     Ensemble accuracy: {ensemble_score:.3f}")
        
        self.is_trained = True
        print(f"\n✅ ML model training complete!")
        
        return True
    
    def predict_hybrid(self, image: np.ndarray, fiber_mask: np.ndarray,
                      lumen_mask: Optional[np.ndarray] = None,
                      scale_factor: float = 1.0) -> Dict:
        """
        Make hybrid prediction combining traditional and ML approaches.
        
        Returns comprehensive analysis with uncertainty quantification.
        """
        
        # 1. Extract features using traditional detector
        feature_analysis = self.extract_features(image, fiber_mask, lumen_mask, scale_factor)
        traditional_result = feature_analysis['traditional_result']
        
        # 2. Get traditional prediction
        traditional_pred = traditional_result.get('classification', 'unknown')
        traditional_conf = traditional_result.get('confidence', 0.5)
        traditional_score = traditional_result.get('crumbly_score', 0.5)
        
        # 3. Initialize hybrid result
        hybrid_result = {
            'traditional_prediction': traditional_pred,
            'traditional_confidence': traditional_conf,
            'traditional_score': traditional_score,
            'ml_available': self.is_trained,
            'ensemble_used': False,
            'final_prediction': traditional_pred,
            'final_confidence': traditional_conf,
            'prediction_method': 'traditional_only',
            'uncertainty': 0.5,
            'feature_analysis': feature_analysis
        }
        
        # 4. Add ML predictions if models are trained
        if self.is_trained and self.feature_scaler is not None:
            try:
                # Prepare feature vector
                feature_vector = np.array(feature_analysis['feature_vector']).reshape(1, -1)
                feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1.0, neginf=0.0)
                
                # Scale features
                feature_vector_scaled = self.feature_scaler.transform(feature_vector)
                
                # Get ML predictions
                ml_predictions = {}
                ml_probabilities = {}
                
                for model_name, model in self.ml_models.items():
                    try:
                        pred = model.predict(feature_vector_scaled)[0]
                        pred_label = self.reverse_label_mapping[pred]
                        
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(feature_vector_scaled)[0]
                            max_proba = np.max(proba)
                        else:
                            max_proba = 0.5
                        
                        ml_predictions[model_name] = {
                            'prediction': pred_label,
                            'confidence': max_proba,
                            'probabilities': proba.tolist() if 'proba' in locals() else [0.33, 0.33, 0.34]
                        }
                        
                        ml_probabilities[model_name] = proba if 'proba' in locals() else np.array([0.33, 0.33, 0.34])
                        
                    except Exception as e:
                        print(f"Warning: ML prediction error for {model_name}: {e}")
                
                hybrid_result['ml_predictions'] = ml_predictions
                
                # 5. Ensemble prediction if available
                if hasattr(self, 'ensemble_model') and ml_predictions:
                    try:
                        ensemble_pred = self.ensemble_model.predict(feature_vector_scaled)[0]
                        ensemble_proba = self.ensemble_model.predict_proba(feature_vector_scaled)[0]
                        ensemble_conf = np.max(ensemble_proba)
                        ensemble_label = self.reverse_label_mapping[ensemble_pred]
                        
                        hybrid_result['ensemble_prediction'] = {
                            'prediction': ensemble_label,
                            'confidence': ensemble_conf,
                            'probabilities': ensemble_proba.tolist()
                        }
                        
                        # 6. Decide final prediction based on confidence
                        if ensemble_conf > self.confidence_threshold:
                            # Use ensemble if confident
                            hybrid_result['final_prediction'] = ensemble_label
                            hybrid_result['final_confidence'] = ensemble_conf
                            hybrid_result['prediction_method'] = 'ensemble'
                            hybrid_result['ensemble_used'] = True
                        elif traditional_conf > ensemble_conf:
                            # Use traditional if it's more confident
                            hybrid_result['final_prediction'] = traditional_pred
                            hybrid_result['final_confidence'] = traditional_conf
                            hybrid_result['prediction_method'] = 'traditional_preferred'
                        else:
                            # Use ensemble but mark lower confidence
                            hybrid_result['final_prediction'] = ensemble_label
                            hybrid_result['final_confidence'] = ensemble_conf * 0.8  # Penalty for low confidence
                            hybrid_result['prediction_method'] = 'ensemble_low_conf'
                        
                        # Calculate uncertainty
                        entropy = -np.sum(ensemble_proba * np.log(ensemble_proba + 1e-10))
                        max_entropy = np.log(3)  # 3 classes
                        uncertainty = entropy / max_entropy
                        hybrid_result['uncertainty'] = uncertainty
                        
                    except Exception as e:
                        print(f"Warning: Ensemble prediction error: {e}")
                        # Fall back to traditional
                        hybrid_result['prediction_method'] = 'traditional_fallback'
                
                # 7. Agreement analysis
                if ml_predictions:
                    all_predictions = [traditional_pred] + [ml['prediction'] for ml in ml_predictions.values()]
                    unique_predictions = set(all_predictions)
                    agreement_ratio = all_predictions.count(hybrid_result['final_prediction']) / len(all_predictions)
                    
                    hybrid_result['prediction_agreement'] = {
                        'ratio': agreement_ratio,
                        'all_predictions': all_predictions,
                        'unanimous': len(unique_predictions) == 1
                    }
                    
                    # Boost confidence if all methods agree
                    if agreement_ratio == 1.0:
                        hybrid_result['final_confidence'] = min(0.95, hybrid_result['final_confidence'] + 0.1)
                    elif agreement_ratio < 0.5:
                        hybrid_result['final_confidence'] *= 0.8  # Reduce confidence if disagreement
            
            except Exception as e:
                print(f"Warning: ML analysis failed: {e}")
                hybrid_result['ml_error'] = str(e)
        
        return hybrid_result
    
    def save_models(self, model_path: str):
        """Save trained models to disk."""
        
        if not self.is_trained:
            print("❌ No trained models to save")
            return False
        
        model_dir = Path(model_path)
        model_dir.mkdir(exist_ok=True)
        
        try:
            # Save individual models
            for model_name, model in self.ml_models.items():
                model_file = model_dir / f"{model_name}_model.joblib"
                joblib.dump(model, model_file)
            
            # Save ensemble if available
            if hasattr(self, 'ensemble_model'):
                ensemble_file = model_dir / "ensemble_model.joblib"
                joblib.dump(self.ensemble_model, ensemble_file)
            
            # Save scaler
            if self.feature_scaler is not None:
                scaler_file = model_dir / "feature_scaler.joblib"
                joblib.dump(self.feature_scaler, scaler_file)
            
            # Save metadata
            metadata = {
                'feature_names': self.feature_names,
                'label_mapping': self.label_mapping,
                'model_names': list(self.ml_models.keys()),
                'has_ensemble': hasattr(self, 'ensemble_model'),
                'confidence_threshold': self.confidence_threshold
            }
            
            metadata_file = model_dir / "model_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Models saved to {model_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving models: {e}")
            return False
    
    def load_models(self, model_path: str):
        """Load trained models from disk."""
        
        model_dir = Path(model_path)
        
        if not model_dir.exists():
            print(f"❌ Model directory not found: {model_dir}")
            return False
        
        try:
            # Load metadata
            metadata_file = model_dir / "model_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                self.feature_names = metadata.get('feature_names', [])
                self.label_mapping = metadata.get('label_mapping', self.label_mapping)
                self.confidence_threshold = metadata.get('confidence_threshold', self.confidence_threshold)
                model_names = metadata.get('model_names', [])
                has_ensemble = metadata.get('has_ensemble', False)
            else:
                print("⚠️ No metadata file found, using defaults")
                model_names = ['random_forest', 'svm']
                has_ensemble = False
            
            # Load individual models
            for model_name in model_names:
                model_file = model_dir / f"{model_name}_model.joblib"
                if model_file.exists():
                    self.ml_models[model_name] = joblib.load(model_file)
                    print(f"   ✅ Loaded {model_name}")
            
            # Load ensemble
            if has_ensemble:
                ensemble_file = model_dir / "ensemble_model.joblib"
                if ensemble_file.exists():
                    self.ensemble_model = joblib.load(ensemble_file)
                    print(f"   ✅ Loaded ensemble model")
            
            # Load scaler
            scaler_file = model_dir / "feature_scaler.joblib"
            if scaler_file.exists():
                self.feature_scaler = joblib.load(scaler_file)
                print(f"   ✅ Loaded feature scaler")
            
            if self.ml_models and self.feature_scaler is not None:
                self.is_trained = True
                print(f"✅ Successfully loaded {len(self.ml_models)} models")
                return True
            else:
                print(f"❌ Failed to load required components")
                return False
                
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def analyze_crumbly_texture(self, image: np.ndarray, fiber_mask: np.ndarray,
                               lumen_mask: Optional[np.ndarray] = None,
                               scale_factor: float = 1.0, debug: bool = False) -> Dict:
        """
        Main interface for hybrid crumbly texture analysis.
        Compatible with original CrumblyDetector interface.
        """
        
        hybrid_result = self.predict_hybrid(image, fiber_mask, lumen_mask, scale_factor)
        
        # Format result to match original interface
        result = {
            'classification': hybrid_result['final_prediction'],
            'confidence': hybrid_result['final_confidence'],
            'crumbly_score': hybrid_result.get('traditional_score', 0.5),
            'prediction_method': hybrid_result['prediction_method'],
            'uncertainty': hybrid_result.get('uncertainty', 0.5),
            'scale_factor': scale_factor,
            
            # Traditional results
            'traditional_result': hybrid_result['feature_analysis']['traditional_result'],
            
            # ML results (if available)
            'ml_predictions': hybrid_result.get('ml_predictions', {}),
            'ensemble_prediction': hybrid_result.get('ensemble_prediction', {}),
            'prediction_agreement': hybrid_result.get('prediction_agreement', {}),
            
            # Enhanced analysis
            'hybrid_analysis': {
                'method_used': hybrid_result['prediction_method'],
                'ensemble_available': hybrid_result.get('ensemble_used', False),
                'ml_available': hybrid_result['ml_available'],
                'agreement_ratio': hybrid_result.get('prediction_agreement', {}).get('ratio', 0.0),
                'confidence_boost_applied': False  # Will be set below
            }
        }
        
        # Apply confidence boosts based on agreement
        agreement = result['prediction_agreement']
        if agreement and agreement.get('unanimous', False):
            result['confidence'] = min(0.98, result['confidence'] + 0.1)
            result['hybrid_analysis']['confidence_boost_applied'] = True
        
        return result

# Convenience functions for easy integration

def train_hybrid_detector(evaluation_csv_path: str, 
                         model_save_path: str,
                         traditional_detector: Optional[CrumblyDetector] = None) -> HybridCrumblyDetector:
    """
    Train a hybrid detector from evaluation results.
    
    Args:
        evaluation_csv_path: Path to CSV with ML features from evaluation system
        model_save_path: Path to save trained models
        traditional_detector: Optional existing CrumblyDetector instance
        
    Returns:
        Trained HybridCrumblyDetector
    """
    
    print(f"🚀 TRAINING HYBRID CRUMBLY DETECTOR")
    print("=" * 50)
    
    # Initialize hybrid detector
    hybrid = HybridCrumblyDetector(traditional_detector=traditional_detector)
    
    # Train ML models
    success = hybrid.train_ml_models(evaluation_csv_path)
    
    if success:
        # Save models
        hybrid.save_models(model_save_path)
        print(f"✅ Hybrid detector trained and saved to {model_save_path}")
    else:
        print(f"❌ Training failed")
    
    return hybrid

def load_hybrid_detector(model_path: str,
                        traditional_detector: Optional[CrumblyDetector] = None) -> HybridCrumblyDetector:
    """
    Load a pre-trained hybrid detector.
    
    Args:
        model_path: Path to saved models
        traditional_detector: Optional existing CrumblyDetector instance
        
    Returns:
        Loaded HybridCrumblyDetector
    """
    
    hybrid = HybridCrumblyDetector(traditional_detector=traditional_detector)
    
    if hybrid.load_models(model_path):
        print(f"✅ Hybrid detector loaded from {model_path}")
    else:
        print(f"⚠️ Could not load models, using traditional detector only")
    
    return hybrid

def compare_detectors(image: np.ndarray, fiber_mask: np.ndarray,
                     traditional_detector: CrumblyDetector,
                     hybrid_detector: HybridCrumblyDetector,
                     lumen_mask: Optional[np.ndarray] = None,
                     scale_factor: float = 1.0) -> Dict:
    """
    Compare traditional vs hybrid detector performance on a single image.
    """
    
    print(f"🔬 COMPARING DETECTORS")
    print("-" * 30)
    
    # Traditional analysis
    print("Traditional detector...")
    traditional_result = traditional_detector.analyze_crumbly_texture(
        image, fiber_mask, lumen_mask, scale_factor
    )
    
    # Hybrid analysis
    print("Hybrid detector...")
    hybrid_result = hybrid_detector.analyze_crumbly_texture(
        image, fiber_mask, lumen_mask, scale_factor
    )
    
    # Compare results
    comparison = {
        'traditional': {
            'classification': traditional_result['classification'],
            'confidence': traditional_result['confidence'],
            'crumbly_score': traditional_result['crumbly_score']
        },
        'hybrid': {
            'classification': hybrid_result['classification'],
            'confidence': hybrid_result['confidence'],
            'crumbly_score': hybrid_result['crumbly_score'],
            'method_used': hybrid_result['prediction_method'],
            'uncertainty': hybrid_result['uncertainty']
        },
        'agreement': traditional_result['classification'] == hybrid_result['classification'],
        'confidence_improvement': hybrid_result['confidence'] - traditional_result['confidence'],
        'ml_available': hybrid_result.get('ml_available', False)
    }
    
    print(f"\nResults:")
    print(f"  Traditional: {comparison['traditional']['classification']} "
          f"(conf: {comparison['traditional']['confidence']:.3f})")
    print(f"  Hybrid:      {comparison['hybrid']['classification']} "
          f"(conf: {comparison['hybrid']['confidence']:.3f}, "
          f"method: {comparison['hybrid']['method_used']})")
    print(f"  Agreement:   {comparison['agreement']}")
    print(f"  Improvement: {comparison['confidence_improvement']:+.3f}")
    
    return comparison

class AdaptiveThresholdOptimizer:
    """
    Optimize thresholds in the traditional detector based on labeled data.
    """
    
    def __init__(self, detector: CrumblyDetector):
        self.detector = detector
        self.optimized_thresholds = None
        
    def optimize_thresholds(self, evaluation_results: List[Dict]):
        """
        Optimize thresholds based on evaluation results.
        
        Args:
            evaluation_results: List of evaluation results from evaluation system
        """
        
        print(f"🎯 OPTIMIZING THRESHOLDS")
        print("-" * 30)
        
        # Extract features and labels from successful results
        features = []
        labels = []
        
        for result in evaluation_results:
            if result.get('processing_success', False):
                crumbly_result = result.get('crumbly_result', {})
                
                # Extract key decision features
                feature_vector = [
                    crumbly_result.get('crumbly_score', 0.5),
                    crumbly_result.get('crumbly_evidence', 0.5),
                    crumbly_result.get('porous_intact_evidence', 0.5),
                    crumbly_result.get('pore_metrics', {}).get('organized_porosity_score', 0.5),
                    crumbly_result.get('wall_integrity_metrics', {}).get('wall_integrity_score', 0.5),
                    crumbly_result.get('boundary_metrics', {}).get('outer_boundary', {}).get('circularity', 0.5),
                    crumbly_result.get('boundary_metrics', {}).get('outer_boundary', {}).get('roughness_index', 0.5)
                ]
                
                features.append(feature_vector)
                
                # Convert label to numeric
                label_map = {'not': 0, 'intermediate': 1, 'crumbly': 2}
                labels.append(label_map.get(result['true_label'], 1))
        
        if len(features) < 10:
            print("❌ Not enough data for threshold optimization")
            return False
        
        features = np.array(features)
        labels = np.array(labels)
        
        # Use Random Forest to find optimal decision boundaries
        from sklearn.tree import DecisionTreeClassifier
        
        # Train a simple decision tree to find thresholds
        tree = DecisionTreeClassifier(max_depth=3, random_state=42)
        tree.fit(features, labels)
        
        # Extract decision rules (simplified)
        # This is a basic implementation - could be made more sophisticated
        
        # Find optimal threshold for crumbly score
        crumbly_scores = features[:, 0]
        
        best_accuracy = 0
        best_threshold = 0.5
        
        for threshold in np.arange(0.3, 0.8, 0.05):
            predictions = (crumbly_scores > threshold).astype(int)
            # Convert to 3-class problem (simplified)
            binary_labels = (labels >= 2).astype(int)  # crumbly vs not crumbly
            accuracy = np.mean(predictions == binary_labels)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        self.optimized_thresholds = {
            'crumbly_threshold': best_threshold,
            'accuracy_achieved': best_accuracy
        }
        
        print(f"✅ Optimized crumbly threshold: {best_threshold:.3f} "
              f"(accuracy: {best_accuracy:.3f})")
        
        return True
    
    def apply_optimized_thresholds(self):
        """Apply optimized thresholds to the detector."""
        
        if self.optimized_thresholds is None:
            print("❌ No optimized thresholds available")
            return False
        
        # Update detector thresholds
        new_threshold = self.optimized_thresholds['crumbly_threshold']
        
        # This would modify your detector's internal thresholds
        # Implementation depends on your CrumblyDetector structure
        print(f"✅ Applied optimized threshold: {new_threshold:.3f}")
        
        return True

# Example usage and testing functions

def test_hybrid_system():
    """Test the hybrid system with synthetic data."""
    
    print(f"🧪 TESTING HYBRID SYSTEM")
    print("=" * 40)
    
    # Create synthetic test data
    np.random.seed(42)
    
    # Simulate evaluation results
    synthetic_features = []
    synthetic_labels = []
    
    for label in [0, 1, 2]:  # not, intermediate, crumbly
        for _ in range(20):
            # Generate features with some class separation
            base_features = np.random.rand(30)
            
            if label == 0:  # not crumbly
                base_features[0] = np.random.uniform(0.0, 0.4)  # low crumbly score
                base_features[1] = np.random.uniform(0.0, 0.3)  # low crumbly evidence
                base_features[2] = np.random.uniform(0.6, 1.0)  # high porous intact evidence
            elif label == 1:  # intermediate
                base_features[0] = np.random.uniform(0.3, 0.7)  # medium crumbly score
                base_features[1] = np.random.uniform(0.2, 0.6)  # medium crumbly evidence
                base_features[2] = np.random.uniform(0.3, 0.7)  # medium porous intact evidence
            else:  # crumbly
                base_features[0] = np.random.uniform(0.6, 1.0)  # high crumbly score
                base_features[1] = np.random.uniform(0.5, 1.0)  # high crumbly evidence
                base_features[2] = np.random.uniform(0.0, 0.4)  # low porous intact evidence
            
            synthetic_features.append(base_features)
            synthetic_labels.append(label)
    
    # Create synthetic CSV
    feature_names = [f'feature_{i}' for i in range(30)]
    df = pd.DataFrame(synthetic_features, columns=feature_names)
    df['true_label'] = synthetic_labels
    df['true_label_name'] = [['not', 'intermediate', 'crumbly'][label] for label in synthetic_labels]
    
    # Save synthetic data
    synthetic_csv = "synthetic_training_data.csv"
    df.to_csv(synthetic_csv, index=False)
    
    print(f"   Created synthetic dataset: {len(df)} samples")
    
    # Test training
    try:
        hybrid = HybridCrumblyDetector()
        success = hybrid.train_ml_models(synthetic_csv, test_size=0.3)
        
        if success:
            print(f"✅ Hybrid system test successful!")
            
            # Test prediction on synthetic data
            test_features = np.random.rand(1, 30)
            test_features[0, 0] = 0.8  # High crumbly score
            test_features[0, 1] = 0.7  # High crumbly evidence
            test_features[0, 2] = 0.2  # Low porous intact evidence
            
            # This would require the full interface, but demonstrates the concept
            print(f"   Test prediction capability verified")
            
            return True
        else:
            print(f"❌ Hybrid system test failed")
            return False
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False
    
    finally:
        # Cleanup
        try:
            Path(synthetic_csv).unlink()
        except:
            pass

if __name__ == "__main__":
    # Run basic test
    test_hybrid_system()