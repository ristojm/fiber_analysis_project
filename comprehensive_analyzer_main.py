#!/usr/bin/env python3
"""
Comprehensive SEM Fiber Analyzer - Main Application
Production application for complete fiber characterization with fast refined porosity analysis

This is the main user-facing application that orchestrates all analysis modules
to provide complete SEM fiber analysis with reporting and visualization.

Usage:
    python comprehensive_analyzer_main.py --image path/to/image.jpg
    python comprehensive_analyzer_main.py --batch sample_images/
    python comprehensive_analyzer_main.py --help
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import argparse
import time
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Setup paths for module imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "modules"))

# Import analysis modules
print("🔧 Loading SEM Fiber Analysis modules...")

try:
    # Core analysis modules
    from modules.scale_detection import ScaleBarDetector, detect_scale_bar
    from modules.fiber_type_detection import FiberTypeDetector, detect_fiber_type
    from modules.image_preprocessing import load_image, preprocess_pipeline
    print("✅ Core modules loaded successfully")
except ImportError as e:
    print(f"❌ Could not import core modules: {e}")
    sys.exit(1)

try:
    # Updated porosity analysis module (fast refined method)
    from modules.porosity_analysis import PorosityAnalyzer, analyze_fiber_porosity, quick_porosity_check
    print("✅ Fast refined porosity module loaded")
    POROSITY_AVAILABLE = True
    POROSITY_TYPE = "fast_refined"
except ImportError:
    print("⚠️ Fast refined porosity module not found, trying legacy versions...")
    try:
        # Fallback to enhanced porosity
        from modules.porosity_analysis import EnhancedPorosityAnalyzer, analyze_fiber_porosity_enhanced
        print("✅ Enhanced porosity module loaded")
        POROSITY_AVAILABLE = True
        POROSITY_TYPE = "enhanced"
    except ImportError:
        try:
            # Fallback to basic porosity
            from modules.porosity_analysis import PorosityAnalyzer
            print("✅ Basic porosity module loaded")
            POROSITY_AVAILABLE = True
            POROSITY_TYPE = "basic"
        except ImportError:
            print("❌ No porosity analysis available")
            POROSITY_AVAILABLE = False
            POROSITY_TYPE = None


class ComprehensiveFiberAnalyzer:
    """
    Main application class that orchestrates all analysis modules
    for comprehensive SEM fiber characterization with fast refined porosity analysis.
    """
    
    def __init__(self, config: Optional[Dict] = None, debug: bool = True):
        """
        Initialize the comprehensive analyzer.
        
        Args:
            config: Configuration dictionary for all modules
            debug: Enable debug output
        """
        self.debug = debug
        self.config = self._get_default_config()
        
        if config:
            self._update_config(config)
        
        # Initialize modules
        self.scale_detector = ScaleBarDetector(
            ocr_backend=self.config.get('scale_detection', {}).get('ocr_backend'),
            use_enhanced_detection=True
        )
        
        self.fiber_detector = FiberTypeDetector(
            **self.config.get('fiber_detection', {})
        )
        
        # Initialize porosity analyzer based on available type
        if POROSITY_AVAILABLE:
            if POROSITY_TYPE == "fast_refined":
                # New fast refined porosity analyzer
                self.porosity_analyzer = PorosityAnalyzer(
                    config=self.config.get('porosity_analysis', {})
                )
            elif POROSITY_TYPE == "enhanced":
                # Enhanced porosity analyzer (legacy)
                self.porosity_analyzer = EnhancedPorosityAnalyzer(
                    config=self.config.get('porosity_analysis', {})
                )
            else:
                # Basic porosity analyzer (legacy)
                self.porosity_analyzer = PorosityAnalyzer()
        else:
            self.porosity_analyzer = None
        
        if self.debug:
            print(f"🔬 Comprehensive Fiber Analyzer initialized")
            print(f"   Scale detection: {self.scale_detector.ocr_backend or 'legacy'}")
            print(f"   Fiber detection: Adaptive algorithms")
            print(f"   Porosity analysis: {POROSITY_TYPE or 'Not available'}")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for all modules."""
        return {
            'scale_detection': {
                'ocr_backend': None,  # Auto-select best available
                'use_enhanced_detection': True,
            },
            'fiber_detection': {
                'min_fiber_ratio': 0.001,
                'max_fiber_ratio': 0.8,
                'lumen_area_threshold': 0.02,
                'circularity_threshold': 0.2,
                'confidence_threshold': 0.6,
            },
            'porosity_analysis': {
                # Fast refined porosity configuration
                'pore_detection': {
                    'intensity_percentile': 28,
                    'min_pore_area_pixels': 3,
                    'max_pore_area_ratio': 0.1,
                    'fast_filtering': True,
                    'early_size_filter': True,
                    'vectorized_operations': True,
                },
                'performance': {
                    'max_candidates_per_stage': 5000,
                    'use_simplified_morphology': True,
                    'batch_processing': True,
                    'enable_timing': False,  # Disable timing in production
                },
                'quality_control': {
                    'circularity_threshold': 0.05,
                    'aspect_ratio_threshold': 8,
                    'solidity_threshold': 0.25,
                    'intensity_validation': True,
                    'size_dependent_validation': True,
                },
                'fiber_integration': {
                    'use_individual_fibers': True,
                    'exclude_lumen': True,
                    'lumen_buffer_pixels': 3,
                    'min_fiber_area_analysis': 1000,
                },
                'analysis': {
                    'calculate_size_distribution': True,
                    'calculate_spatial_metrics': True,
                    'detailed_reporting': True,
                    'save_individual_pore_data': True,
                }
            },
            'output': {
                'save_visualizations': False,  # CHANGED: Disable by default
                'save_data': True,  # Keep data export (JSON, Excel)
                'create_report': False,  # CHANGED: Disable by default
                'dpi': 300,
            }
        }

    def analyze_single_image(self, image_path: str, 
                           output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a single SEM image.
        
        Args:
            image_path: Path to SEM image
            output_dir: Directory for saving results
            
        Returns:
            Dictionary containing complete analysis results
        """
        start_time = time.time()
        
        if self.debug:
            print(f"\n🔍 COMPREHENSIVE FIBER ANALYSIS")
            print(f"Image: {Path(image_path).name}")
            print("=" * 70)
        
        # Initialize result structure
        result = {
            'image_path': str(image_path),
            'image_name': Path(image_path).name,
            'analysis_timestamp': datetime.now().isoformat(),
            'success': False,
            'total_processing_time': 0.0,
            'modules_used': [],
            'porosity_method': POROSITY_TYPE
        }
        
        try:
            # Setup output directory
            if output_dir is None:
                output_dir = Path(image_path).parent / 'analysis_results'
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            
            # Step 1: Image loading and preprocessing
            if self.debug:
                print("📸 Step 1: Loading and preprocessing image...")
            
            step_start = time.time()
            
            # Load image
            image = load_image(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Basic preprocessing for analysis
            preprocessed = self._preprocess_for_analysis(image)
            
            result.update({
                'image_shape': image.shape,
                'image_size_mb': os.path.getsize(image_path) / (1024 * 1024),
                'preprocessing_time': time.time() - step_start
            })
            
            if self.debug:
                print(f"   ✅ Image processed: {image.shape}")
                print(f"   Processing time: {result['preprocessing_time']:.3f}s")
            
            # Step 2: Scale detection (NO DEBUG IMAGES)
            if self.debug:
                print("📏 Step 2: Scale bar detection...")
            
            step_start = time.time()
            scale_result = self.scale_detector.detect_scale_bar(
                image,
                debug=False,  # CHANGED: Always disable debug output
                save_debug_image=False,  # CHANGED: Never save debug images
                output_dir=None  # CHANGED: No debug output directory
            )
            scale_time = time.time() - step_start
            
            result['scale_detection'] = scale_result
            result['scale_processing_time'] = scale_time
            result['modules_used'].append('scale_detection')
            
            # Extract scale factor
            if scale_result['scale_detected']:
                scale_factor = scale_result['micrometers_per_pixel']
                if self.debug:
                    print(f"   ✅ Scale detected: {scale_factor:.4f} μm/pixel")
                    scale_info = scale_result.get('scale_info', {})
                    print(f"   Scale text: '{scale_info.get('text', 'N/A')}'")
                    print(f"   Confidence: {scale_result.get('confidence', 0):.2%}")
            else:
                scale_factor = 1.0  # Fallback
                if self.debug:
                    print(f"   ⚠️ Scale detection failed: {scale_result.get('error', 'Unknown')}")
                    print(f"   Using fallback scale: 1.0 μm/pixel")
            
            # Step 3: Fiber type detection
            if self.debug:
                print("🧬 Step 3: Fiber type detection...")
            
            step_start = time.time()
            fiber_type, fiber_confidence, fiber_analysis_data = self.fiber_detector.classify_fiber_type(preprocessed)
            fiber_time = time.time() - step_start
            
            result['fiber_detection'] = {
                'fiber_type': fiber_type,
                'confidence': fiber_confidence,
                'total_fibers': fiber_analysis_data.get('total_fibers', 0),
                'hollow_fibers': fiber_analysis_data.get('hollow_fibers', 0),
                'filaments': fiber_analysis_data.get('filaments', 0),
                'thresholds_used': fiber_analysis_data.get('thresholds', {}),
                'classification_method': fiber_analysis_data.get('classification_method', 'unknown'),
                'processing_time': fiber_time,
                'analysis_data': fiber_analysis_data
            }
            result['fiber_processing_time'] = fiber_time
            result['modules_used'].append('fiber_detection')
            
            if self.debug:
                print(f"   ✅ Fiber type: {fiber_type} (confidence: {fiber_confidence:.3f})")
                print(f"   Total fibers: {fiber_analysis_data.get('total_fibers', 0)}")
                print(f"   Hollow: {fiber_analysis_data.get('hollow_fibers', 0)}")
                print(f"   Filaments: {fiber_analysis_data.get('filaments', 0)}")
                print(f"   Processing time: {fiber_time:.3f}s")
            
            # Step 4: Fast refined porosity analysis
            porosity_result = None
            if self.porosity_analyzer and POROSITY_AVAILABLE:
                if self.debug:
                    print(f"🕳️  Step 4: Porosity analysis ({POROSITY_TYPE})...")
                
                step_start = time.time()
                
                # Get fiber mask from detection results
                fiber_mask = self.fix_fiber_mask_extraction(image, fiber_analysis_data, self.debug)
                
                if np.sum(fiber_mask) > 1000:  # Minimum area threshold
                    try:
                        if POROSITY_TYPE == "fast_refined":
                            # New fast refined porosity analyzer
                            porosity_result = self.porosity_analyzer.analyze_fiber_porosity(
                                preprocessed, 
                                fiber_mask.astype(np.uint8), 
                                scale_factor, 
                                fiber_type,
                                fiber_analysis_data
                            )
                        elif POROSITY_TYPE == "enhanced":
                            # Enhanced porosity analyzer (legacy)
                            porosity_result = self.porosity_analyzer.analyze_fiber_porosity(
                                preprocessed, 
                                fiber_mask.astype(np.uint8), 
                                scale_factor, 
                                fiber_type,
                                fiber_analysis_data
                            )
                        else:
                            # Basic porosity analyzer (fallback)
                            porosity_result = {
                                'porosity_metrics': {
                                    'total_porosity_percent': 0.0,
                                    'pore_count': 0,
                                    'average_pore_size_um2': 0.0,
                                    'pore_density_per_mm2': 0.0,
                                    'method': 'basic_fallback'
                                },
                                'note': 'Basic porosity analysis - limited functionality'
                            }
                        
                        porosity_time = time.time() - step_start
                        result['porosity_analysis'] = porosity_result
                        result['porosity_processing_time'] = porosity_time
                        result['modules_used'].append('porosity_analysis')
                        
                        if self.debug and 'porosity_metrics' in porosity_result:
                            pm = porosity_result['porosity_metrics']
                            print(f"   ✅ Porosity analysis completed:")
                            print(f"   Total porosity: {pm.get('total_porosity_percent', 0):.2f}%")
                            print(f"   Pore count: {pm.get('pore_count', 0)}")
                            print(f"   Average pore size: {pm.get('average_pore_size_um2', 0):.2f} μm²")
                            print(f"   Method: {pm.get('method', 'unknown')}")
                            print(f"   Processing time: {porosity_time:.3f}s")
                            
                            # Show performance info for fast refined method
                            if POROSITY_TYPE == "fast_refined" and 'performance_stats' in porosity_result:
                                perf = porosity_result['performance_stats']
                                candidates = perf.get('candidates_processed', 0)
                                if candidates > 0:
                                    print(f"   Performance: {candidates:,} candidates processed, {candidates/porosity_time:.0f} candidates/sec")
                    
                    except Exception as e:
                        if self.debug:
                            print(f"   ❌ Porosity analysis failed: {e}")
                        porosity_result = {'error': str(e)}
                
                else:
                    if self.debug:
                        print(f"   ⚠️ Insufficient fiber area for porosity analysis")
                        print(f"   Fiber area: {np.sum(fiber_mask)} pixels (min: 1000)")
                    porosity_result = {'error': 'Insufficient fiber area'}
            
            else:
                if self.debug:
                    print("⚠️ Step 4: Porosity analysis not available")
                porosity_result = {'error': 'Porosity analysis module not available'}
            
            # Step 5: Generate comprehensive metrics
            if self.debug:
                print("📊 Step 5: Generating comprehensive metrics...")
            
            comprehensive_metrics = self._generate_comprehensive_metrics(
                result, scale_factor, fiber_type, fiber_confidence, 
                fiber_analysis_data, porosity_result
            )
            result['comprehensive_metrics'] = comprehensive_metrics
            
            # Step 6: Export results (ONLY DATA, NO IMAGES)
            if self.config['output']['save_data']:
                if self.debug:
                    print("💾 Step 6: Exporting data results...")
                
                export_paths = self._export_results(
                    result, output_dir, image, preprocessed
                )
                result['export_paths'] = export_paths
            
            # Mark as successful
            result['success'] = True
            result['total_processing_time'] = time.time() - start_time
            
            if self.debug:
                print(f"✅ Analysis completed successfully!")
                print(f"⏱️ Total time: {result['total_processing_time']:.2f}s")
                self._print_summary(result)
        
        except Exception as e:
            result['error'] = str(e)
            result['total_processing_time'] = time.time() - start_time
            
            if self.debug:
                print(f"❌ Analysis failed: {e}")
                import traceback
                traceback.print_exc()
        
        return result

    def _update_config(self, new_config: Dict):
        """Update configuration recursively."""
        for key, value in new_config.items():
            if key in self.config and isinstance(self.config[key], dict):
                self.config[key].update(value)
            else:
                self.config[key] = value
    
    def analyze_single_image(self, image_path: str, 
                           output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a single SEM image.
        
        Args:
            image_path: Path to SEM image
            output_dir: Directory for saving results
            
        Returns:
            Dictionary containing complete analysis results
        """
        start_time = time.time()
        
        if self.debug:
            print(f"\n🔍 COMPREHENSIVE FIBER ANALYSIS")
            print(f"Image: {Path(image_path).name}")
            print("=" * 70)
        
        # Initialize result structure
        result = {
            'image_path': str(image_path),
            'image_name': Path(image_path).name,
            'analysis_timestamp': datetime.now().isoformat(),
            'success': False,
            'total_processing_time': 0.0,
            'modules_used': [],
            'porosity_method': POROSITY_TYPE
        }
        
        try:
            # Setup output directory
            if output_dir is None:
                output_dir = Path(image_path).parent / 'analysis_results'
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            
            # Step 1: Image loading and preprocessing
            if self.debug:
                print("📸 Step 1: Loading and preprocessing image...")
            
            step_start = time.time()
            
            # Load image
            image = load_image(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Basic preprocessing for analysis
            preprocessed = self._preprocess_for_analysis(image)
            
            result.update({
                'image_shape': image.shape,
                'image_size_mb': os.path.getsize(image_path) / (1024 * 1024),
                'preprocessing_time': time.time() - step_start
            })
            
            if self.debug:
                print(f"   ✅ Image processed: {image.shape}")
                print(f"   Processing time: {result['preprocessing_time']:.3f}s")
            
            # Step 2: Scale detection
            if self.debug:
                print("📏 Step 2: Scale bar detection...")
            
            step_start = time.time()
            scale_result = self.scale_detector.detect_scale_bar(
                image,
                debug=self.debug,
                save_debug_image=self.config['output']['save_visualizations'],
                output_dir=output_dir
            )
            scale_time = time.time() - step_start
            
            result['scale_detection'] = scale_result
            result['scale_processing_time'] = scale_time
            result['modules_used'].append('scale_detection')
            
            # Extract scale factor
            if scale_result['scale_detected']:
                scale_factor = scale_result['micrometers_per_pixel']
                if self.debug:
                    print(f"   ✅ Scale detected: {scale_factor:.4f} μm/pixel")
                    scale_info = scale_result.get('scale_info', {})
                    print(f"   Scale text: '{scale_info.get('text', 'N/A')}'")
                    print(f"   Confidence: {scale_result.get('confidence', 0):.2%}")
            else:
                scale_factor = 1.0  # Fallback
                if self.debug:
                    print(f"   ⚠️ Scale detection failed: {scale_result.get('error', 'Unknown')}")
                    print(f"   Using fallback scale: 1.0 μm/pixel")
            
            # Step 3: Fiber type detection
            if self.debug:
                print("🧬 Step 3: Fiber type detection...")
            
            step_start = time.time()
            fiber_type, fiber_confidence, fiber_analysis_data = self.fiber_detector.classify_fiber_type(preprocessed)
            fiber_time = time.time() - step_start
            
            result['fiber_detection'] = {
                'fiber_type': fiber_type,
                'confidence': fiber_confidence,
                'total_fibers': fiber_analysis_data.get('total_fibers', 0),
                'hollow_fibers': fiber_analysis_data.get('hollow_fibers', 0),
                'filaments': fiber_analysis_data.get('filaments', 0),
                'thresholds_used': fiber_analysis_data.get('thresholds', {}),
                'classification_method': fiber_analysis_data.get('classification_method', 'unknown'),
                'processing_time': fiber_time,
                'analysis_data': fiber_analysis_data
            }
            result['fiber_processing_time'] = fiber_time
            result['modules_used'].append('fiber_detection')
            
            if self.debug:
                print(f"   ✅ Fiber type: {fiber_type} (confidence: {fiber_confidence:.3f})")
                print(f"   Total fibers: {fiber_analysis_data.get('total_fibers', 0)}")
                print(f"   Hollow: {fiber_analysis_data.get('hollow_fibers', 0)}")
                print(f"   Filaments: {fiber_analysis_data.get('filaments', 0)}")
                print(f"   Processing time: {fiber_time:.3f}s")
            
            # Step 4: Fast refined porosity analysis
            porosity_result = None
            if self.porosity_analyzer and POROSITY_AVAILABLE:
                if self.debug:
                    print(f"🕳️  Step 4: Porosity analysis ({POROSITY_TYPE})...")
                
                step_start = time.time()
                
                # Get fiber mask from detection results
                fiber_mask = fiber_analysis_data.get('fiber_mask', np.zeros_like(image, dtype=bool))
                
                if np.sum(fiber_mask) > 1000:  # Minimum area threshold
                    try:
                        if POROSITY_TYPE == "fast_refined":
                            # New fast refined porosity analyzer
                            porosity_result = self.porosity_analyzer.analyze_fiber_porosity(
                                preprocessed, 
                                fiber_mask.astype(np.uint8), 
                                scale_factor, 
                                fiber_type,
                                fiber_analysis_data
                            )
                        elif POROSITY_TYPE == "enhanced":
                            # Enhanced porosity analyzer (legacy)
                            porosity_result = self.porosity_analyzer.analyze_fiber_porosity(
                                preprocessed, 
                                fiber_mask.astype(np.uint8), 
                                scale_factor, 
                                fiber_type,
                                fiber_analysis_data
                            )
                        else:
                            # Basic porosity analyzer (fallback)
                            porosity_result = {
                                'porosity_metrics': {
                                    'total_porosity_percent': 0.0,
                                    'pore_count': 0,
                                    'average_pore_size_um2': 0.0,
                                    'pore_density_per_mm2': 0.0,
                                    'method': 'basic_fallback'
                                },
                                'note': 'Basic porosity analysis - limited functionality'
                            }
                        
                        porosity_time = time.time() - step_start
                        result['porosity_analysis'] = porosity_result
                        result['porosity_processing_time'] = porosity_time
                        result['modules_used'].append('porosity_analysis')
                        
                        if self.debug and 'porosity_metrics' in porosity_result:
                            pm = porosity_result['porosity_metrics']
                            print(f"   ✅ Porosity analysis completed:")
                            print(f"   Total porosity: {pm.get('total_porosity_percent', 0):.2f}%")
                            print(f"   Pore count: {pm.get('pore_count', 0)}")
                            print(f"   Average pore size: {pm.get('average_pore_size_um2', 0):.2f} μm²")
                            print(f"   Method: {pm.get('method', 'unknown')}")
                            print(f"   Processing time: {porosity_time:.3f}s")
                            
                            # Show performance info for fast refined method
                            if POROSITY_TYPE == "fast_refined" and 'performance_stats' in porosity_result:
                                perf = porosity_result['performance_stats']
                                candidates = perf.get('candidates_processed', 0)
                                if candidates > 0:
                                    print(f"   Performance: {candidates:,} candidates processed, {candidates/porosity_time:.0f} candidates/sec")
                    
                    except Exception as e:
                        if self.debug:
                            print(f"   ❌ Porosity analysis failed: {e}")
                        porosity_result = {'error': str(e)}
                
                else:
                    if self.debug:
                        print(f"   ⚠️ Insufficient fiber area for porosity analysis")
                        print(f"   Fiber area: {np.sum(fiber_mask)} pixels (min: 1000)")
                    porosity_result = {'error': 'Insufficient fiber area'}
            
            else:
                if self.debug:
                    print("⚠️ Step 4: Porosity analysis not available")
                porosity_result = {'error': 'Porosity analysis module not available'}
            
            # Step 5: Generate comprehensive metrics
            if self.debug:
                print("📊 Step 5: Generating comprehensive metrics...")
            
            comprehensive_metrics = self._generate_comprehensive_metrics(
                result, scale_factor, fiber_type, fiber_confidence, 
                fiber_analysis_data, porosity_result
            )
            result['comprehensive_metrics'] = comprehensive_metrics
            
            # Step 6: Export results
            if self.config['output']['save_data'] or self.config['output']['save_visualizations']:
                if self.debug:
                    print("💾 Step 6: Exporting results...")
                
                export_paths = self._export_results(
                    result, output_dir, image, preprocessed
                )
                result['export_paths'] = export_paths
            
            # Mark as successful
            result['success'] = True
            result['total_processing_time'] = time.time() - start_time
            
            if self.debug:
                print(f"✅ Analysis completed successfully!")
                print(f"⏱️ Total time: {result['total_processing_time']:.2f}s")
                self._print_summary(result)
        
        except Exception as e:
            result['error'] = str(e)
            result['total_processing_time'] = time.time() - start_time
            
            if self.debug:
                print(f"❌ Analysis failed: {e}")
                import traceback
                traceback.print_exc()
        
        return result
    
    def _preprocess_for_analysis(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image optimally for all analysis modules."""
        
        # Use bilateral filter for edge preservation while reducing noise
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        # CLAHE for contrast enhancement (balanced for all modules)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        return enhanced
    
    def _generate_comprehensive_metrics(self, result: Dict, scale_factor: float,
                                      fiber_type: str, fiber_confidence: float,
                                      fiber_analysis_data: Dict, 
                                      porosity_result: Optional[Dict]) -> Dict:
        """Generate comprehensive analysis metrics."""
        
        metrics = {
            'analysis_quality': 'unknown',
            'quality_score': 0.0,
            'scale_factor_um_per_pixel': scale_factor,
            'fiber_type': fiber_type,
            'fiber_confidence': fiber_confidence,
            'porosity_method': POROSITY_TYPE,
        }
        
        # Calculate overall quality score
        quality_score = 0.0
        quality_factors = []
        
        # Scale detection quality (25%)
        scale_result = result.get('scale_detection', {})
        if scale_result.get('scale_detected', False):
            scale_conf = scale_result.get('confidence', 0.0)
            quality_score += scale_conf * 0.25
            quality_factors.append(f"Scale: {scale_conf:.2f}")
        else:
            quality_factors.append("Scale: failed")
        
        # Fiber detection quality (35%)
        quality_score += fiber_confidence * 0.35
        quality_factors.append(f"Fiber: {fiber_confidence:.2f}")
        
        # Porosity analysis quality (40%)
        if porosity_result and 'porosity_metrics' in porosity_result:
            pm = porosity_result['porosity_metrics']
            pore_count = pm.get('pore_count', 0)
            
            # Quality assessment based on pore count and method
            if POROSITY_TYPE == "fast_refined":
                # More lenient for fast refined (expects more pores)
                if pore_count >= 100:
                    porosity_quality = 1.0
                elif pore_count >= 50:
                    porosity_quality = 0.9
                elif pore_count >= 20:
                    porosity_quality = 0.8
                elif pore_count >= 10:
                    porosity_quality = 0.6
                elif pore_count > 0:
                    porosity_quality = 0.4
                else:
                    porosity_quality = 0.0
            else:
                # Standard quality assessment
                if pore_count >= 20:
                    porosity_quality = 1.0
                elif pore_count >= 10:
                    porosity_quality = 0.8
                elif pore_count >= 5:
                    porosity_quality = 0.6
                elif pore_count > 0:
                    porosity_quality = 0.4
                else:
                    porosity_quality = 0.0
            
            quality_score += porosity_quality * 0.4
            quality_factors.append(f"Porosity: {porosity_quality:.2f} ({POROSITY_TYPE})")
        else:
            quality_factors.append(f"Porosity: unavailable ({POROSITY_TYPE or 'none'})")
        
        # Determine overall quality level
        if quality_score >= 0.85:
            quality_level = "excellent"
        elif quality_score >= 0.70:
            quality_level = "good"
        elif quality_score >= 0.50:
            quality_level = "moderate"
        elif quality_score >= 0.30:
            quality_level = "poor"
        else:
            quality_level = "very_poor"
        
        metrics.update({
            'analysis_quality': quality_level,
            'quality_score': quality_score,
            'quality_factors': quality_factors
        })
        
        # Add physical measurements if scale is available
        if scale_result.get('scale_detected', False):
            physical_measurements = self._calculate_physical_measurements(
                fiber_analysis_data, porosity_result, scale_factor
            )
            metrics['physical_measurements'] = physical_measurements
        
        return metrics
    
    def _calculate_physical_measurements(self, fiber_data: Dict, 
                                       porosity_data: Optional[Dict],
                                       scale_factor: float) -> Dict:
        """Calculate physical measurements in real units."""
        
        measurements = {}
        
        # Fiber measurements
        individual_results = fiber_data.get('individual_results', [])
        if individual_results:
            fiber_areas_um2 = []
            fiber_diameters_um = []
            lumen_areas_um2 = []
            
            for result in individual_results:
                fiber_props = result.get('fiber_properties', {})
                area_pixels = fiber_props.get('area', 0)
                
                if area_pixels > 0:
                    area_um2 = area_pixels * (scale_factor ** 2)
                    diameter_um = 2 * np.sqrt(area_um2 / np.pi)
                    
                    fiber_areas_um2.append(area_um2)
                    fiber_diameters_um.append(diameter_um)
                    
                    # Lumen measurements for hollow fibers
                    if result.get('has_lumen', False):
                        lumen_props = result.get('lumen_properties', {})
                        lumen_area_pixels = lumen_props.get('area', 0)
                        if lumen_area_pixels > 0:
                            lumen_area_um2 = lumen_area_pixels * (scale_factor ** 2)
                            lumen_areas_um2.append(lumen_area_um2)
            
            # Fiber statistics
            if fiber_areas_um2:
                measurements['fiber_statistics'] = {
                    'count': len(fiber_areas_um2),
                    'mean_area_um2': np.mean(fiber_areas_um2),
                    'std_area_um2': np.std(fiber_areas_um2),
                    'mean_diameter_um': np.mean(fiber_diameters_um),
                    'std_diameter_um': np.std(fiber_diameters_um),
                    'min_diameter_um': np.min(fiber_diameters_um),
                    'max_diameter_um': np.max(fiber_diameters_um),
                    'median_diameter_um': np.median(fiber_diameters_um)
                }
            
            # Lumen statistics
            if lumen_areas_um2:
                measurements['lumen_statistics'] = {
                    'count': len(lumen_areas_um2),
                    'mean_area_um2': np.mean(lumen_areas_um2),
                    'std_area_um2': np.std(lumen_areas_um2)
                }
        
        # Porosity measurements
        if porosity_data and 'porosity_metrics' in porosity_data:
            pm = porosity_data['porosity_metrics']
            measurements['porosity_statistics'] = {
                'total_porosity_percent': pm.get('total_porosity_percent', 0),
                'pore_count': pm.get('pore_count', 0),
                'average_pore_size_um2': pm.get('average_pore_size_um2', 0),
                'pore_density_per_mm2': pm.get('pore_density_per_mm2', 0),
                'method_used': pm.get('method', POROSITY_TYPE)
            }
        
        return measurements
    
    def _export_results(self, result: Dict, output_dir: Path,
                       original_image: np.ndarray, 
                       preprocessed_image: np.ndarray) -> Dict:
        """Export analysis results to files."""
        
        export_paths = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(result['image_path']).stem
        
        try:
            # 1. Save JSON results
            if self.config['output']['save_data']:
                json_path = output_dir / f"{base_name}_analysis_{timestamp}.json"
                with open(json_path, 'w') as f:
                    # Convert numpy arrays to lists for JSON serialization
                    json_result = self._prepare_for_json(result)
                    json.dump(json_result, f, indent=2, default=str)
                export_paths['json_results'] = str(json_path)
                
                if self.debug:
                    print(f"   💾 Results saved: {json_path.name}")
            
            # 2. Save Excel summary
            if self.config['output']['save_data']:
                excel_path = output_dir / f"{base_name}_summary_{timestamp}.xlsx"
                self._create_excel_summary(result, excel_path)
                export_paths['excel_summary'] = str(excel_path)
                
                if self.debug:
                    print(f"   📊 Excel summary: {excel_path.name}")
            
            # 3. Create visualization
            if self.config['output']['save_visualizations']:
                viz_path = output_dir / f"{base_name}_visualization_{timestamp}.png"
                self._create_visualization(result, original_image, viz_path)
                export_paths['visualization'] = str(viz_path)
                
                if self.debug:
                    print(f"   🎨 Visualization: {viz_path.name}")
            
            # 4. Generate report
            if self.config['output']['create_report']:
                report_path = output_dir / f"{base_name}_report_{timestamp}.txt"
                self._create_report(result, report_path)
                export_paths['report'] = str(report_path)
                
                if self.debug:
                    print(f"   📄 Report: {report_path.name}")
        
        except Exception as e:
            if self.debug:
                print(f"   ⚠️ Export error: {e}")
        
        return export_paths
    
    def _prepare_for_json(self, obj):
        """Prepare object for JSON serialization by converting numpy arrays."""
        if isinstance(obj, dict):
            return {key: self._prepare_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    def _create_excel_summary(self, result: Dict, excel_path: Path):
        """Create Excel summary of results."""
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Overview sheet
            overview_data = {
                'Metric': ['Image Name', 'Analysis Quality', 'Quality Score', 'Total Processing Time',
                          'Fiber Type', 'Fiber Confidence', 'Scale Detected', 'Scale Factor',
                          'Porosity Method'],
                'Value': [
                    result['image_name'],
                    result.get('comprehensive_metrics', {}).get('analysis_quality', 'unknown'),
                    result.get('comprehensive_metrics', {}).get('quality_score', 0),
                    f"{result.get('total_processing_time', 0):.2f}s",
                    result.get('fiber_detection', {}).get('fiber_type', 'unknown'),
                    result.get('fiber_detection', {}).get('confidence', 0),
                    result.get('scale_detection', {}).get('scale_detected', False),
                    result.get('comprehensive_metrics', {}).get('scale_factor_um_per_pixel', 0),
                    result.get('porosity_method', 'unknown')
                ]
            }
            overview_df = pd.DataFrame(overview_data)
            overview_df.to_excel(writer, sheet_name='Overview', index=False)
            
            # Fiber detection details
            fiber_data = result.get('fiber_detection', {})
            fiber_summary = pd.DataFrame([{
                'Total Fibers': fiber_data.get('total_fibers', 0),
                'Hollow Fibers': fiber_data.get('hollow_fibers', 0),
                'Filaments': fiber_data.get('filaments', 0),
                'Classification Method': fiber_data.get('classification_method', 'unknown'),
                'Processing Time (s)': fiber_data.get('processing_time', 0)
            }])
            fiber_summary.to_excel(writer, sheet_name='Fiber_Detection', index=False)
            
            # Porosity results if available
            porosity_data = result.get('porosity_analysis', {})
            if porosity_data and 'porosity_metrics' in porosity_data:
                pm = porosity_data['porosity_metrics']
                porosity_summary = pd.DataFrame([{
                    'Total Porosity (%)': pm.get('total_porosity_percent', 0),
                    'Pore Count': pm.get('pore_count', 0),
                    'Average Pore Size (μm²)': pm.get('average_pore_size_um2', 0),
                    'Pore Density (/mm²)': pm.get('pore_density_per_mm2', 0),
                    'Method': pm.get('method', POROSITY_TYPE),
                    'Processing Time (s)': result.get('porosity_processing_time', 0)
                }])
                porosity_summary.to_excel(writer, sheet_name='Porosity_Analysis', index=False)
            
            # Physical measurements if available
            measurements = result.get('comprehensive_metrics', {}).get('physical_measurements', {})
            if 'fiber_statistics' in measurements:
                fs = measurements['fiber_statistics']
                fiber_measurements = pd.DataFrame([fs])
                fiber_measurements.to_excel(writer, sheet_name='Fiber_Measurements', index=False)
    
    def _create_visualization(self, result: Dict, original_image: np.ndarray, viz_path: Path):
        """Create comprehensive visualization."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original image
        axes[0, 0].imshow(original_image, cmap='gray')
        axes[0, 0].set_title('Original SEM Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Analysis summary
        scale_result = result.get('scale_detection', {})
        fiber_result = result.get('fiber_detection', {})
        porosity_result = result.get('porosity_analysis', {})
        comprehensive = result.get('comprehensive_metrics', {})
        
        # Create summary text
        summary_text = f"Analysis Summary:\n\n"
        summary_text += f"Quality: {comprehensive.get('analysis_quality', 'unknown').title()}\n"
        summary_text += f"Score: {comprehensive.get('quality_score', 0):.2f}/1.0\n\n"
        
        if scale_result.get('scale_detected', False):
            summary_text += f"Scale: {scale_result['micrometers_per_pixel']:.4f} μm/pixel ✅\n"
        else:
            summary_text += f"Scale: Detection failed ❌\n"
        
        summary_text += f"Fiber Type: {fiber_result.get('fiber_type', 'unknown')}\n"
        summary_text += f"Confidence: {fiber_result.get('confidence', 0):.3f}\n"
        summary_text += f"Total Fibers: {fiber_result.get('total_fibers', 0)}\n\n"
        
        if porosity_result and 'porosity_metrics' in porosity_result:
            pm = porosity_result['porosity_metrics']
            summary_text += f"Porosity: {pm.get('total_porosity_percent', 0):.2f}%\n"
            summary_text += f"Pore Count: {pm.get('pore_count', 0)}\n"
            summary_text += f"Method: {pm.get('method', POROSITY_TYPE)}\n"
        else:
            summary_text += f"Porosity: Analysis failed\n"
        
        summary_text += f"\nTotal Time: {result.get('total_processing_time', 0):.2f}s"
        
        axes[0, 1].text(0.05, 0.95, summary_text, transform=axes[0, 1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue'))
        axes[0, 1].set_title('Analysis Results', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Fiber detection visualization
        fiber_analysis = fiber_result.get('analysis_data', {})
        if 'fiber_mask' in fiber_analysis:
            fiber_mask = fiber_analysis['fiber_mask']
            axes[1, 0].imshow(fiber_mask, cmap='gray')
            axes[1, 0].set_title('Detected Fibers', fontsize=12, fontweight='bold')
        else:
            axes[1, 0].text(0.5, 0.5, 'Fiber mask\nnot available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Fiber Detection', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Quality assessment and performance
        quality_factors = comprehensive.get('quality_factors', [])
        quality_text = "Quality Factors:\n" + "\n".join(quality_factors) + "\n\n"
        
        # Add performance info for fast refined method
        if (POROSITY_TYPE == "fast_refined" and 
            porosity_result and 'performance_stats' in porosity_result):
            perf = porosity_result['performance_stats']
            candidates = perf.get('candidates_processed', 0)
            porosity_time = result.get('porosity_processing_time', 0)
            if candidates > 0 and porosity_time > 0:
                quality_text += f"Performance:\n"
                quality_text += f"  {candidates:,} candidates\n"
                quality_text += f"  {candidates/porosity_time:.0f} candidates/sec"
        
        axes[1, 1].text(0.05, 0.95, quality_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen'))
        axes[1, 1].set_title('Quality & Performance', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.suptitle(f"SEM Fiber Analysis: {result['image_name']}", 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(viz_path, dpi=self.config['output']['dpi'], bbox_inches='tight')
        plt.close()
    
    def _create_report(self, result: Dict, report_path: Path):
        """Create detailed analysis report."""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("SEM FIBER ANALYSIS REPORT\n")
            f.write("=" * 30 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Image: {result['image_name']}\n")
            f.write(f"Analysis Duration: {result.get('total_processing_time', 0):.2f} seconds\n")
            f.write(f"Porosity Method: {POROSITY_TYPE}\n\n")
            
            # Overall assessment
            comprehensive = result.get('comprehensive_metrics', {})
            f.write("OVERALL ASSESSMENT\n")
            f.write("-" * 20 + "\n")
            f.write(f"Analysis Quality: {comprehensive.get('analysis_quality', 'unknown').title()}\n")
            f.write(f"Quality Score: {comprehensive.get('quality_score', 0):.2f}/1.0\n")
            f.write(f"Fiber Type: {comprehensive.get('fiber_type', 'unknown')}\n")
            f.write(f"Scale Factor: {comprehensive.get('scale_factor_um_per_pixel', 0):.4f} μm/pixel\n\n")
            
            # Module results
            self._write_module_results(f, result)
    
    def _write_module_results(self, f, result: Dict):
        """Write detailed module results to report."""
        
        # Scale detection
        f.write("SCALE DETECTION\n")
        f.write("-" * 15 + "\n")
        scale_result = result.get('scale_detection', {})
        if scale_result.get('scale_detected', False):
            f.write(f"Status: Successful\n")
            f.write(f"Scale Factor: {scale_result['micrometers_per_pixel']:.4f} μm/pixel\n")
            info = scale_result.get('scale_info', {})
            f.write(f"Scale Text: '{info.get('text', 'N/A')}'\n")
            f.write(f"Confidence: {scale_result.get('confidence', 0):.1%}\n")
        else:
            f.write(f"Status: Failed\n")
            f.write(f"Error: {scale_result.get('error', 'Unknown')}\n")
        f.write(f"Processing Time: {result.get('scale_processing_time', 0):.3f}s\n\n")
        
        # Fiber detection
        f.write("FIBER TYPE DETECTION\n")
        f.write("-" * 20 + "\n")
        fiber_result = result.get('fiber_detection', {})
        f.write(f"Fiber Type: {fiber_result.get('fiber_type', 'Unknown')}\n")
        f.write(f"Confidence: {fiber_result.get('confidence', 0):.3f}\n")
        f.write(f"Total Fibers: {fiber_result.get('total_fibers', 0)}\n")
        f.write(f"Hollow Fibers: {fiber_result.get('hollow_fibers', 0)}\n")
        f.write(f"Filaments: {fiber_result.get('filaments', 0)}\n")
        f.write(f"Processing Time: {fiber_result.get('processing_time', 0):.3f}s\n\n")
        
        # Porosity analysis
        f.write(f"POROSITY ANALYSIS ({POROSITY_TYPE})\n")
        f.write("-" * (17 + len(POROSITY_TYPE or '')) + "\n")
        porosity_result = result.get('porosity_analysis', {})
        if porosity_result and 'porosity_metrics' in porosity_result:
            pm = porosity_result['porosity_metrics']
            f.write(f"Status: Successful\n")
            f.write(f"Method: {pm.get('method', POROSITY_TYPE)}\n")
            f.write(f"Total Porosity: {pm.get('total_porosity_percent', 0):.2f}%\n")
            f.write(f"Pore Count: {pm.get('pore_count', 0)}\n")
            f.write(f"Average Pore Size: {pm.get('average_pore_size_um2', 0):.2f} μm²\n")
            f.write(f"Pore Density: {pm.get('pore_density_per_mm2', 0):.1f}/mm²\n")
            
            # Performance info for fast refined
            if (POROSITY_TYPE == "fast_refined" and 'performance_stats' in porosity_result):
                perf = porosity_result['performance_stats']
                candidates = perf.get('candidates_processed', 0)
                porosity_time = result.get('porosity_processing_time', 0)
                if candidates > 0:
                    f.write(f"Candidates Processed: {candidates:,}\n")
                    if porosity_time > 0:
                        f.write(f"Processing Speed: {candidates/porosity_time:.0f} candidates/sec\n")
        else:
            f.write(f"Status: Failed or Not Available\n")
            error = porosity_result.get('error', 'Unknown') if porosity_result else 'Not performed'
            f.write(f"Reason: {error}\n")
        f.write(f"Processing Time: {result.get('porosity_processing_time', 0):.3f}s\n\n")
    
    def _print_summary(self, result: Dict):
        """Print analysis summary."""
        
        print(f"\n📋 ANALYSIS SUMMARY")
        print("=" * 30)
        
        comprehensive = result.get('comprehensive_metrics', {})
        print(f"Quality: {comprehensive.get('analysis_quality', 'unknown').title()}")
        print(f"Score: {comprehensive.get('quality_score', 0):.2f}/1.0")
        print(f"Total Time: {result.get('total_processing_time', 0):.2f}s")
        print(f"Porosity Method: {POROSITY_TYPE}")
        
        # Key results
        scale_result = result.get('scale_detection', {})
        if scale_result.get('scale_detected', False):
            print(f"Scale: {scale_result['micrometers_per_pixel']:.4f} μm/pixel ✅")
        else:
            print(f"Scale: Detection failed ❌")
        
        fiber_result = result.get('fiber_detection', {})
        print(f"Fiber Type: {fiber_result.get('fiber_type', 'unknown')}")
        print(f"Confidence: {fiber_result.get('confidence', 0):.3f}")
        
        porosity_result = result.get('porosity_analysis', {})
        if porosity_result and 'porosity_metrics' in porosity_result:
            pm = porosity_result['porosity_metrics']
            print(f"Porosity: {pm.get('total_porosity_percent', 0):.2f}% ({pm.get('pore_count', 0)} pores)")
            
            # Show performance for fast refined
            if POROSITY_TYPE == "fast_refined" and 'performance_stats' in porosity_result:
                perf = porosity_result['performance_stats']
                candidates = perf.get('candidates_processed', 0)
                if candidates > 0:
                    print(f"Performance: {candidates:,} candidates processed")
        else:
            print(f"Porosity: Analysis failed or unavailable")

    def analyze_batch(self, image_directory: str, 
                     output_dir: Optional[str] = None) -> Dict:
        """
        Perform analysis on multiple images in a directory.
        
        Args:
            image_directory: Directory containing SEM images
            output_dir: Output directory for results
            
        Returns:
            Dictionary containing batch analysis results
        """
        print(f"🧪 BATCH ANALYSIS")
        print("=" * 40)
        
        # Setup directories
        image_dir = Path(image_directory)
        if not image_dir.exists():
            print(f"❌ Image directory not found: {image_dir}")
            return {'error': f'Directory not found: {image_dir}'}
        
        if output_dir is None:
            output_dir = image_dir.parent / 'batch_analysis_results'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        # Find image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(image_dir.glob(f'*{ext}'))
            image_files.extend(image_dir.glob(f'*{ext.upper()}'))
        
        image_files = sorted(set(image_files))
        
        if not image_files:
            print(f"❌ No image files found in {image_dir}")
            return {'error': f'No images found in {image_dir}'}
        
        print(f"📁 Analyzing {len(image_files)} images")
        print(f"📊 Results will be saved to: {output_dir}")
        print(f"🚀 Using {POROSITY_TYPE} porosity analysis")
        
        # Process each image
        results = []
        successful = 0
        total_time = 0
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] 📸 {image_path.name}")
            print("-" * 50)
            
            result = self.analyze_single_image(str(image_path), output_dir)
            results.append(result)
            
            total_time += result.get('total_processing_time', 0)
            if result['success']:
                successful += 1
        
        # Generate batch summary
        summary = {
            'batch_info': {
                'timestamp': datetime.now().isoformat(),
                'input_directory': str(image_dir),
                'output_directory': str(output_dir),
                'total_images': len(image_files),
                'successful_analyses': successful,
                'success_rate': successful / len(image_files) * 100 if image_files else 0,
                'total_processing_time': total_time,
                'average_time_per_image': total_time / len(image_files) if image_files else 0,
                'porosity_method': POROSITY_TYPE
            },
            'individual_results': results
        }
        
        # Save batch results JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_json = output_dir / f'batch_analysis_{timestamp}.json'
        
        with open(batch_json, 'w') as f:
            json.dump(self._prepare_for_json(summary), f, indent=2, default=str)
        
        # CREATE CENTRAL EXCEL REPORT (NEW)
        try:
            central_excel_path = output_dir / f'CENTRAL_BATCH_REPORT_{timestamp}.xlsx'
            self._create_central_excel_report(summary, central_excel_path)
            print(f"\n📊 CENTRAL EXCEL REPORT CREATED: {central_excel_path.name}")
        except Exception as e:
            print(f"⚠️ Could not create central Excel report: {e}")
        
        print(f"\n🎯 BATCH ANALYSIS COMPLETE!")
        print(f"📊 Success Rate: {successful}/{len(image_files)} ({successful/len(image_files)*100:.1f}%)")
        print(f"⏱️ Total Time: {total_time:.2f} seconds")
        print(f"🚀 Method Used: {POROSITY_TYPE}")
        print(f"💾 Results saved to: {output_dir}")
        
        return summary

    def _create_central_excel_report(self, batch_summary: Dict, excel_path: Path):
        """
        Create a central Excel file with all sample data in organized sheets.
        This replaces individual reports with a comprehensive comparison view.
        """
        
        results = batch_summary['individual_results']
        batch_info = batch_summary['batch_info']
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            
            # 1. OVERVIEW SHEET - Batch Summary
            overview_data = {
                'Metric': [
                    'Analysis Date', 'Input Directory', 'Total Images', 'Successful Analyses', 
                    'Success Rate (%)', 'Total Processing Time (s)', 'Average Time per Image (s)',
                    'Porosity Method', 'Analysis Quality'
                ],
                'Value': [
                    batch_info['timestamp'][:19].replace('T', ' '),
                    batch_info['input_directory'],
                    batch_info['total_images'],
                    batch_info['successful_analyses'],
                    f"{batch_info['success_rate']:.1f}%",
                    f"{batch_info['total_processing_time']:.2f}",
                    f"{batch_info['average_time_per_image']:.2f}",
                    batch_info['porosity_method'],
                    'See individual results →'
                ]
            }
            overview_df = pd.DataFrame(overview_data)
            overview_df.to_excel(writer, sheet_name='Batch_Overview', index=False)
            
            # 2. MAIN RESULTS SHEET - All Sample Comparisons
            main_results = []
            for result in results:
                if result.get('success', False):
                    # Extract all key data
                    scale_data = result.get('scale_detection', {})
                    fiber_data = result.get('fiber_detection', {})
                    porosity_data = result.get('porosity_analysis', {})
                    comprehensive = result.get('comprehensive_metrics', {})
                    pm = porosity_data.get('porosity_metrics', {}) if porosity_data else {}
                    quality_assessment = porosity_data.get('quality_assessment', {}) if porosity_data else {}
                    
                    row = {
                        # Basic Info
                        'Image_Name': result['image_name'],
                        'Analysis_Quality': comprehensive.get('analysis_quality', 'unknown'),
                        'Quality_Score': comprehensive.get('quality_score', 0),
                        'Total_Processing_Time_s': result.get('total_processing_time', 0),
                        'Analysis_Success': result.get('success', False),
                        
                        # Scale Detection
                        'Scale_Detected': scale_data.get('scale_detected', False),
                        'Scale_Factor_um_per_pixel': scale_data.get('micrometers_per_pixel', 0),
                        'Scale_Confidence': scale_data.get('confidence', 0),
                        'Scale_Text': scale_data.get('scale_info', {}).get('text', ''),
                        'Scale_Method': scale_data.get('method_used', ''),
                        'OCR_Backend': scale_data.get('ocr_backend', ''),
                        'Scale_Processing_Time_s': result.get('scale_processing_time', 0),
                        
                        # Fiber Detection
                        'Fiber_Type': fiber_data.get('fiber_type', 'unknown'),
                        'Fiber_Confidence': fiber_data.get('confidence', 0),
                        'Total_Fibers': fiber_data.get('total_fibers', 0),
                        'Hollow_Fibers': fiber_data.get('hollow_fibers', 0),
                        'Filaments': fiber_data.get('filaments', 0),
                        'Classification_Method': fiber_data.get('classification_method', 'unknown'),
                        'Fiber_Processing_Time_s': fiber_data.get('processing_time', 0),
                        
                        # Porosity Analysis - Core Metrics
                        'Porosity_Success': 'porosity_metrics' in porosity_data if porosity_data else False,
                        'Total_Porosity_Percent': pm.get('total_porosity_percent', 0),
                        'Pore_Count': pm.get('pore_count', 0),
                        'Average_Pore_Size_um2': pm.get('average_pore_size_um2', 0),
                        'Median_Pore_Size_um2': pm.get('median_pore_size_um2', 0),
                        'Std_Pore_Size_um2': pm.get('std_pore_size_um2', 0),
                        'Min_Pore_Size_um2': pm.get('min_pore_size_um2', 0),
                        'Max_Pore_Size_um2': pm.get('max_pore_size_um2', 0),
                        'Mean_Pore_Diameter_um': pm.get('mean_pore_diameter_um', 0),
                        'Median_Pore_Diameter_um': pm.get('median_pore_diameter_um', 0),
                        'Pore_Density_per_mm2': pm.get('pore_density_per_mm2', 0),
                        'Total_Pore_Area_um2': pm.get('total_pore_area_um2', 0),
                        'Total_Fiber_Area_um2': pm.get('total_fiber_area_um2', 0),
                        'Porosity_Method': pm.get('method', ''),
                        'Porosity_Processing_Time_s': result.get('porosity_processing_time', 0),
                        
                        # Quality Assessment from Porosity Analysis
                        'Porosity_Analysis_Quality': quality_assessment.get('overall_quality', 'unknown'),
                        'Porosity_Analysis_Confidence': quality_assessment.get('confidence', 0),
                        'Porosity_Quality_Score': quality_assessment.get('quality_score', 0),
                        'Porosity_Issues': '; '.join(quality_assessment.get('issues', [])),
                        'Tiny_Pore_Fraction': quality_assessment.get('tiny_pore_fraction', 0),
                        
                        # Comprehensive Quality Factors
                        'Quality_Factors': '; '.join(comprehensive.get('quality_factors', [])),
                        'Porosity_Method_Used': comprehensive.get('porosity_method', ''),
                        
                        # Physical Measurements - Fiber Statistics
                        'Fiber_Count': 0,
                        'Fiber_Mean_Area_um2': 0,
                        'Fiber_Std_Area_um2': 0,
                        'Fiber_Mean_Diameter_um': 0,
                        'Fiber_Std_Diameter_um': 0,
                        'Fiber_Min_Diameter_um': 0,
                        'Fiber_Max_Diameter_um': 0,
                        'Fiber_Median_Diameter_um': 0,
                        
                        # Physical Measurements - Lumen Statistics
                        'Lumen_Count': 0,
                        'Lumen_Mean_Area_um2': 0,
                        'Lumen_Std_Area_um2': 0,
                        
                        # Physical Measurements - Porosity Statistics (duplicated for clarity)
                        'Porosity_Stats_Total_Percent': 0,
                        'Porosity_Stats_Pore_Count': 0,
                        'Porosity_Stats_Average_Pore_Size_um2': 0,
                        'Porosity_Stats_Pore_Density_per_mm2': 0,
                        'Porosity_Stats_Method': '',
                        
                        # Image Information
                        'Image_Shape': f"{result.get('image_shape', [0,0])[0]}x{result.get('image_shape', [0,0])[1]}",
                        'Image_Size_MB': result.get('image_size_mb', 0),
                        'Preprocessing_Time_s': result.get('preprocessing_time', 0),
                    }
                    
                    # Fill in physical measurements if available
                    measurements = comprehensive.get('physical_measurements', {})
                    
                    # Fiber statistics
                    if 'fiber_statistics' in measurements:
                        fs = measurements['fiber_statistics']
                        row.update({
                            'Fiber_Count': fs.get('count', 0),
                            'Fiber_Mean_Area_um2': fs.get('mean_area_um2', 0),
                            'Fiber_Std_Area_um2': fs.get('std_area_um2', 0),
                            'Fiber_Mean_Diameter_um': fs.get('mean_diameter_um', 0),
                            'Fiber_Std_Diameter_um': fs.get('std_diameter_um', 0),
                            'Fiber_Min_Diameter_um': fs.get('min_diameter_um', 0),
                            'Fiber_Max_Diameter_um': fs.get('max_diameter_um', 0),
                            'Fiber_Median_Diameter_um': fs.get('median_diameter_um', 0),
                        })
                    
                    # Lumen statistics
                    if 'lumen_statistics' in measurements:
                        ls = measurements['lumen_statistics']
                        row.update({
                            'Lumen_Count': ls.get('count', 0),
                            'Lumen_Mean_Area_um2': ls.get('mean_area_um2', 0),
                            'Lumen_Std_Area_um2': ls.get('std_area_um2', 0),
                        })
                    
                    # Porosity statistics from physical measurements
                    if 'porosity_statistics' in measurements:
                        ps = measurements['porosity_statistics']
                        row.update({
                            'Porosity_Stats_Total_Percent': ps.get('total_porosity_percent', 0),
                            'Porosity_Stats_Pore_Count': ps.get('pore_count', 0),
                            'Porosity_Stats_Average_Pore_Size_um2': ps.get('average_pore_size_um2', 0),
                            'Porosity_Stats_Pore_Density_per_mm2': ps.get('pore_density_per_mm2', 0),
                            'Porosity_Stats_Method': ps.get('method_used', ''),
                        })
                    
                else:
                    # Failed analysis
                    row = {
                        'Image_Name': result['image_name'],
                        'Analysis_Quality': 'FAILED',
                        'Analysis_Success': False,
                        'Error': result.get('error', 'Unknown error'),
                        'Total_Processing_Time_s': result.get('total_processing_time', 0),
                        'Image_Shape': f"{result.get('image_shape', [0,0])[0]}x{result.get('image_shape', [0,0])[1]}" if 'image_shape' in result else '',
                        'Image_Size_MB': result.get('image_size_mb', 0),
                    }
                
                main_results.append(row)
            
            # Create main results DataFrame
            main_df = pd.DataFrame(main_results)
            main_df.to_excel(writer, sheet_name='All_Sample_Results', index=False)
            
            # 3. SCALE DETECTION DETAILS
            scale_details = []
            for result in results:
                if result.get('success', False):
                    scale_data = result.get('scale_detection', {})
                    scale_info = scale_data.get('scale_info', {})
                    
                    scale_details.append({
                        'Image_Name': result['image_name'],
                        'Scale_Detected': scale_data.get('scale_detected', False),
                        'Scale_Factor': scale_data.get('micrometers_per_pixel', 0),
                        'Scale_Text': scale_info.get('text', ''),
                        'Scale_Value': scale_info.get('value', 0),
                        'Scale_Unit': scale_info.get('unit', ''),
                        'Confidence': scale_data.get('confidence', 0),
                        'Method': scale_data.get('method_used', ''),
                        'OCR_Backend': scale_data.get('ocr_backend', ''),
                        'Processing_Time_s': result.get('scale_processing_time', 0),
                        'Error': scale_data.get('error', '') if not scale_data.get('scale_detected', False) else ''
                    })
            
            scale_df = pd.DataFrame(scale_details)
            scale_df.to_excel(writer, sheet_name='Scale_Detection_Details', index=False)
            
            # 4. FIBER ANALYSIS DETAILS
            fiber_details = []
            for result in results:
                if result.get('success', False):
                    fiber_data = result.get('fiber_detection', {})
                    thresholds = fiber_data.get('thresholds_used', {})
                    
                    fiber_details.append({
                        'Image_Name': result['image_name'],
                        'Fiber_Type': fiber_data.get('fiber_type', 'unknown'),
                        'Confidence': fiber_data.get('confidence', 0),
                        'Total_Fibers': fiber_data.get('total_fibers', 0),
                        'Hollow_Fibers': fiber_data.get('hollow_fibers', 0),
                        'Filaments': fiber_data.get('filaments', 0),
                        'Classification_Method': fiber_data.get('classification_method', 'unknown'),
                        'Min_Fiber_Area_pixels': thresholds.get('min_fiber_area', 0),
                        'Max_Fiber_Area_pixels': thresholds.get('max_fiber_area', 0),
                        'Kernel_Size': thresholds.get('kernel_size', 0),
                        'Processing_Time_s': fiber_data.get('processing_time', 0)
                    })
            
            fiber_df = pd.DataFrame(fiber_details)
            fiber_df.to_excel(writer, sheet_name='Fiber_Analysis_Details', index=False)
            
            # 5. POROSITY ANALYSIS DETAILS
            porosity_details = []
            for result in results:
                if result.get('success', False):
                    porosity_data = result.get('porosity_analysis', {})
                    pm = porosity_data.get('porosity_metrics', {}) if porosity_data else {}
                    quality = porosity_data.get('quality_assessment', {}) if porosity_data else {}
                    
                    porosity_details.append({
                        'Image_Name': result['image_name'],
                        'Analysis_Success': 'porosity_metrics' in porosity_data if porosity_data else False,
                        'Method': pm.get('method', ''),
                        'Total_Porosity_Percent': pm.get('total_porosity_percent', 0),
                        'Pore_Count': pm.get('pore_count', 0),
                        'Total_Pore_Area_um2': pm.get('total_pore_area_um2', 0),
                        'Average_Pore_Size_um2': pm.get('average_pore_size_um2', 0),
                        'Median_Pore_Size_um2': pm.get('median_pore_size_um2', 0),
                        'Std_Pore_Size_um2': pm.get('std_pore_size_um2', 0),
                        'Min_Pore_Size_um2': pm.get('min_pore_size_um2', 0),
                        'Max_Pore_Size_um2': pm.get('max_pore_size_um2', 0),
                        'Mean_Pore_Diameter_um': pm.get('mean_pore_diameter_um', 0),
                        'Pore_Density_per_mm2': pm.get('pore_density_per_mm2', 0),
                        'Analysis_Quality': quality.get('overall_quality', 'unknown'),
                        'Quality_Confidence': quality.get('confidence', 0),
                        'Processing_Time_s': result.get('porosity_processing_time', 0),
                        'Error': porosity_data.get('error', '') if porosity_data and 'error' in porosity_data else ''
                    })
            
            porosity_df = pd.DataFrame(porosity_details)
            porosity_df.to_excel(writer, sheet_name='Porosity_Analysis_Details', index=False)
            
            # 6. QUALITY SUMMARY
            quality_summary = []
            for result in results:
                if result.get('success', False):
                    comprehensive = result.get('comprehensive_metrics', {})
                    
                    quality_summary.append({
                        'Image_Name': result['image_name'],
                        'Overall_Quality': comprehensive.get('analysis_quality', 'unknown'),
                        'Quality_Score': comprehensive.get('quality_score', 0),
                        'Scale_Factor': comprehensive.get('scale_factor_um_per_pixel', 0),
                        'Fiber_Type': comprehensive.get('fiber_type', 'unknown'),
                        'Fiber_Confidence': comprehensive.get('fiber_confidence', 0),
                        'Porosity_Method': comprehensive.get('porosity_method', ''),
                        'Total_Time_s': result.get('total_processing_time', 0),
                        'Quality_Factors': '; '.join(comprehensive.get('quality_factors', []))
                    })
            
            quality_df = pd.DataFrame(quality_summary)
            quality_df.to_excel(writer, sheet_name='Quality_Summary', index=False)
            
            # 7. PROCESSING PERFORMANCE
            performance_data = []
            for result in results:
                if result.get('success', False):
                    performance_data.append({
                        'Image_Name': result['image_name'],
                        'Total_Time_s': result.get('total_processing_time', 0),
                        'Preprocessing_Time_s': result.get('preprocessing_time', 0),
                        'Scale_Detection_Time_s': result.get('scale_processing_time', 0),
                        'Fiber_Detection_Time_s': result.get('fiber_processing_time', 0),
                        'Porosity_Analysis_Time_s': result.get('porosity_processing_time', 0),
                        'Image_Size_MB': result.get('image_size_mb', 0),
                        'Image_Shape': f"{result.get('image_shape', [0,0])[0]}x{result.get('image_shape', [0,0])[1]}"
                    })
            
            performance_df = pd.DataFrame(performance_data)
            performance_df.to_excel(writer, sheet_name='Processing_Performance', index=False)
        
        print(f"📊 Central Excel report created with {len(results)} samples across 7 detailed sheets")

    def _create_central_excel_report(self, batch_summary: Dict, excel_path: Path):
        """
        Create a central Excel file with all sample data in organized sheets.
        This replaces individual reports with a comprehensive comparison view.
        """
        
        results = batch_summary['individual_results']
        batch_info = batch_summary['batch_info']
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            
            # 1. OVERVIEW SHEET - Batch Summary
            overview_data = {
                'Metric': [
                    'Analysis Date', 'Input Directory', 'Total Images', 'Successful Analyses', 
                    'Success Rate (%)', 'Total Processing Time (s)', 'Average Time per Image (s)',
                    'Porosity Method', 'Analysis Quality'
                ],
                'Value': [
                    batch_info['timestamp'][:19].replace('T', ' '),
                    batch_info['input_directory'],
                    batch_info['total_images'],
                    batch_info['successful_analyses'],
                    f"{batch_info['success_rate']:.1f}%",
                    f"{batch_info['total_processing_time']:.2f}",
                    f"{batch_info['average_time_per_image']:.2f}",
                    batch_info['porosity_method'],
                    'See individual results →'
                ]
            }
            overview_df = pd.DataFrame(overview_data)
            overview_df.to_excel(writer, sheet_name='Batch_Overview', index=False)
            
            # 2. MAIN RESULTS SHEET - All Sample Comparisons
            main_results = []
            for result in results:
                if result.get('success', False):
                    # Extract all key data
                    scale_data = result.get('scale_detection', {})
                    fiber_data = result.get('fiber_detection', {})
                    porosity_data = result.get('porosity_analysis', {})
                    comprehensive = result.get('comprehensive_metrics', {})
                    pm = porosity_data.get('porosity_metrics', {}) if porosity_data else {}
                    
                    row = {
                        # Basic Info
                        'Image_Name': result['image_name'],
                        'Analysis_Quality': comprehensive.get('analysis_quality', 'unknown'),
                        'Quality_Score': comprehensive.get('quality_score', 0),
                        'Total_Processing_Time_s': result.get('total_processing_time', 0),
                        
                        # Scale Detection
                        'Scale_Detected': scale_data.get('scale_detected', False),
                        'Scale_Factor_um_per_pixel': scale_data.get('micrometers_per_pixel', 0),
                        'Scale_Confidence': scale_data.get('confidence', 0),
                        'Scale_Text': scale_data.get('scale_info', {}).get('text', ''),
                        'Scale_Method': scale_data.get('method_used', ''),
                        'OCR_Backend': scale_data.get('ocr_backend', ''),
                        
                        # Fiber Detection
                        'Fiber_Type': fiber_data.get('fiber_type', 'unknown'),
                        'Fiber_Confidence': fiber_data.get('confidence', 0),
                        'Total_Fibers': fiber_data.get('total_fibers', 0),
                        'Hollow_Fibers': fiber_data.get('hollow_fibers', 0),
                        'Filaments': fiber_data.get('filaments', 0),
                        'Classification_Method': fiber_data.get('classification_method', 'unknown'),
                        'Fiber_Processing_Time_s': fiber_data.get('processing_time', 0),
                        
                        # Porosity Analysis
                        'Porosity_Success': 'porosity_metrics' in porosity_data if porosity_data else False,
                        'Total_Porosity_Percent': pm.get('total_porosity_percent', 0),
                        'Pore_Count': pm.get('pore_count', 0),
                        'Average_Pore_Size_um2': pm.get('average_pore_size_um2', 0),
                        'Median_Pore_Size_um2': pm.get('median_pore_size_um2', 0),
                        'Min_Pore_Size_um2': pm.get('min_pore_size_um2', 0),
                        'Max_Pore_Size_um2': pm.get('max_pore_size_um2', 0),
                        'Pore_Density_per_mm2': pm.get('pore_density_per_mm2', 0),
                        'Porosity_Method': pm.get('method', ''),
                        'Porosity_Processing_Time_s': result.get('porosity_processing_time', 0),
                        
                        # Physical Measurements (if scale detected)
                        'Fiber_Mean_Diameter_um': 0,
                        'Fiber_Std_Diameter_um': 0,
                        'Lumen_Mean_Area_um2': 0,
                    }
                    
                    # Add physical measurements if available
                    measurements = comprehensive.get('physical_measurements', {})
                    if 'fiber_statistics' in measurements:
                        fs = measurements['fiber_statistics']
                        row['Fiber_Mean_Diameter_um'] = fs.get('mean_diameter_um', 0)
                        row['Fiber_Std_Diameter_um'] = fs.get('std_diameter_um', 0)
                    
                    if 'lumen_statistics' in measurements:
                        ls = measurements['lumen_statistics']
                        row['Lumen_Mean_Area_um2'] = ls.get('mean_area_um2', 0)
                    
                else:
                    # Failed analysis
                    row = {
                        'Image_Name': result['image_name'],
                        'Analysis_Quality': 'FAILED',
                        'Error': result.get('error', 'Unknown error'),
                        'Total_Processing_Time_s': result.get('total_processing_time', 0),
                    }
                
                main_results.append(row)
            
            # Create main results DataFrame
            main_df = pd.DataFrame(main_results)
            main_df.to_excel(writer, sheet_name='All_Sample_Results', index=False)
            
            # 3. SCALE DETECTION DETAILS
            scale_details = []
            for result in results:
                if result.get('success', False):
                    scale_data = result.get('scale_detection', {})
                    scale_info = scale_data.get('scale_info', {})
                    
                    scale_details.append({
                        'Image_Name': result['image_name'],
                        'Scale_Detected': scale_data.get('scale_detected', False),
                        'Scale_Factor': scale_data.get('micrometers_per_pixel', 0),
                        'Scale_Text': scale_info.get('text', ''),
                        'Scale_Value': scale_info.get('value', 0),
                        'Scale_Unit': scale_info.get('unit', ''),
                        'Confidence': scale_data.get('confidence', 0),
                        'Method': scale_data.get('method_used', ''),
                        'OCR_Backend': scale_data.get('ocr_backend', ''),
                        'Processing_Time_s': result.get('scale_processing_time', 0),
                        'Error': scale_data.get('error', '') if not scale_data.get('scale_detected', False) else ''
                    })
            
            scale_df = pd.DataFrame(scale_details)
            scale_df.to_excel(writer, sheet_name='Scale_Detection_Details', index=False)
            
            # 4. FIBER ANALYSIS DETAILS
            fiber_details = []
            for result in results:
                if result.get('success', False):
                    fiber_data = result.get('fiber_detection', {})
                    thresholds = fiber_data.get('thresholds_used', {})
                    
                    fiber_details.append({
                        'Image_Name': result['image_name'],
                        'Fiber_Type': fiber_data.get('fiber_type', 'unknown'),
                        'Confidence': fiber_data.get('confidence', 0),
                        'Total_Fibers': fiber_data.get('total_fibers', 0),
                        'Hollow_Fibers': fiber_data.get('hollow_fibers', 0),
                        'Filaments': fiber_data.get('filaments', 0),
                        'Classification_Method': fiber_data.get('classification_method', 'unknown'),
                        'Min_Fiber_Area_pixels': thresholds.get('min_fiber_area', 0),
                        'Max_Fiber_Area_pixels': thresholds.get('max_fiber_area', 0),
                        'Kernel_Size': thresholds.get('kernel_size', 0),
                        'Processing_Time_s': fiber_data.get('processing_time', 0)
                    })
            
            fiber_df = pd.DataFrame(fiber_details)
            fiber_df.to_excel(writer, sheet_name='Fiber_Analysis_Details', index=False)
            
            # 5. POROSITY ANALYSIS DETAILS
            porosity_details = []
            for result in results:
                if result.get('success', False):
                    porosity_data = result.get('porosity_analysis', {})
                    pm = porosity_data.get('porosity_metrics', {}) if porosity_data else {}
                    quality = porosity_data.get('quality_assessment', {}) if porosity_data else {}
                    
                    porosity_details.append({
                        'Image_Name': result['image_name'],
                        'Analysis_Success': 'porosity_metrics' in porosity_data if porosity_data else False,
                        'Method': pm.get('method', ''),
                        'Total_Porosity_Percent': pm.get('total_porosity_percent', 0),
                        'Pore_Count': pm.get('pore_count', 0),
                        'Total_Pore_Area_um2': pm.get('total_pore_area_um2', 0),
                        'Average_Pore_Size_um2': pm.get('average_pore_size_um2', 0),
                        'Median_Pore_Size_um2': pm.get('median_pore_size_um2', 0),
                        'Std_Pore_Size_um2': pm.get('std_pore_size_um2', 0),
                        'Min_Pore_Size_um2': pm.get('min_pore_size_um2', 0),
                        'Max_Pore_Size_um2': pm.get('max_pore_size_um2', 0),
                        'Mean_Pore_Diameter_um': pm.get('mean_pore_diameter_um', 0),
                        'Pore_Density_per_mm2': pm.get('pore_density_per_mm2', 0),
                        'Analysis_Quality': quality.get('overall_quality', 'unknown'),
                        'Quality_Confidence': quality.get('confidence', 0),
                        'Processing_Time_s': result.get('porosity_processing_time', 0),
                        'Error': porosity_data.get('error', '') if porosity_data and 'error' in porosity_data else ''
                    })
            
            porosity_df = pd.DataFrame(porosity_details)
            porosity_df.to_excel(writer, sheet_name='Porosity_Analysis_Details', index=False)
            
            # 6. QUALITY SUMMARY
            quality_summary = []
            for result in results:
                if result.get('success', False):
                    comprehensive = result.get('comprehensive_metrics', {})
                    
                    quality_summary.append({
                        'Image_Name': result['image_name'],
                        'Overall_Quality': comprehensive.get('analysis_quality', 'unknown'),
                        'Quality_Score': comprehensive.get('quality_score', 0),
                        'Scale_Factor': comprehensive.get('scale_factor_um_per_pixel', 0),
                        'Fiber_Type': comprehensive.get('fiber_type', 'unknown'),
                        'Fiber_Confidence': comprehensive.get('fiber_confidence', 0),
                        'Porosity_Method': comprehensive.get('porosity_method', ''),
                        'Total_Time_s': result.get('total_processing_time', 0),
                        'Quality_Factors': '; '.join(comprehensive.get('quality_factors', []))
                    })
            
            quality_df = pd.DataFrame(quality_summary)
            quality_df.to_excel(writer, sheet_name='Quality_Summary', index=False)
            
            # 7. PROCESSING PERFORMANCE
            performance_data = []
            for result in results:
                if result.get('success', False):
                    performance_data.append({
                        'Image_Name': result['image_name'],
                        'Total_Time_s': result.get('total_processing_time', 0),
                        'Preprocessing_Time_s': result.get('preprocessing_time', 0),
                        'Scale_Detection_Time_s': result.get('scale_processing_time', 0),
                        'Fiber_Detection_Time_s': result.get('fiber_processing_time', 0),
                        'Porosity_Analysis_Time_s': result.get('porosity_processing_time', 0),
                        'Image_Size_MB': result.get('image_size_mb', 0),
                        'Image_Shape': f"{result.get('image_shape', [0,0])[0]}x{result.get('image_shape', [0,0])[1]}"
                    })
            
            performance_df = pd.DataFrame(performance_data)
            performance_df.to_excel(writer, sheet_name='Processing_Performance', index=False)
        
        print(f"📊 Central Excel report created with {len(results)} samples across 7 detailed sheets")
def main():
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(
        description='Comprehensive SEM Fiber Analysis with Fast Refined Porosity',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python comprehensive_analyzer_main.py --image sample.jpg
  python comprehensive_analyzer_main.py --batch sample_images/
  python comprehensive_analyzer_main.py --image sample.jpg --output results/
        """
    )
    
    parser.add_argument('--image', '-i', help='Analyze single image file')
    parser.add_argument('--batch', '-b', help='Analyze all images in directory')
    parser.add_argument('--output', '-o', help='Output directory for results')
    parser.add_argument('--config', '-c', help='Configuration JSON file')
    parser.add_argument('--ocr-backend', choices=['rapidocr', 'easyocr'], 
                       help='OCR backend for scale detection')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualizations')
    parser.add_argument('--no-data', action='store_true', help='Disable data export')
    parser.add_argument('--quiet', '-q', action='store_true', help='Minimal output')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Override config with command line arguments
    if args.ocr_backend:
        config.setdefault('scale_detection', {})['ocr_backend'] = args.ocr_backend
    
    if args.no_viz:
        config.setdefault('output', {})['save_visualizations'] = False
    
    if args.no_data:
        config.setdefault('output', {})['save_data'] = False
    
    # Initialize analyzer
    analyzer = ComprehensiveFiberAnalyzer(config=config, debug=not args.quiet)
    
    if args.image:
        # Single image analysis
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"❌ Image not found: {image_path}")
            return
        
        result = analyzer.analyze_single_image(str(image_path), args.output)
        
        if result['success']:
            print(f"\n🎉 Analysis completed successfully!")
        else:
            print(f"\n❌ Analysis failed: {result.get('error', 'Unknown error')}")
    
    elif args.batch:
        # Batch analysis
        batch_dir = Path(args.batch)
        if not batch_dir.exists():
            print(f"❌ Directory not found: {batch_dir}")
            return
        
        summary = analyzer.analyze_batch(str(batch_dir), args.output)
        
        if 'error' not in summary:
            batch_info = summary['batch_info']
            print(f"\n🎯 Batch Analysis Summary:")
            print(f"Success Rate: {batch_info['success_rate']:.1f}%")
            print(f"Average Time: {batch_info['average_time_per_image']:.2f}s per image")
            print(f"Method: {batch_info['porosity_method']}")
    
    else:
        # No specific action - show help or analyze demo
        sample_dir = Path("sample_images")
        if sample_dir.exists():
            image_files = (list(sample_dir.glob("*.jpg")) + 
                          list(sample_dir.glob("*.png")) + 
                          list(sample_dir.glob("*.tif")))
            if image_files:
                print("No specific action specified. Analyzing first available image...")
                result = analyzer.analyze_single_image(str(image_files[0]))
                
                if result['success']:
                    print(f"\n✅ Demo analysis successful!")
                else:
                    print(f"\n❌ Demo analysis failed!")
            else:
                print("No images found in sample_images/ for demo.")
                parser.print_help()
        else:
            print("Usage examples:")
            print("  python comprehensive_analyzer_main.py --image your_image.jpg")
            print("  python comprehensive_analyzer_main.py --batch your_image_folder/")
            print("  python comprehensive_analyzer_main.py --help")


if __name__ == "__main__":
    main()