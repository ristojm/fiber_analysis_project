#!/usr/bin/env python3
"""
Comprehensive SEM Fiber Analyzer - Main Integration Script
Uses modular components for complete fiber characterization

This script demonstrates how to use the existing modules together:
1. Scale detection (optimized)
2. Fiber type detection (adaptive) 
3. Enhanced porosity analysis
4. Comprehensive reporting and visualization

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
    # Enhanced porosity analysis
    from modules.porosity_analysis import EnhancedPorosityAnalyzer, analyze_fiber_porosity_enhanced
    print("✅ Enhanced porosity module loaded")
    POROSITY_AVAILABLE = True
except ImportError:
    print("⚠️ Enhanced porosity module not found, using basic analysis")
    try:
        from modules.porosity_analysis import PorosityAnalyzer, analyze_fiber_porosity
        POROSITY_AVAILABLE = True
        ENHANCED_POROSITY = False
    except ImportError:
        print("❌ No porosity analysis available")
        POROSITY_AVAILABLE = False
        ENHANCED_POROSITY = False

class ComprehensiveFiberAnalyzer:
    """
    Comprehensive analyzer that orchestrates all analysis modules.
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
            ocr_backend=self.config.get('scale_detection', {}).get('ocr_backend')
        )
        
        self.fiber_detector = FiberTypeDetector(
            **self.config.get('fiber_detection', {})
        )
        
        if POROSITY_AVAILABLE:
            if 'ENHANCED_POROSITY' in globals() and ENHANCED_POROSITY:
                self.porosity_analyzer = EnhancedPorosityAnalyzer(
                    config=self.config.get('porosity_analysis', {})
                )
            else:
                try:
                    self.porosity_analyzer = PorosityAnalyzer(
                        config=self.config.get('porosity_analysis', {})
                    )
                except:
                    self.porosity_analyzer = None
        else:
            self.porosity_analyzer = None
        
        if self.debug:
            print(f"🔬 Comprehensive Fiber Analyzer initialized")
            print(f"   Scale detection: {self.scale_detector.ocr_backend or 'legacy'}")
            print(f"   Fiber detection: Adaptive algorithms")
            print(f"   Porosity analysis: {'Enhanced' if POROSITY_AVAILABLE else 'Not available'}")
    
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
                'pore_detection': {
                    'min_pore_area': 5,
                    'max_pore_area': 50000,
                    'contrast_threshold': 0.3,
                    'adaptive_thresholds': True,
                },
                'segmentation': {
                    'method': 'multi_otsu',
                    'fiber_type_aware': True,
                },
                'analysis': {
                    'spatial_analysis': True,
                    'percentiles': [25, 50, 75, 90, 95],
                },
                'fiber_integration': {
                    'use_individual_fibers': True,
                    'hollow_fiber_lumen_exclusion': True,
                }
            },
            'output': {
                'save_visualizations': True,
                'save_data': True,
                'create_report': True,
                'dpi': 300,
            }
        }
    
    def _update_config(self, new_config: Dict):
        """Update configuration recursively."""
        for key, value in new_config.items():
            if key in self.config and isinstance(self.config[key], dict):
                self.config[key].update(value)
            else:
                self.config[key] = value
    
    def analyze_comprehensive(self, image_path: str, 
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
            'modules_used': []
        }
        
        try:
            # Setup output directory
            if output_dir is None:
                output_dir = Path(image_path).parent / 'comprehensive_analysis'
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
            
            # Enhanced preprocessing for comprehensive analysis
            preprocessed = self._preprocess_for_comprehensive_analysis(image)
            
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
            
            # Step 4: Porosity analysis
            porosity_result = None
            if self.porosity_analyzer and POROSITY_AVAILABLE:
                if self.debug:
                    print("🕳️  Step 4: Enhanced porosity analysis...")
                
                step_start = time.time()
                
                # Get fiber mask from detection results
                fiber_mask = fiber_analysis_data.get('fiber_mask', np.zeros_like(image, dtype=bool))
                
                if np.sum(fiber_mask) > 1000:  # Minimum area threshold
                    try:
                        if hasattr(self.porosity_analyzer, 'analyze_fiber_porosity'):
                            # Enhanced porosity analyzer
                            porosity_result = self.porosity_analyzer.analyze_fiber_porosity(
                                preprocessed, 
                                fiber_mask.astype(np.uint8), 
                                scale_factor, 
                                fiber_type,
                                fiber_analysis_data
                            )
                        else:
                            # Standard porosity analyzer
                            porosity_result = self.porosity_analyzer.analyze_porosity(
                                preprocessed, 
                                fiber_mask.astype(np.uint8), 
                                scale_factor, 
                                fiber_type
                            )
                        
                        porosity_time = time.time() - step_start
                        result['porosity_analysis'] = porosity_result
                        result['porosity_processing_time'] = porosity_time
                        result['modules_used'].append('porosity_analysis')
                        
                        if self.debug:
                            pm = porosity_result['porosity_metrics']
                            print(f"   ✅ Porosity analysis completed:")
                            print(f"   Total porosity: {pm['total_porosity_percent']:.2f}%")
                            print(f"   Pore count: {pm['pore_count']}")
                            if 'average_pore_size_um2' in pm:
                                print(f"   Avg pore size: {pm['average_pore_size_um2']:.2f} μm²")
                            print(f"   Processing time: {porosity_time:.3f}s")
                    
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
            
            # Step 5: Comprehensive analysis and metrics
            if self.debug:
                print("📊 Step 5: Generating comprehensive metrics...")
            
            comprehensive_metrics = self._generate_comprehensive_metrics(
                result, scale_factor, fiber_type, fiber_confidence, 
                fiber_analysis_data, porosity_result
            )
            result['comprehensive_metrics'] = comprehensive_metrics
            
            # Step 6: Export results and visualizations
            if self.config['output']['save_data'] or self.config['output']['save_visualizations']:
                if self.debug:
                    print("💾 Step 6: Exporting results...")
                
                export_paths = self._export_comprehensive_results(
                    result, output_dir, image, preprocessed
                )
                result['export_paths'] = export_paths
            
            # Mark as successful
            result['success'] = True
            result['total_processing_time'] = time.time() - start_time
            
            if self.debug:
                print(f"✅ Comprehensive analysis completed successfully!")
                print(f"⏱️ Total time: {result['total_processing_time']:.2f}s")
                self._print_comprehensive_summary(result)
        
        except Exception as e:
            result['error'] = str(e)
            result['total_processing_time'] = time.time() - start_time
            
            if self.debug:
                print(f"❌ Comprehensive analysis failed: {e}")
                import traceback
                traceback.print_exc()
        
        return result
    
    def _preprocess_for_comprehensive_analysis(self, image: np.ndarray) -> np.ndarray:
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
        
        # Fiber detection quality (25%)
        quality_score += fiber_confidence * 0.25
        quality_factors.append(f"Fiber: {fiber_confidence:.2f}")
        
        # Porosity analysis quality (35%)
        if porosity_result and 'porosity_metrics' in porosity_result:
            pm = porosity_result['porosity_metrics']
            pore_count = pm.get('pore_count', 0)
            
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
            
            quality_score += porosity_quality * 0.35
            quality_factors.append(f"Porosity: {porosity_quality:.2f}")
        else:
            quality_factors.append("Porosity: unavailable")
        
        # Processing efficiency (15%)
        total_time = result.get('total_processing_time', 0)
        if total_time < 5:
            efficiency_score = 1.0
        elif total_time < 15:
            efficiency_score = 0.8
        elif total_time < 30:
            efficiency_score = 0.6
        else:
            efficiency_score = 0.4
        
        quality_score += efficiency_score * 0.15
        quality_factors.append(f"Efficiency: {efficiency_score:.2f}")
        
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
        
        # Add material characterization
        material_char = self._characterize_material(
            fiber_type, fiber_confidence, porosity_result, scale_factor
        )
        metrics['material_characterization'] = material_char
        
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
                'pore_density_per_mm2': pm.get('pore_density_per_mm2', 0)
            }
        
        return measurements
    
    def _characterize_material(self, fiber_type: str, fiber_confidence: float,
                             porosity_data: Optional[Dict], scale_factor: float) -> Dict:
        """Characterize material properties and applications."""
        
        characterization = {
            'primary_structure': fiber_type,
            'structure_confidence': fiber_confidence,
            'scale_calibrated': scale_factor != 1.0
        }
        
        # Porosity-based characterization
        if porosity_data and 'porosity_metrics' in porosity_data:
            pm = porosity_data['porosity_metrics']
            porosity_percent = pm.get('total_porosity_percent', 0)
            
            if porosity_percent > 15:
                porosity_level = 'high'
                quality_impact = 'significant_impact'
            elif porosity_percent > 5:
                porosity_level = 'moderate'
                quality_impact = 'moderate_impact'
            elif porosity_percent > 1:
                porosity_level = 'low'
                quality_impact = 'minimal_impact'
            else:
                porosity_level = 'minimal'
                quality_impact = 'no_impact'
            
            characterization.update({
                'porosity_level': porosity_level,
                'quality_impact': quality_impact
            })
        else:
            characterization.update({
                'porosity_level': 'unknown',
                'quality_impact': 'unknown'
            })
        
        # Application recommendations
        applications = []
        if fiber_type == 'hollow_fiber':
            if characterization.get('porosity_level') in ['minimal', 'low']:
                applications.extend(['filtration', 'separation_membranes', 'medical_devices'])
            else:
                applications.extend(['insulation', 'lightweight_composites'])
        elif fiber_type == 'filament':
            if characterization.get('porosity_level') in ['minimal', 'low']:
                applications.extend(['structural_composites', 'reinforcement', 'textiles'])
            else:
                applications.extend(['acoustic_materials', 'thermal_insulation'])
        else:
            applications = ['requires_further_analysis']
        
        characterization['recommended_applications'] = applications
        
        return characterization
    
    def _export_comprehensive_results(self, result: Dict, output_dir: Path,
                                    original_image: np.ndarray, 
                                    preprocessed_image: np.ndarray) -> Dict:
        """Export comprehensive results to files."""
        
        export_paths = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(result['image_path']).stem
        
        try:
            # 1. Save complete JSON results
            if self.config['output']['save_data']:
                json_path = output_dir / f"{base_name}_complete_analysis_{timestamp}.json"
                with open(json_path, 'w') as f:
                    # Convert numpy arrays to lists for JSON serialization
                    json_result = self._prepare_for_json(result)
                    json.dump(json_result, f, indent=2, default=str)
                export_paths['complete_json'] = str(json_path)
                
                if self.debug:
                    print(f"   💾 Complete results: {json_path.name}")
            
            # 2. Save Excel summary
            if self.config['output']['save_data']:
                excel_path = output_dir / f"{base_name}_analysis_summary_{timestamp}.xlsx"
                self._create_excel_summary(result, excel_path)
                export_paths['excel_summary'] = str(excel_path)
                
                if self.debug:
                    print(f"   📊 Excel summary: {excel_path.name}")
            
            # 3. Create comprehensive visualization
            if self.config['output']['save_visualizations']:
                viz_path = output_dir / f"{base_name}_comprehensive_viz_{timestamp}.png"
                self._create_comprehensive_visualization(result, original_image, viz_path)
                export_paths['comprehensive_visualization'] = str(viz_path)
                
                if self.debug:
                    print(f"   🎨 Visualization: {viz_path.name}")
            
            # 4. Generate analysis report
            if self.config['output']['create_report']:
                report_path = output_dir / f"{base_name}_analysis_report_{timestamp}.txt"
                self._create_analysis_report(result, report_path)
                export_paths['analysis_report'] = str(report_path)
                
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
        """Create Excel summary of all results."""
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Overview sheet
            overview_data = {
                'Metric': ['Image Name', 'Analysis Quality', 'Quality Score', 'Total Processing Time',
                          'Fiber Type', 'Fiber Confidence', 'Scale Detected', 'Scale Factor'],
                'Value': [
                    result['image_name'],
                    result.get('comprehensive_metrics', {}).get('analysis_quality', 'unknown'),
                    result.get('comprehensive_metrics', {}).get('quality_score', 0),
                    f"{result.get('total_processing_time', 0):.2f}s",
                    result.get('fiber_detection', {}).get('fiber_type', 'unknown'),
                    result.get('fiber_detection', {}).get('confidence', 0),
                    result.get('scale_detection', {}).get('scale_detected', False),
                    result.get('comprehensive_metrics', {}).get('scale_factor_um_per_pixel', 0)
                ]
            }
            overview_df = pd.DataFrame(overview_data)
            overview_df.to_excel(writer, sheet_name='Overview', index=False)
            
            # Fiber detection results
            fiber_data = result.get('fiber_detection', {})
            fiber_summary = pd.DataFrame([{
                'Total Fibers': fiber_data.get('total_fibers', 0),
                'Hollow Fibers': fiber_data.get('hollow_fibers', 0),
                'Filaments': fiber_data.get('filaments', 0),
                'Classification Method': fiber_data.get('classification_method', 'unknown'),
                'Processing Time (s)': fiber_data.get('processing_time', 0)
            }])
            fiber_summary.to_excel(writer, sheet_name='Fiber_Detection', index=False)
            
            # Porosity results
            porosity_data = result.get('porosity_analysis', {})
            if porosity_data and 'porosity_metrics' in porosity_data:
                pm = porosity_data['porosity_metrics']
                porosity_summary = pd.DataFrame([{
                    'Total Porosity (%)': pm.get('total_porosity_percent', 0),
                    'Pore Count': pm.get('pore_count', 0),
                    'Average Pore Size (μm²)': pm.get('average_pore_size_um2', 0),
                    'Pore Density (/mm²)': pm.get('pore_density_per_mm2', 0)
                }])
                porosity_summary.to_excel(writer, sheet_name='Porosity_Analysis', index=False)
            
            # Physical measurements
            measurements = result.get('comprehensive_metrics', {}).get('physical_measurements', {})
            if 'fiber_statistics' in measurements:
                fs = measurements['fiber_statistics']
                fiber_measurements = pd.DataFrame([fs])
                fiber_measurements.to_excel(writer, sheet_name='Fiber_Measurements', index=False)
    
    def _create_comprehensive_visualization(self, result: Dict, 
                                          original_image: np.ndarray, 
                                          viz_path: Path):
        """Create comprehensive visualization of all results."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        axes[0, 0].imshow(original_image, cmap='gray')
        axes[0, 0].set_title('Original SEM Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Scale detection info
        scale_result = result.get('scale_detection', {})
        scale_text = self._format_scale_summary(scale_result)
        axes[0, 1].text(0.1, 0.5, scale_text, transform=axes[0, 1].transAxes,
                       fontsize=10, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue'))
        axes[0, 1].set_title('Scale Detection', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Fiber detection info
        fiber_result = result.get('fiber_detection', {})
        fiber_text = self._format_fiber_summary(fiber_result)
        axes[0, 2].text(0.1, 0.5, fiber_text, transform=axes[0, 2].transAxes,
                       fontsize=10, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen'))
        axes[0, 2].set_title('Fiber Type Detection', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Porosity analysis info
        porosity_result = result.get('porosity_analysis', {})
        porosity_text = self._format_porosity_summary(porosity_result)
        axes[1, 0].text(0.1, 0.5, porosity_text, transform=axes[1, 0].transAxes,
                       fontsize=10, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow'))
        axes[1, 0].set_title('Porosity Analysis', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Quality assessment
        comprehensive = result.get('comprehensive_metrics', {})
        quality_text = self._format_quality_summary(comprehensive, result)
        axes[1, 1].text(0.1, 0.5, quality_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcoral'))
        axes[1, 1].set_title('Analysis Quality', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Material characterization
        char_text = self._format_characterization_summary(comprehensive)
        axes[1, 2].text(0.1, 0.5, char_text, transform=axes[1, 2].transAxes,
                       fontsize=10, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='lightsteelblue'))
        axes[1, 2].set_title('Material Characterization', fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.suptitle(f"Comprehensive SEM Fiber Analysis: {result['image_name']}", 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(viz_path, dpi=self.config['output']['dpi'], bbox_inches='tight')
        plt.close()
    
    def _format_scale_summary(self, scale_result: Dict) -> str:
        """Format scale detection summary."""
        if scale_result.get('scale_detected', False):
            info = scale_result.get('scale_info', {})
            text = f"✅ Scale Detected\n\n"
            text += f"Factor: {scale_result['micrometers_per_pixel']:.4f} μm/pixel\n"
            text += f"Text: '{info.get('text', 'N/A')}'\n"
            text += f"Confidence: {scale_result.get('confidence', 0):.1%}\n"
            text += f"Method: {scale_result.get('detection_method', 'unknown')}\n"
            text += f"OCR: {scale_result.get('ocr_backend', 'unknown')}"
        else:
            text = f"❌ Scale Detection Failed\n\n"
            text += f"Error: {scale_result.get('error', 'Unknown')}\n"
            text += f"Using fallback: 1.0 μm/pixel"
        return text
    
    def _format_fiber_summary(self, fiber_result: Dict) -> str:
        """Format fiber detection summary."""
        text = f"Type: {fiber_result.get('fiber_type', 'Unknown')}\n"
        text += f"Confidence: {fiber_result.get('confidence', 0):.3f}\n\n"
        text += f"Detection Results:\n"
        text += f"• Total: {fiber_result.get('total_fibers', 0)}\n"
        text += f"• Hollow: {fiber_result.get('hollow_fibers', 0)}\n"
        text += f"• Filaments: {fiber_result.get('filaments', 0)}\n\n"
        text += f"Method: {fiber_result.get('classification_method', 'unknown')}\n"
        text += f"Time: {fiber_result.get('processing_time', 0):.3f}s"
        return text
    
    def _format_porosity_summary(self, porosity_result: Dict) -> str:
        """Format porosity analysis summary."""
        if porosity_result and 'porosity_metrics' in porosity_result:
            pm = porosity_result['porosity_metrics']
            text = f"✅ Porosity Analysis\n\n"
            text += f"Total Porosity: {pm.get('total_porosity_percent', 0):.2f}%\n"
            text += f"Pore Count: {pm.get('pore_count', 0)}\n"
            text += f"Avg Size: {pm.get('average_pore_size_um2', 0):.2f} μm²\n"
            text += f"Density: {pm.get('pore_density_per_mm2', 0):.1f}/mm²\n\n"
            
            # Analysis quality
            if 'analysis_quality' in porosity_result:
                aq = porosity_result['analysis_quality']
                text += f"Quality: {aq.get('overall_quality', 'unknown')}\n"
                text += f"Confidence: {aq.get('confidence_score', 0):.2f}"
        else:
            error = porosity_result.get('error', 'Unknown error') if porosity_result else 'Not performed'
            text = f"⚠️ Porosity Analysis\n\n"
            text += f"Status: Failed\n"
            text += f"Reason: {error}"
        return text
    
    def _format_quality_summary(self, comprehensive: Dict, result: Dict) -> str:
        """Format analysis quality summary."""
        text = f"Overall Quality: {comprehensive.get('analysis_quality', 'unknown').title()}\n"
        text += f"Quality Score: {comprehensive.get('quality_score', 0):.2f}/1.0\n\n"
        
        factors = comprehensive.get('quality_factors', [])
        if factors:
            text += "Component Scores:\n"
            for factor in factors:
                text += f"• {factor}\n"
        
        text += f"\nProcessing Time: {result.get('total_processing_time', 0):.2f}s\n"
        text += f"Modules Used: {len(result.get('modules_used', []))}"
        
        return text
    
    def _format_characterization_summary(self, comprehensive: Dict) -> str:
        """Format material characterization summary."""
        char = comprehensive.get('material_characterization', {})
        
        text = f"Structure: {char.get('primary_structure', 'Unknown')}\n"
        text += f"Confidence: {char.get('structure_confidence', 0):.3f}\n"
        text += f"Porosity: {char.get('porosity_level', 'unknown').title()}\n"
        text += f"Quality Impact: {char.get('quality_impact', 'unknown').replace('_', ' ').title()}\n\n"
        
        apps = char.get('recommended_applications', [])
        if apps:
            text += "Applications:\n"
            for app in apps[:3]:
                text += f"• {app.replace('_', ' ').title()}\n"
        
        text += f"\nScale Calibrated: {'Yes' if char.get('scale_calibrated', False) else 'No'}"
        
        return text
    
    def _create_analysis_report(self, result: Dict, report_path: Path):
        """Create detailed analysis report."""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("COMPREHENSIVE SEM FIBER ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Image: {result['image_name']}\n")
            f.write(f"Analysis Duration: {result.get('total_processing_time', 0):.2f} seconds\n")
            f.write(f"Modules Used: {', '.join(result.get('modules_used', []))}\n\n")
            
            # Overall assessment
            comprehensive = result.get('comprehensive_metrics', {})
            f.write("OVERALL ASSESSMENT\n")
            f.write("-" * 20 + "\n")
            f.write(f"Analysis Quality: {comprehensive.get('analysis_quality', 'unknown').title()}\n")
            f.write(f"Quality Score: {comprehensive.get('quality_score', 0):.2f}/1.0\n")
            f.write(f"Fiber Type: {comprehensive.get('fiber_type', 'unknown')}\n")
            f.write(f"Scale Factor: {comprehensive.get('scale_factor_um_per_pixel', 0):.4f} μm/pixel\n\n")
            
            # Detailed results
            self._write_module_results(f, result)
            
            # Material characterization
            self._write_material_characterization(f, comprehensive)
            
            # Recommendations
            self._write_recommendations(f, result, comprehensive)
    
    def _write_module_results(self, f, result: Dict):
        """Write detailed module results to report."""
        
        # Scale detection
        f.write("SCALE DETECTION RESULTS\n")
        f.write("-" * 25 + "\n")
        scale_result = result.get('scale_detection', {})
        if scale_result.get('scale_detected', False):
            f.write(f"Status: Successful\n")
            f.write(f"Scale Factor: {scale_result['micrometers_per_pixel']:.4f} μm/pixel\n")
            info = scale_result.get('scale_info', {})
            f.write(f"Scale Text: '{info.get('text', 'N/A')}'\n")
            f.write(f"Confidence: {scale_result.get('confidence', 0):.1%}\n")
            f.write(f"Detection Method: {scale_result.get('detection_method', 'unknown')}\n")
        else:
            f.write(f"Status: Failed\n")
            f.write(f"Error: {scale_result.get('error', 'Unknown')}\n")
        f.write(f"Processing Time: {result.get('scale_processing_time', 0):.3f}s\n\n")
        
        # Fiber detection
        f.write("FIBER TYPE DETECTION RESULTS\n")
        f.write("-" * 30 + "\n")
        fiber_result = result.get('fiber_detection', {})
        f.write(f"Fiber Type: {fiber_result.get('fiber_type', 'Unknown')}\n")
        f.write(f"Confidence: {fiber_result.get('confidence', 0):.3f}\n")
        f.write(f"Total Fibers Detected: {fiber_result.get('total_fibers', 0)}\n")
        f.write(f"Hollow Fibers: {fiber_result.get('hollow_fibers', 0)}\n")
        f.write(f"Filaments: {fiber_result.get('filaments', 0)}\n")
        f.write(f"Classification Method: {fiber_result.get('classification_method', 'unknown')}\n")
        f.write(f"Processing Time: {fiber_result.get('processing_time', 0):.3f}s\n\n")
        
        # Porosity analysis
        f.write("POROSITY ANALYSIS RESULTS\n")
        f.write("-" * 27 + "\n")
        porosity_result = result.get('porosity_analysis', {})
        if porosity_result and 'porosity_metrics' in porosity_result:
            pm = porosity_result['porosity_metrics']
            f.write(f"Status: Successful\n")
            f.write(f"Total Porosity: {pm.get('total_porosity_percent', 0):.2f}%\n")
            f.write(f"Pore Count: {pm.get('pore_count', 0)}\n")
            f.write(f"Average Pore Size: {pm.get('average_pore_size_um2', 0):.2f} μm²\n")
            f.write(f"Pore Density: {pm.get('pore_density_per_mm2', 0):.1f} pores/mm²\n")
            
            if 'analysis_quality' in porosity_result:
                aq = porosity_result['analysis_quality']
                f.write(f"Analysis Quality: {aq.get('overall_quality', 'unknown')}\n")
        else:
            f.write(f"Status: Failed or Not Available\n")
            error = porosity_result.get('error', 'Unknown') if porosity_result else 'Not performed'
            f.write(f"Reason: {error}\n")
        f.write(f"Processing Time: {result.get('porosity_processing_time', 0):.3f}s\n\n")
    
    def _write_material_characterization(self, f, comprehensive: Dict):
        """Write material characterization to report."""
        
        f.write("MATERIAL CHARACTERIZATION\n")
        f.write("-" * 26 + "\n")
        
        char = comprehensive.get('material_characterization', {})
        f.write(f"Primary Structure: {char.get('primary_structure', 'Unknown')}\n")
        f.write(f"Structure Confidence: {char.get('structure_confidence', 0):.3f}\n")
        f.write(f"Porosity Level: {char.get('porosity_level', 'unknown').title()}\n")
        f.write(f"Quality Impact: {char.get('quality_impact', 'unknown').replace('_', ' ').title()}\n")
        f.write(f"Scale Calibrated: {'Yes' if char.get('scale_calibrated', False) else 'No'}\n\n")
        
        # Physical measurements
        measurements = comprehensive.get('physical_measurements', {})
        if 'fiber_statistics' in measurements:
            f.write("PHYSICAL MEASUREMENTS\n")
            f.write("-" * 21 + "\n")
            fs = measurements['fiber_statistics']
            f.write(f"Fiber Count: {fs.get('count', 0)}\n")
            f.write(f"Mean Diameter: {fs.get('mean_diameter_um', 0):.1f} ± {fs.get('std_diameter_um', 0):.1f} μm\n")
            f.write(f"Diameter Range: {fs.get('min_diameter_um', 0):.1f} - {fs.get('max_diameter_um', 0):.1f} μm\n")
            f.write(f"Median Diameter: {fs.get('median_diameter_um', 0):.1f} μm\n\n")
        
        # Applications
        apps = char.get('recommended_applications', [])
        if apps:
            f.write("RECOMMENDED APPLICATIONS\n")
            f.write("-" * 25 + "\n")
            for i, app in enumerate(apps, 1):
                f.write(f"{i}. {app.replace('_', ' ').title()}\n")
            f.write("\n")
    
    def _write_recommendations(self, f, result: Dict, comprehensive: Dict):
        """Write analysis recommendations to report."""
        
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 15 + "\n")
        
        quality = comprehensive.get('analysis_quality', 'unknown')
        
        if quality in ['excellent', 'good']:
            f.write("✓ Analysis quality is sufficient for quantitative conclusions\n")
            f.write("✓ Results can be used for material optimization\n")
            f.write("✓ Data is suitable for comparison studies\n")
        elif quality == 'moderate':
            f.write("△ Analysis quality is moderate - trends are reliable\n")
            f.write("△ Consider additional validation for critical decisions\n")
            f.write("△ Absolute values should be interpreted with caution\n")
        else:
            f.write("✗ Analysis quality is low - results should be validated\n")
            f.write("✗ Consider re-imaging with better conditions\n")
            f.write("✗ Check imaging parameters and sample preparation\n")
        
        f.write("\n")
        
        # Processing recommendations
        porosity_result = result.get('porosity_analysis', {})
        if porosity_result and 'porosity_metrics' in porosity_result:
            porosity_percent = porosity_result['porosity_metrics'].get('total_porosity_percent', 0)
            
            if porosity_percent > 10:
                f.write("PROCESS OPTIMIZATION SUGGESTIONS:\n")
                f.write("• High porosity detected - consider optimizing spinning conditions\n")
                f.write("• Check for process-induced defects\n")
                f.write("• Evaluate temperature and pressure parameters\n")
            elif porosity_percent < 1:
                f.write("PROCESS ASSESSMENT:\n")
                f.write("• Low porosity indicates good structural integrity\n")
                f.write("• Current process conditions appear optimal\n")
                f.write("• Material suitable for structural applications\n")
        
        f.write("\n")
        f.write("For detailed analysis data, refer to the accompanying Excel file.\n")
    
    def _print_comprehensive_summary(self, result: Dict):
        """Print comprehensive analysis summary."""
        
        print(f"\n📋 COMPREHENSIVE ANALYSIS SUMMARY")
        print("=" * 50)
        
        comprehensive = result.get('comprehensive_metrics', {})
        print(f"Analysis Quality: {comprehensive.get('analysis_quality', 'unknown').title()}")
        print(f"Quality Score: {comprehensive.get('quality_score', 0):.2f}/1.0")
        print(f"Total Processing Time: {result.get('total_processing_time', 0):.2f}s")
        
        # Module results
        modules = result.get('modules_used', [])
        print(f"Modules Used: {', '.join(modules)}")
        
        # Key results
        scale_result = result.get('scale_detection', {})
        if scale_result.get('scale_detected', False):
            print(f"Scale: {scale_result['micrometers_per_pixel']:.4f} μm/pixel ✅")
        else:
            print(f"Scale: Detection failed ❌")
        
        fiber_result = result.get('fiber_detection', {})
        print(f"Fiber Type: {fiber_result.get('fiber_type', 'unknown')}")
        print(f"Fiber Confidence: {fiber_result.get('confidence', 0):.3f}")
        
        porosity_result = result.get('porosity_analysis', {})
        if porosity_result and 'porosity_metrics' in porosity_result:
            pm = porosity_result['porosity_metrics']
            print(f"Porosity: {pm.get('total_porosity_percent', 0):.2f}%")
            print(f"Pore Count: {pm.get('pore_count', 0)}")
        else:
            print(f"Porosity: Analysis failed or unavailable")
        
        # Material assessment
        char = comprehensive.get('material_characterization', {})
        apps = char.get('recommended_applications', [])
        if apps:
            print(f"Applications: {', '.join(apps[:2])}")
    
    def batch_analyze(self, image_directory: str, 
                     output_dir: Optional[str] = None) -> Dict:
        """
        Perform comprehensive analysis on multiple images.
        
        Args:
            image_directory: Directory containing SEM images
            output_dir: Output directory for results
            
        Returns:
            Dictionary containing batch analysis results
        """
        print(f"🧪 COMPREHENSIVE BATCH ANALYSIS")
        print("=" * 60)
        
        # Setup directories
        image_dir = Path(image_directory)
        if not image_dir.exists():
            print(f"❌ Image directory not found: {image_dir}")
            return {'error': f'Directory not found: {image_dir}'}
        
        if output_dir is None:
            output_dir = image_dir.parent / 'comprehensive_batch_analysis'
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
        
        # Process each image
        results = []
        successful = 0
        total_time = 0
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] 📸 {image_path.name}")
            print("-" * 50)
            
            result = self.analyze_comprehensive(str(image_path), output_dir)
            results.append(result)
            
            total_time += result.get('total_processing_time', 0)
            if result['success']:
                successful += 1
        
        # Generate batch summary
        summary = self._generate_batch_summary(results, image_dir, output_dir, total_time)
        
        # Save batch results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_json = output_dir / f'batch_analysis_results_{timestamp}.json'
        
        with open(batch_json, 'w') as f:
            json.dump(self._prepare_for_json(summary), f, indent=2, default=str)
        
        print(f"\n🎯 BATCH ANALYSIS COMPLETE!")
        print(f"📊 Success Rate: {successful}/{len(image_files)} ({successful/len(image_files)*100:.1f}%)")
        print(f"⏱️ Total Time: {total_time:.2f} seconds")
        print(f"💾 Results saved to: {output_dir}")
        
        return summary
    
    def _generate_batch_summary(self, results: List[Dict], image_dir: Path,
                               output_dir: Path, total_time: float) -> Dict:
        """Generate comprehensive batch analysis summary."""
        
        successful_results = [r for r in results if r['success']]
        
        summary = {
            'batch_info': {
                'timestamp': datetime.now().isoformat(),
                'input_directory': str(image_dir),
                'output_directory': str(output_dir),
                'total_images': len(results),
                'successful_analyses': len(successful_results),
                'success_rate': len(successful_results) / len(results) * 100 if results else 0,
                'total_processing_time': total_time,
                'average_time_per_image': total_time / len(results) if results else 0
            },
            'aggregate_statistics': {},
            'individual_results': results
        }
        
        if successful_results:
            # Aggregate fiber type statistics
            fiber_types = [r.get('fiber_detection', {}).get('fiber_type', 'unknown') 
                          for r in successful_results]
            from collections import Counter
            type_counts = Counter(fiber_types)
            
            # Aggregate porosity statistics
            porosities = []
            for r in successful_results:
                porosity_data = r.get('porosity_analysis', {})
                if porosity_data and 'porosity_metrics' in porosity_data:
                    porosities.append(porosity_data['porosity_metrics'].get('total_porosity_percent', 0))
            
            # Aggregate quality scores
            quality_scores = []
            for r in successful_results:
                comprehensive = r.get('comprehensive_metrics', {})
                if 'quality_score' in comprehensive:
                    quality_scores.append(comprehensive['quality_score'])
            
            summary['aggregate_statistics'] = {
                'fiber_type_distribution': dict(type_counts),
                'porosity_statistics': {
                    'mean': np.mean(porosities) if porosities else 0,
                    'std': np.std(porosities) if porosities else 0,
                    'min': np.min(porosities) if porosities else 0,
                    'max': np.max(porosities) if porosities else 0,
                    'count': len(porosities)
                } if porosities else None,
                'quality_statistics': {
                    'mean': np.mean(quality_scores) if quality_scores else 0,
                    'std': np.std(quality_scores) if quality_scores else 0,
                    'min': np.min(quality_scores) if quality_scores else 0,
                    'max': np.max(quality_scores) if quality_scores else 0,
                    'count': len(quality_scores)
                } if quality_scores else None
            }
        
        return summary


def main():
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(
        description='Comprehensive SEM Fiber Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python comprehensive_analyzer_main.py --image sample.jpg
  python comprehensive_analyzer_main.py --batch sample_images/
  python comprehensive_analyzer_main.py --image sample.jpg --config config.json
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
        
        result = analyzer.analyze_comprehensive(str(image_path), args.output)
        
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
        
        summary = analyzer.batch_analyze(str(batch_dir), args.output)
        
        if 'error' not in summary:
            batch_info = summary['batch_info']
            print(f"\n🎯 Batch Analysis Summary:")
            print(f"Success Rate: {batch_info['success_rate']:.1f}%")
            print(f"Average Time: {batch_info['average_time_per_image']:.2f}s per image")
    
    else:
        # Default: try to find and analyze one image
        sample_dir = Path("sample_images")
        if sample_dir.exists():
            image_files = (list(sample_dir.glob("*.jpg")) + 
                          list(sample_dir.glob("*.png")) + 
                          list(sample_dir.glob("*.tif")))
            if image_files:
                print("No specific action specified. Analyzing first available image...")
                result = analyzer.analyze_comprehensive(str(image_files[0]))
                
                if result['success']:
                    print(f"\n✅ Demo analysis successful!")
                else:
                    print(f"\n❌ Demo analysis failed!")
            else:
                print("No images found in sample_images/")
        else:
            print("Usage: python comprehensive_analyzer_main.py --help")


if __name__ == "__main__":
    main()