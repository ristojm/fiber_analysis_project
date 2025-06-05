#!/usr/bin/env python3
"""
Comprehensive SEM Fiber Analyzer - Main Application
Production application for complete fiber characterization

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
    # Enhanced porosity analysis
    from modules.porosity_analysis import EnhancedPorosityAnalyzer, analyze_fiber_porosity_enhanced
    print("✅ Enhanced porosity module loaded")
    POROSITY_AVAILABLE = True
    ENHANCED_POROSITY = True
except ImportError:
    print("⚠️ Enhanced porosity module not found, trying basic version...")
    try:
        from modules.porosity_analysis import PorosityAnalyzer
        POROSITY_AVAILABLE = True
        ENHANCED_POROSITY = False
        print("✅ Basic porosity module loaded")
    except ImportError:
        print("❌ No porosity analysis available")
        POROSITY_AVAILABLE = False
        ENHANCED_POROSITY = False


class ComprehensiveFiberAnalyzer:
    """
    Main application class that orchestrates all analysis modules
    for comprehensive SEM fiber characterization.
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
        
        if POROSITY_AVAILABLE:
            if ENHANCED_POROSITY:
                self.porosity_analyzer = EnhancedPorosityAnalyzer(
                    config=self.config.get('porosity_analysis', {})
                )
            else:
                self.porosity_analyzer = PorosityAnalyzer()
        else:
            self.porosity_analyzer = None
        
        if self.debug:
            print(f"🔬 Comprehensive Fiber Analyzer initialized")
            print(f"   Scale detection: {self.scale_detector.ocr_backend or 'legacy'}")
            print(f"   Fiber detection: Adaptive algorithms")
            print(f"   Porosity analysis: {'Enhanced' if ENHANCED_POROSITY else 'Basic' if POROSITY_AVAILABLE else 'Not available'}")
    
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
            'modules_used': []
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
            
            # Step 4: Porosity analysis
            porosity_result = None
            if self.porosity_analyzer and POROSITY_AVAILABLE:
                if self.debug:
                    print("🕳️  Step 4: Porosity analysis...")
                
                step_start = time.time()
                
                # Get fiber mask from detection results
                fiber_mask = fiber_analysis_data.get('fiber_mask', np.zeros_like(image, dtype=bool))
                
                if np.sum(fiber_mask) > 1000:  # Minimum area threshold
                    try:
                        if ENHANCED_POROSITY:
                            # Enhanced porosity analyzer
                            porosity_result = self.porosity_analyzer.analyze_fiber_porosity(
                                preprocessed, 
                                fiber_mask.astype(np.uint8), 
                                scale_factor, 
                                fiber_type,
                                fiber_analysis_data
                            )
                        else:
                            # Basic porosity analyzer - need to implement basic interface
                            porosity_result = {
                                'porosity_metrics': {
                                    'total_porosity_percent': 0.0,
                                    'pore_count': 0,
                                    'average_pore_size_um2': 0.0,
                                    'pore_density_per_mm2': 0.0
                                },
                                'error': 'Basic porosity analysis not fully implemented'
                            }
                        
                        porosity_time = time.time() - step_start
                        result['porosity_analysis'] = porosity_result
                        result['porosity_processing_time'] = porosity_time
                        result['modules_used'].append('porosity_analysis')
                        
                        if self.debug and 'porosity_metrics' in porosity_result:
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
            quality_factors.append(f"Porosity: {porosity_quality:.2f}")
        else:
            quality_factors.append("Porosity: unavailable")
        
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
        
        return metrics
    
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
                    'Pore Density (/mm²)': pm.get('pore_density_per_mm2', 0)
                }])
                porosity_summary.to_excel(writer, sheet_name='Porosity_Analysis', index=False)
    
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
        
        # Quality assessment
        quality_factors = comprehensive.get('quality_factors', [])
        if quality_factors:
            quality_text = "Quality Factors:\n\n" + "\n".join(quality_factors)
        else:
            quality_text = "Quality assessment\nnot available"
        
        axes[1, 1].text(0.05, 0.95, quality_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen'))
        axes[1, 1].set_title('Quality Assessment', fontsize=12, fontweight='bold')
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
            f.write(f"Analysis Duration: {result.get('total_processing_time', 0):.2f} seconds\n\n")
            
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
        f.write("POROSITY ANALYSIS\n")
        f.write("-" * 17 + "\n")
        porosity_result = result.get('porosity_analysis', {})
        if porosity_result and 'porosity_metrics' in porosity_result:
            pm = porosity_result['porosity_metrics']
            f.write(f"Status: Successful\n")
            f.write(f"Total Porosity: {pm.get('total_porosity_percent', 0):.2f}%\n")
            f.write(f"Pore Count: {pm.get('pore_count', 0)}\n")
            f.write(f"Average Pore Size: {pm.get('average_pore_size_um2', 0):.2f} μm²\n")
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
            print(f"Porosity: {pm.get('total_porosity_percent', 0):.2f}%")
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
                'average_time_per_image': total_time / len(image_files) if image_files else 0
            },
            'individual_results': results
        }
        
        # Save batch results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_json = output_dir / f'batch_analysis_{timestamp}.json'
        
        with open(batch_json, 'w') as f:
            json.dump(self._prepare_for_json(summary), f, indent=2, default=str)
        
        print(f"\n🎯 BATCH ANALYSIS COMPLETE!")
        print(f"📊 Success Rate: {successful}/{len(image_files)} ({successful/len(image_files)*100:.1f}%)")
        print(f"⏱️ Total Time: {total_time:.2f} seconds")
        print(f"💾 Results saved to: {output_dir}")
        
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