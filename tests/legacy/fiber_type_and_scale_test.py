#!/usr/bin/env python3
"""
Integrated Test Script - Scale Detection + Fiber Type Detection
Tests the updated scale detection module with the existing fiber type detection.

This script demonstrates:
1. Loading SEM images
2. Detecting scale bars with the optimized module
3. Detecting fiber types (hollow fiber vs filament)
4. Combining results for comprehensive analysis

Usage:
    python integrated_test_script.py
    python integrated_test_script.py --image path/to/image.jpg
    python integrated_test_script.py --batch  # Process all images in sample_images/
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import argparse
import time
from typing import Dict, List, Tuple, Optional, Any

# Setup paths for module imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "modules"))

# Import analysis modules
print("🔧 Importing analysis modules...")

try:
    # Import the updated scale detection module
    from modules.scale_detection import ScaleBarDetector, detect_scale_bar, detect_scale_factor_only
    print("✅ Optimized scale detection module imported successfully")
except ImportError as e:
    print(f"❌ Could not import optimized scale detection: {e}")
    sys.exit(1)

try:
    # Import image preprocessing
    from modules.image_preprocessing import load_image, preprocess_pipeline
    print("✅ Image preprocessing module imported")
except ImportError as e:
    print(f"⚠️ Could not import image preprocessing: {e}")
    print("Will use basic OpenCV loading")

try:
    # Import fiber type detection
    from modules.fiber_type_detection import FiberTypeDetector, detect_fiber_type, visualize_fiber_type_analysis
    print("✅ Fiber type detection module imported")
except ImportError as e:
    print(f"❌ Could not import fiber type detection: {e}")
    sys.exit(1)

class IntegratedAnalyzer:
    """
    Integrated analyzer combining optimized scale detection with fiber type detection.
    """
    
    def __init__(self, ocr_backend=None, debug=True):
        """
        Initialize the integrated analyzer.
        
        Args:
            ocr_backend: OCR backend for scale detection ('rapidocr', 'easyocr', or None)
            debug: Enable debug output
        """
        self.debug = debug
        
        # Initialize detectors
        self.scale_detector = ScaleBarDetector(ocr_backend=ocr_backend)
        self.fiber_detector = FiberTypeDetector()
        
        if debug:
            print(f"🔬 Integrated Analyzer initialized")
            print(f"   Scale detection: {self.scale_detector.ocr_backend or 'legacy'} backend")
            print(f"   Fiber detection: Adaptive algorithms")
    
    def analyze_single_image(self, image_path: str, save_debug: bool = True) -> Dict[str, Any]:
        """
        Analyze a single SEM image for both scale and fiber type.
        
        Args:
            image_path: Path to SEM image
            save_debug: Save debug visualizations
            
        Returns:
            Dictionary containing complete analysis results
        """
        start_time = time.time()
        
        if self.debug:
            print(f"\n🔍 ANALYZING: {Path(image_path).name}")
            print("=" * 60)
        
        result = {
            'image_path': str(image_path),
            'image_name': Path(image_path).name,
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'error': None,
            'processing_time': 0.0
        }
        
        try:
            # Step 1: Load image
            if self.debug:
                print("📸 Step 1: Loading image...")
            
            try:
                # Try using the preprocessing module if available
                image = load_image(str(image_path))
            except:
                # Fallback to OpenCV
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            if self.debug:
                print(f"   ✅ Image loaded: {image.shape}")
                print(f"   File size: {os.path.getsize(image_path) / (1024*1024):.2f} MB")
            
            result.update({
                'image_shape': image.shape,
                'image_size_mb': os.path.getsize(image_path) / (1024 * 1024)
            })
            
            # Step 2: Scale detection
            if self.debug:
                print("📏 Step 2: Detecting scale bar...")
            
            scale_start = time.time()
            scale_result = detect_scale_bar(
                str(image_path),  # Pass path for better error handling
                debug=self.debug,
                save_debug_image=save_debug,
                output_dir=Path(image_path).parent / "debug_output" if save_debug else None
            )
            scale_time = time.time() - scale_start
            
            result['scale_detection'] = scale_result
            result['scale_processing_time'] = scale_time
            
            if scale_result['scale_detected']:
                scale_factor = scale_result['micrometers_per_pixel']
                if self.debug:
                    print(f"   ✅ Scale detected: {scale_factor:.4f} μm/pixel")
                    scale_info = scale_result.get('scale_info', {})
                    print(f"   Scale text: '{scale_info.get('text', 'N/A')}'")
                    print(f"   Method: {scale_result.get('detection_method', 'unknown')}")
                    print(f"   Confidence: {scale_result.get('confidence', 0):.2%}")
                    print(f"   Processing time: {scale_time:.3f}s")
            else:
                scale_factor = 1.0  # Fallback
                if self.debug:
                    print(f"   ⚠️ Scale detection failed: {scale_result.get('error', 'Unknown error')}")
                    print(f"   Using fallback scale: 1.0 μm/pixel")
                    print(f"   Processing time: {scale_time:.3f}s")
            
            # Step 3: Preprocess for fiber detection
            if self.debug:
                print("🔧 Step 3: Preprocessing for fiber detection...")
            
            # Enhanced preprocessing for fiber detection
            preprocessed = self._preprocess_for_fiber_detection(image)
            
            # Step 4: Fiber type detection
            if self.debug:
                print("🧬 Step 4: Detecting fiber type...")
            
            fiber_start = time.time()
            fiber_type, confidence, analysis_data = self.fiber_detector.classify_fiber_type(preprocessed)
            fiber_time = time.time() - fiber_start
            
            result['fiber_detection'] = {
                'fiber_type': fiber_type,
                'confidence': confidence,
                'total_fibers': analysis_data.get('total_fibers', 0),
                'hollow_fibers': analysis_data.get('hollow_fibers', 0),
                'filaments': analysis_data.get('filaments', 0),
                'thresholds_used': analysis_data.get('thresholds', {}),
                'classification_method': analysis_data.get('classification_method', 'unknown'),
                'processing_time': fiber_time
            }
            
            if self.debug:
                print(f"   ✅ Fiber type: {fiber_type} (confidence: {confidence:.3f})")
                print(f"   Total fibers detected: {analysis_data.get('total_fibers', 0)}")
                print(f"   Hollow fibers: {analysis_data.get('hollow_fibers', 0)}")
                print(f"   Filaments: {analysis_data.get('filaments', 0)}")
                print(f"   Method: {analysis_data.get('classification_method', 'unknown')}")
                print(f"   Processing time: {fiber_time:.3f}s")
            
            # Step 5: Combined analysis
            if self.debug:
                print("📊 Step 5: Generating combined analysis...")
            
            combined_analysis = self._generate_combined_analysis(
                scale_result, fiber_type, confidence, analysis_data, scale_factor
            )
            
            result['combined_analysis'] = combined_analysis
            
            # Step 6: Create visualizations if requested
            if save_debug:
                if self.debug:
                    print("🎨 Step 6: Creating visualizations...")
                
                viz_paths = self._create_visualizations(
                    image, preprocessed, analysis_data, scale_result, result
                )
                result['visualization_paths'] = viz_paths
            
            # Mark as successful
            result['success'] = True
            result['processing_time'] = time.time() - start_time
            
            if self.debug:
                print(f"✅ Analysis completed successfully!")
                print(f"⏱️ Total processing time: {result['processing_time']:.2f} seconds")
                print(f"   Scale detection: {scale_time:.3f}s")
                print(f"   Fiber detection: {fiber_time:.3f}s")
        
        except Exception as e:
            result['error'] = str(e)
            result['processing_time'] = time.time() - start_time
            
            if self.debug:
                print(f"❌ Analysis failed: {e}")
                import traceback
                traceback.print_exc()
        
        return result
    
    def _preprocess_for_fiber_detection(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for optimal fiber detection."""
        
        # Use bilateral filter to preserve edges while reducing noise
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        return enhanced
    
    def _generate_combined_analysis(self, scale_result: Dict, fiber_type: str, 
                                   confidence: float, analysis_data: Dict,
                                   scale_factor: float) -> Dict:
        """Generate combined analysis metrics."""
        
        combined = {
            'fiber_type': fiber_type,
            'fiber_confidence': confidence,
            'scale_factor_um_per_pixel': scale_factor,
            'scale_detected': scale_result['scale_detected'],
            'scale_confidence': scale_result.get('confidence', 0.0)
        }
        
        # Calculate overall analysis quality
        quality_score = 0.0
        quality_factors = []
        
        # Scale detection quality (40% weight)
        if scale_result['scale_detected']:
            scale_conf = scale_result.get('confidence', 0.0)
            quality_score += scale_conf * 0.4
            quality_factors.append(f"Scale: {scale_conf:.2f}")
        else:
            quality_factors.append("Scale: failed")
        
        # Fiber detection quality (40% weight)
        fiber_conf = confidence
        quality_score += fiber_conf * 0.4
        quality_factors.append(f"Fiber: {fiber_conf:.2f}")
        
        # Fiber count quality (20% weight)
        total_fibers = analysis_data.get('total_fibers', 0)
        if total_fibers > 0:
            fiber_count_score = min(0.2, total_fibers * 0.05)  # Max 0.2 for 4+ fibers
        else:
            fiber_count_score = 0.0
        
        quality_score += fiber_count_score
        quality_factors.append(f"Count: {fiber_count_score:.2f}")
        
        # Determine quality level
        if quality_score >= 0.8:
            quality_level = "excellent"
        elif quality_score >= 0.6:
            quality_level = "good"
        elif quality_score >= 0.4:
            quality_level = "moderate"
        else:
            quality_level = "poor"
        
        combined.update({
            'analysis_quality': quality_level,
            'quality_score': quality_score,
            'quality_factors': quality_factors
        })
        
        # Add physical measurements if we have scale
        if scale_result['scale_detected'] and total_fibers > 0:
            physical_measurements = self._calculate_physical_measurements(
                analysis_data, scale_factor
            )
            combined['physical_measurements'] = physical_measurements
        
        return combined
    
    def _calculate_physical_measurements(self, analysis_data: Dict, 
                                       scale_factor: float) -> Dict:
        """Calculate physical measurements from fiber analysis."""
        
        measurements = {}
        
        # Extract individual fiber results if available
        individual_results = analysis_data.get('individual_results', [])
        
        if individual_results:
            fiber_areas_um2 = []
            fiber_diameters_um = []
            lumen_areas_um2 = []
            
            for result in individual_results:
                fiber_props = result.get('fiber_properties', {})
                area_pixels = fiber_props.get('area', 0)
                
                if area_pixels > 0:
                    # Convert to physical units
                    area_um2 = area_pixels * (scale_factor ** 2)
                    diameter_um = 2 * np.sqrt(area_um2 / np.pi)  # Equivalent circular diameter
                    
                    fiber_areas_um2.append(area_um2)
                    fiber_diameters_um.append(diameter_um)
                    
                    # Lumen measurements for hollow fibers
                    if result.get('has_lumen', False):
                        lumen_props = result.get('lumen_properties', {})
                        lumen_area_pixels = lumen_props.get('area', 0)
                        if lumen_area_pixels > 0:
                            lumen_area_um2 = lumen_area_pixels * (scale_factor ** 2)
                            lumen_areas_um2.append(lumen_area_um2)
            
            # Calculate statistics
            if fiber_areas_um2:
                measurements.update({
                    'fiber_count': len(fiber_areas_um2),
                    'mean_fiber_area_um2': np.mean(fiber_areas_um2),
                    'std_fiber_area_um2': np.std(fiber_areas_um2),
                    'mean_fiber_diameter_um': np.mean(fiber_diameters_um),
                    'std_fiber_diameter_um': np.std(fiber_diameters_um),
                    'min_fiber_diameter_um': np.min(fiber_diameters_um),
                    'max_fiber_diameter_um': np.max(fiber_diameters_um)
                })
            
            if lumen_areas_um2:
                measurements.update({
                    'lumen_count': len(lumen_areas_um2),
                    'mean_lumen_area_um2': np.mean(lumen_areas_um2),
                    'std_lumen_area_um2': np.std(lumen_areas_um2)
                })
        
        return measurements
    
    def _create_visualizations(self, original_image: np.ndarray, preprocessed_image: np.ndarray,
                              analysis_data: Dict, scale_result: Dict, result: Dict) -> Dict:
        """Create analysis visualizations."""
        
        viz_paths = {}
        
        try:
            # Create output directory
            image_path = Path(result['image_path'])
            output_dir = image_path.parent / "analysis_output"
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = image_path.stem
            
            # 1. Fiber type analysis visualization
            if 'individual_results' in analysis_data:
                try:
                    fig_path = output_dir / f"{base_name}_fiber_analysis_{timestamp}.png"
                    visualize_fiber_type_analysis(original_image, analysis_data, figsize=(15, 10))
                    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    viz_paths['fiber_analysis'] = str(fig_path)
                    
                    if self.debug:
                        print(f"   📊 Fiber analysis saved: {fig_path.name}")
                except Exception as e:
                    if self.debug:
                        print(f"   ⚠️ Failed to save fiber analysis: {e}")
            
            # 2. Combined summary visualization
            try:
                fig_path = output_dir / f"{base_name}_summary_{timestamp}.png"
                self._create_summary_plot(original_image, result, fig_path)
                viz_paths['summary'] = str(fig_path)
                
                if self.debug:
                    print(f"   📈 Summary plot saved: {fig_path.name}")
            except Exception as e:
                if self.debug:
                    print(f"   ⚠️ Failed to save summary plot: {e}")
            
        except Exception as e:
            if self.debug:
                print(f"   ⚠️ Visualization creation failed: {e}")
        
        return viz_paths
    
    def _create_summary_plot(self, image: np.ndarray, result: Dict, output_path: Path):
        """Create a summary plot showing key results."""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original SEM Image')
        axes[0, 0].axis('off')
        
        # Scale detection result
        scale_result = result.get('scale_detection', {})
        if scale_result.get('scale_detected', False):
            scale_text = f"✅ Scale Detected\n"
            scale_text += f"{scale_result['micrometers_per_pixel']:.4f} μm/pixel\n"
            scale_info = scale_result.get('scale_info', {})
            scale_text += f"Text: '{scale_info.get('text', 'N/A')}'\n"
            scale_text += f"Confidence: {scale_result.get('confidence', 0):.1%}"
            color = 'green'
        else:
            scale_text = f"❌ Scale Detection Failed\n"
            scale_text += f"Error: {scale_result.get('error', 'Unknown')}"
            color = 'red'
        
        axes[0, 1].text(0.1, 0.5, scale_text, transform=axes[0, 1].transAxes,
                       fontsize=11, verticalalignment='center', color=color,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
        axes[0, 1].set_title('Scale Detection Results')
        axes[0, 1].axis('off')
        
        # Fiber detection result
        fiber_result = result.get('fiber_detection', {})
        fiber_text = f"Fiber Type: {fiber_result.get('fiber_type', 'Unknown')}\n"
        fiber_text += f"Confidence: {fiber_result.get('confidence', 0):.3f}\n"
        fiber_text += f"Total Fibers: {fiber_result.get('total_fibers', 0)}\n"
        fiber_text += f"Hollow: {fiber_result.get('hollow_fibers', 0)}\n"
        fiber_text += f"Filaments: {fiber_result.get('filaments', 0)}\n"
        fiber_text += f"Method: {fiber_result.get('classification_method', 'Unknown')}"
        
        confidence = fiber_result.get('confidence', 0)
        if confidence > 0.7:
            fiber_color = 'green'
        elif confidence > 0.4:
            fiber_color = 'orange'
        else:
            fiber_color = 'red'
        
        axes[1, 0].text(0.1, 0.5, fiber_text, transform=axes[1, 0].transAxes,
                       fontsize=11, verticalalignment='center', color=fiber_color,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
        axes[1, 0].set_title('Fiber Type Detection Results')
        axes[1, 0].axis('off')
        
        # Combined analysis summary
        combined = result.get('combined_analysis', {})
        summary_text = f"Analysis Quality: {combined.get('analysis_quality', 'Unknown')}\n"
        summary_text += f"Quality Score: {combined.get('quality_score', 0):.2f}\n\n"
        
        if 'physical_measurements' in combined:
            pm = combined['physical_measurements']
            summary_text += f"Physical Measurements:\n"
            summary_text += f"Mean Diameter: {pm.get('mean_fiber_diameter_um', 0):.1f} μm\n"
            summary_text += f"Std Diameter: {pm.get('std_fiber_diameter_um', 0):.1f} μm\n"
            if 'mean_lumen_area_um2' in pm:
                summary_text += f"Mean Lumen Area: {pm['mean_lumen_area_um2']:.1f} μm²\n"
        
        processing_time = result.get('processing_time', 0)
        summary_text += f"\nProcessing Time: {processing_time:.2f}s"
        
        axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
        axes[1, 1].set_title('Analysis Summary')
        axes[1, 1].axis('off')
        
        plt.suptitle(f"SEM Fiber Analysis: {result['image_name']}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def batch_analyze(self, image_directory: str = "sample_images", 
                     output_dir: Optional[str] = None,
                     save_debug: bool = True) -> Dict:
        """
        Analyze all images in a directory.
        
        Args:
            image_directory: Directory containing SEM images
            output_dir: Output directory for results
            save_debug: Save debug visualizations
            
        Returns:
            Dictionary containing batch analysis results
        """
        print(f"🧪 BATCH INTEGRATED ANALYSIS")
        print("=" * 60)
        
        # Setup directories
        image_dir = Path(image_directory)
        if not image_dir.exists():
            print(f"❌ Image directory not found: {image_dir}")
            return {'error': f'Directory not found: {image_dir}'}
        
        if output_dir is None:
            output_dir = image_dir.parent / 'integrated_analysis_results'
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
        
        print(f"📁 Analyzing {len(image_files)} images from: {image_dir}")
        print(f"📊 Results will be saved to: {output_dir}")
        
        # Process each image
        results = []
        successful = 0
        scale_successful = 0
        fiber_successful = 0
        total_time = 0
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] 📸 Processing: {image_path.name}")
            print("-" * 40)
            
            result = self.analyze_single_image(str(image_path), save_debug=save_debug)
            results.append(result)
            
            total_time += result.get('processing_time', 0)
            
            if result['success']:
                successful += 1
                
                if result.get('scale_detection', {}).get('scale_detected', False):
                    scale_successful += 1
                
                fiber_conf = result.get('fiber_detection', {}).get('confidence', 0)
                if fiber_conf > 0.5:  # Reasonable confidence threshold
                    fiber_successful += 1
        
        # Generate summary
        print(f"\n" + "=" * 60)
        print("📊 BATCH ANALYSIS SUMMARY")
        print("=" * 60)
        
        overall_success_rate = successful / len(image_files) * 100 if image_files else 0
        scale_success_rate = scale_successful / len(image_files) * 100 if image_files else 0
        fiber_success_rate = fiber_successful / len(image_files) * 100 if image_files else 0
        avg_time = total_time / len(image_files) if image_files else 0
        
        print(f"Total images: {len(image_files)}")
        print(f"Overall successful: {successful} ({overall_success_rate:.1f}%)")
        print(f"Scale detection successful: {scale_successful} ({scale_success_rate:.1f}%)")
        print(f"Fiber detection successful: {fiber_successful} ({fiber_success_rate:.1f}%)")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Average time per image: {avg_time:.3f} seconds")
        
        # Analyze results
        successful_results = [r for r in results if r['success']]
        
        if successful_results:
            # Scale factor analysis
            scale_factors = []
            for r in successful_results:
                if r.get('scale_detection', {}).get('scale_detected', False):
                    scale_factors.append(r['scale_detection']['micrometers_per_pixel'])
            
            if scale_factors:
                print(f"\n📏 Scale Factor Analysis:")
                print(f"   Range: {min(scale_factors):.4f} - {max(scale_factors):.4f} μm/pixel")
                print(f"   Mean: {np.mean(scale_factors):.4f} μm/pixel")
            
            # Fiber type analysis
            fiber_types = []
            for r in successful_results:
                fiber_type = r.get('fiber_detection', {}).get('fiber_type', 'unknown')
                if fiber_type != 'unknown':
                    fiber_types.append(fiber_type)
            
            if fiber_types:
                from collections import Counter
                type_counts = Counter(fiber_types)
                print(f"\n🧬 Fiber Type Analysis:")
                for fiber_type, count in type_counts.items():
                    print(f"   {fiber_type}: {count} images")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        summary = {
            'timestamp': timestamp,
            'test_directory': str(image_dir),
            'output_directory': str(output_dir),
            'total_images': len(image_files),
            'overall_successful': successful,
            'scale_successful': scale_successful,
            'fiber_successful': fiber_successful,
            'success_rates': {
                'overall': overall_success_rate,
                'scale_detection': scale_success_rate,
                'fiber_detection': fiber_success_rate
            },
            'processing_time': {
                'total': total_time,
                'average': avg_time
            },
            'results': results
        }
        
        # Save as JSON
        json_file = output_dir / f'integrated_analysis_{timestamp}.json'
        try:
            with open(json_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"\n💾 Results saved to: {json_file}")
        except Exception as e:
            print(f"\n⚠️ Could not save results: {e}")
        
        print(f"\n🎯 BATCH ANALYSIS COMPLETE!")
        
        return summary

def main():
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(description='Integrated SEM Fiber Analysis Test')
    parser.add_argument('--image', '-i', help='Analyze single image')
    parser.add_argument('--batch', '-b', action='store_true', help='Analyze all images in sample_images/')
    parser.add_argument('--directory', '-d', default='sample_images', help='Directory containing images')
    parser.add_argument('--output', '-o', help='Output directory for results')
    parser.add_argument('--ocr-backend', choices=['rapidocr', 'easyocr'], help='OCR backend for scale detection')
    parser.add_argument('--no-debug', action='store_true', help='Disable debug visualizations')
    parser.add_argument('--quiet', '-q', action='store_true', help='Minimal output')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = IntegratedAnalyzer(
        ocr_backend=args.ocr_backend,
        debug=not args.quiet
    )
    
    save_debug = not args.no_debug
    
    if args.image:
        # Single image analysis
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"❌ Image not found: {image_path}")
            return
        
        print("🔍 SINGLE IMAGE ANALYSIS")
        print("=" * 40)
        
        result = analyzer.analyze_single_image(str(image_path), save_debug=save_debug)
        
        if result['success']:
            print(f"\n🎉 Analysis completed successfully!")
            combined = result.get('combined_analysis', {})
            print(f"Quality: {combined.get('analysis_quality', 'unknown')}")
            
            # Print key results
            scale_result = result.get('scale_detection', {})
            if scale_result.get('scale_detected', False):
                print(f"Scale: {scale_result['micrometers_per_pixel']:.4f} μm/pixel")
            
            fiber_result = result.get('fiber_detection', {})
            print(f"Fiber type: {fiber_result.get('fiber_type', 'unknown')}")
            print(f"Confidence: {fiber_result.get('confidence', 0):.3f}")
            
        else:
            print(f"\n❌ Analysis failed: {result.get('error', 'Unknown error')}")
    
    elif args.batch:
        # Batch analysis
        summary = analyzer.batch_analyze(
            image_directory=args.directory,
            output_dir=args.output,
            save_debug=save_debug
        )
        
        if 'error' not in summary:
            success_rates = summary['success_rates']
            print(f"\n🎯 Final Results:")
            print(f"Overall success: {success_rates['overall']:.1f}%")
            print(f"Scale detection: {success_rates['scale_detection']:.1f}%")
            print(f"Fiber detection: {success_rates['fiber_detection']:.1f}%")
    
    else:
        # Default: try to find and analyze one image
        sample_dir = Path("sample_images")
        if sample_dir.exists():
            image_files = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png")) + list(sample_dir.glob("*.tif"))
            if image_files:
                print("No specific action specified. Analyzing first available image...")
                result = analyzer.analyze_single_image(str(image_files[0]), save_debug=save_debug)
                
                if result['success']:
                    print(f"\n✅ Test successful! Check output for detailed results.")
                else:
                    print(f"\n❌ Test failed: {result.get('error', 'Unknown error')}")
            else:
                print("No images found in sample_images/. Use --image or --batch options.")
        else:
            print("No sample_images/ directory found. Use --image to specify an image file.")
            print("Usage: python integrated_test_script.py --image path/to/image.jpg")

if __name__ == "__main__":
    main()