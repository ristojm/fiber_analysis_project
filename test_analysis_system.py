#!/usr/bin/env python3
"""
Updated Test Script for SEM Fiber Analysis System
Tests with real images from sample_images folder and generates debug visualizations

- Uses actual SEM images from sample_images/
- Outputs debug images from each processing stage
- Tests comprehensive analyzer with batch processing
- Saves batch results to Excel
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import time
import traceback
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Setup paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "modules"))

print("🔧 SEM Fiber Analysis System - Real Image Test Suite")
print("=" * 70)
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print(f"Project root: {project_root}")

# Global variables to track available modules
MODULES_AVAILABLE = {}
OCR_BACKENDS = {}

def test_imports():
    """Test all module imports with detailed reporting"""
    global MODULES_AVAILABLE, OCR_BACKENDS
    
    print("\n📦 Testing module imports...")
    
    # Test core modules
    try:
        from modules.image_preprocessing import load_image, preprocess_pipeline, load_and_preprocess
        print("✅ Image Preprocessing: Available")
        MODULES_AVAILABLE['image_preprocessing'] = True
    except ImportError as e:
        print(f"❌ Image Preprocessing: {e}")
        MODULES_AVAILABLE['image_preprocessing'] = False
    
    try:
        from modules.scale_detection import ScaleBarDetector, detect_scale_bar
        print("✅ Scale Detection: Available")
        MODULES_AVAILABLE['scale_detection'] = True
    except ImportError as e:
        print(f"❌ Scale Detection: {e}")
        MODULES_AVAILABLE['scale_detection'] = False
    
    try:
        from modules.fiber_type_detection import FiberTypeDetector, detect_fiber_type
        print("✅ Fiber Type Detection: Available")
        MODULES_AVAILABLE['fiber_type_detection'] = True
    except ImportError as e:
        print(f"❌ Fiber Type Detection: {e}")
        MODULES_AVAILABLE['fiber_type_detection'] = False
    
    try:
        from modules.porosity_analysis import PorosityAnalyzer, analyze_fiber_porosity
        print("✅ Porosity Analysis: Available")
        MODULES_AVAILABLE['porosity_analysis'] = True
    except ImportError as e:
        print(f"❌ Porosity Analysis: {e}")
        MODULES_AVAILABLE['porosity_analysis'] = False
    
    try:
        from comprehensive_analyzer_main import ComprehensiveFiberAnalyzer
        print("✅ Comprehensive Analyzer: Available")
        MODULES_AVAILABLE['comprehensive_analyzer'] = True
    except ImportError as e:
        print(f"⚠️ Comprehensive Analyzer: {e}")
        MODULES_AVAILABLE['comprehensive_analyzer'] = False
    
    # Test OCR backends
    print("\n🔍 Checking OCR backends...")
    try:
        from rapidocr_onnxruntime import RapidOCR
        print("✅ RapidOCR: Available")
        OCR_BACKENDS['rapidocr'] = True
    except ImportError:
        print("⚠️ RapidOCR: Not available")
        OCR_BACKENDS['rapidocr'] = False
    
    try:
        import easyocr
        print("✅ EasyOCR: Available")
        OCR_BACKENDS['easyocr'] = True
    except ImportError:
        print("⚠️ EasyOCR: Not available")
        OCR_BACKENDS['easyocr'] = False
    
    return MODULES_AVAILABLE, OCR_BACKENDS

def find_sample_image():
    """Find a sample image from the sample_images folder"""
    print("\n📁 Looking for sample images...")
    
    # Look for sample_images folder
    sample_dirs = ["sample_images", "test", "images"]
    sample_extensions = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp"]
    
    for sample_dir in sample_dirs:
        sample_path = Path(sample_dir)
        if sample_path.exists():
            print(f"   Found directory: {sample_path}")
            
            for ext in sample_extensions:
                image_files = list(sample_path.glob(ext))
                image_files.extend(list(sample_path.glob(ext.upper())))
                
                if image_files:
                    selected_image = image_files[0]
                    print(f"✅ Selected sample image: {selected_image}")
                    print(f"   Available images: {len(image_files)} total")
                    return selected_image, list(set(image_files))  # Remove duplicates
    
    print("❌ No sample images found!")
    print("   Please add SEM images to one of these folders:")
    for sample_dir in sample_dirs:
        print(f"   - {sample_dir}/")
    
    return None, []

def create_debug_output_dir():
    """Create debug output directory"""
    debug_dir = Path("test_debug_output")
    debug_dir.mkdir(exist_ok=True)
    
    # Create timestamp subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = debug_dir / f"test_session_{timestamp}"
    session_dir.mkdir(exist_ok=True)
    
    print(f"📁 Debug output directory: {session_dir}")
    return session_dir

def test_image_preprocessing_with_debug(image_path, debug_dir, verbose=True):
    """Test image preprocessing and save debug images"""
    if not MODULES_AVAILABLE.get('image_preprocessing', False):
        return False, {'error': 'Image preprocessing module not available'}
    
    if verbose:
        print("\n📸 Testing Image Preprocessing with Debug Output...")
        print("-" * 55)
    
    try:
        from modules.image_preprocessing import load_image, preprocess_pipeline
        
        start_time = time.time()
        
        # Test preprocessing pipeline
        result = preprocess_pipeline(str(image_path))
        
        processing_time = time.time() - start_time
        
        if result.get('preprocessing_complete', False):
            if verbose:
                print(f"✅ Preprocessing successful!")
                print(f"   Original shape: {result['image_shape']}")
                print(f"   Processing steps: {len(result.get('processing_steps', []))}")
                print(f"   Processing time: {processing_time:.3f}s")
                
                for step in result.get('processing_steps', []):
                    print(f"     - {step}")
            
            # Save debug images
            debug_images = {
                'original': result.get('original'),
                'contrast_enhanced': result.get('contrast_enhanced'),
                'denoised': result.get('denoised'),
                'normalized': result.get('normalized'),
                'processed': result.get('processed')
            }
            
            # Create preprocessing debug visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, (name, img) in enumerate(debug_images.items()):
                if img is not None and i < len(axes):
                    axes[i].imshow(img, cmap='gray')
                    axes[i].set_title(f'{name.replace("_", " ").title()}')
                    axes[i].axis('off')
            
            # Hide unused subplots
            for j in range(len(debug_images), len(axes)):
                axes[j].axis('off')
            
            plt.suptitle(f'Image Preprocessing Steps - {Path(image_path).name}', fontsize=16)
            plt.tight_layout()
            
            debug_file = debug_dir / 'preprocessing_steps.png'
            plt.savefig(debug_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            if verbose:
                print(f"   💾 Debug image saved: {debug_file.name}")
            
            return True, result
        else:
            if verbose:
                print(f"❌ Preprocessing failed")
            return False, result
            
    except Exception as e:
        if verbose:
            print(f"💥 Preprocessing error: {e}")
            traceback.print_exc()
        return False, {'error': str(e)}

def test_scale_detection_with_debug(image, debug_dir, verbose=True):
    """Test scale detection and save debug images"""
    if not MODULES_AVAILABLE.get('scale_detection', False):
        return False, {'error': 'Scale detection module not available'}
    
    if verbose:
        print("\n📏 Testing Scale Detection with Debug Output...")
        print("-" * 50)
    
    try:
        from modules.scale_detection import detect_scale_bar
        
        start_time = time.time()
        
        # Test scale detection with debug output
        result = detect_scale_bar(
            image,
            use_enhanced=True,
            debug=False,  # Don't print debug info
            save_debug_image=True,
            output_dir=str(debug_dir)
        )
        
        processing_time = time.time() - start_time
        
        if result.get('scale_detected', False):
            if verbose:
                print(f"✅ Scale detection successful!")
                print(f"   Scale factor: {result['micrometers_per_pixel']:.4f} μm/pixel")
                print(f"   Method: {result.get('method_used', 'unknown')}")
                print(f"   Confidence: {result.get('confidence', 0):.2%}")
                print(f"   OCR backend: {result.get('ocr_backend', 'unknown')}")
                print(f"   Processing time: {processing_time:.3f}s")
                
                scale_info = result.get('scale_info', {})
                if scale_info:
                    print(f"   Scale text detected: '{scale_info.get('text', 'N/A')}'")
                    print(f"   Scale value: {scale_info.get('value', 0)} {scale_info.get('unit', '')}")
            
            # Check if debug image was saved
            if 'debug_image_path' in result:
                if verbose:
                    print(f"   💾 Scale debug image: {Path(result['debug_image_path']).name}")
            
            return True, result
        else:
            if verbose:
                print(f"❌ Scale detection failed: {result.get('error', 'Unknown error')}")
            return False, result
            
    except Exception as e:
        if verbose:
            print(f"💥 Scale detection error: {e}")
            traceback.print_exc()
        return False, {'error': str(e)}

def test_fiber_detection_with_debug(image, debug_dir, verbose=True):
    """Test fiber type detection and save debug images"""
    if not MODULES_AVAILABLE.get('fiber_type_detection', False):
        return False, {'error': 'Fiber type detection module not available'}
    
    if verbose:
        print("\n🧬 Testing Fiber Type Detection with Debug Output...")
        print("-" * 55)
    
    try:
        from modules.fiber_type_detection import FiberTypeDetector, visualize_fiber_type_analysis
        
        start_time = time.time()
        
        # Initialize detector with adaptive settings
        detector = FiberTypeDetector()
        
        # Run classification
        fiber_type, confidence, analysis_data = detector.classify_fiber_type(image)
        
        processing_time = time.time() - start_time
        
        if verbose:
            print(f"✅ Fiber type detection completed!")
            print(f"   Detected type: {fiber_type}")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Total fibers: {analysis_data.get('total_fibers', 0)}")
            print(f"   Hollow fibers: {analysis_data.get('hollow_fibers', 0)}")
            print(f"   Filaments: {analysis_data.get('filaments', 0)}")
            print(f"   Classification method: {analysis_data.get('classification_method', 'unknown')}")
            print(f"   Processing time: {processing_time:.3f}s")
            
            # Show adaptive thresholds
            thresholds = analysis_data.get('thresholds', {})
            if thresholds:
                print(f"   Adaptive thresholds used:")
                print(f"     Min fiber area: {thresholds.get('min_fiber_area', 0):,} pixels")
                print(f"     Max fiber area: {thresholds.get('max_fiber_area', 0):,} pixels")
                print(f"     Kernel size: {thresholds.get('kernel_size', 0)}")
        
        # Create debug visualization
        try:
            # Redirect the visualization to our debug directory
            plt.ioff()  # Turn off interactive mode
            visualize_fiber_type_analysis(image, analysis_data, figsize=(15, 10))
            
            debug_file = debug_dir / 'fiber_detection_analysis.png'
            plt.savefig(debug_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            if verbose:
                print(f"   💾 Debug image saved: {debug_file.name}")
        except Exception as viz_error:
            if verbose:
                print(f"   ⚠️ Could not save debug visualization: {viz_error}")
        
        return True, {
            'fiber_type': fiber_type,
            'confidence': confidence,
            'analysis_data': analysis_data,
            'processing_time': processing_time
        }
        
    except Exception as e:
        if verbose:
            print(f"💥 Fiber type detection error: {e}")
            traceback.print_exc()
        return False, {'error': str(e)}

def test_porosity_analysis_with_debug(image, fiber_mask, scale_factor, fiber_type, fiber_analysis_data, debug_dir, verbose=True):
    """Test porosity analysis and save debug images"""
    if not MODULES_AVAILABLE.get('porosity_analysis', False):
        return False, {'error': 'Porosity analysis module not available'}
    
    if verbose:
        print("\n🕳️ Testing Porosity Analysis with Debug Output...")
        print("-" * 50)
    
    try:
        from modules.porosity_analysis import PorosityAnalyzer, visualize_porosity_results
        
        start_time = time.time()
        
        # Initialize analyzer with debug-friendly config
        config = {
            'pore_detection': {
                'intensity_percentile': 28,
                'min_pore_area_pixels': 3,
                'max_pore_area_ratio': 0.1,
            },
            'performance': {
                'enable_timing': False,  # Disable timing output
            },
            'analysis': {
                'calculate_size_distribution': True,
                'calculate_spatial_metrics': True,
                'save_individual_pore_data': True,
            }
        }
        
        analyzer = PorosityAnalyzer(config=config)
        
        # Run analysis
        result = analyzer.analyze_fiber_porosity(
            image,
            fiber_mask.astype(np.uint8),
            scale_factor,
            fiber_type,
            fiber_analysis_data
        )
        
        processing_time = time.time() - start_time
        
        if result.get('success', False) and 'porosity_metrics' in result:
            pm = result['porosity_metrics']
            
            if verbose:
                print(f"✅ Porosity analysis completed!")
                print(f"   Method: {pm.get('method', 'unknown')}")
                print(f"   Total porosity: {pm.get('total_porosity_percent', 0):.2f}%")
                print(f"   Pore count: {pm.get('pore_count', 0)}")
                print(f"   Average pore size: {pm.get('average_pore_size_um2', 0):.2f} μm²")
                print(f"   Pore density: {pm.get('pore_density_per_mm2', 0):.1f}/mm²")
                print(f"   Processing time: {processing_time:.3f}s")
                
                # Quality assessment
                quality = result.get('quality_assessment', {})
                if quality:
                    print(f"   Analysis quality: {quality.get('overall_quality', 'unknown')}")
                    print(f"   Confidence: {quality.get('confidence', 0):.2f}")
            
            # Create debug visualization
            try:
                plt.ioff()  # Turn off interactive mode
                visualize_porosity_results(image, result, figsize=(15, 10))
                
                debug_file = debug_dir / 'porosity_analysis.png'
                plt.savefig(debug_file, dpi=150, bbox_inches='tight')
                plt.close()
                
                if verbose:
                    print(f"   💾 Debug image saved: {debug_file.name}")
            except Exception as viz_error:
                if verbose:
                    print(f"   ⚠️ Could not save debug visualization: {viz_error}")
            
            return True, result
        else:
            error = result.get('error', 'Unknown error')
            if verbose:
                print(f"❌ Porosity analysis failed: {error}")
            return False, result
            
    except Exception as e:
        if verbose:
            print(f"💥 Porosity analysis error: {e}")
            traceback.print_exc()
        return False, {'error': str(e)}

def test_comprehensive_analyzer_batch(sample_images, verbose=True):
    """Test comprehensive analyzer with batch processing (no debug output)"""
    if not MODULES_AVAILABLE.get('comprehensive_analyzer', False):
        return False, {'error': 'Comprehensive analyzer not available'}
    
    if verbose:
        print("\n🔬 Testing Comprehensive Analyzer - Batch Processing...")
        print("-" * 60)
    
    try:
        from comprehensive_analyzer_main import ComprehensiveFiberAnalyzer
        
        # Create output directory for batch results
        batch_dir = Path("test_batch_results")
        batch_dir.mkdir(exist_ok=True)
        
        # Configure analyzer for batch processing (no debug output)
        config = {
            'output': {
                'save_visualizations': False,  # Disable individual visualizations
                'save_data': True,
                'create_report': False,  # Disable individual reports
            },
            'performance': {
                'enable_timing': False,  # Disable timing output
            }
        }
        
        # Initialize analyzer with quiet mode
        analyzer = ComprehensiveFiberAnalyzer(config=config, debug=False)
        
        if verbose:
            print(f"   Processing {len(sample_images)} images...")
            print(f"   Output directory: {batch_dir}")
        
        # Process sample_images directory
        sample_dir = Path("sample_images")
        if sample_dir.exists():
            start_time = time.time()
            
            # Run batch analysis
            batch_result = analyzer.analyze_batch(str(sample_dir), str(batch_dir))
            
            total_time = time.time() - start_time
            
            if 'error' not in batch_result:
                batch_info = batch_result['batch_info']
                
                if verbose:
                    print(f"✅ Batch analysis completed!")
                    print(f"   Total images: {batch_info['total_images']}")
                    print(f"   Successful: {batch_info['successful_analyses']}")
                    print(f"   Success rate: {batch_info['success_rate']:.1f}%")
                    print(f"   Total time: {batch_info['total_processing_time']:.2f}s")
                    print(f"   Average per image: {batch_info['average_time_per_image']:.2f}s")
                    print(f"   Porosity method: {batch_info.get('porosity_method', 'unknown')}")
                
                # Create summary Excel file
                try:
                    import pandas as pd
                    
                    # Compile results into DataFrame
                    summary_data = []
                    for result in batch_result['individual_results']:
                        if result.get('success', False):
                            row = {
                                'image_name': result['image_name'],
                                'processing_time': result.get('total_processing_time', 0),
                                'scale_detected': result.get('scale_detection', {}).get('scale_detected', False),
                                'scale_factor': result.get('scale_detection', {}).get('micrometers_per_pixel', 0),
                                'fiber_type': result.get('fiber_detection', {}).get('fiber_type', 'unknown'),
                                'fiber_confidence': result.get('fiber_detection', {}).get('confidence', 0),
                                'total_fibers': result.get('fiber_detection', {}).get('total_fibers', 0),
                                'hollow_fibers': result.get('fiber_detection', {}).get('hollow_fibers', 0),
                                'filaments': result.get('fiber_detection', {}).get('filaments', 0),
                                'porosity_percent': result.get('porosity_analysis', {}).get('porosity_metrics', {}).get('total_porosity_percent', 0),
                                'pore_count': result.get('porosity_analysis', {}).get('porosity_metrics', {}).get('pore_count', 0),
                                'avg_pore_size': result.get('porosity_analysis', {}).get('porosity_metrics', {}).get('average_pore_size_um2', 0),
                                'analysis_quality': result.get('comprehensive_metrics', {}).get('analysis_quality', 'unknown'),
                                'quality_score': result.get('comprehensive_metrics', {}).get('quality_score', 0),
                            }
                        else:
                            row = {
                                'image_name': result['image_name'],
                                'processing_time': result.get('total_processing_time', 0),
                                'error': result.get('error', 'Unknown error')
                            }
                        summary_data.append(row)
                    
                    # Save to Excel
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    excel_file = batch_dir / f'batch_analysis_summary_{timestamp}.xlsx'
                    
                    df = pd.DataFrame(summary_data)
                    df.to_excel(excel_file, index=False)
                    
                    if verbose:
                        print(f"   📊 Excel summary saved: {excel_file.name}")
                        print(f"   📁 Full batch results: {batch_dir}")
                
                except ImportError:
                    if verbose:
                        print(f"   ⚠️ pandas not available - Excel export skipped")
                except Exception as excel_error:
                    if verbose:
                        print(f"   ⚠️ Excel export failed: {excel_error}")
                
                return True, batch_result
            else:
                if verbose:
                    print(f"❌ Batch analysis failed: {batch_result.get('error', 'Unknown error')}")
                return False, batch_result
        else:
            if verbose:
                print(f"❌ sample_images directory not found")
            return False, {'error': 'sample_images directory not found'}
            
    except Exception as e:
        if verbose:
            print(f"💥 Comprehensive analyzer error: {e}")
            traceback.print_exc()
        return False, {'error': str(e)}

def run_individual_module_tests(sample_image, debug_dir):
    """Run tests on individual modules with debug output"""
    print(f"\n🧪 INDIVIDUAL MODULE TESTS")
    print(f"Image: {sample_image.name}")
    print("=" * 70)
    
    results = {}
    
    # Test 1: Image Preprocessing
    preprocessing_success, preprocessing_result = test_image_preprocessing_with_debug(
        sample_image, debug_dir, verbose=True
    )
    results['preprocessing'] = preprocessing_success
    
    if preprocessing_success:
        processed_image = preprocessing_result.get('processed')
        if processed_image is None:
            processed_image = preprocessing_result.get('original')
    else:
        # Fallback: load image directly
        if MODULES_AVAILABLE.get('image_preprocessing', False):
            from modules.image_preprocessing import load_image
            processed_image = load_image(str(sample_image))
        else:
            processed_image = cv2.imread(str(sample_image), cv2.IMREAD_GRAYSCALE)
    
    if processed_image is None:
        print("❌ Could not load or process image - aborting tests")
        return results
    
    # Test 2: Scale Detection
    scale_success, scale_result = test_scale_detection_with_debug(
        processed_image, debug_dir, verbose=True
    )
    results['scale_detection'] = scale_success
    scale_factor = scale_result.get('micrometers_per_pixel', 1.0) if scale_success else 1.0
    
    # Test 3: Fiber Type Detection
    fiber_success, fiber_result = test_fiber_detection_with_debug(
        processed_image, debug_dir, verbose=True
    )
    results['fiber_detection'] = fiber_success
    
    if fiber_success:
        fiber_type = fiber_result.get('fiber_type', 'unknown')
        fiber_analysis_data = fiber_result.get('analysis_data', {})
        fiber_mask = fiber_analysis_data.get('fiber_mask', np.zeros_like(processed_image, dtype=bool))
    else:
        fiber_type = 'unknown'
        fiber_analysis_data = {}
        # Create a simple circular mask for testing
        fiber_mask = np.zeros_like(processed_image, dtype=bool)
        center = (processed_image.shape[1]//2, processed_image.shape[0]//2)
        cv2.circle(fiber_mask.astype(np.uint8), center, 300, 1, -1)
        fiber_mask = fiber_mask.astype(bool)
    
    # Test 4: Porosity Analysis
    if np.sum(fiber_mask) > 1000:  # Ensure sufficient fiber area
        porosity_success, porosity_result = test_porosity_analysis_with_debug(
            processed_image, fiber_mask, scale_factor, fiber_type, 
            fiber_analysis_data, debug_dir, verbose=True
        )
        results['porosity_analysis'] = porosity_success
    else:
        print("\n⚠️ Insufficient fiber area for porosity analysis test")
        results['porosity_analysis'] = False
    
    return results

def run_batch_test(sample_images):
    """Run comprehensive analyzer batch test"""
    print(f"\n🔬 COMPREHENSIVE ANALYZER BATCH TEST")
    print("=" * 70)
    
    batch_success, batch_result = test_comprehensive_analyzer_batch(
        sample_images, verbose=True
    )
    
    return {'batch_analysis': batch_success}

def main():
    """Main test runner"""
    print(f"\n🚀 Starting Real Image Tests")
    
    # Test imports
    test_imports()
    
    # Find sample image
    sample_image, all_sample_images = find_sample_image()
    if sample_image is None:
        print("\n❌ Cannot run tests without sample images!")
        return False
    
    # Create debug output directory
    debug_dir = create_debug_output_dir()
    
    # Run individual module tests with debug output
    individual_results = run_individual_module_tests(sample_image, debug_dir)
    
    # Run batch test if we have multiple images
    batch_results = {}
    if len(all_sample_images) > 1:
        batch_results = run_batch_test(all_sample_images)
    else:
        print(f"\n⚠️ Only one sample image found - skipping batch test")
        print(f"   Add more images to sample_images/ for batch testing")
    
    # Combine results
    all_results = {**individual_results, **batch_results}
    
    # Final summary
    print(f"\n🎯 FINAL TEST SUMMARY")
    print("=" * 70)
    
    for test_name, success in all_results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    overall_success = all(all_results.values())
    
    print(f"\nOverall Result: {'🎉 ALL TESTS PASSED!' if overall_success else '⚠️ SOME TESTS FAILED'}")
    print(f"Debug outputs saved to: {debug_dir}")
    
    if batch_results.get('batch_analysis', False):
        print(f"Batch results saved to: test_batch_results/")
    
    if overall_success:
        print("\n✅ Your SEM Fiber Analysis System is working correctly!")
    else:
        print("\n🔧 Some tests failed - check the output above for details")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)