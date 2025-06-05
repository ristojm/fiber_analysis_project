#!/usr/bin/env python3
"""
Test Script for SEM Fiber Analysis System
Run comprehensive tests on fiber detection, scale detection, and porosity analysis
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import time
import traceback

# Setup paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "modules"))

print("🔧 SEM Fiber Analysis System - Test Suite")
print("=" * 60)

# Test imports
print("📦 Testing module imports...")
try:
    from modules.scale_detection import detect_scale_bar, ScaleBarDetector
    print("✅ Scale detection module imported")
except ImportError as e:
    print(f"❌ Scale detection import failed: {e}")
    sys.exit(1)

try:
    from modules.fiber_type_detection import FiberTypeDetector, detect_fiber_type
    print("✅ Fiber type detection module imported")
except ImportError as e:
    print(f"❌ Fiber type detection import failed: {e}")
    sys.exit(1)

try:
    from modules.image_preprocessing import load_image, preprocess_pipeline
    print("✅ Image preprocessing module imported")
except ImportError as e:
    print(f"❌ Image preprocessing import failed: {e}")
    sys.exit(1)

try:
    from modules.porosity_analysis import EnhancedPorosityAnalyzer, analyze_fiber_porosity_enhanced
    print("✅ Enhanced porosity analysis module imported")
    POROSITY_AVAILABLE = True
except ImportError:
    print("⚠️ Enhanced porosity module not found, trying basic version...")
    try:
        from modules.porosity_analysis import PorosityAnalyzer
        print("✅ Basic porosity analysis module imported")
        POROSITY_AVAILABLE = True
        ENHANCED_POROSITY = False
    except ImportError as e:
        print(f"❌ Porosity analysis import failed: {e}")
        POROSITY_AVAILABLE = False

try:
    # Import the ComprehensiveFiberAnalyzer class directly
    sys.path.insert(0, str(project_root))
    from comprehensive_analyzer_main import ComprehensiveFiberAnalyzer
    print("✅ Comprehensive analyzer class imported")
    COMPREHENSIVE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Comprehensive analyzer class import failed: {e}")
    print("   Will test comprehensive_analyzer_main.py as standalone script instead")
    COMPREHENSIVE_AVAILABLE = False

print("\n🎯 All required modules imported successfully!")

def create_test_image():
    """Create a synthetic test image with realistic fiber structures and scale bar"""
    print("\n🔬 Creating synthetic test image...")
    
    # Create base image
    height, width = 2048, 2048
    img = np.ones((height, width), dtype=np.uint8) * 40  # Dark background
    
    # Add multiple fiber-like structures with realistic SEM appearance
    
    # Main hollow fiber (large, center-left)
    center_x, center_y = width // 2 - 200, height // 2 - 100
    
    # Create hollow fiber with wall thickness
    wall_thickness = 80
    outer_radius = 250
    inner_radius = outer_radius - wall_thickness
    
    # Outer fiber boundary (bright)
    cv2.circle(img, (center_x, center_y), outer_radius, 180, -1)
    # Inner lumen (dark - should be excluded from porosity)
    cv2.circle(img, (center_x, center_y), inner_radius, 45, -1)
    
    # Add some small pores in the fiber wall
    pore_positions = [
        (center_x - 100, center_y - 50, 8),
        (center_x + 80, center_y + 60, 6),
        (center_x - 50, center_y + 100, 10),
        (center_x + 120, center_y - 80, 7),
        (center_x - 130, center_y + 30, 5)
    ]
    
    for px, py, pore_size in pore_positions:
        cv2.circle(img, (px, py), pore_size, 45, -1)  # Dark pores
    
    # Add a solid filament for comparison (smaller, right side)
    solid_center_x, solid_center_y = width // 2 + 300, height // 2 + 200
    cv2.circle(img, (solid_center_x, solid_center_y), 120, 170, -1)
    
    # Add a few small pores to the solid filament too
    cv2.circle(img, (solid_center_x - 40, solid_center_y - 30), 4, 45, -1)
    cv2.circle(img, (solid_center_x + 35, solid_center_y + 40), 6, 45, -1)
    
    # Add realistic noise and texture
    noise = np.random.normal(0, 8, (height, width))
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    # Apply slight blur for more realistic SEM appearance
    img = cv2.GaussianBlur(img, (3, 3), 0.5)
    
    # Add scale bar at bottom (more realistic)
    scale_y = height - 120
    scale_start_x = 200
    scale_length = 400
    
    # Scale bar rectangle (white)
    cv2.rectangle(img, (scale_start_x, scale_y), 
                  (scale_start_x + scale_length, scale_y + 15), 255, -1)
    
    # Add scale text
    cv2.putText(img, "100 μm", (scale_start_x, scale_y + 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 2)
    
    print(f"✅ Synthetic test image created: {img.shape}")
    print(f"   Features: 1 hollow fiber + 1 solid filament + pores + scale bar")
    return img

def test_scale_detection(image, verbose=True):
    """Test scale detection functionality"""
    if verbose:
        print("\n📏 Testing Scale Detection...")
        print("-" * 30)
    
    try:
        start_time = time.time()
        
        # Test with enhanced detection
        result = detect_scale_bar(image, use_enhanced=True, debug=verbose)
        
        processing_time = time.time() - start_time
        
        if result['scale_detected']:
            if verbose:
                print(f"✅ Scale detection successful!")
                print(f"   Scale factor: {result['micrometers_per_pixel']:.4f} μm/pixel")
                print(f"   Method: {result.get('method_used', 'unknown')}")
                print(f"   Confidence: {result.get('confidence', 0):.2%}")
                print(f"   Processing time: {processing_time:.3f}s")
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

def debug_fiber_segmentation(image, fiber_result, save_debug=True):
    """Debug function to visualize fiber segmentation results"""
    print("\n🔍 Debugging Fiber Segmentation...")
    
    if not fiber_result or 'analysis_data' not in fiber_result:
        print("   ❌ No fiber analysis data available")
        return
    
    analysis_data = fiber_result['analysis_data']
    
    # Check what masks are available
    print(f"   Available data keys: {list(analysis_data.keys())}")
    
    if 'fiber_mask' in analysis_data:
        fiber_mask = analysis_data['fiber_mask']
        print(f"   Fiber mask shape: {fiber_mask.shape}")
        print(f"   Fiber mask type: {type(fiber_mask)}")
        print(f"   Fiber mask unique values: {np.unique(fiber_mask)}")
        print(f"   Total fiber area: {np.sum(fiber_mask > 0):,} pixels")
    else:
        print("   ❌ No fiber_mask found in analysis data")
    
    # Check individual results for lumen information
    individual_results = analysis_data.get('individual_results', [])
    print(f"   Individual fibers detected: {len(individual_results)}")
    
    for i, result in enumerate(individual_results):
        fiber_props = result.get('fiber_properties', {})
        has_lumen = result.get('has_lumen', False)
        lumen_props = result.get('lumen_properties', {})
        
        print(f"   Fiber {i+1}:")
        print(f"     Has lumen: {has_lumen}")
        print(f"     Fiber area: {fiber_props.get('area', 0):,} pixels")
        if has_lumen and lumen_props:
            print(f"     Lumen area: {lumen_props.get('area', 0):,} pixels")
            print(f"     Lumen ratio: {lumen_props.get('area_ratio', 0):.3f}")
    
    if save_debug:
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(image, cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Fiber mask
            if 'fiber_mask' in analysis_data:
                axes[1].imshow(analysis_data['fiber_mask'], cmap='gray')
                axes[1].set_title('Fiber Mask')
                axes[1].axis('off')
            
            # Overlay
            if 'fiber_mask' in analysis_data:
                overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                
                # Draw fiber contours
                for result in individual_results:
                    contour = result.get('fiber_properties', {}).get('contour')
                    has_lumen = result.get('has_lumen', False)
                    
                    if contour is not None:
                        color = (0, 255, 0) if has_lumen else (255, 0, 0)
                        cv2.drawContours(overlay, [contour], -1, color, 3)
                        
                        # Draw lumen if present
                        if has_lumen:
                            lumen_contour = result.get('lumen_properties', {}).get('contour')
                            if lumen_contour is not None:
                                cv2.drawContours(overlay, [lumen_contour], -1, (0, 255, 255), 2)
                
                axes[2].imshow(overlay)
                axes[2].set_title('Fiber Detection\n(Green=Hollow, Red=Solid, Cyan=Lumen)')
                axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig('debug_fiber_segmentation.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("   📸 Debug visualization saved: debug_fiber_segmentation.png")
            
        except Exception as e:
            print(f"   ⚠️ Could not save debug visualization: {e}")

def test_fiber_type_detection(image, verbose=True):
    """Test fiber type detection functionality with enhanced debugging"""
    if verbose:
        print("\n🧬 Testing Fiber Type Detection...")
        print("-" * 35)
    
    try:
        start_time = time.time()
        
        # Initialize detector
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
            print(f"   Method: {analysis_data.get('classification_method', 'unknown')}")
            print(f"   Processing time: {processing_time:.3f}s")
        
        result = {
            'fiber_type': fiber_type,
            'confidence': confidence,
            'analysis_data': analysis_data,
            'processing_time': processing_time
        }
        
        # Add debugging
        if verbose:
            debug_fiber_segmentation(image, result)
        
        return True, result
        
    except Exception as e:
        if verbose:
            print(f"💥 Fiber type detection error: {e}")
            traceback.print_exc()
        return False, {'error': str(e)}

def test_porosity_analysis(image, fiber_mask, scale_factor, fiber_type, verbose=True):
    """Test porosity analysis functionality"""
    if not POROSITY_AVAILABLE:
        if verbose:
            print("\n⚠️ Porosity analysis not available - module not found")
        return False, {'error': 'Porosity module not available'}
    
    if verbose:
        print("\n🕳️ Testing Porosity Analysis...")
        print("-" * 30)
    
    try:
        start_time = time.time()
        
        # Initialize analyzer
        analyzer = EnhancedPorosityAnalyzer()
        
        # Run analysis
        result = analyzer.analyze_fiber_porosity(
            image, 
            fiber_mask.astype(np.uint8), 
            scale_factor, 
            fiber_type,
            None  # No fiber analysis data for this test
        )
        
        processing_time = time.time() - start_time
        
        if 'porosity_metrics' in result:
            pm = result['porosity_metrics']
            if verbose:
                print(f"✅ Porosity analysis completed!")
                print(f"   Total porosity: {pm.get('total_porosity_percent', 0):.2f}%")
                print(f"   Pore count: {pm.get('pore_count', 0)}")
                print(f"   Average pore size: {pm.get('average_pore_size_um2', 0):.2f} μm²")
                print(f"   Pore density: {pm.get('pore_density_per_mm2', 0):.1f}/mm²")
                print(f"   Analysis quality: {result.get('analysis_quality', {}).get('overall_quality', 'unknown')}")
                print(f"   Processing time: {processing_time:.3f}s")
            
            return True, result
        else:
            if verbose:
                print(f"❌ Porosity analysis failed: {result.get('error', 'Unknown error')}")
            return False, result
            
    except Exception as e:
        if verbose:
            print(f"💥 Porosity analysis error: {e}")
            traceback.print_exc()
        return False, {'error': str(e)}

def test_comprehensive_analyzer(image_path, verbose=True):
    """Test the comprehensive analyzer"""
    if verbose:
        print("\n🔬 Testing Comprehensive Analyzer...")
        print("-" * 40)
    
    try:
        if COMPREHENSIVE_AVAILABLE:
            # Test by importing the class directly
            analyzer = ComprehensiveFiberAnalyzer(debug=verbose)
            result = analyzer.analyze_single_image(image_path)
            
            if result['success']:
                if verbose:
                    print(f"✅ Comprehensive analysis successful!")
                    comprehensive = result.get('comprehensive_metrics', {})
                    print(f"   Overall quality: {comprehensive.get('analysis_quality', 'unknown')}")
                    print(f"   Quality score: {comprehensive.get('quality_score', 0):.2f}/1.0")
                    print(f"   Total time: {result.get('total_processing_time', 0):.2f}s")
                return True, result
            else:
                if verbose:
                    print(f"❌ Comprehensive analysis failed: {result.get('error', 'Unknown error')}")
                return False, result
        else:
            # Test by running as subprocess
            import subprocess
            
            if verbose:
                print("   Running comprehensive_analyzer_main.py as standalone script...")
            
            # Run the comprehensive analyzer as a subprocess
            cmd = [sys.executable, "comprehensive_analyzer_main.py", "--image", image_path, "--quiet"]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    if verbose:
                        print(f"✅ Comprehensive analyzer script ran successfully!")
                        print(f"   Output: {len(result.stdout)} characters")
                    return True, {'output': result.stdout, 'success': True}
                else:
                    if verbose:
                        print(f"❌ Comprehensive analyzer script failed with return code: {result.returncode}")
                        if result.stderr:
                            print(f"   Error: {result.stderr}")
                    return False, {'error': result.stderr, 'returncode': result.returncode}
            
            except subprocess.TimeoutExpired:
                if verbose:
                    print(f"❌ Comprehensive analyzer script timed out after 60 seconds")
                return False, {'error': 'Timeout after 60 seconds'}
            except FileNotFoundError:
                if verbose:
                    print(f"❌ comprehensive_analyzer_main.py not found")
                return False, {'error': 'Script file not found'}
            
    except Exception as e:
        if verbose:
            print(f"💥 Comprehensive analyzer error: {e}")
            traceback.print_exc()
        return False, {'error': str(e)}

def run_individual_tests():
    """Run individual module tests"""
    print("\n🧪 INDIVIDUAL MODULE TESTS")
    print("=" * 60)
    
    # Create test image
    test_img = create_test_image()
    
    # Test 1: Scale Detection
    scale_success, scale_result = test_scale_detection(test_img)
    scale_factor = scale_result.get('micrometers_per_pixel', 1.0) if scale_success else 1.0
    
    # Test 2: Fiber Type Detection
    fiber_success, fiber_result = test_fiber_type_detection(test_img)
    fiber_type = fiber_result.get('fiber_type', 'unknown') if fiber_success else 'unknown'
    
    # Extract fiber mask for porosity test
    fiber_mask = np.zeros_like(test_img, dtype=bool)
    if fiber_success and 'analysis_data' in fiber_result:
        analysis_data = fiber_result['analysis_data']
        if 'fiber_mask' in analysis_data:
            fiber_mask = analysis_data['fiber_mask']
        else:
            # Create simple circular mask for testing
            center = (test_img.shape[1]//2, test_img.shape[0]//2 - 200)
            cv2.circle(fiber_mask.astype(np.uint8), center, 300, 1, -1)
            fiber_mask = fiber_mask.astype(bool)
    
    # Test 3: Porosity Analysis
    porosity_success, porosity_result = test_porosity_analysis(
        test_img, fiber_mask, scale_factor, fiber_type
    )
    
    # Summary
    print(f"\n📊 INDIVIDUAL TEST SUMMARY")
    print("-" * 30)
    print(f"Scale Detection: {'✅ PASS' if scale_success else '❌ FAIL'}")
    print(f"Fiber Detection: {'✅ PASS' if fiber_success else '❌ FAIL'}")
    print(f"Porosity Analysis: {'✅ PASS' if porosity_success else '❌ FAIL'}")
    
    overall_success = scale_success and fiber_success and porosity_success
    print(f"Overall: {'✅ ALL TESTS PASSED' if overall_success else '⚠️ SOME TESTS FAILED'}")
    
    return overall_success

def run_comprehensive_test():
    """Run comprehensive analyzer test"""
    print("\n🔬 COMPREHENSIVE ANALYZER TEST")
    print("=" * 60)
    
    # Save test image to file
    test_img = create_test_image()
    test_path = "test_fiber_image.png"
    cv2.imwrite(test_path, test_img)
    print(f"📁 Test image saved: {test_path}")
    
    try:
        # Test comprehensive analyzer
        comp_success, comp_result = test_comprehensive_analyzer(test_path)
        
        # Cleanup
        if os.path.exists(test_path):
            os.remove(test_path)
        
        print(f"\n📊 COMPREHENSIVE TEST SUMMARY")
        print("-" * 35)
        print(f"Comprehensive Analyzer: {'✅ PASS' if comp_success else '❌ FAIL'}")
        
        return comp_success
        
    except Exception as e:
        print(f"💥 Comprehensive test error: {e}")
        return False
    finally:
        # Cleanup
        if os.path.exists(test_path):
            os.remove(test_path)

def test_with_real_image():
    """Test with real SEM image if available"""
    print("\n📸 REAL IMAGE TEST")
    print("=" * 60)
    
    # Look for sample images
    sample_dirs = ["sample_images", "images", "test_images"]
    sample_extensions = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]
    
    real_image_path = None
    
    for sample_dir in sample_dirs:
        sample_path = Path(sample_dir)
        if sample_path.exists():
            for ext in sample_extensions:
                image_files = list(sample_path.glob(ext))
                if image_files:
                    real_image_path = image_files[0]
                    break
            if real_image_path:
                break
    
    if real_image_path:
        print(f"📁 Found real image: {real_image_path}")
        
        try:
            # Load image
            img = load_image(str(real_image_path))
            if img is not None:
                print(f"✅ Image loaded successfully: {img.shape}")
                
                # Test individual modules
                print("\n🔍 Testing with real image...")
                
                scale_success, scale_result = test_scale_detection(img, verbose=False)
                fiber_success, fiber_result = test_fiber_type_detection(img, verbose=False)
                
                print(f"Scale Detection: {'✅ PASS' if scale_success else '❌ FAIL'}")
                print(f"Fiber Detection: {'✅ PASS' if fiber_success else '❌ FAIL'}")
                
                if scale_success:
                    print(f"   Scale factor: {scale_result.get('micrometers_per_pixel', 0):.4f} μm/pixel")
                
                if fiber_success:
                    print(f"   Fiber type: {fiber_result.get('fiber_type', 'unknown')}")
                    print(f"   Confidence: {fiber_result.get('confidence', 0):.3f}")
                
                return True
            else:
                print(f"❌ Could not load image: {real_image_path}")
                return False
                
        except Exception as e:
            print(f"💥 Real image test error: {e}")
            return False
    else:
        print("📁 No real SEM images found in common directories")
        print("   Looked in: sample_images/, images/, test_images/")
        print("   Supported formats: .jpg, .jpeg, .png, .tif, .tiff")
        return False

def main():
    """Main test runner"""
    print(f"\n🚀 Starting SEM Fiber Analysis System Tests")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    results = {}
    
    # Run individual module tests
    results['individual'] = run_individual_tests()
    
    # Run comprehensive test
    results['comprehensive'] = run_comprehensive_test()
    
    # Test with real image if available
    results['real_image'] = test_with_real_image()
    
    # Final summary
    print(f"\n🎯 FINAL TEST SUMMARY")
    print("=" * 60)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    overall_success = all(results.values())
    print(f"\nOverall Result: {'🎉 ALL SYSTEMS GO!' if overall_success else '⚠️ SOME ISSUES DETECTED'}")
    
    if overall_success:
        print("\n✅ Your SEM Fiber Analysis System is ready to use!")
        print("   You can now run:")
        print("   python comprehensive_analyzer_main.py --image your_image.jpg")
        print("   python comprehensive_analyzer_main.py --batch your_image_folder/")
    else:
        print("\n🔧 System needs attention. Check the failed tests above.")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)