#!/usr/bin/env python3
"""
Debug test to isolate the fiber detection issue
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import time

# Add paths
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "modules"))
sys.path.insert(0, str(current_dir))

try:
    from modules.fiber_type_detection import FiberTypeDetector
    from modules.image_preprocessing import load_image, preprocess_pipeline
    print("✅ Modules imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def test_fiber_detection_detailed():
    """Test fiber detection with detailed debugging"""
    print("\n🧬 DETAILED FIBER DETECTION DEBUG")
    print("=" * 50)
    
    # Test image path
    test_image_path = "C:/Users/risto/Nextcloud/KalvoTech/Operations/Python/fiber_analysis_project/sample_images/Crumble_test_score/crumbly/14a_001.jpg"
    
    if not Path(test_image_path).exists():
        print(f"❌ Test image not found: {test_image_path}")
        return False
    
    try:
        # Step 1: Load image
        print("\n1️⃣ Loading image...")
        image = load_image(test_image_path)
        print(f"   ✅ Raw image loaded: {image.shape}")
        
        # Step 2: Preprocess like evaluation system does
        print("\n2️⃣ Preprocessing image...")
        preprocessed_result = preprocess_pipeline(test_image_path)
        preprocessed = preprocessed_result['processed']
        print(f"   ✅ Preprocessed image: {preprocessed.shape}")
        print(f"   Processing steps: {preprocessed_result.get('processing_steps', [])}")
        
        # Step 3: Initialize detector
        print("\n3️⃣ Initializing FiberTypeDetector...")
        detector = FiberTypeDetector()
        print(f"   ✅ Detector initialized")
        print(f"   Min fiber ratio: {detector.min_fiber_ratio}")
        print(f"   Max fiber ratio: {detector.max_fiber_ratio}")
        print(f"   Confidence threshold: {detector.confidence_threshold}")
        
        # Step 4: Test classification (like comprehensive_analyzer_main.py does)
        print("\n4️⃣ Running classify_fiber_type (comprehensive_analyzer_main.py style)...")
        start_time = time.time()
        
        try:
            fiber_type, confidence, detailed_analysis = detector.classify_fiber_type(preprocessed)
            processing_time = time.time() - start_time
            
            print(f"   ✅ Method returned successfully in {processing_time:.3f}s")
            print(f"   Fiber type: {fiber_type}")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Detailed analysis type: {type(detailed_analysis)}")
            
            # Step 5: Check detailed_analysis content
            print("\n5️⃣ Analyzing detailed_analysis content...")
            if detailed_analysis is None:
                print("   ❌ detailed_analysis is None")
                return False
            elif not isinstance(detailed_analysis, dict):
                print(f"   ❌ detailed_analysis is not a dict: {type(detailed_analysis)}")
                return False
            else:
                print(f"   ✅ detailed_analysis is a dict with {len(detailed_analysis)} keys")
                
                # Check for success flag
                success_flag = detailed_analysis.get('success', None)
                print(f"   Success flag: {success_flag}")
                
                # Check for masks
                fiber_mask = detailed_analysis.get('fiber_mask', None)
                lumen_mask = detailed_analysis.get('lumen_mask', None)
                
                if fiber_mask is not None:
                    print(f"   ✅ fiber_mask found: {type(fiber_mask)}, shape: {fiber_mask.shape}")
                else:
                    print(f"   ❌ fiber_mask is None")
                
                if lumen_mask is not None:
                    print(f"   ✅ lumen_mask found: {type(lumen_mask)}, shape: {lumen_mask.shape}")
                else:
                    print(f"   ⚠️ lumen_mask is None (might be normal for filaments)")
                
                # Print all keys in detailed_analysis
                print(f"   All keys in detailed_analysis: {list(detailed_analysis.keys())}")
                
                # Check if this matches the expected pattern
                if success_flag and fiber_mask is not None:
                    print("\n🎉 FIBER DETECTION WORKING CORRECTLY!")
                    return True
                else:
                    print("\n❌ FIBER DETECTION ISSUE IDENTIFIED:")
                    if not success_flag:
                        print("   - Missing or False 'success' flag")
                    if fiber_mask is None:
                        print("   - Missing 'fiber_mask'")
                    return False
                
        except Exception as e:
            print(f"   ❌ classify_fiber_type failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fiber_detection_simple():
    """Test with simple synthetic image"""
    print("\n🧪 SIMPLE FIBER DETECTION TEST")
    print("=" * 40)
    
    try:
        # Create simple test image
        test_image = np.ones((500, 500), dtype=np.uint8) * 128
        # Add a simple circular "fiber"
        center = (250, 250)
        cv2.circle(test_image, center, 100, 50, -1)  # Dark circle
        cv2.circle(test_image, center, 30, 200, -1)  # Light center (lumen)
        
        print(f"   ✅ Synthetic image created: {test_image.shape}")
        
        # Test with detector
        detector = FiberTypeDetector()
        fiber_type, confidence, detailed_analysis = detector.classify_fiber_type(test_image)
        
        print(f"   Fiber type: {fiber_type}")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Success flag: {detailed_analysis.get('success', False) if detailed_analysis else False}")
        
        return detailed_analysis.get('success', False) if detailed_analysis else False
        
    except Exception as e:
        print(f"   ❌ Simple test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔬 Fiber Detection Debug Suite")
    print("=" * 40)
    
    # Test 1: Simple synthetic image
    simple_success = test_fiber_detection_simple()
    
    # Test 2: Real image with detailed analysis
    if simple_success:
        detailed_success = test_fiber_detection_detailed()
    else:
        print("\n⚠️ Skipping detailed test due to simple test failure")
        detailed_success = False
    
    # Summary
    print("\n📊 DEBUG RESULTS:")
    print(f"   Simple test: {'✅ PASSED' if simple_success else '❌ FAILED'}")
    print(f"   Detailed test: {'✅ PASSED' if detailed_success else '❌ FAILED'}")
    
    if simple_success and detailed_success:
        print("\n🎉 Fiber detection is working correctly!")
        print("   The issue might be in the evaluation system logic.")
    elif simple_success and not detailed_success:
        print("\n⚠️ Fiber detection works on simple images but fails on real images.")
        print("   Check image preprocessing or detector parameters.")
    else:
        print("\n❌ Fiber detection has fundamental issues.")
        print("   Check FiberTypeDetector implementation.")