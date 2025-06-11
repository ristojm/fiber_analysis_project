#!/usr/bin/env python3
"""
Test Day 3 Workflow Reorganization
Tests the new orchestrator approach and verifies the workflow reorganization.
"""

import sys
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "modules"))
sys.path.insert(0, str(project_root / "Crumble_ML"))  # Add Crumble_ML directory

print(f"🔧 Testing Day 3 Workflow Reorganization")
print(f"   Project root: {project_root}")

def test_orchestrator_function():
    """Test the new process_single_image_orchestrator function."""
    print(f"\n📋 Testing Orchestrator Function...")
    
    try:
        # Import the new orchestrator from Crumble_ML directory
        from multiprocessing_crumbly_workflow import process_single_image_orchestrator
        print(f"✅ Orchestrator function imported successfully")
        
        # Create a test image
        test_dir = Path(tempfile.mkdtemp())
        test_image_path = test_dir / "test_image.png"
        
        # Create and save test image
        import cv2
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(test_image_path), test_image)
        
        print(f"✅ Created test image: {test_image_path}")
        
        # Test orchestrator with debug mode
        worker_args = {
            'image_path': test_image_path,
            'debug': True
        }
        
        print(f"\n🔄 Running orchestrator with debug mode...")
        result = process_single_image_orchestrator(worker_args)
        
        # Validate result structure
        required_keys = ['image_path', 'success', 'processing_steps']
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
        
        # Check processing steps
        expected_steps = ['image_loaded', 'scale_detection', 'preprocessing', 
                         'fiber_detection', 'mask_creation']
        
        processing_steps = result.get('processing_steps', [])
        for step in expected_steps:
            assert step in processing_steps, f"Missing processing step: {step}"
        
        print(f"✅ Orchestrator result structure correct")
        print(f"   Processing steps: {processing_steps}")
        print(f"   Success: {result.get('success', False)}")
        
        # Check individual result sections
        sections_to_check = ['scale_detection', 'preprocessing', 'fiber_detection', 'mask_creation']
        for section in sections_to_check:
            assert section in result, f"Missing result section: {section}"
        
        print(f"✅ All result sections present")
        
        # Cleanup
        shutil.rmtree(test_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ Orchestrator function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_workflow_class_integration():
    """Test the updated MultiprocessingCrumblyWorkflow class."""
    print(f"\n📋 Testing Workflow Class Integration...")
    
    try:
        # Import the updated workflow class
        from multiprocessing_crumbly_workflow import MultiprocessingCrumblyWorkflow
        print(f"✅ Workflow class imported successfully")
        
        # Create workflow instance
        workflow = MultiprocessingCrumblyWorkflow(num_processes=1)  # Use 1 process for testing
        print(f"✅ Workflow instance created")
        
        # Check that it has the expected attributes
        assert hasattr(workflow, 'run_parallel_evaluation'), "Should have run_parallel_evaluation method"
        assert hasattr(workflow, 'output_dir'), "Should have output_dir attribute"
        assert hasattr(workflow, 'num_processes'), "Should have num_processes attribute"
        
        print(f"✅ Workflow class has expected interface")
        
        return True
        
    except Exception as e:
        print(f"❌ Workflow class integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_module_imports_in_workflow():
    """Test that the workflow properly imports reorganized modules."""
    print(f"\n📋 Testing Module Imports in Workflow...")
    
    try:
        # Test that workflow can import all reorganized functions
        from multiprocessing_crumbly_workflow import MODULES_LOADED
        print(f"✅ MODULES_LOADED imported: {MODULES_LOADED}")
        
        # Check expected module availability
        expected_modules = ['core', 'image_preprocessing', 'fiber_type_detection', 
                           'scale_detection', 'crumbly_detection']
        
        for module in expected_modules:
            assert module in MODULES_LOADED, f"Missing module: {module}"
        
        # Check that core modules are loaded
        assert MODULES_LOADED['core'] == True, "Core modules should be loaded"
        
        print(f"✅ All expected modules present in MODULES_LOADED")
        
        # Test that we can import the functions directly
        try:
            from modules.image_preprocessing import load_image
            from modules.fiber_type_detection import detect_fiber_type
            from modules.scale_detection import detect_scale_bar_from_crop
            from modules.crumbly_detection import improve_crumbly_classification
            print(f"✅ All reorganized functions can be imported from modules")
        except ImportError as e:
            print(f"❌ Some reorganized functions missing: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Module imports test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backwards_compatibility():
    """Test that the reorganization maintains backwards compatibility."""
    print(f"\n📋 Testing Backwards Compatibility...")
    
    try:
        # Test that old function names still work through modules
        from modules.image_preprocessing import load_image
        from modules.fiber_type_detection import detect_fiber_type
        print(f"✅ Legacy function names still available")
        
        # Test that results format is compatible
        test_image = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        
        # This should work with the new functions
        # Use FiberTypeDetector class for consistent interface
        from modules.fiber_type_detection import FiberTypeDetector
        detector = FiberTypeDetector()
        fiber_type, confidence, analysis_data = detector.classify_fiber_type(test_image)
        
        assert isinstance(fiber_type, str), "Fiber type should be string"
        assert isinstance(confidence, (int, float)), "Confidence should be numeric"
        assert isinstance(analysis_data, dict), "Analysis data should be dict"
        
        print(f"✅ Function interfaces remain compatible")
        print(f"   Fiber type: {fiber_type}")
        print(f"   Confidence: {confidence:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Backwards compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Day 3 workflow reorganization tests."""
    print(f"🧪 Day 3 Workflow Reorganization Tests")
    print(f"=" * 50)
    
    # Enable debug for testing
    try:
        from modules import enable_global_debug, disable_global_debug
        enable_global_debug(save_images=False, show_plots=False)
    except:
        pass  # Debug not available yet
    
    tests = [
        ("Orchestrator Function", test_orchestrator_function),
        ("Workflow Class Integration", test_workflow_class_integration),
        ("Module Imports in Workflow", test_module_imports_in_workflow),
        ("Backwards Compatibility", test_backwards_compatibility)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        success = test_func()
        results.append((test_name, success))
        
        if success:
            print(f"✅ {test_name}: PASSED")
        else:
            print(f"❌ {test_name}: FAILED")
    
    try:
        disable_global_debug()
    except:
        pass
    
    # Summary
    print(f"\n📊 Test Summary:")
    print(f"=" * 50)
    
    passed = sum(1 for name, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print(f"🎉 All Day 3 tests passed! Workflow reorganization complete!")
        print(f"🚀 Ready for Day 4: Final validation and testing.")
        return True
    else:
        print(f"⚠️ Some tests failed. Please fix issues before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)