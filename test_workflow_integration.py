#!/usr/bin/env python3
"""Test script to verify the workflow integration is working correctly."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all imports work correctly."""
    print("🧪 Testing module imports...")
    
    try:
        # Test importing from modules package
        from modules import (
            load_image, preprocess_for_analysis,
            detect_scale_bar_from_crop,
            FiberTypeDetector, create_optimal_fiber_mask,
            analyze_fiber_porosity, standardize_porosity_result,
            CrumblyDetector, improve_crumbly_classification,
            enable_global_debug, disable_global_debug,
            HAS_ENHANCED_PREPROCESSING, POROSITY_AVAILABLE
        )
        
        print("✅ All imports successful!")
        print(f"   Preprocessing available: {HAS_ENHANCED_PREPROCESSING}")
        print(f"   Porosity available: {POROSITY_AVAILABLE}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_workflow():
    """Test the complete workflow with a dummy image."""
    print("\n🧪 Testing workflow integration...")
    
    try:
        from multiprocessing_crumbly_workflow import process_single_image_orchestrator
        import numpy as np
        from pathlib import Path
        
        # Create a dummy image file
        test_dir = Path("test_workflow_temp")
        test_dir.mkdir(exist_ok=True)
        
        test_image_path = test_dir / "test_image.png"
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        import cv2
        cv2.imwrite(str(test_image_path), dummy_image)
        
        # Test the orchestrator
        worker_args = {
            'image_path': test_image_path,
            'debug': True
        }
        
        print("\n🚀 Running orchestrator test...")
        result = process_single_image_orchestrator(worker_args)
        
        # Check results
        if result.get('success'):
            print("\n✅ Workflow test successful!")
            print(f"   Processing steps completed: {len(result.get('processing_steps', []))}")
            print(f"   Steps: {', '.join(result.get('processing_steps', []))}")
        else:
            print("\n⚠️ Workflow completed with issues:")
            print(f"   Error: {result.get('error', 'Unknown error')}")
        
        # Cleanup
        test_image_path.unlink()
        test_dir.rmdir()
        
        return result.get('success', False)
        
    except Exception as e:
        print(f"\n❌ Workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🔧 SEM Fiber Analysis Workflow Integration Test")
    print("=" * 50)
    
    # Test imports
    import_success = test_imports()
    
    if not import_success:
        print("\n❌ Cannot proceed without successful imports")
        return False
    
    # Test workflow
    workflow_success = test_workflow()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY:")
    print(f"   Module imports: {'✅ PASS' if import_success else '❌ FAIL'}")
    print(f"   Workflow test: {'✅ PASS' if workflow_success else '❌ FAIL'}")
    
    overall_success = import_success and workflow_success
    print(f"\n{'🎉 ALL TESTS PASSED!' if overall_success else '❌ SOME TESTS FAILED'}")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)