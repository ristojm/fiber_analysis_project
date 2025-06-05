#!/usr/bin/env python3
"""
Verify that the scale detection module has been updated
"""

def check_module_version():
    """Check if the scale detection module has the new features."""
    
    print("🔍 CHECKING MODULE VERSION")
    print("=" * 30)
    
    try:
        import sys
        from pathlib import Path
        
        # Add modules to path
        project_root = Path(__file__).parent.parent
        if (project_root / "modules").exists():
            sys.path.insert(0, str(project_root / "modules"))
        else:
            sys.path.insert(0, str(project_root))
        
        # Import the module
        from scale_detection import ScaleBarDetector
        
        # Check if the new methods exist
        detector = ScaleBarDetector()
        
        # Check for new methods
        new_methods = [
            'find_any_scale_text',
            'estimate_line_length_from_text',
            '_find_any_text_tesseract'
        ]
        
        has_new_methods = all(hasattr(detector, method) for method in new_methods)
        
        if has_new_methods:
            print("✅ NEW VERSION DETECTED!")
            print("   - Has find_any_scale_text method")
            print("   - Has estimate_line_length_from_text method")
            print("   - Has dual strategy approach")
            print("\n🎯 Your module is updated and ready!")
            return True
        else:
            print("❌ OLD VERSION DETECTED!")
            print("   Missing new methods:")
            for method in new_methods:
                if not hasattr(detector, method):
                    print(f"   - Missing: {method}")
            
            print("\n🔧 ACTION NEEDED:")
            print("   1. Replace modules/scale_detection.py with the updated version")
            print("   2. Copy the code from the 'Updated Scale Detection Module' artifact")
            print("   3. Run this test again to verify")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    success = check_module_version()
    
    if success:
        print("\n✅ Ready to test! The batch test should now work better.")
    else:
        print("\n⚠️  Update needed before testing.")