#!/usr/bin/env python3
"""
Debug what's actually being loaded by the module system
"""

import sys
from pathlib import Path

# Setup paths
current_dir = Path(__file__).parent
if current_dir.name == 'tests':
    project_root = current_dir.parent
else:
    project_root = current_dir

modules_dir = project_root / "modules"
sys.path.insert(0, str(modules_dir))

print("="*60)
print("DEBUGGING MODULE LOADING")
print("="*60)

# Check what Python is actually importing
print(f"Python path includes: {modules_dir}")
print(f"Module file exists: {(modules_dir / 'fiber_type_detection.py').exists()}")

# Import and check the actual class
from fiber_type_detection import FiberTypeDetector

# Create instance and check parameters
detector = FiberTypeDetector()
print(f"\n📋 ACTUAL DETECTOR PARAMETERS:")
print(f"  min_fiber_area: {detector.min_fiber_area}")
print(f"  lumen_area_threshold: {detector.lumen_area_threshold}")
print(f"  circularity_threshold: {detector.circularity_threshold}")
print(f"  confidence_threshold: {detector.confidence_threshold}")

# Check the module file location
import fiber_type_detection
print(f"\n📁 MODULE INFO:")
print(f"  Module file: {fiber_type_detection.__file__}")

# Check the __init__ method source (first few lines)
import inspect
init_source = inspect.getsource(FiberTypeDetector.__init__)
print(f"\n🔍 __init__ METHOD SOURCE:")
print("  " + "\n  ".join(init_source.split('\n')[:10]))

# Test a quick classification to see behavior
print(f"\n🧪 QUICK CLASSIFICATION TEST:")
try:
    from image_preprocessing import load_image
    img = load_image(str(project_root / "sample_images" / "hollow_fiber_sample.jpg"))
    
    # Preprocess and segment (check intermediate steps)
    preprocessed = detector.preprocess_for_detection(img)
    fiber_mask, fiber_properties = detector.segment_fibers(preprocessed)
    
    print(f"  Image shape: {img.shape}")
    print(f"  Fibers found: {len(fiber_properties)}")
    print(f"  Min area filter: {detector.min_fiber_area}")
    
    for i, props in enumerate(fiber_properties):
        area = props['area']
        print(f"    Fiber {i+1}: {area:.0f} pixels")
        
    # Check if large fiber passes filter
    all_areas = [props['area'] for props in fiber_properties]
    if all_areas:
        max_area = max(all_areas)
        print(f"  Largest area: {max_area:.0f}")
        print(f"  Passes filter: {max_area >= detector.min_fiber_area}")

except Exception as e:
    print(f"  Error in test: {e}")

print("\n" + "="*60)