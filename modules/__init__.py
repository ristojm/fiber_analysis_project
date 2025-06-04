"""
SEM Fiber Analysis System - Modules Package

This package contains all the core analysis modules for the SEM fiber analysis system.

Modules:
- image_preprocessing: Image loading, enhancement, and preparation
- fiber_type_detection: Automatic fiber type classification (hollow vs solid)
- scale_detection: Scale bar recognition and calibration
- porosity_analysis: Pore detection and porosity quantification (future)
- texture_analysis: Crumbly texture detection (future)
- defect_detection: Hole and ear defect identification (future)
"""

# Import main functions for easy access
from .image_preprocessing import load_and_preprocess, preprocess_pipeline
from .fiber_type_detection import FiberTypeDetector, detect_fiber_type
from .scale_detection import ScaleBarDetector, detect_scale_bar, manual_scale_calibration

# Version info
__version__ = "1.0.0"
__author__ = "SEM Fiber Analysis Team"

# Make key classes and functions available at package level
__all__ = [
    'load_and_preprocess',
    'preprocess_pipeline', 
    'FiberTypeDetector',
    'detect_fiber_type',
    'ScaleBarDetector', 
    'detect_scale_bar',
    'manual_scale_calibration'
]

print(f"SEM Fiber Analysis Modules v{__version__} loaded successfully!")