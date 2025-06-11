"""
SEM Fiber Analysis System - Enhanced Modules Package
Now includes unified debug system and reorganized functions.
"""

# Import debug system first
from .debug_config import DEBUG_CONFIG, enable_global_debug, disable_global_debug, is_debug_enabled

# Import all functions from modules (including legacy functions)
from .image_preprocessing import load_image, load_and_preprocess, preprocess_pipeline
from .fiber_type_detection import FiberTypeDetector, detect_fiber_type
from .scale_detection import ScaleBarDetector, detect_scale_bar, manual_scale_calibration
from .crumbly_detection import CrumblyDetector

# Import enhanced functions (new reorganized functions)
try:
    from .image_preprocessing import preprocess_for_analysis
    HAS_ENHANCED_PREPROCESSING = True
except ImportError:
    HAS_ENHANCED_PREPROCESSING = False
    print("⚠️ Enhanced preprocessing not yet implemented")

try:
    from .fiber_type_detection import extract_fiber_mask_from_analysis, create_optimal_fiber_mask
    HAS_ENHANCED_FIBER_DETECTION = True
except ImportError:
    HAS_ENHANCED_FIBER_DETECTION = False
    print("⚠️ Enhanced fiber detection not yet implemented")

try:
    from .scale_detection import detect_scale_bar_from_crop
    HAS_ENHANCED_SCALE_DETECTION = True
except ImportError:
    HAS_ENHANCED_SCALE_DETECTION = False
    print("⚠️ Enhanced scale detection not yet implemented")

try:
    from .crumbly_detection import improve_crumbly_classification
    HAS_ENHANCED_CRUMBLY_DETECTION = True
except ImportError:
    HAS_ENHANCED_CRUMBLY_DETECTION = False
    print("⚠️ Enhanced crumbly detection not yet implemented")

# Try to import porosity analysis
try:
    from .porosity_analysis import PorosityAnalyzer, analyze_fiber_porosity
    POROSITY_AVAILABLE = True
    POROSITY_TYPE = 'advanced'
except ImportError:
    try:
        from .porosity_analysis import quick_porosity_check
        POROSITY_AVAILABLE = True
        POROSITY_TYPE = 'basic'
    except ImportError:
        POROSITY_AVAILABLE = False
        POROSITY_TYPE = None

# Version info
__version__ = "2.0.0"  # Updated for reorganization
__author__ = "SEM Fiber Analysis Team"

# Export all functions
__all__ = [
    # Core legacy functions (backward compatibility)
    'load_image', 'load_and_preprocess', 'preprocess_pipeline', 'detect_scale_bar', 'manual_scale_calibration',
    'detect_fiber_type',
    # Classes  
    'FiberTypeDetector', 'ScaleBarDetector', 'CrumblyDetector',
    # Debug control
    'enable_global_debug', 'disable_global_debug', 'is_debug_enabled', 'DEBUG_CONFIG',
    # Module info
    'POROSITY_AVAILABLE', 'POROSITY_TYPE',
    'HAS_ENHANCED_PREPROCESSING', 'HAS_ENHANCED_FIBER_DETECTION', 
    'HAS_ENHANCED_SCALE_DETECTION', 'HAS_ENHANCED_CRUMBLY_DETECTION'
]

# Add enhanced functions to exports as they become available
if HAS_ENHANCED_PREPROCESSING:
    __all__.append('preprocess_for_analysis')
if HAS_ENHANCED_FIBER_DETECTION:
    __all__.extend(['extract_fiber_mask_from_analysis', 'create_optimal_fiber_mask'])
if HAS_ENHANCED_SCALE_DETECTION:
    __all__.append('detect_scale_bar_from_crop')
if HAS_ENHANCED_CRUMBLY_DETECTION:
    __all__.append('improve_crumbly_classification')

print(f"SEM Fiber Analysis Modules v{__version__} loaded successfully!")
print(f"Enhanced with unified debug system and reorganized architecture")
print(f"Enhanced functions available: preprocessing={HAS_ENHANCED_PREPROCESSING}, "
      f"fiber={HAS_ENHANCED_FIBER_DETECTION}, scale={HAS_ENHANCED_SCALE_DETECTION}, "
      f"crumbly={HAS_ENHANCED_CRUMBLY_DETECTION}")