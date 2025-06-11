"""
SEM Fiber Analysis System - Unified Module Interface
This file exposes all public functions and classes from the modules package.
"""

# ===== CORE IMPORTS =====
# Image preprocessing
from .image_preprocessing import (
    load_image, 
    preprocess_for_analysis,
    preprocess_pipeline,
    remove_scale_bar_region,
    enhance_contrast,
    denoise_image,
    normalize_image,
    load_and_preprocess
)

# Scale detection
from .scale_detection import (
    detect_scale_bar,
    detect_scale_bar_from_crop,
    ScaleBarDetector,
    detect_scale_factor_only
)

# Fiber type detection
from .fiber_type_detection import (
    FiberTypeDetector,
    detect_fiber_type,
    create_optimal_fiber_mask,
    extract_fiber_mask_from_analysis,
    visualize_fiber_type_analysis
)

# Porosity analysis
try:
    from .porosity_analysis import (
        PorosityAnalyzer,
        analyze_fiber_porosity,
        quick_porosity_check,
        visualize_enhanced_porosity_results
    )
    POROSITY_AVAILABLE = True
    POROSITY_TYPE = "fast_refined"
except ImportError:
    POROSITY_AVAILABLE = False
    POROSITY_TYPE = None
    # Define dummy functions for compatibility
    def analyze_fiber_porosity(*args, **kwargs):
        return {'error': 'Porosity analysis not available'}
    def quick_porosity_check(*args, **kwargs):
        return 0.0

# Crumbly detection
from .crumbly_detection import (
    CrumblyDetector,
    detect_crumbly_texture,
    improve_crumbly_classification,
    batch_analyze_crumbly_texture
)

# Debug configuration
from .debug_config import (
    enable_global_debug,
    disable_global_debug,
    is_debug_enabled,
    DEBUG_CONFIG
)

# ===== MODULE FLAGS =====
# These flags indicate which enhanced features are available
HAS_ENHANCED_PREPROCESSING = True
HAS_ENHANCED_SCALE_DETECTION = True
HAS_ENHANCED_FIBER_DETECTION = True
HAS_ENHANCED_CRUMBLY_DETECTION = True

# ===== UTILITY FUNCTIONS =====
def standardize_porosity_result(result):
    """
    Ensure porosity result is always a properly formatted dictionary.
    
    Args:
        result: Raw porosity analysis result (dict, float, or other)
        
    Returns:
        dict: Standardized porosity result dictionary
    """
    if isinstance(result, dict) and 'porosity_metrics' in result:
        # Full porosity analyzer result
        metrics = result['porosity_metrics']
        return {
            'porosity_percentage': metrics.get('total_porosity_percent', 0.0),
            'total_pores': metrics.get('pore_count', 0),
            'average_pore_size': metrics.get('average_pore_size_um2', 0.0),
            'pore_density': metrics.get('pore_density_per_mm2', 0.0),
            'method': 'full_analysis',
            'full_result': result
        }
    elif isinstance(result, dict):
        # Already a dictionary, ensure required fields
        return {
            'porosity_percentage': result.get('porosity_percentage', 0.0),
            'total_pores': result.get('total_pores', 0),
            'average_pore_size': result.get('average_pore_size', 0.0),
            'method': result.get('method', 'dict_passthrough'),
            **result
        }
    elif isinstance(result, (int, float, np.number)):
        # Single porosity value
        return {
            'porosity_percentage': float(result),
            'total_pores': 0,
            'average_pore_size': 0.0,
            'method': 'single_value'
        }
    else:
        # Unknown format
        return {
            'porosity_percentage': 0.0,
            'total_pores': 0,
            'average_pore_size': 0.0,
            'method': 'unknown',
            'error': f'Unknown result type: {type(result)}'
        }

# ===== EXPORT LIST =====
__all__ = [
    # Image preprocessing
    'load_image', 'preprocess_for_analysis', 'preprocess_pipeline',
    'remove_scale_bar_region', 'enhance_contrast', 'denoise_image',
    'normalize_image', 'load_and_preprocess',
    
    # Scale detection
    'detect_scale_bar', 'detect_scale_bar_from_crop', 'ScaleBarDetector',
    'detect_scale_factor_only',
    
    # Fiber detection
    'FiberTypeDetector', 'detect_fiber_type', 'create_optimal_fiber_mask',
    'extract_fiber_mask_from_analysis', 'visualize_fiber_type_analysis',
    
    # Porosity analysis
    'PorosityAnalyzer', 'analyze_fiber_porosity', 'quick_porosity_check',
    'visualize_enhanced_porosity_results', 'standardize_porosity_result',
    
    # Crumbly detection
    'CrumblyDetector', 'detect_crumbly_texture', 
    'improve_crumbly_classification', 'batch_analyze_crumbly_texture',
    
    # Debug
    'enable_global_debug', 'disable_global_debug', 'is_debug_enabled',
    'DEBUG_CONFIG',
    
    # Flags
    'HAS_ENHANCED_PREPROCESSING', 'HAS_ENHANCED_SCALE_DETECTION',
    'HAS_ENHANCED_FIBER_DETECTION', 'HAS_ENHANCED_CRUMBLY_DETECTION',
    'POROSITY_AVAILABLE', 'POROSITY_TYPE'
]

# Import numpy for utility functions
import numpy as np

print("✅ SEM Fiber Analysis modules loaded successfully")