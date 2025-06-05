"""
Results Directory Configuration for SEM Fiber Analysis System
Centralizes all output paths to the results/ folder with organized subdirectories

This module provides a unified interface for all analysis modules to save their results
in the correct location within the results/ folder structure.
"""

from pathlib import Path
from datetime import datetime
import os
from typing import Optional, Union

# ===== CONFIGURATION =====
# Base results directory - change this to relocate all outputs
RESULTS_BASE_DIR = Path("results")

# ===== SUBDIRECTORY STRUCTURE =====
# All analysis outputs are organized into these subdirectories
BATCH_ANALYSIS_DIR = RESULTS_BASE_DIR / "batch_analysis"
SCALE_DETECTION_DIR = RESULTS_BASE_DIR / "scale_detection" 
DEBUG_OUTPUT_DIR = RESULTS_BASE_DIR / "debug_output"
INDIVIDUAL_ANALYSIS_DIR = RESULTS_BASE_DIR / "individual_analysis"
VISUALIZATION_DIR = RESULTS_BASE_DIR / "visualizations"
MULTIPROCESSING_DIR = RESULTS_BASE_DIR / "multiprocessing_results"
TEST_RESULTS_DIR = RESULTS_BASE_DIR / "test_results"
COMPREHENSIVE_DIR = RESULTS_BASE_DIR / "comprehensive_analysis"

# ===== INITIALIZATION =====
def initialize_results_directories():
    """Create all results directories if they don't exist."""
    directories = [
        RESULTS_BASE_DIR,
        BATCH_ANALYSIS_DIR,
        SCALE_DETECTION_DIR,
        DEBUG_OUTPUT_DIR,
        INDIVIDUAL_ANALYSIS_DIR,
        VISUALIZATION_DIR,
        MULTIPROCESSING_DIR,
        TEST_RESULTS_DIR,
        COMPREHENSIVE_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    return len(directories)

# Initialize directories on import
num_dirs = initialize_results_directories()

# ===== UTILITY FUNCTIONS =====
def get_timestamped_filename(base_name: str, extension: str, subdir: Optional[Path] = None) -> Path:
    """
    Generate timestamped filename in appropriate results subdirectory.
    
    Args:
        base_name: Base name for the file (without extension)
        extension: File extension (without dot)
        subdir: Specific subdirectory (optional)
    
    Returns:
        Full path with timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{timestamp}.{extension}"
    
    if subdir:
        return subdir / filename
    else:
        return RESULTS_BASE_DIR / filename

def ensure_directory_exists(path: Union[str, Path]) -> Path:
    """Ensure a directory exists, create if necessary."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

# ===== BATCH ANALYSIS FUNCTIONS =====
def get_batch_analysis_path(filename: str) -> Path:
    """Get path for batch analysis results."""
    return BATCH_ANALYSIS_DIR / filename

def get_multiprocessing_path(filename: str) -> Path:
    """Get path for multiprocessing analyzer results."""
    return MULTIPROCESSING_DIR / filename

def get_comprehensive_analysis_path(filename: str) -> Path:
    """Get path for comprehensive analysis results."""
    return COMPREHENSIVE_DIR / filename

# ===== SCALE DETECTION FUNCTIONS =====
def get_scale_detection_path(filename: str) -> Path:
    """Get path for scale detection results."""
    return SCALE_DETECTION_DIR / filename

def get_scale_batch_test_path(filename: str) -> Path:
    """Get path for scale detection batch test results."""
    return SCALE_DETECTION_DIR / "batch_tests" / filename

# ===== DEBUG AND TESTING FUNCTIONS =====
def get_debug_output_path(filename: str) -> Path:
    """Get path for debug output."""
    return DEBUG_OUTPUT_DIR / filename

def get_test_session_dir() -> Path:
    """Create a new timestamped test session directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = TEST_RESULTS_DIR / f"test_session_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir

def get_test_results_path(filename: str) -> Path:
    """Get path for test results."""
    return TEST_RESULTS_DIR / filename

# ===== INDIVIDUAL ANALYSIS FUNCTIONS =====
def get_individual_analysis_path(filename: str) -> Path:
    """Get path for individual analysis results."""
    return INDIVIDUAL_ANALYSIS_DIR / filename

def get_individual_session_dir(session_name: Optional[str] = None) -> Path:
    """Create a new individual analysis session directory."""
    if session_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = f"individual_session_{timestamp}"
    
    session_dir = INDIVIDUAL_ANALYSIS_DIR / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir

# ===== VISUALIZATION FUNCTIONS =====
def get_visualization_path(filename: str) -> Path:
    """Get path for visualization output."""
    return VISUALIZATION_DIR / filename

def get_visualization_session_dir(session_name: Optional[str] = None) -> Path:
    """Create a new visualization session directory."""
    if session_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = f"viz_session_{timestamp}"
    
    session_dir = VISUALIZATION_DIR / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir

# ===== SPECIALIZED FILENAME GENERATORS =====
def get_excel_report_path(report_type: str = "COMPREHENSIVE_ANALYSIS") -> Path:
    """Generate timestamped Excel report path."""
    return get_timestamped_filename(report_type, "xlsx", MULTIPROCESSING_DIR)

def get_json_results_path(result_type: str = "batch_results") -> Path:
    """Generate timestamped JSON results path."""
    return get_timestamped_filename(result_type, "json", MULTIPROCESSING_DIR)

def get_debug_image_path(image_name: str) -> Path:
    """Generate timestamped debug image path."""
    base_name = Path(image_name).stem
    return get_timestamped_filename(f"debug_{base_name}", "png", DEBUG_OUTPUT_DIR)

def get_scale_debug_path(debug_type: str = "scale_detection") -> Path:
    """Generate scale detection debug path."""
    return get_timestamped_filename(debug_type, "png", SCALE_DETECTION_DIR)

# ===== CONFIGURATION INFO =====
def get_results_info() -> dict:
    """Get information about the results configuration."""
    return {
        'base_directory': str(RESULTS_BASE_DIR.absolute()),
        'subdirectories': {
            'batch_analysis': str(BATCH_ANALYSIS_DIR),
            'scale_detection': str(SCALE_DETECTION_DIR),
            'debug_output': str(DEBUG_OUTPUT_DIR),
            'individual_analysis': str(INDIVIDUAL_ANALYSIS_DIR),
            'visualizations': str(VISUALIZATION_DIR),
            'multiprocessing': str(MULTIPROCESSING_DIR),
            'test_results': str(TEST_RESULTS_DIR),
            'comprehensive': str(COMPREHENSIVE_DIR)
        },
        'directories_created': num_dirs,
        'initialized': True
    }

def print_results_structure():
    """Print the results directory structure."""
    print("📁 SEM Analysis Results Directory Structure:")
    print(f"   Base: {RESULTS_BASE_DIR.absolute()}")
    print("   Subdirectories:")
    print(f"     📊 batch_analysis/     - Batch processing results")
    print(f"     📏 scale_detection/    - Scale bar detection results")
    print(f"     🐛 debug_output/       - Debug images and logs")
    print(f"     🔬 individual_analysis/ - Single image analysis")
    print(f"     📈 visualizations/     - Plots and charts")
    print(f"     ⚡ multiprocessing_results/ - Parallel processing results")
    print(f"     🧪 test_results/       - Test outputs")
    print(f"     🎯 comprehensive_analysis/ - Full pipeline results")

# ===== CLEANUP FUNCTIONS =====
def cleanup_old_results(days_old: int = 30, dry_run: bool = True) -> dict:
    """
    Clean up old result files (optional maintenance function).
    
    Args:
        days_old: Files older than this many days will be cleaned
        dry_run: If True, only report what would be deleted
    
    Returns:
        Dictionary with cleanup statistics
    """
    import time
    from datetime import timedelta
    
    cutoff_time = time.time() - (days_old * 24 * 60 * 60)
    stats = {'files_found': 0, 'files_deleted': 0, 'total_size_mb': 0}
    
    if not RESULTS_BASE_DIR.exists():
        return stats
    
    for file_path in RESULTS_BASE_DIR.rglob('*'):
        if file_path.is_file():
            try:
                file_time = file_path.stat().st_mtime
                if file_time < cutoff_time:
                    stats['files_found'] += 1
                    stats['total_size_mb'] += file_path.stat().st_size / (1024 * 1024)
                    
                    if not dry_run:
                        file_path.unlink()
                        stats['files_deleted'] += 1
            except Exception:
                pass  # Skip files we can't process
    
    return stats

# ===== MODULE METADATA =====
__version__ = "1.0.0"
__author__ = "SEM Fiber Analysis Team"

# Export main functions
__all__ = [
    # Directory paths
    'RESULTS_BASE_DIR', 'BATCH_ANALYSIS_DIR', 'SCALE_DETECTION_DIR', 
    'DEBUG_OUTPUT_DIR', 'INDIVIDUAL_ANALYSIS_DIR', 'VISUALIZATION_DIR',
    'MULTIPROCESSING_DIR', 'TEST_RESULTS_DIR', 'COMPREHENSIVE_DIR',
    
    # Main functions
    'get_batch_analysis_path', 'get_multiprocessing_path', 'get_comprehensive_analysis_path',
    'get_scale_detection_path', 'get_scale_batch_test_path',
    'get_debug_output_path', 'get_test_session_dir', 'get_test_results_path',
    'get_individual_analysis_path', 'get_individual_session_dir',
    'get_visualization_path', 'get_visualization_session_dir',
    
    # Specialized generators
    'get_excel_report_path', 'get_json_results_path', 'get_debug_image_path',
    'get_scale_debug_path', 'get_timestamped_filename',
    
    # Utilities
    'initialize_results_directories', 'ensure_directory_exists',
    'get_results_info', 'print_results_structure', 'cleanup_old_results'
]

# Print configuration on import (only in debug mode)
if os.environ.get('SEM_ANALYSIS_DEBUG', '').lower() in ['true', '1', 'yes']:
    print_results_structure()
else:
    # Minimal notification
    print(f"✅ Results configuration loaded: {num_dirs} directories initialized in {RESULTS_BASE_DIR}")