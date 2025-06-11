"""
Centralized debug configuration for SEM Fiber Analysis System.
Provides unified debug control across all modules.
"""

import os
from pathlib import Path
from typing import Optional

class DebugConfig:
    """Centralized debug configuration for all modules."""
    
    def __init__(self):
        self.enabled = False
        self.save_images = False
        self.show_plots = False
        self.verbose_output = False
        self.output_dir = None
    
    def enable_debug(self, save_images=True, show_plots=False, 
                    verbose=True, output_dir=None):
        """Enable debug mode with specified options."""
        self.enabled = True
        self.save_images = save_images
        self.show_plots = show_plots
        self.verbose_output = verbose
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔧 Debug mode enabled:")
        print(f"   Save images: {save_images}")
        print(f"   Show plots: {show_plots}")
        print(f"   Verbose output: {verbose}")
        print(f"   Output directory: {self.output_dir}")
    
    def disable_debug(self):
        """Disable all debug output."""
        self.enabled = False
        self.save_images = False
        self.show_plots = False
        self.verbose_output = False
        print("🔧 Debug mode disabled")
    
    def get_debug_path(self, filename: str) -> Optional[Path]:
        """Get path for debug file output."""
        if self.output_dir and self.save_images:
            return self.output_dir / filename
        return None

# Global debug instance
DEBUG_CONFIG = DebugConfig()

# Convenience functions
def enable_global_debug(save_images=True, show_plots=False, output_dir=None):
    """Enable debug mode across all modules."""
    DEBUG_CONFIG.enable_debug(save_images, show_plots, True, output_dir)

def disable_global_debug():
    """Disable debug mode across all modules."""
    DEBUG_CONFIG.disable_debug()

def is_debug_enabled():
    """Check if debug mode is enabled."""
    return DEBUG_CONFIG.enabled