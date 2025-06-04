#!/usr/bin/env python3
"""
SEM Fiber Analysis - Main Runner Script
Run this from the project root directory to ensure correct paths.
"""

import sys
from pathlib import Path

# Ensure we're running from project root
project_root = Path(__file__).parent
print(f"Running from project root: {project_root}")

# Add modules to path
modules_dir = project_root / "modules"
sys.path.insert(0, str(modules_dir))

# Now run the main analysis
if __name__ == "__main__":
    # Import and run the main analysis
    import os
    os.chdir(project_root)  # Ensure working directory is project root
    
    # Import the main analysis script
    notebooks_dir = project_root / "notebooks"
    sys.path.insert(0, str(notebooks_dir))
    
    print("="*60)
    print("SEM FIBER ANALYSIS SYSTEM")
    print("="*60)
    print(f"Project root: {project_root}")
    print(f"Sample images: {project_root / 'sample_images'}")
    print(f"Results output: {project_root / 'analysis_results'}")
    print("="*60)
    
    # Execute the main analysis
    exec(open(notebooks_dir / "fiber_analysis_main.py").read())