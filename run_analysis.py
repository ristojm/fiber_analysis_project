#!/usr/bin/env python3
"""
Fixed run_analysis.py - Solves Unicode encoding issues
SEM Fiber Analysis System - Main Entry Point

This script fixes the Unicode encoding error and provides a robust entry point
for the fiber analysis system.
"""

import sys
import os
from pathlib import Path
import traceback

def main():
    """Main entry point with proper encoding handling"""
    try:
        # Get the current directory and project structure
        current_dir = Path(__file__).parent
        
        # Check multiple possible locations for the main analysis script
        possible_locations = [
            current_dir / "notebooks" / "fiber_analysis_main.py",
            current_dir / "notebooks" / "fiber_analysis_main.ipynb", 
            current_dir / "fiber_analysis_main.py",
            current_dir / "main.py",
            current_dir / "notebooks" / "main.py"
        ]
        
        main_analysis_file = None
        notebooks_dir = current_dir / "notebooks"
        
        # Find the main analysis file
        for location in possible_locations:
            if location.exists():
                main_analysis_file = location
                if "notebooks" in str(location):
                    notebooks_dir = location.parent
                else:
                    notebooks_dir = current_dir
                break
        
        if not main_analysis_file.exists():
            print(f"Error: {main_analysis_file} not found!")
            print(f"Please ensure the file exists in: {notebooks_dir}")
            return
        
        print("=" * 60)
        print("SEM Fiber Analysis System")
        print("=" * 60)
        print(f"Loading analysis from: {main_analysis_file}")
        
        # Read the file with proper UTF-8 encoding
        try:
            with open(main_analysis_file, 'r', encoding='utf-8') as f:
                code = f.read()
        except UnicodeDecodeError:
            # Fallback to latin-1 if UTF-8 fails
            print("UTF-8 failed, trying latin-1 encoding...")
            with open(main_analysis_file, 'r', encoding='latin-1') as f:
                code = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            print("Trying with error handling...")
            with open(main_analysis_file, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
        
        # Change to the notebooks directory for proper imports
        original_cwd = os.getcwd()
        os.chdir(notebooks_dir)
        
        # Add the project root and modules to Python path
        sys.path.insert(0, str(current_dir))
        sys.path.insert(0, str(current_dir / "modules"))
        
        print("Executing fiber analysis...")
        print("-" * 60)
        
        # Execute the code with proper globals
        exec(code, {'__file__': str(main_analysis_file)})
        
        print("-" * 60)
        print("Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        print("\nTroubleshooting tips:")
        print("1. Check if all required modules are installed: pip install -r requirements.txt")
        print("2. Ensure sample images are in the sample_images/ directory")
        print("3. Check that all module files exist in the modules/ directory")
        print("4. Try running individual modules to isolate the issue")
        
    finally:
        # Restore original working directory
        if 'original_cwd' in locals():
            os.chdir(original_cwd)

if __name__ == "__main__":
    main()