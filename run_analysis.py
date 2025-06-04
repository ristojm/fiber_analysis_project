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
from datetime import datetime

def setup_directories(project_root):
    """Create necessary directories"""
    output_dir = project_root / 'analysis_results'
    output_dir.mkdir(exist_ok=True)
    (output_dir / 'visualizations').mkdir(exist_ok=True)
    (output_dir / 'data').mkdir(exist_ok=True)
    (output_dir / 'reports').mkdir(exist_ok=True)
    print(f"✓ Output directories ready: {output_dir}")

def get_image_files(input_dir):
    """Get list of image files"""
    if not input_dir.exists():
        input_dir.mkdir(exist_ok=True)
        return []
    
    extensions = ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg']
    image_files = []
    for ext in extensions:
        image_files.extend(input_dir.glob(ext))
        image_files.extend(input_dir.glob(ext.upper()))
    return sorted(image_files)

def analyze_image_basic(image_path, project_root, has_porosity=False):
    """Run basic analysis on an image"""
    print(f"\nAnalyzing: {image_path.name}")
    print("-" * 40)
    
    try:
        # Load and preprocess
        from modules.image_preprocessing import load_and_preprocess
        from modules.fiber_type_detection import detect_fiber_type
        from modules.scale_detection import detect_scale_bar
        
        image = load_and_preprocess(str(image_path))
        print(f"✓ Image loaded: {image.shape}")
        
        # Detect fiber type
        fiber_type, confidence = detect_fiber_type(image)
        print(f"✓ Fiber type: {fiber_type} (confidence: {confidence:.3f})")
        
        # Detect scale
        # FIXED: Detect scale - handle both path and image array properly
        try:
            scale_info = detect_scale_bar(str(image_path))  # ✅ Now handles string paths correctly
            
            # Extract scale factor safely
            if isinstance(scale_info, dict):
                scale_factor = scale_info.get('micrometers_per_pixel', 0.0)
                if scale_info.get('scale_detected', False):
                    print(f"✓ Scale factor: {scale_factor:.4f} μm/pixel")
                else:
                    print(f"⚠️ Scale detection failed: {scale_info.get('error', 'Unknown error')}")
                    scale_factor = 1.0  # Default fallback
            else:
                # Handle case where function returns a float (old behavior)
                scale_factor = float(scale_info) if scale_info > 0 else 1.0
                if scale_factor > 0:
                    print(f"✓ Scale factor: {scale_factor:.4f} μm/pixel")
                else:
                    print("⚠️ Scale detection failed, using default scale")
                    scale_factor = 1.0
                    
        except Exception as scale_error:
            print(f"⚠️ Scale detection error: {scale_error}")
            scale_factor = 1.0
        
        # Basic porosity analysis if available
        if has_porosity:
            try:
                from modules.porosity_analysis import analyze_porosity
                import cv2
                import numpy as np
                
                # Create basic fiber mask for demo
                blurred = cv2.GaussianBlur(image, (5, 5), 0)
                _, binary = cv2.threshold((blurred * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                fiber_mask = binary > 128
                
                # Run porosity analysis
                porosity_results = analyze_porosity(image, fiber_mask, scale_factor, fiber_type, visualize=False)
                porosity_stats = porosity_results['porosity_statistics']
                
                print(f"✓ Porosity: {porosity_stats['porosity_percentage']:.2f}%")
                print(f"✓ Pore count: {porosity_stats['pore_count']}")
                print(f"✓ Mean pore diameter: {porosity_stats['mean_pore_diameter_um']:.3f} μm")
                
                # Save results
                output_dir = project_root / 'analysis_results'
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Save summary
                summary_file = output_dir / 'reports' / f"{image_path.stem}_analysis_{timestamp}.txt"
                with open(summary_file, 'w') as f:
                    f.write(f"SEM Fiber Analysis Results\n")
                    f.write(f"{'='*30}\n")
                    f.write(f"File: {image_path.name}\n")
                    f.write(f"Fiber Type: {fiber_type}\n")
                    f.write(f"Confidence: {confidence:.3f}\n")
                    f.write(f"Scale Factor: {scale_factor:.4f} μm/pixel\n")
                    f.write(f"Porosity: {porosity_stats['porosity_percentage']:.2f}%\n")
                    f.write(f"Pore Count: {porosity_stats['pore_count']}\n")
                    f.write(f"Mean Pore Diameter: {porosity_stats['mean_pore_diameter_um']:.3f} μm\n")
                
                print(f"✓ Results saved to: {summary_file}")
                
            except Exception as e:
                print(f"⚠️  Porosity analysis failed: {e}")
        
        print("✓ Analysis completed successfully!")
        
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

def run_integrated_analysis(project_root):
    """Run analysis directly when no external main file exists"""
    print("=" * 60)
    print("SEM Fiber Analysis System - Integrated Mode")
    print("=" * 60)
    
    # Setup directories
    setup_directories(project_root)
    
    # Check for images
    input_dir = project_root / 'sample_images'
    image_files = get_image_files(input_dir)
    
    if not image_files:
        print("No SEM images found for analysis.")
        print(f"Please add images to: {input_dir}")
        print("Supported formats: .tif, .tiff, .png, .jpg, .jpeg")
        return
    
    print(f"Found {len(image_files)} image(s) to analyze")
    
    # Add modules to path
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "modules"))
    
    # Try to run analysis
    try:
        # Import analysis modules
        from modules.image_preprocessing import load_and_preprocess
        from modules.fiber_type_detection import detect_fiber_type
        from modules.scale_detection import detect_scale_bar
        
        # Try to import porosity analysis
        try:
            from modules.porosity_analysis import analyze_porosity
            has_porosity = True
            print("✓ Porosity analysis module available")
        except ImportError:
            has_porosity = False
            print("⚠️  Porosity analysis module not found - basic analysis only")
        
        # Run basic analysis on first image
        analyze_image_basic(image_files[0], project_root, has_porosity)
        
        # If multiple images, ask user
        if len(image_files) > 1:
            print(f"\nFound {len(image_files)} images. Analyzing all...")
            for image_file in image_files[1:]:
                analyze_image_basic(image_file, project_root, has_porosity)
        
    except ImportError as e:
        print(f"✗ Missing required modules: {e}")
        print("Please ensure all modules exist in the modules/ directory")
        print("Required modules:")
        print("  - modules/image_preprocessing.py")
        print("  - modules/fiber_type_detection.py")
        print("  - modules/scale_detection.py")

def main():
    """Main entry point with proper encoding handling"""
    try:
        # Get the current directory and project structure
        current_dir = Path(__file__).parent
        
        # Check for Setup_Guides directory and PROJECT_SUMMARY.md
        setup_guides_dir = current_dir / "Setup_Guides"
        if setup_guides_dir.exists():
            project_summary = setup_guides_dir / "PROJECT_SUMMARY.md"
            if project_summary.exists():
                print("✓ Found Setup_Guides/PROJECT_SUMMARY.md")
                print("✓ Using project structure according to Setup_Guides")
            else:
                print("⚠️  Setup_Guides found but PROJECT_SUMMARY.md missing")
        else:
            print("⚠️  Setup_Guides directory not found, using standard structure")
        
        # Check for existing main analysis file (but don't require it)
        possible_locations = [
            current_dir / "fiber_analysis_main.py",           # Root directory (preferred)
            current_dir / "main.py",                          # Alternative main file
        ]
        
        main_analysis_file = None
        for location in possible_locations:
            if location.exists():
                main_analysis_file = location
                break
        
        if main_analysis_file:
            print("=" * 60)
            print("SEM Fiber Analysis System - External Script Mode")
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
            
            # Change to the script's directory for proper imports
            original_cwd = os.getcwd()
            script_dir = main_analysis_file.parent
            os.chdir(script_dir)
            
            # Add the project root and modules to Python path
            sys.path.insert(0, str(current_dir))
            sys.path.insert(0, str(current_dir / "modules"))
            
            print("Executing fiber analysis...")
            print("-" * 60)
            
            # Execute the code with proper globals
            exec(code, {'__file__': str(main_analysis_file)})
            
            print("-" * 60)
            print("Analysis completed successfully!")
            
            # Restore original working directory
            os.chdir(original_cwd)
        else:
            # No external file found, run integrated analysis
            print("No external main analysis file found.")
            print("Running integrated analysis from run_analysis.py")
            run_integrated_analysis(current_dir)
        
    except Exception as e:
        print(f"Error during execution: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        print("\nTroubleshooting tips:")
        print("1. Check if all required modules are installed: pip install -r requirements.txt")
        print("2. Ensure sample images are in the sample_images/ directory")
        print("3. Check that all module files exist in the modules/ directory")
        print("4. Try running individual modules to isolate the issue")
        print("5. Make sure you're running from the project root directory")

if __name__ == "__main__":
    main()