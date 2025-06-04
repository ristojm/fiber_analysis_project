# %%
"""
SEM Fiber Analysis System - Main Analysis Script
Automated Characterization of Hollow Fibers and Filaments

This script provides a complete pipeline for analyzing SEM images of hollow fibers and filaments.

Key Features:
- Automatic fiber type detection (hollow vs. solid)
- Scale bar detection and calibration
- Porosity analysis
- Texture analysis (crumbly detection)
- Defect detection (holes, ears)
- Batch processing capabilities

Phase 1 Implementation: Foundation with defect-free samples
"""

# %%
# 1. Setup and Imports

# Core imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add modules folder to Python path (simple approach)
import sys
from pathlib import Path

# Add the modules directory to Python path
modules_dir = Path("./modules")
if modules_dir.exists():
    sys.path.insert(0, str(modules_dir))
    print(f"✓ Added modules directory: {modules_dir.absolute()}")
else:
    print(f"⚠ Modules directory not found: {modules_dir.absolute()}")
    print("Please ensure the modules folder exists in the same directory as this script")

# Import our custom modules
try:
    from image_preprocessing import load_and_preprocess, preprocess_pipeline, visualize_preprocessing_steps
    from fiber_type_detection import FiberTypeDetector, detect_fiber_type, visualize_fiber_type_analysis
    from scale_detection import ScaleBarDetector, detect_scale_bar, manual_scale_calibration
    print("✓ All modules imported successfully!")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Please check that all .py files are in the modules/ folder")

# Set up matplotlib for better plots
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("SEM Fiber Analysis System - Phase 1")
print("Modules loaded successfully!")
print("\nCapabilities in this phase:")
print("✓ Image preprocessing and enhancement")
print("✓ Automatic fiber type detection (hollow vs. solid)")
print("✓ Scale bar detection and calibration")
print("✓ Basic fiber segmentation")
print("\nPlanned for future phases:")
print("○ Porosity analysis")
print("○ Texture analysis (crumbly detection)")
print("○ Defect detection (holes, ears)")

# %%
# 2. Configuration and Parameters

# Analysis configuration
CONFIG = {
    # File paths
    'input_directory': './sample_images/',  # Directory containing SEM images
    'output_directory': './analysis_results/',  # Directory for saving results
    
    # Image preprocessing
    'preprocessing': {
        'enhance_contrast_method': 'clahe',
        'denoise_method': 'bilateral',
        'remove_scale_bar': True,
        'normalize': True
    },
    
    # Fiber type detection (Updated parameters for better irregular lumen detection)
    'fiber_detection': {
        'min_fiber_area': 1000,
        'lumen_area_threshold': 0.02,  # Reduced for irregular lumens
        'circularity_threshold': 0.2,  # More lenient for irregular shapes  
        'confidence_threshold': 0.6    # Lower threshold for acceptance
    },
    
    # Scale detection
    'scale_detection': {
        'scale_region_fraction': 0.15,
        'min_bar_length': 50,
        'max_bar_thickness': 20
    },
    
    # Analysis options
    'show_intermediate_steps': True,
    'save_visualizations': True,
    'export_data': True
}

# Create output directory if it doesn't exist
Path(CONFIG['output_directory']).mkdir(parents=True, exist_ok=True)

print("Configuration loaded:")
for key, value in CONFIG.items():
    if isinstance(value, dict):
        print(f"  {key}:")
        for subkey, subvalue in value.items():
            print(f"    {subkey}: {subvalue}")
    else:
        print(f"  {key}: {value}")

# %%
# 3. Single Image Analysis Example

# Load and analyze a single image
# The system will look for images in the input directory specified in CONFIG
input_dir = Path(CONFIG['input_directory'])

# Option 1: Specify just the filename (it will look in input_directory)
IMAGE_FILENAME = 'your_image.jpg'  # Change this to your actual filename
IMAGE_PATH = input_dir / IMAGE_FILENAME

# Option 2: Auto-detect first image in input directory
# Uncomment these lines to automatically use the first image found:
# image_files = list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.tif')) + list(input_dir.glob('*.png'))
# if image_files:
#     IMAGE_PATH = image_files[0]
#     print(f"Auto-detected image: {IMAGE_PATH.name}")
# else:
#     print(f"No images found in {input_dir}")

print(f"Image path: {IMAGE_PATH}")

try:
    print(f"Analyzing image: {IMAGE_PATH}")
    
    # Check if the image file exists
    if not IMAGE_PATH.exists():
        print(f"❌ Image file not found: {IMAGE_PATH}")
        print(f"📁 Looking in directory: {input_dir.absolute()}")
        print(f"📋 Available files:")
        for file in input_dir.glob('*'):
            if file.is_file():
                print(f"   - {file.name}")
        raise FileNotFoundError(f"Image file not found: {IMAGE_PATH}")
    
    # Step 1: Load and preprocess image
    print("\n1. Image Preprocessing...")
    preprocessing_result = preprocess_pipeline(str(IMAGE_PATH), **CONFIG['preprocessing'])
    
    if CONFIG['show_intermediate_steps']:
        visualize_preprocessing_steps(preprocessing_result)
    
    # Get the main processed image (without scale bar)
    main_image = preprocessing_result['main_region']
    original_image = preprocessing_result['original']
    
    print(f"Original image shape: {original_image.shape}")
    print(f"Main analysis region shape: {main_image.shape}")
    
    # Step 2: Detect fiber type (PRIORITY #1)
    print("\n2. Fiber Type Detection...")
    detector = FiberTypeDetector(**CONFIG['fiber_detection'])
    fiber_type, confidence, analysis_data = detector.classify_fiber_type(main_image)
    
    print(f"Detected fiber type: {fiber_type}")
    print(f"Classification confidence: {confidence:.3f}")
    print(f"Number of fibers detected: {analysis_data['total_fibers']}")
    
    if fiber_type == 'hollow_fiber':
        print(f"Hollow fibers: {analysis_data['hollow_fibers']}")
        print(f"Solid filaments: {analysis_data['filaments']}")
    
    # Visualize fiber type analysis
    if CONFIG['show_intermediate_steps']:
        visualize_fiber_type_analysis(main_image, analysis_data)
    
    # Step 3: Scale bar detection and calibration
    print("\n3. Scale Bar Detection and Calibration...")
    scale_detector = ScaleBarDetector(**CONFIG['scale_detection'])
    scale_result = scale_detector.detect_scale_bar(original_image)
    
    if scale_result['scale_detected']:
        micrometers_per_pixel = scale_result['micrometers_per_pixel']
        scale_info = scale_result['scale_info']
        print(f"Scale detection: SUCCESS")
        print(f"Scale bar value: {scale_info['value']} {scale_info['unit']}")
        print(f"Calibration factor: {micrometers_per_pixel:.4f} μm/pixel")
    else:
        print(f"Scale detection: FAILED - {scale_result['error']}")
        # For demonstration, use manual calibration
        micrometers_per_pixel = 0.1  # Example: 0.1 μm/pixel
        print(f"Using manual calibration: {micrometers_per_pixel} μm/pixel")
    
    # Visualize scale detection
    if CONFIG['show_intermediate_steps']:
        scale_detector.visualize_scale_detection(original_image, scale_result)
    
    # Step 4: Compile results
    results = {
        'image_path': IMAGE_PATH,
        'fiber_type': fiber_type,
        'confidence': confidence,
        'total_fibers': analysis_data['total_fibers'],
        'hollow_fibers': analysis_data['hollow_fibers'],
        'filaments': analysis_data['filaments'],
        'scale_detected': scale_result['scale_detected'],
        'micrometers_per_pixel': micrometers_per_pixel,
        'scale_info': scale_result.get('scale_info', {})
    }
    
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    print(f"Image: {IMAGE_PATH}")
    print(f"Fiber Type: {fiber_type} (confidence: {confidence:.3f})")
    print(f"Total Fibers: {analysis_data['total_fibers']}")
    if fiber_type == 'hollow_fiber':
        print(f"Hollow Fibers: {analysis_data['hollow_fibers']}")
        print(f"Solid Filaments: {analysis_data['filaments']}")
    print(f"Scale Calibration: {micrometers_per_pixel:.4f} μm/pixel")
    print("="*50)
    
except Exception as e:
    print(f"Error analyzing image: {e}")
    print("\nPlease check:")
    print("1. Image file path is correct")
    print("2. Image file is readable")
    print("3. Required dependencies are installed")

# %%
# 4. Batch Processing Multiple Images

def analyze_image_batch(input_directory, file_extensions=('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
    """
    Process multiple images in a directory.
    """
    input_path = Path(input_directory)
    
    # Find all image files
    image_files = []
    for ext in file_extensions:
        image_files.extend(list(input_path.glob(f'*{ext}')))
        image_files.extend(list(input_path.glob(f'*{ext.upper()}')))
    
    if not image_files:
        print(f"No image files found in {input_directory}")
        return pd.DataFrame()
    
    print(f"Found {len(image_files)} image files")
    
    # Initialize results storage
    results_list = []
    
    # Initialize detectors
    fiber_detector = FiberTypeDetector(**CONFIG['fiber_detection'])
    scale_detector = ScaleBarDetector(**CONFIG['scale_detection'])
    
    # Process each image
    for i, image_path in enumerate(image_files, 1):
        print(f"\nProcessing {i}/{len(image_files)}: {image_path.name}")
        
        try:
            # Preprocess image
            preprocessing_result = preprocess_pipeline(str(image_path), **CONFIG['preprocessing'])
            main_image = preprocessing_result['main_region']
            original_image = preprocessing_result['original']
            
            # Fiber type detection
            fiber_type, confidence, analysis_data = fiber_detector.classify_fiber_type(main_image)
            
            # Scale detection
            scale_result = scale_detector.detect_scale_bar(original_image)
            micrometers_per_pixel = scale_result.get('micrometers_per_pixel', 0.0)
            
            # Compile results
            result = {
                'image_name': image_path.name,
                'image_path': str(image_path),
                'fiber_type': fiber_type,
                'confidence': confidence,
                'total_fibers': analysis_data['total_fibers'],
                'hollow_fibers': analysis_data['hollow_fibers'],
                'filaments': analysis_data['filaments'],
                'scale_detected': scale_result['scale_detected'],
                'micrometers_per_pixel': micrometers_per_pixel,
                'scale_value': scale_result.get('scale_info', {}).get('value', None),
                'scale_unit': scale_result.get('scale_info', {}).get('unit', None),
                'analysis_success': True,
                'error_message': None
            }
            
            results_list.append(result)
            print(f"  ✓ Type: {fiber_type} (confidence: {confidence:.3f})")
            print(f"  ✓ Fibers: {analysis_data['total_fibers']}")
            print(f"  ✓ Scale: {micrometers_per_pixel:.4f} μm/pixel")
            
        except Exception as e:
            # Record failed analysis
            result = {
                'image_name': image_path.name,
                'image_path': str(image_path),
                'fiber_type': 'error',
                'confidence': 0.0,
                'total_fibers': 0,
                'hollow_fibers': 0,
                'filaments': 0,
                'scale_detected': False,
                'micrometers_per_pixel': 0.0,
                'scale_value': None,
                'scale_unit': None,
                'analysis_success': False,
                'error_message': str(e)
            }
            results_list.append(result)
            print(f"  ✗ Error: {e}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Save results if requested
    if CONFIG['export_data']:
        output_file = Path(CONFIG['output_directory']) / 'batch_analysis_results.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
    
    return results_df

# Run batch analysis if input directory exists
input_dir = Path(CONFIG['input_directory'])
if input_dir.exists() and input_dir.is_dir():
    print(f"Starting batch analysis of directory: {input_dir}")
    batch_results = analyze_image_batch(input_dir)
    
    if not batch_results.empty:
        print(f"\nBatch Analysis Complete!")
        print(f"Successfully analyzed: {batch_results['analysis_success'].sum()} images")
        print(f"Failed analyses: {(~batch_results['analysis_success']).sum()} images")
        
        # Summary statistics
        successful_results = batch_results[batch_results['analysis_success']]
        if not successful_results.empty:
            print(f"\nSummary Statistics:")
            print(f"Hollow fibers detected: {(successful_results['fiber_type'] == 'hollow_fiber').sum()}")
            print(f"Filaments detected: {(successful_results['fiber_type'] == 'filament').sum()}")
            print(f"Average confidence: {successful_results['confidence'].mean():.3f}")
            print(f"Scale detection success rate: {successful_results['scale_detected'].mean()*100:.1f}%")
        
        # Display results table
        display_columns = ['image_name', 'fiber_type', 'confidence', 'total_fibers', 
                          'hollow_fibers', 'filaments', 'scale_detected', 'micrometers_per_pixel']
        print(f"\nDetailed Results:")
        print(batch_results[display_columns].to_string(index=False))
        
else:
    print(f"Input directory not found: {input_dir}")
    print("Skipping batch analysis. Update CONFIG['input_directory'] with the correct path.")

# %%
# 5. Demo Analysis with Provided Sample Images

# Demo with uploaded sample images
print("="*60)
print("DEMONSTRATION WITH PROVIDED SAMPLE IMAGES")
print("="*60)

# Simulated analysis results based on visual inspection
simulated_results = pd.DataFrame([
    {
        'image_name': 'hollow_fiber_sample.tif',
        'fiber_type': 'hollow_fiber',
        'confidence': 0.95,
        'total_fibers': 1,
        'hollow_fibers': 1,
        'filaments': 0,
        'scale_detected': True,
        'micrometers_per_pixel': 1.33,  # 400μm scale bar / ~300 pixels
        'scale_value': 400.0,
        'scale_unit': 'micrometer'
    },
    {
        'image_name': 'solid_filament_sample.tif', 
        'fiber_type': 'filament',
        'confidence': 0.92,
        'total_fibers': 1,
        'hollow_fibers': 0,
        'filaments': 1,
        'scale_detected': True,
        'micrometers_per_pixel': 1.67,  # 500μm scale bar / ~300 pixels
        'scale_value': 500.0,
        'scale_unit': 'micrometer'
    }
])

print(f"Simulated Analysis Results:")
print(simulated_results.to_string(index=False))

# %%
# 6. Phase 2 & 3 Development Roadmap

print("\n" + "="*60)
print("DEVELOPMENT ROADMAP - NEXT PHASES")
print("="*60)

print("""
Phase 2A: Porosity Analysis (Ready to implement)
- Pore detection and measurement
- Porosity quantification  
- Pore size distribution analysis
- Can work with current samples

Phase 2B: Texture Analysis (Awaiting crumbly samples)
- Surface roughness analysis
- Crumbly vs smooth classification
- Texture feature extraction

Phase 3: Defect Detection (Awaiting defect samples)
- 4-fold symmetrical hole detection
- Ear defect identification
- Defect size and position analysis
""")

# %%
# 7. Utility Functions and Tools

def list_available_images(directory=None):
    """List all image files in the specified directory."""
    if directory is None:
        directory = CONFIG['input_directory']
    
    input_dir = Path(directory)
    if not input_dir.exists():
        print(f"Directory not found: {input_dir}")
        return []
    
    # Common image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp']
    image_files = []
    
    for ext in extensions:
        image_files.extend(list(input_dir.glob(ext)))
        image_files.extend(list(input_dir.glob(ext.upper())))
    
    if image_files:
        print(f"📁 Found {len(image_files)} images in {input_dir}:")
        for i, img in enumerate(image_files, 1):
            print(f"   {i}. {img.name}")
    else:
        print(f"📁 No images found in {input_dir}")
    
    return image_files

def analyze_specific_image(filename):
    """Analyze a specific image by filename."""
    image_path = Path(CONFIG['input_directory']) / filename
    
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return None
    
    try:
        result = quick_analysis(str(image_path))
        print(f"✅ Analysis complete for {filename}:")
        print(f"   Type: {result['fiber_type']} (confidence: {result['confidence']:.3f})")
        print(f"   Scale: {result['scale_factor']:.4f} μm/pixel")
        return result
    except Exception as e:
        print(f"❌ Analysis failed for {filename}: {e}")
        return None

def quick_analysis(image_path):
    """Quick analysis function for single images."""
    try:
        # Load and preprocess
        processed_img = load_and_preprocess(image_path, **CONFIG['preprocessing'])
        
        # Detect fiber type
        fiber_type, confidence = detect_fiber_type(processed_img, **CONFIG['fiber_detection'])
        
        # Detect scale
        original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        scale_factor = detect_scale_bar(original_img, **CONFIG['scale_detection'])
        
        return {
            'fiber_type': fiber_type,
            'confidence': confidence,
            'scale_factor': scale_factor,
            'success': True
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def update_config(new_settings):
    """Update analysis configuration."""
    global CONFIG
    for key, value in new_settings.items():
        if key in CONFIG:
            if isinstance(CONFIG[key], dict) and isinstance(value, dict):
                CONFIG[key].update(value)
            else:
                CONFIG[key] = value
    print("Configuration updated successfully")

def export_results_excel(results_df, filename=None):
    """Export results to Excel with formatting."""
    if filename is None:
        filename = Path(CONFIG['output_directory']) / 'fiber_analysis_results.xlsx'
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Analysis Results', index=False)
        
        # Add summary sheet
        summary_data = {
            'Metric': ['Total Images', 'Successful Analyses', 'Hollow Fibers', 'Filaments', 
                      'Average Confidence', 'Scale Detection Rate'],
            'Value': [
                len(results_df),
                results_df['analysis_success'].sum(),
                (results_df['fiber_type'] == 'hollow_fiber').sum(),
                (results_df['fiber_type'] == 'filament').sum(),
                results_df[results_df['analysis_success']]['confidence'].mean(),
                results_df['scale_detected'].mean()
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"Results exported to: {filename}")

# Example usage of utility functions
print("Utility functions loaded:")
print("- list_available_images(): Show all images in input directory")
print("- analyze_specific_image(filename): Analyze a specific image")
print("- quick_analysis(image_path): Fast single image analysis")
print("- update_config(new_settings): Update analysis parameters") 
print("- export_results_excel(df): Export to formatted Excel file")

print("\n" + "="*50)
print("QUICK START COMMANDS")
print("="*50)
print("# List your images:")
print("list_available_images()")
print()
print("# Analyze a specific image:")
print("analyze_specific_image('your_image.jpg')")
print()
print("# Or update the IMAGE_FILENAME variable above and run the full analysis")

# %%
# 8. Next Steps and Sample Requirements

print("\n" + "="*60)
print("NEXT STEPS AND SAMPLE REQUIREMENTS")
print("="*60)

print("""
Immediate Next Steps:
1. Test fiber type detection on your samples
2. Validate scale bar detection accuracy
3. Implement porosity analysis module
4. Fine-tune parameters based on your data
5. Add batch processing for your image sets

For Phase 2 (Texture Analysis):
Please provide: SEM images showing fibers with crumbly textures alongside smooth ones
- This will enable training/tuning of texture classification algorithms
- Different degrees of "crumbly" texture would be ideal

For Phase 3 (Defect Detection):  
Please provide: SEM images containing:
- Fibers with 4-fold symmetrical hole patterns
- Fibers with "ear" defects (surface protrusions)
- Various defect sizes and positions

Validation Requirements:
- Ground truth data: Manual measurements for comparison
- Multiple magnifications: To test scale detection robustness  
- Various image qualities: Different contrast, noise levels
- Edge cases: Partially visible fibers, multiple fibers per image

The modular design allows us to iteratively improve each component as we get 
more sample types and validation data.

Ready to test with your samples! 🔬
""")