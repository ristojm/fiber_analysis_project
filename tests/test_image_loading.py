#!/usr/bin/env python3
"""
Diagnostic script to test image loading and identify encoding issues.
Run this to troubleshoot the UnicodeDecodeError.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

def test_image_loading():
    """Test image loading with various methods."""
    
    # Get project paths
    project_root = Path(__file__).parent
    sample_dir = project_root / "sample_images"
    
    print("="*60)
    print("SEM FIBER ANALYSIS - IMAGE LOADING DIAGNOSTIC")
    print("="*60)
    print(f"Project root: {project_root}")
    print(f"Sample directory: {sample_dir}")
    print(f"Sample directory exists: {sample_dir.exists()}")
    
    if not sample_dir.exists():
        print(f"❌ Sample directory not found: {sample_dir}")
        print("Please create the sample_images folder and add your JPG files")
        return
    
    # Find image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(sample_dir.glob(ext)))
        image_files.extend(list(sample_dir.glob(ext.upper())))
    
    print(f"\nFound {len(image_files)} image files:")
    for i, img_file in enumerate(image_files, 1):
        print(f"  {i}. {img_file.name} ({img_file.stat().st_size} bytes)")
    
    if not image_files:
        print("❌ No image files found!")
        print("Please add your JPG files to the sample_images folder")
        return
    
    # Test loading each image
    for img_file in image_files:
        print(f"\n" + "-"*50)
        print(f"Testing: {img_file.name}")
        print("-"*50)
        
        # Test 1: OpenCV
        print("1. Testing OpenCV loading...")
        try:
            img_cv = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img_cv is not None:
                print(f"   ✓ OpenCV success: {img_cv.shape}, dtype: {img_cv.dtype}")
            else:
                print(f"   ❌ OpenCV failed: returned None")
        except Exception as e:
            print(f"   ❌ OpenCV error: {e}")
        
        # Test 2: PIL
        print("2. Testing PIL loading...")
        try:
            from PIL import Image
            pil_img = Image.open(img_file)
            print(f"   ✓ PIL success: {pil_img.size}, mode: {pil_img.mode}")
            
            # Convert to grayscale array
            if pil_img.mode != 'L':
                pil_img = pil_img.convert('L')
            img_array = np.array(pil_img)
            print(f"   ✓ PIL array: {img_array.shape}, dtype: {img_array.dtype}")
            
        except Exception as e:
            print(f"   ❌ PIL error: {e}")
        
        # Test 3: Matplotlib
        print("3. Testing Matplotlib loading...")
        try:
            from matplotlib.image import imread
            img_mpl = imread(str(img_file))
            print(f"   ✓ Matplotlib success: {img_mpl.shape}, dtype: {img_mpl.dtype}")
        except Exception as e:
            print(f"   ❌ Matplotlib error: {e}")
        
        # Test 4: File properties
        print("4. File properties...")
        try:
            file_size = img_file.stat().st_size
            print(f"   File size: {file_size} bytes")
            
            # Read first few bytes to check file header
            with open(img_file, 'rb') as f:
                header = f.read(20)
                header_hex = ' '.join(f'{b:02x}' for b in header)
                print(f"   File header: {header_hex}")
                
                # Check for common image formats
                if header.startswith(b'\xff\xd8\xff'):
                    print(f"   ✓ JPEG format detected")
                elif header.startswith(b'\x89PNG'):
                    print(f"   ✓ PNG format detected")
                elif header.startswith(b'II*\x00') or header.startswith(b'MM\x00*'):
                    print(f"   ✓ TIFF format detected")
                else:
                    print(f"   ⚠ Unknown format")
                    
        except Exception as e:
            print(f"   ❌ File properties error: {e}")

def test_module_imports():
    """Test if all required modules can be imported."""
    print("\n" + "="*60)
    print("MODULE IMPORT TEST")
    print("="*60)
    
    modules_to_test = [
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
        ('matplotlib', 'Matplotlib'),
        ('skimage', 'scikit-image'),
        ('scipy', 'SciPy'),
        ('pandas', 'Pandas')
    ]
    
    for module_name, display_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✓ {display_name} ({module_name})")
        except ImportError as e:
            print(f"❌ {display_name} ({module_name}): {e}")

if __name__ == "__main__":
    print("Python version:", sys.version)
    print("Platform:", sys.platform)
    
    test_module_imports()
    test_image_loading()
    
    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)
    print("If you see errors above, they indicate the source of the UnicodeDecodeError.")
    print("Most likely causes:")
    print("1. Corrupted image file")
    print("2. Unsupported image format")
    print("3. File path encoding issues on Windows")
    print("4. Missing image processing libraries")