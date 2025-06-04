"""
SEM Fiber Analysis System - Image Preprocessing Module
Handles image loading, enhancement, and preparation for analysis.
"""

import cv2
import numpy as np
from skimage import filters, morphology, restoration, exposure
from scipy import ndimage
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union
import warnings

def load_image(image_path: str) -> np.ndarray:
    """
    Load SEM image from file path with robust encoding handling.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Loaded image as numpy array
    """
    try:
        # Try OpenCV first (handles most formats well)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            # Fallback to matplotlib for other formats
            try:
                from matplotlib.image import imread
                img = imread(image_path)
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                # Convert to uint8 if needed
                if img.dtype != np.uint8:
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
            except Exception as e:
                # Final fallback - try PIL
                try:
                    from PIL import Image
                    pil_img = Image.open(image_path)
                    if pil_img.mode != 'L':
                        pil_img = pil_img.convert('L')
                    img = np.array(pil_img)
                except Exception as pil_e:
                    raise ValueError(f"Could not load image from {image_path}. OpenCV error: Failed to load. PIL error: {pil_e}")
                
        return img
    except Exception as e:
        raise ValueError(f"Could not load image from {image_path}: {str(e)}")

def enhance_contrast(image: np.ndarray, method: str = 'clahe') -> np.ndarray:
    """
    Enhance image contrast using various methods.
    
    Args:
        image: Input grayscale image
        method: Enhancement method ('clahe', 'histogram_eq', 'adaptive_eq')
        
    Returns:
        Contrast-enhanced image
    """
    if method == 'clahe':
        # Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    
    elif method == 'histogram_eq':
        # Global histogram equalization
        return cv2.equalizeHist(image)
    
    elif method == 'adaptive_eq':
        # Adaptive histogram equalization using skimage
        return exposure.equalize_adapthist(image, clip_limit=0.02)
    
    else:
        raise ValueError(f"Unknown enhancement method: {method}")

def denoise_image(image: np.ndarray, method: str = 'gaussian', **kwargs) -> np.ndarray:
    """
    Remove noise from SEM images.
    
    Args:
        image: Input image
        method: Denoising method ('gaussian', 'bilateral', 'non_local_means', 'wiener')
        **kwargs: Additional parameters for specific methods
        
    Returns:
        Denoised image
    """
    if method == 'gaussian':
        sigma = kwargs.get('sigma', 1.0)
        return filters.gaussian(image, sigma=sigma, preserve_range=True).astype(np.uint8)
    
    elif method == 'bilateral':
        d = kwargs.get('d', 9)
        sigma_color = kwargs.get('sigma_color', 75)
        sigma_space = kwargs.get('sigma_space', 75)
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    elif method == 'non_local_means':
        h = kwargs.get('h', 10)
        template_window_size = kwargs.get('template_window_size', 7)
        search_window_size = kwargs.get('search_window_size', 21)
        return cv2.fastNlMeansDenoising(image, None, h, 
                                       template_window_size, search_window_size)
    
    elif method == 'wiener':
        # Wiener filter approximation using noise estimation
        noise_var = kwargs.get('noise_var', None)
        if noise_var is None:
            # Estimate noise variance
            noise_var = restoration.estimate_sigma(image, average_sigmas=True) ** 2
        
        return restoration.wiener(image, np.ones((5, 5)) / 25, noise_var)
    
    else:
        raise ValueError(f"Unknown denoising method: {method}")

def normalize_image(image: np.ndarray, target_range: Tuple[int, int] = (0, 255)) -> np.ndarray:
    """
    Normalize image intensity to specified range.
    
    Args:
        image: Input image
        target_range: Target intensity range (min, max)
        
    Returns:
        Normalized image
    """
    min_val, max_val = target_range
    img_min, img_max = image.min(), image.max()
    
    if img_max == img_min:
        # Uniform image
        return np.full_like(image, min_val)
    
    normalized = (image - img_min) / (img_max - img_min)
    normalized = normalized * (max_val - min_val) + min_val
    
    return normalized.astype(np.uint8)

def remove_scale_bar_region(image: np.ndarray, bottom_fraction: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separate main fiber region from scale bar region.
    
    Args:
        image: Input SEM image
        bottom_fraction: Fraction of image height considered as scale bar region
        
    Returns:
        Tuple of (main_region, scale_bar_region)
    """
    height = image.shape[0]
    split_row = int(height * (1 - bottom_fraction))
    
    main_region = image[:split_row, :]
    scale_bar_region = image[split_row:, :]
    
    return main_region, scale_bar_region

def adaptive_threshold(image: np.ndarray, method: str = 'otsu') -> np.ndarray:
    """
    Apply adaptive thresholding for segmentation preparation.
    
    Args:
        image: Input grayscale image
        method: Thresholding method ('otsu', 'adaptive_mean', 'adaptive_gaussian')
        
    Returns:
        Binary thresholded image
    """
    if method == 'otsu':
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    
    elif method == 'adaptive_mean':
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    elif method == 'adaptive_gaussian':
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    else:
        raise ValueError(f"Unknown thresholding method: {method}")

def preprocess_pipeline(image_path: str, 
                       enhance_contrast_method: str = 'clahe',
                       denoise_method: str = 'bilateral',
                       remove_scale_bar: bool = True,
                       normalize: bool = True) -> dict:
    """
    Complete preprocessing pipeline for SEM images with robust file handling.
    
    Args:
        image_path: Path to input image
        enhance_contrast_method: Contrast enhancement method
        denoise_method: Denoising method
        remove_scale_bar: Whether to separate scale bar region
        normalize: Whether to normalize intensity
        
    Returns:
        Dictionary containing processed images and metadata
    """
    try:
        # Load original image with robust handling
        print(f"Loading image: {image_path}")
        original = load_image(image_path)
        print(f"✓ Image loaded successfully: {original.shape}")
        
        # Store processing steps
        result = {
            'original': original.copy(),
            'processing_steps': []
        }
        
        # Enhance contrast
        enhanced = enhance_contrast(original, method=enhance_contrast_method)
        result['contrast_enhanced'] = enhanced
        result['processing_steps'].append(f"Contrast enhancement: {enhance_contrast_method}")
        
        # Denoise
        denoised = denoise_image(enhanced, method=denoise_method)
        result['denoised'] = denoised
        result['processing_steps'].append(f"Denoising: {denoise_method}")
        
        # Normalize
        if normalize:
            normalized = normalize_image(denoised)
            result['normalized'] = normalized
            result['processing_steps'].append("Intensity normalization")
            current_image = normalized
        else:
            current_image = denoised
        
        # Separate scale bar region
        if remove_scale_bar:
            main_region, scale_bar_region = remove_scale_bar_region(current_image)
            result['main_region'] = main_region
            result['scale_bar_region'] = scale_bar_region
            result['processing_steps'].append("Scale bar separation")
            result['processed'] = main_region  # Main analysis image
        else:
            result['processed'] = current_image
        
        # Add metadata
        result['image_shape'] = original.shape
        result['preprocessing_complete'] = True
        
        return result
        
    except Exception as e:
        print(f"❌ Error in preprocessing pipeline: {e}")
        raise

def load_and_preprocess(image_path: str, **kwargs) -> np.ndarray:
    """
    Convenience function for quick preprocessing.
    
    Args:
        image_path: Path to input image
        **kwargs: Arguments for preprocess_pipeline
        
    Returns:
        Preprocessed image ready for analysis
    """
    result = preprocess_pipeline(image_path, **kwargs)
    return result['processed']

def visualize_preprocessing_steps(preprocessing_result: dict, figsize: Tuple[int, int] = (15, 10)):
    """
    Visualize the preprocessing pipeline steps.
    
    Args:
        preprocessing_result: Result from preprocess_pipeline
        figsize: Figure size for visualization
    """
    steps = ['original', 'contrast_enhanced', 'denoised', 'normalized', 'processed']
    available_steps = [step for step in steps if step in preprocessing_result]
    
    n_steps = len(available_steps)
    fig, axes = plt.subplots(2, (n_steps + 1) // 2, figsize=figsize)
    axes = axes.flatten() if n_steps > 1 else [axes]
    
    for i, step in enumerate(available_steps):
        if i < len(axes):
            axes[i].imshow(preprocessing_result[step], cmap='gray')
            axes[i].set_title(step.replace('_', ' ').title())
            axes[i].axis('off')
    
    # Hide unused subplots
    for j in range(len(available_steps), len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print processing steps
    print("Processing Steps Applied:")
    for i, step in enumerate(preprocessing_result.get('processing_steps', []), 1):
        print(f"{i}. {step}")