"""
SEM Fiber Analysis System - Scale Detection Module
Automatic detection and calibration of scale bars in SEM images.
"""

import cv2
import numpy as np
import re
from typing import Tuple, Dict, Optional, List
import matplotlib.pyplot as plt
from skimage import morphology, measure
from scipy import ndimage

# Optional OCR imports - will handle gracefully if not available
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    print("Warning: pytesseract not available. Scale text detection will be limited.")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("Warning: easyocr not available. Scale text detection will be limited.")

# If no OCR is available, we'll use manual calibration
if not PYTESSERACT_AVAILABLE and not EASYOCR_AVAILABLE:
    print("No OCR engines available. Using manual scale calibration mode.")

class ScaleBarDetector:
    """
    Detects scale bars and extracts calibration information from SEM images.
    """
    
    def __init__(self, 
                 scale_region_fraction: float = 0.15,
                 min_bar_length: int = 50,
                 max_bar_thickness: int = 20,
                 text_search_region: float = 0.3):
        """
        Initialize scale bar detector.
        
        Args:
            scale_region_fraction: Fraction of image height to search for scale bar
            min_bar_length: Minimum length of scale bar in pixels
            max_bar_thickness: Maximum thickness of scale bar in pixels
            text_search_region: Fraction of scale region to search for text
        """
        self.scale_region_fraction = scale_region_fraction
        self.min_bar_length = min_bar_length
        self.max_bar_thickness = max_bar_thickness
        self.text_search_region = text_search_region
        
        # Initialize OCR reader if available
        self.easyocr_reader = None
        if EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(['en'])
            except:
                pass
    
    def extract_scale_region(self, image: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Extract the bottom region likely to contain scale bar.
        
        Args:
            image: Input SEM image
            
        Returns:
            Tuple of (scale_region, y_offset)
        """
        height = image.shape[0]
        y_start = int(height * (1 - self.scale_region_fraction))
        
        scale_region = image[y_start:, :]
        return scale_region, y_start
    
    def detect_scale_bar_line(self, scale_region: np.ndarray) -> List[Dict]:
        """
        Detect horizontal line segments that could be scale bars.
        
        Args:
            scale_region: Region of image to search for scale bar
            
        Returns:
            List of potential scale bar candidates with properties
        """
        # Preprocess for line detection
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(scale_region, (3, 3), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # Morphological operations to connect line segments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size constraints
            if w < self.min_bar_length or h > self.max_bar_thickness:
                continue
            
            # Calculate aspect ratio
            aspect_ratio = w / h if h > 0 else 0
            
            # Should be very horizontal (high aspect ratio)
            if aspect_ratio < 5:
                continue
            
            # Calculate other properties
            area = cv2.contourArea(contour)
            rect_area = w * h
            fill_ratio = area / rect_area if rect_area > 0 else 0
            
            candidate = {
                'contour': contour,
                'bbox': (x, y, w, h),
                'length': w,
                'thickness': h,
                'aspect_ratio': aspect_ratio,
                'fill_ratio': fill_ratio,
                'area': area,
                'center_x': x + w // 2,
                'center_y': y + h // 2
            }
            
            candidates.append(candidate)
        
        # Sort by confidence score
        candidates.sort(key=self._calculate_bar_confidence, reverse=True)
        
        return candidates
    
    def _calculate_bar_confidence(self, candidate: Dict) -> float:
        """
        Calculate confidence score for scale bar candidate.
        
        Args:
            candidate: Scale bar candidate properties
            
        Returns:
            Confidence score
        """
        score = 0.0
        
        # Length score (longer is better, up to a point)
        length_score = min(1.0, candidate['length'] / 200)
        score += 0.3 * length_score
        
        # Aspect ratio score (should be very horizontal)
        aspect_score = min(1.0, candidate['aspect_ratio'] / 20)
        score += 0.3 * aspect_score
        
        # Fill ratio score (should be fairly solid)
        fill_score = min(1.0, candidate['fill_ratio'] / 0.7)
        score += 0.2 * fill_score
        
        # Position score (should be in reasonable location)
        # Prefer bars that are somewhat centered horizontally
        relative_x = candidate['center_x'] / 640  # Assume reasonable image width
        position_score = 1.0 - 2 * abs(relative_x - 0.5)  # Penalty for being off-center
        score += 0.2 * max(0, position_score)
        
        return score
    
    def extract_scale_text_pytesseract(self, scale_region: np.ndarray, bar_candidates: List[Dict]) -> List[str]:
        """
        Extract text from scale region using Tesseract OCR with proper encoding handling.
        
        Args:
            scale_region: Scale region image
            bar_candidates: Detected scale bar candidates
            
        Returns:
            List of extracted text strings
        """
        if not PYTESSERACT_AVAILABLE:
            return []
        
        try:
            # Preprocess for OCR
            # Increase contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(scale_region)
            
            # OCR configuration for better number recognition
            config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789µμmnmkMKMm.μ '
            
            # Use encoding parameter to avoid Windows Unicode issues
            import tempfile
            import os
            
            # Save image temporarily 
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                cv2.imwrite(temp_file.name, enhanced)
                temp_path = temp_file.name
            
            try:
                # Extract text with explicit encoding handling
                text = pytesseract.image_to_string(temp_path, config=config)
                
                # Clean and split text
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                
                return lines
                
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass
            
        except Exception as e:
            print(f"Tesseract OCR failed: {e}")
            return []
    
    def extract_scale_text_easyocr(self, scale_region: np.ndarray, bar_candidates: List[Dict]) -> List[str]:
        """
        Extract text from scale region using EasyOCR.
        
        Args:
            scale_region: Scale region image
            bar_candidates: Detected scale bar candidates
            
        Returns:
            List of extracted text strings
        """
        if not self.easyocr_reader:
            return []
        
        try:
            # Extract text
            results = self.easyocr_reader.readtext(scale_region)
            
            # Extract just the text strings
            texts = [result[1] for result in results if result[2] > 0.5]  # Confidence > 0.5
            
            return texts
            
        except Exception as e:
            print(f"EasyOCR failed: {e}")
            return []
    
    def parse_scale_text(self, text_lines: List[str]) -> Optional[Dict]:
        """
        Parse scale text to extract numerical value and unit.
        
        Args:
            text_lines: List of text strings from OCR
            
        Returns:
            Dictionary with scale value and unit, or None if not found
        """
        # Common unit patterns
        unit_patterns = {
            r'(\d+\.?\d*)\s*μm': ('micrometer', 1.0),
            r'(\d+\.?\d*)\s*µm': ('micrometer', 1.0),
            r'(\d+\.?\d*)\s*um': ('micrometer', 1.0),
            r'(\d+\.?\d*)\s*nm': ('nanometer', 0.001),
            r'(\d+\.?\d*)\s*mm': ('millimeter', 1000.0),
            r'(\d+\.?\d*)\s*cm': ('centimeter', 10000.0),
            r'(\d+\.?\d*)\s*m(?!m)': ('meter', 1000000.0),
        }
        
        for text in text_lines:
            text = text.replace(' ', '').lower()  # Remove spaces and lowercase
            
            for pattern, (unit_name, conversion_factor) in unit_patterns.items():
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        value = float(match.group(1))
                        return {
                            'value': value,
                            'unit': unit_name,
                            'micrometers': value * conversion_factor,
                            'original_text': text
                        }
                    except ValueError:
                        continue
        
        return None
    
    def calculate_pixel_scale(self, scale_info: Dict, bar_length_pixels: int) -> float:
        """
        Calculate micrometers per pixel conversion factor.
        
        Args:
            scale_info: Parsed scale information
            bar_length_pixels: Length of scale bar in pixels
            
        Returns:
            Micrometers per pixel
        """
        if bar_length_pixels <= 0:
            return 0.0
        
        micrometers = scale_info['micrometers']
        return micrometers / bar_length_pixels
    
    def detect_scale_bar(self, image: np.ndarray) -> Dict:
        """
        Main function to detect scale bar and calculate calibration.
        
        Args:
            image: Input SEM image
            
        Returns:
            Dictionary containing scale detection results
        """
        # Extract scale region
        scale_region, y_offset = self.extract_scale_region(image)
        
        # Detect scale bar lines
        bar_candidates = self.detect_scale_bar_line(scale_region)
        
        result = {
            'scale_detected': False,
            'scale_region': scale_region,
            'y_offset': y_offset,
            'bar_candidates': bar_candidates,
            'micrometers_per_pixel': None,
            'scale_info': None,
            'error': None
        }
        
        if not bar_candidates:
            result['error'] = "No scale bar candidates detected"
            return result
        
        # Try OCR to extract scale text
        text_lines = []
        
        # Try EasyOCR first (generally more robust)
        if self.easyocr_reader:
            text_lines.extend(self.extract_scale_text_easyocr(scale_region, bar_candidates))
        
        # Try Tesseract as backup
        if not text_lines and PYTESSERACT_AVAILABLE:
            text_lines.extend(self.extract_scale_text_pytesseract(scale_region, bar_candidates))
        
        result['extracted_text'] = text_lines
        
        if not text_lines:
            result['error'] = "No text extracted from scale region"
            return result
        
        # Parse scale information
        scale_info = self.parse_scale_text(text_lines)
        
        if not scale_info:
            result['error'] = "Could not parse scale information from text"
            return result
        
        # Use the best scale bar candidate
        best_bar = bar_candidates[0]
        bar_length = best_bar['length']
        
        # Calculate calibration
        micrometers_per_pixel = self.calculate_pixel_scale(scale_info, bar_length)
        
        result.update({
            'scale_detected': True,
            'micrometers_per_pixel': micrometers_per_pixel,
            'scale_info': scale_info,
            'best_bar': best_bar,
            'bar_length_pixels': bar_length
        })
        
        return result
    
    def visualize_scale_detection(self, image: np.ndarray, detection_result: Dict, figsize: Tuple[int, int] = (12, 8)):
        """
        Visualize scale detection results.
        
        Args:
            image: Original image
            detection_result: Result from detect_scale_bar
            figsize: Figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Original image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Scale region
        scale_region = detection_result['scale_region']
        axes[0, 1].imshow(scale_region, cmap='gray')
        axes[0, 1].set_title('Scale Region')
        axes[0, 1].axis('off')
        
        # Scale detection overlay
        if detection_result['bar_candidates']:
            overlay = cv2.cvtColor(scale_region, cv2.COLOR_GRAY2RGB)
            
            for i, candidate in enumerate(detection_result['bar_candidates'][:3]):  # Show top 3
                x, y, w, h = candidate['bbox']
                color = (0, 255, 0) if i == 0 else (255, 255, 0)  # Green for best, yellow for others
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            
            axes[1, 0].imshow(overlay)
            axes[1, 0].set_title('Detected Scale Bars')
        else:
            axes[1, 0].imshow(scale_region, cmap='gray')
            axes[1, 0].set_title('No Scale Bars Detected')
        axes[1, 0].axis('off')
        
        # Results summary
        ax_text = axes[1, 1]
        ax_text.axis('off')
        
        if detection_result['scale_detected']:
            info = detection_result['scale_info']
            text = f"Scale Detection: SUCCESS\n\n"
            text += f"Scale Value: {info['value']} {info['unit']}\n"
            text += f"Bar Length: {detection_result['bar_length_pixels']} pixels\n"
            text += f"Calibration: {detection_result['micrometers_per_pixel']:.4f} μm/pixel\n\n"
            text += f"Extracted Text:\n"
            for line in detection_result.get('extracted_text', []):
                text += f"  '{line}'\n"
        else:
            text = f"Scale Detection: FAILED\n\n"
            text += f"Error: {detection_result.get('error', 'Unknown error')}\n\n"
            if detection_result.get('extracted_text'):
                text += f"Extracted Text:\n"
                for line in detection_result['extracted_text']:
                    text += f"  '{line}'\n"
        
        ax_text.text(0.05, 0.95, text, transform=ax_text.transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.show()

def detect_scale_bar(image_input, **kwargs) -> Dict:
    """
    Convenience function to detect scale bar and return full results.
    
    Args:
        image_input: Either np.ndarray (image) or str (path to image)
        **kwargs: Additional parameters for ScaleBarDetector
        
    Returns:
        Dictionary containing scale detection results
    """
    # Handle both image arrays and file paths
    if isinstance(image_input, str):
        # Load image from path
        import cv2
        image = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return {
                'scale_detected': False,
                'micrometers_per_pixel': 0.0,
                'scale_factor': 0.0,  # For backward compatibility
                'error': f'Could not load image from path: {image_input}'
            }
    elif isinstance(image_input, np.ndarray):
        # Already an image array
        image = image_input
    else:
        return {
            'scale_detected': False,
            'micrometers_per_pixel': 0.0,
            'scale_factor': 0.0,
            'error': f'Invalid input type: {type(image_input)}. Expected str or np.ndarray'
        }
    
    # Run detection
    detector = ScaleBarDetector(**kwargs)
    result = detector.detect_scale_bar(image)
    
    # Ensure consistent return format
    if not isinstance(result, dict):
        # Handle unexpected return type
        result = {
            'scale_detected': False,
            'micrometers_per_pixel': 0.0,
            'scale_factor': 0.0,
            'error': f'Unexpected return type from detector: {type(result)}'
        }
    
    # Add scale_factor for backward compatibility
    if 'scale_factor' not in result:
        result['scale_factor'] = result.get('micrometers_per_pixel', 0.0)
    
    return result

def detect_scale_factor_only(image_input, **kwargs) -> float:
    """
    Convenience function that returns only the scale factor (for backward compatibility).
    
    Args:
        image_input: Either np.ndarray (image) or str (path to image)
        **kwargs: Additional parameters for ScaleBarDetector
        
    Returns:
        Micrometers per pixel conversion factor (0.0 if detection failed)
    """
    result = detect_scale_bar(image_input, **kwargs)
    
    if result['scale_detected']:
        return result['micrometers_per_pixel']
    else:
        print(f"Scale detection failed: {result.get('error', 'Unknown error')}")
        return 0.0

def manual_scale_calibration(bar_length_pixels: int, bar_length_micrometers: float) -> float:
    """
    Manual calibration when automatic detection fails.
    
    Args:
        bar_length_pixels: Length of scale bar in pixels
        bar_length_micrometers: Real length of scale bar in micrometers
        
    Returns:
        Micrometers per pixel conversion factor
    """
    if bar_length_pixels <= 0:
        raise ValueError("Bar length in pixels must be positive")
    
    return bar_length_micrometers / bar_length_pixels

def pixels_to_micrometers(pixel_measurement: float, micrometers_per_pixel: float) -> float:
    """
    Convert pixel measurements to micrometers.
    
    Args:
        pixel_measurement: Measurement in pixels
        micrometers_per_pixel: Conversion factor
        
    Returns:
        Measurement in micrometers
    """
    return pixel_measurement * micrometers_per_pixel

def micrometers_to_pixels(micrometer_measurement: float, micrometers_per_pixel: float) -> float:
    """
    Convert micrometer measurements to pixels.
    
    Args:
        micrometer_measurement: Measurement in micrometers
        micrometers_per_pixel: Conversion factor
        
    Returns:
        Measurement in pixels
    """
    if micrometers_per_pixel <= 0:
        raise ValueError("Micrometers per pixel must be positive")
    
    return micrometer_measurement / micrometers_per_pixel