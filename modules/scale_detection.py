"""
SEM Fiber Analysis System - Enhanced Scale Detection Module
Automatic detection and calibration of scale bars in SEM images.
UPDATED: Smart bottom-right corner detection approach.
"""

import cv2
import numpy as np
import re
from typing import Tuple, Dict, Optional, List
import matplotlib.pyplot as plt

# Optional OCR imports - will handle gracefully if not available
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

EASYOCR_AVAILABLE = False
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except (ImportError, KeyboardInterrupt, Exception) as e:
    # Handle various import failures gracefully
    EASYOCR_AVAILABLE = False
    print(f"EasyOCR not available: {type(e).__name__}")

# If no OCR is available, warn user
if not PYTESSERACT_AVAILABLE and not EASYOCR_AVAILABLE:
    print("Warning: No OCR engines available. Scale detection will be limited to manual calibration.")

class ScaleBarDetector:
    """
    Enhanced scale bar detector with smart corner detection.
    """
    
    def __init__(self, 
                 scale_region_fraction: float = 0.15,
                 min_bar_length: int = 50,
                 max_bar_thickness: int = 20,
                 use_enhanced_detection: bool = True,
                 **kwargs):
        """
        Initialize scale bar detector.
        
        Args:
            scale_region_fraction: Fraction of image height to search for scale bar
            min_bar_length: Minimum length of scale bar in pixels
            max_bar_thickness: Maximum thickness of scale bar in pixels
            use_enhanced_detection: Compatibility parameter (ignored - always uses smart detection)
            **kwargs: Additional compatibility parameters
        """
        self.scale_region_fraction = scale_region_fraction
        self.min_bar_length = min_bar_length
        self.max_bar_thickness = max_bar_thickness
        
        # Initialize OCR reader if available
        self.easyocr_reader = None
        if EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(['en'], verbose=False)
                print("✅ EasyOCR initialized successfully")
            except Exception as e:
                print(f"⚠️ EasyOCR initialization failed: {e}")
                self.easyocr_reader = None
    
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
    
    def extract_corner_region(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Extract bottom-right corner where scale bars are typically located.
        
        Args:
            image: Input SEM image
            
        Returns:
            Tuple of (corner_region, (x_offset, y_offset))
        """
        height, width = image.shape
        
        # Take bottom 20% and right 40% of image
        corner_height = int(height * 0.2)
        corner_width = int(width * 0.4)
        
        y_start = height - corner_height
        x_start = width - corner_width
        
        corner_region = image[y_start:, x_start:]
        corner_offset = (x_start, y_start)
        
        return corner_region, corner_offset
    
    def find_scale_lines(self, corner_region: np.ndarray) -> List[Dict]:
        """
        Find horizontal white lines that could be scale bars.
        
        Args:
            corner_region: Bottom-right corner region
            
        Returns:
            List of potential scale line dictionaries
        """
        lines = []
        
        # Use high threshold to find bright white lines
        _, binary = cv2.threshold(corner_region, 240, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Filter for scale bar characteristics:
            # - Very horizontal (aspect ratio > 8)
            # - Reasonable length (> min_bar_length)
            # - Not too thick (< max_bar_thickness)
            if aspect_ratio > 8 and w > self.min_bar_length and h <= self.max_bar_thickness:
                lines.append({
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'length': w,
                    'center_x': x + w // 2,
                    'center_y': y + h // 2,
                    'aspect_ratio': aspect_ratio,
                    'contour': contour
                })
        
        # Sort by length (longer lines are more likely to be scale bars)
        lines.sort(key=lambda x: x['length'], reverse=True)
        
        return lines
    
    def find_text_near_line(self, corner_region: np.ndarray, line: Dict) -> Optional[Dict]:
        """
        Find scale text near a detected line.
        
        Args:
            corner_region: Corner region image
            line: Line information dictionary
            
        Returns:
            Scale text information or None
        """
        if not self.easyocr_reader:
            # Try Tesseract as fallback
            if PYTESSERACT_AVAILABLE:
                return self._find_text_tesseract(corner_region, line)
            else:
                print("⚠️ No OCR available for text detection")
                return None
        
        # Define search area around the line
        line_x = line['x']
        line_y = line['y']
        line_w = line['width']
        line_h = line['height']
        
        # Expand search area around the line
        margin = 30
        search_x1 = max(0, line_x - margin)
        search_y1 = max(0, line_y - margin)
        search_x2 = min(corner_region.shape[1], line_x + line_w + margin)
        search_y2 = min(corner_region.shape[0], line_y + line_h + margin)
        
        search_region = corner_region[search_y1:search_y2, search_x1:search_x2]
        
        try:
            # Use EasyOCR to find text in the search region
            results = self.easyocr_reader.readtext(search_region, detail=1)
            
            for (bbox, text, confidence) in results:
                if confidence > 0.3:  # Reasonable confidence
                    scale_info = self.parse_scale_text([text.strip()])
                    if scale_info:
                        return {
                            'text': text.strip(),
                            'value': scale_info['value'],
                            'unit': scale_info['unit'],
                            'micrometers': scale_info['micrometers'],
                            'confidence': confidence,
                            'original_text': text.strip()
                        }
        
        except Exception as e:
            print(f"⚠️ EasyOCR text detection failed: {e}")
            # Try Tesseract as fallback
            if PYTESSERACT_AVAILABLE:
                return self._find_text_tesseract(corner_region, line)
        
        return None
    
    def _find_text_tesseract(self, corner_region: np.ndarray, line: Dict) -> Optional[Dict]:
        """Fallback text detection using Tesseract."""
        try:
            # Define search area around the line
            line_x = line['x']
            line_y = line['y']
            line_w = line['width']
            line_h = line['height']
            
            margin = 30
            search_x1 = max(0, line_x - margin)
            search_y1 = max(0, line_y - margin)
            search_x2 = min(corner_region.shape[1], line_x + line_w + margin)
            search_y2 = min(corner_region.shape[0], line_y + line_h + margin)
            
            search_region = corner_region[search_y1:search_y2, search_x1:search_x2]
            
            # Enhanced preprocessing for Tesseract
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            enhanced = clahe.apply(search_region)
            
            config = '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789.μµumnmkMm '
            text = pytesseract.image_to_string(enhanced, config=config)
            
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            scale_info = self.parse_scale_text(lines)
            
            if scale_info:
                return {
                    'text': text.strip(),
                    'value': scale_info['value'],
                    'unit': scale_info['unit'],
                    'micrometers': scale_info['micrometers'],
                    'confidence': 0.5,  # Default confidence for Tesseract
                    'original_text': text.strip()
                }
        
        except Exception as e:
            print(f"⚠️ Tesseract text detection failed: {e}")
        
        return None
    
    def parse_scale_text(self, text_lines: List[str]) -> Optional[Dict]:
        """
        Parse scale text with comprehensive OCR error handling.
        
        Args:
            text_lines: List of text lines from OCR
            
        Returns:
            Dictionary with parsed scale information or None
        """
        # Patterns for common scale formats and OCR errors
        unit_patterns = {
            # Standard patterns
            r'(\d+\.?\d*)\s*μm': ('micrometer', 1.0),
            r'(\d+\.?\d*)\s*µm': ('micrometer', 1.0),
            r'(\d+\.?\d*)\s*um': ('micrometer', 1.0),
            
            # Common OCR errors for μm
            r'(\d+\.?\d*)\s*jm': ('micrometer', 1.0),       # jm instead of μm
            r'(\d+\.?\d*)\s*[Oo]jm': ('micrometer', 1.0),   # Ojm instead of μm  
            r'(\d+\.?\d*)\s*[Oo]pm': ('micrometer', 1.0),   # Opm instead of μm
            r'(\d+\.?\d*)\s*Qum': ('micrometer', 1.0),      # Qum instead of μm
            r'(\d+\.?\d*)\s*0μm': ('micrometer', 1.0),      # 0μm instead of μm
            r'(\d+\.?\d*)\s*0jm': ('micrometer', 1.0),      # 0jm instead of μm
            
            # Other units
            r'(\d+\.?\d*)\s*nm': ('nanometer', 0.001),
            r'(\d+\.?\d*)\s*mm': ('millimeter', 1000.0),
            r'(\d+\.?\d*)\s*cm': ('centimeter', 10000.0),
            r'(\d+\.?\d*)\s*m(?!m)': ('meter', 1000000.0),
            
            # Decimal variations
            r'(\d+)[,.](\d+)\s*μm': ('micrometer', 1.0),
            r'(\d+)[,.](\d+)\s*jm': ('micrometer', 1.0),
            r'(\d+)[,.](\d+)\s*[Oo]jm': ('micrometer', 1.0),
            r'(\d+)[,.](\d+)\s*[Oo]pm': ('micrometer', 1.0),
        }
        
        for text in text_lines:
            # Clean up text
            text_clean = text.lower().replace(' ', '').replace('_', '').replace('\n', '')
            
            for pattern, (unit_name, conversion_factor) in unit_patterns.items():
                match = re.search(pattern, text_clean, re.IGNORECASE)
                if match:
                    try:
                        if len(match.groups()) > 1:
                            # Handle decimal patterns like "400.0"
                            value = float(f"{match.group(1)}.{match.group(2)}")
                        else:
                            value = float(match.group(1))
                        
                        micrometers = value * conversion_factor
                        
                        # Sanity check: typical SEM scale range
                        if 1 <= micrometers <= 5000:
                            return {
                                'value': value,
                                'unit': unit_name,
                                'micrometers': micrometers,
                                'original_text': text
                            }
                    
                    except (ValueError, IndexError):
                        continue
        
        return None
    
    def calculate_pixel_scale(self, scale_info: Dict, bar_length_pixels: int) -> float:
        """Calculate micrometers per pixel conversion factor."""
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
        result = {
            'scale_detected': False,
            'scale_region': None,
            'micrometers_per_pixel': 0.0,
            'scale_factor': 0.0,  # For backward compatibility
            'scale_info': None,
            'error': None,
            'method_used': 'smart_corner'
        }
        
        # Try smart corner detection first
        corner_region, corner_offset = self.extract_corner_region(image)
        result['scale_region'] = corner_region
        
        # Strategy 1: Find scale lines and look for nearby text
        print("🔄 Trying Strategy 1: Line-first approach...")
        scale_lines = self.find_scale_lines(corner_region)
        
        if scale_lines:
            print(f"✅ Found {len(scale_lines)} scale lines")
            # Look for text near each line
            for i, line in enumerate(scale_lines):
                print(f"   Checking line {i+1}: {line['width']}x{line['height']} pixels")
                scale_info = self.find_text_near_line(corner_region, line)
                
                if scale_info:
                    print(f"✅ Strategy 1 SUCCESS: Found text near line")
                    # Calculate scale factor
                    micrometers_per_pixel = self.calculate_pixel_scale(scale_info, line['length'])
                    
                    result.update({
                        'scale_detected': True,
                        'micrometers_per_pixel': micrometers_per_pixel,
                        'scale_factor': micrometers_per_pixel,
                        'scale_info': scale_info,
                        'best_bar': line,
                        'bar_length_pixels': line['length'],
                        'corner_offset': corner_offset
                    })
                    
                    return result
                else:
                    print(f"   ❌ No text found near line {i+1}")
            
            print("❌ Strategy 1 failed: Found lines but no nearby text")
        else:
            print("❌ Strategy 1 failed: No scale lines found")
        
        # Strategy 2: If line-first approach fails, try text-first approach
        print("🔄 Trying Strategy 2: Text-first approach...")
        scale_text = self.find_any_scale_text(corner_region)
        
        if scale_text:
            print(f"✅ Found scale text: {scale_text['text']} = {scale_text['micrometers']} μm")
            
            # Look for any reasonable horizontal line in the corner
            estimated_line_length = self.estimate_line_length_from_text(corner_region, scale_text)
            print(f"📏 Estimated line length: {estimated_line_length} pixels")
            
            if estimated_line_length > 0:
                micrometers_per_pixel = scale_text['micrometers'] / estimated_line_length
                
                result.update({
                    'scale_detected': True,
                    'micrometers_per_pixel': micrometers_per_pixel,
                    'scale_factor': micrometers_per_pixel,
                    'scale_info': scale_text,
                    'bar_length_pixels': estimated_line_length,
                    'corner_offset': corner_offset,
                    'method': 'text_first_estimate'
                })
                
                print(f"✅ Strategy 2 SUCCESS: {micrometers_per_pixel:.4f} μm/pixel")
                return result
            else:
                print("❌ Strategy 2 failed: Could not estimate line length")
        else:
            print("❌ Strategy 2 failed: No scale text found")
        
        # Set appropriate error message
        if scale_lines and not scale_text:
            result['error'] = "Found scale lines but no valid text nearby"
        elif scale_text and not scale_lines:
            result['error'] = "Found scale text but no clear scale lines"
        elif not scale_lines and not scale_text:
            result['error'] = "No scale lines or text found in corner region"
        else:
            result['error'] = "Found both lines and text but couldn't match them"
        
        return result
    
    def find_any_scale_text(self, corner_region: np.ndarray) -> Optional[Dict]:
        """
        Find any scale text in the corner region (not necessarily near a line).
        
        Args:
            corner_region: Corner region image
            
        Returns:
            Scale text information or None
        """
        if not self.easyocr_reader:
            if PYTESSERACT_AVAILABLE:
                return self._find_any_text_tesseract(corner_region)
            else:
                return None
        
        try:
            # Search the entire corner region for scale text
            results = self.easyocr_reader.readtext(corner_region, detail=1)
            
            for (bbox, text, confidence) in results:
                if confidence > 0.3:  # Reasonable confidence
                    scale_info = self.parse_scale_text([text.strip()])
                    if scale_info:
                        return {
                            'text': text.strip(),
                            'value': scale_info['value'],
                            'unit': scale_info['unit'],
                            'micrometers': scale_info['micrometers'],
                            'confidence': confidence,
                            'original_text': text.strip()
                        }
        
        except Exception as e:
            print(f"⚠️ EasyOCR text search failed: {e}")
            if PYTESSERACT_AVAILABLE:
                return self._find_any_text_tesseract(corner_region)
        
        return None
    
    def _find_any_text_tesseract(self, corner_region: np.ndarray) -> Optional[Dict]:
        """Fallback: find any scale text using Tesseract."""
        try:
            # Enhanced preprocessing for Tesseract
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            enhanced = clahe.apply(corner_region)
            
            config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.μµumnmkMm '
            text = pytesseract.image_to_string(enhanced, config=config)
            
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            scale_info = self.parse_scale_text(lines)
            
            if scale_info:
                return {
                    'text': text.strip(),
                    'value': scale_info['value'],
                    'unit': scale_info['unit'],
                    'micrometers': scale_info['micrometers'],
                    'confidence': 0.5,  # Default confidence for Tesseract
                    'original_text': text.strip()
                }
        
        except Exception as e:
            print(f"⚠️ Tesseract text search failed: {e}")
        
        return None
    
    def estimate_line_length_from_text(self, corner_region: np.ndarray, scale_text: Dict) -> int:
        """
        Estimate scale bar length when we have text but can't find the exact line.
        
        Args:
            corner_region: Corner region image
            scale_text: Scale text information
            
        Returns:
            Estimated line length in pixels
        """
        # Look for any horizontal bright lines in the corner region
        _, binary = cv2.threshold(corner_region, 200, 255, cv2.THRESH_BINARY)
        
        # Find all contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        horizontal_lines = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Look for any reasonably horizontal line
            if aspect_ratio > 3 and w > 30:  # More lenient criteria
                horizontal_lines.append(w)
        
        if horizontal_lines:
            # Return the longest line found
            return max(horizontal_lines)
        else:
            # If no lines found, estimate based on typical scale bar proportions
            # Typical scale bars are about 1/8 to 1/4 of image width
            estimated_length = int(corner_region.shape[1] * 0.2)  # 20% of corner width
            return max(estimated_length, 100)  # At least 100 pixels
    
    def visualize_scale_detection(self, image: np.ndarray, detection_result: Dict, figsize: Tuple[int, int] = (12, 8)):
        """Visualize scale detection results."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Original image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Scale region
        if detection_result['scale_region'] is not None:
            axes[0, 1].imshow(detection_result['scale_region'], cmap='gray')
            axes[0, 1].set_title('Scale Region (Corner)')
            axes[0, 1].axis('off')
        
        # Detection visualization
        if detection_result['scale_detected']:
            overlay = cv2.cvtColor(detection_result['scale_region'], cv2.COLOR_GRAY2RGB)
            
            if 'best_bar' in detection_result:
                bar = detection_result['best_bar']
                cv2.rectangle(overlay, (bar['x'], bar['y']), 
                            (bar['x'] + bar['width'], bar['y'] + bar['height']), 
                            (0, 255, 0), 2)
            
            axes[1, 0].imshow(overlay)
            axes[1, 0].set_title('Scale Bar Detection')
        else:
            axes[1, 0].imshow(detection_result['scale_region'], cmap='gray')
            axes[1, 0].set_title('Detection Failed')
        axes[1, 0].axis('off')
        
        # Results summary
        ax_text = axes[1, 1]
        ax_text.axis('off')
        
        if detection_result['scale_detected']:
            info = detection_result['scale_info']
            text = f"Scale Detection: SUCCESS\n\n"
            text += f"Method: {detection_result['method_used']}\n"
            text += f"Scale Value: {info.get('value', 'N/A')} {info.get('unit', 'N/A')}\n"
            text += f"Bar Length: {detection_result.get('bar_length_pixels', 'N/A')} pixels\n"
            text += f"Calibration: {detection_result['micrometers_per_pixel']:.4f} μm/pixel\n"
            text += f"Text: '{info.get('original_text', 'N/A')}'\n"
            text += f"Confidence: {info.get('confidence', 'N/A'):.3f}\n"
        else:
            text = f"Scale Detection: FAILED\n\n"
            text += f"Method: {detection_result['method_used']}\n"
            text += f"Error: {detection_result.get('error', 'Unknown error')}\n"
        
        ax_text.text(0.05, 0.95, text, transform=ax_text.transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.show()

# Convenience functions for backward compatibility
def detect_scale_bar(image_input, use_enhanced: bool = True, **kwargs) -> Dict:
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
        image = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return {
                'scale_detected': False,
                'micrometers_per_pixel': 0.0,
                'scale_factor': 0.0,
                'error': f'Could not load image from path: {image_input}'
            }
    elif isinstance(image_input, np.ndarray):
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
    
    return result

def detect_scale_factor_only(image_input, **kwargs) -> float:
    """
    Convenience function that returns only the scale factor.
    
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
    """Convert pixel measurements to micrometers."""
    return pixel_measurement * micrometers_per_pixel

def micrometers_to_pixels(micrometer_measurement: float, micrometers_per_pixel: float) -> float:
    """Convert micrometer measurements to pixels."""
    if micrometers_per_pixel <= 0:
        raise ValueError("Micrometers per pixel must be positive")
    return micrometer_measurement / micrometers_per_pixel