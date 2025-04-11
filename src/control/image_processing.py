"""
Image processing module for the WoW Bot.

This module provides a comprehensive set of image processing tools
for analyzing game screen captures, detecting UI elements, and extracting
meaningful information from the World of Warcraft interface.
"""
import os
import time
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from src.utils.config import get_config
from src.utils.logging import LogCategory, debug, info, warning, error

# Type aliases
RGBColor = Tuple[int, int, int]
BGRColor = Tuple[int, int, int]
Rect = Tuple[int, int, int, int]  # x, y, width, height
Point = Tuple[int, int]  # x, y


class TemplateMatchMethod(Enum):
    """Template matching methods from OpenCV."""
    SQDIFF = cv2.TM_SQDIFF
    SQDIFF_NORMED = cv2.TM_SQDIFF_NORMED
    CCORR = cv2.TM_CCORR
    CCORR_NORMED = cv2.TM_CCORR_NORMED
    CCOEFF = cv2.TM_CCOEFF
    CCOEFF_NORMED = cv2.TM_CCOEFF_NORMED


class ImageProcessor:
    """
    Image processing utilities for analyzing WoW screen captures.
    
    Features:
    - Image transformations and filtering
    - Template matching for UI element detection
    - Color analysis and detection
    - Region of interest extraction
    """
    
    def __init__(self):
        """Initialize the image processor."""
        # Template cache to avoid reloading templates
        self._template_cache = {}
        
        # Load template directory from config
        self._template_dir = get_config(
            'screen_capture.template_dir',
            os.path.join(get_config('system.data_dir', './data'), 'templates')
        )
        
        # Create template directory if it doesn't exist
        os.makedirs(self._template_dir, exist_ok=True)
        
        info("Image processor initialized", LogCategory.CONTROL)
        debug(f"Template directory: {self._template_dir}", LogCategory.CONTROL)
    
    def resize(self, image: np.ndarray, width: int = None, height: int = None,
               scale: float = None) -> np.ndarray:
        """
        Resize an image to the specified dimensions.
        
        Args:
            image: Input image
            width: Target width (maintains aspect ratio if height is None)
            height: Target height (maintains aspect ratio if width is None)
            scale: Scale factor (overrides width/height if provided)
            
        Returns:
            Resized image
        """
        if image is None:
            warning("Cannot resize None image", LogCategory.CONTROL)
            return None
        
        h, w = image.shape[:2]
        
        if scale is not None:
            return cv2.resize(image, (0, 0), fx=scale, fy=scale, 
                              interpolation=cv2.INTER_AREA)
        
        if width is None and height is None:
            return image.copy()
        
        if width is None:
            # Calculate width to maintain aspect ratio
            aspect_ratio = w / h
            width = int(height * aspect_ratio)
        elif height is None:
            # Calculate height to maintain aspect ratio
            aspect_ratio = h / w
            height = int(width * aspect_ratio)
        
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    
    def crop(self, image: np.ndarray, rect: Rect) -> np.ndarray:
        """
        Crop an image to the specified rectangle.
        
        Args:
            image: Input image
            rect: Rectangle (x, y, width, height)
            
        Returns:
            Cropped image
        """
        if image is None:
            warning("Cannot crop None image", LogCategory.CONTROL)
            return None
        
        x, y, w, h = rect
        return image[y:y+h, x:x+w]
    
    def convert_rgb_to_bgr(self, rgb_color: RGBColor) -> BGRColor:
        """Convert RGB color to BGR (OpenCV format)."""
        r, g, b = rgb_color
        return (b, g, r)
    
    def convert_bgr_to_rgb(self, bgr_color: BGRColor) -> RGBColor:
        """Convert BGR color (OpenCV format) to RGB."""
        b, g, r = bgr_color
        return (r, g, b)
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to improve contrast.
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        if image is None:
            warning("Cannot normalize None image", LogCategory.CONTROL)
            return None
        
        # Convert to grayscale if it's a color image
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply histogram equalization
        return cv2.equalizeHist(gray)
    
    def apply_gaussian_blur(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Apply Gaussian blur to an image.
        
        Args:
            image: Input image
            kernel_size: Size of the blur kernel (must be odd)
            
        Returns:
            Blurred image
        """
        if image is None:
            warning("Cannot apply blur to None image", LogCategory.CONTROL)
            return None
        
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def apply_threshold(self, image: np.ndarray, threshold: int = 127, 
                        max_val: int = 255, inverse: bool = False) -> np.ndarray:
        """
        Apply binary threshold to an image.
        
        Args:
            image: Input image (grayscale)
            threshold: Threshold value
            max_val: Maximum value for thresholding
            inverse: Whether to apply inverse thresholding
            
        Returns:
            Thresholded image
        """
        if image is None:
            warning("Cannot apply threshold to None image", LogCategory.CONTROL)
            return None
        
        # Convert to grayscale if it's a color image
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        thresh_type = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
        _, thresh = cv2.threshold(gray, threshold, max_val, thresh_type)
        return thresh
    
    def apply_adaptive_threshold(self, image: np.ndarray, max_val: int = 255,
                                block_size: int = 11, c: int = 2) -> np.ndarray:
        """
        Apply adaptive threshold to handle varying lighting conditions.
        
        Args:
            image: Input image (grayscale)
            max_val: Maximum value for thresholding
            block_size: Size of pixel neighborhood for thresholding
            c: Constant subtracted from mean
            
        Returns:
            Thresholded image
        """
        if image is None:
            warning("Cannot apply adaptive threshold to None image", LogCategory.CONTROL)
            return None
        
        # Convert to grayscale if it's a color image
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        return cv2.adaptiveThreshold(
            gray, max_val, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, c
        )
    
    def detect_edges(self, image: np.ndarray, threshold1: int = 50, 
                     threshold2: int = 150) -> np.ndarray:
        """
        Detect edges in an image using the Canny edge detector.
        
        Args:
            image: Input image
            threshold1: First threshold for the hysteresis procedure
            threshold2: Second threshold for the hysteresis procedure
            
        Returns:
            Edge map
        """
        if image is None:
            warning("Cannot detect edges in None image", LogCategory.CONTROL)
            return None
        
        # Convert to grayscale if it's a color image
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, threshold1, threshold2)
        return edges
    
    def load_template(self, template_name: str) -> np.ndarray:
        """
        Load a template image for template matching.
        
        Args:
            template_name: Name of the template (without extension)
            
        Returns:
            Template image or None if not found
        """
        # Check cache first
        if template_name in self._template_cache:
            return self._template_cache[template_name]
        
        # Try different extensions
        extensions = ['.png', '.jpg', '.jpeg']
        for ext in extensions:
            template_path = os.path.join(self._template_dir, f"{template_name}{ext}")
            if os.path.exists(template_path):
                template = cv2.imread(template_path)
                if template is not None:
                    self._template_cache[template_name] = template
                    debug(f"Loaded template: {template_name}{ext}", LogCategory.CONTROL)
                    return template
        
        warning(f"Template not found: {template_name}", LogCategory.CONTROL)
        return None
    
    def match_template(self, image: np.ndarray, template: Union[str, np.ndarray],
                       method: TemplateMatchMethod = TemplateMatchMethod.CCOEFF_NORMED,
                       threshold: float = 0.8) -> List[Tuple[Rect, float]]:
        """
        Find matches of a template in an image.
        
        Args:
            image: Image to search in
            template: Template image or template name
            method: Template matching method
            threshold: Minimum matching score (0-1)
            
        Returns:
            List of matches as (x, y, w, h, score) tuples
        """
        if image is None:
            warning("Cannot match template on None image", LogCategory.CONTROL)
            return []
        
        # Get template image
        if isinstance(template, str):
            template_img = self.load_template(template)
            if template_img is None:
                return []
        else:
            template_img = template
        
        # Get template dimensions
        h, w = template_img.shape[:2]
        
        # Perform template matching
        start_time = time.time()
        result = cv2.matchTemplate(image, template_img, method.value)
        
        # Find matches based on the method
        matches = []
        
        if method in [TemplateMatchMethod.SQDIFF, TemplateMatchMethod.SQDIFF_NORMED]:
            # For SQDIFF methods, smaller values are better matches
            match_threshold = 1.0 - threshold
            match_compare = lambda x: x <= match_threshold
        else:
            # For other methods, larger values are better matches
            match_threshold = threshold
            match_compare = lambda x: x >= match_threshold
        
        # Get all matches above threshold
        locations = np.where(match_compare(result))
        for pt in zip(*locations[::-1]):  # Swap columns and rows
            match_val = result[pt[1], pt[0]]
            
            # For SQDIFF methods, convert score to a similarity measure
            if method in [TemplateMatchMethod.SQDIFF, TemplateMatchMethod.SQDIFF_NORMED]:
                score = 1.0 - match_val
            else:
                score = match_val
            
            matches.append(((pt[0], pt[1], w, h), score))
        
        elapsed_time = (time.time() - start_time) * 1000  # ms
        debug(f"Template matching completed in {elapsed_time:.2f}ms with {len(matches)} matches", 
              LogCategory.CONTROL)
        
        # Sort matches by score (highest first)
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    def find_ui_element(self, image: np.ndarray, template_name: str, 
                        threshold: float = 0.8) -> Optional[Rect]:
        """
        Find a UI element in the image.
        
        Args:
            image: Image to search in
            template_name: Name of the template to match
            threshold: Minimum matching score (0-1)
            
        Returns:
            Rectangle of the best match or None if not found
        """
        matches = self.match_template(
            image, template_name, TemplateMatchMethod.CCOEFF_NORMED, threshold
        )
        
        if matches:
            return matches[0][0]  # Return the best match
        return None
    
    def get_dominant_colors(self, image: np.ndarray, num_colors: int = 5) -> List[Tuple[BGRColor, float]]:
        """
        Extract dominant colors from an image.
        
        Args:
            image: Input image
            num_colors: Number of dominant colors to extract
            
        Returns:
            List of (color, percentage) tuples
        """
        if image is None:
            warning("Cannot extract dominant colors from None image", LogCategory.CONTROL)
            return []
        
        # Reshape image to be a list of pixels
        pixels = image.reshape(-1, 3)
        
        # Convert to float32 for k-means
        pixels = np.float32(pixels)
        
        # Define criteria, number of clusters and apply kmeans
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert centers to uint8
        centers = np.uint8(centers)
        
        # Count occurrences of each label
        _, counts = np.unique(labels, return_counts=True)
        
        # Calculate percentages
        percentages = counts / len(labels)
        
        # Sort by percentage
        color_percentages = [(tuple(map(int, centers[i])), percentages[i]) for i in range(len(centers))]
        color_percentages.sort(key=lambda x: x[1], reverse=True)
        
        return color_percentages
    
    def find_color_regions(self, image: np.ndarray, target_color: RGBColor, 
                           tolerance: int = 20) -> np.ndarray:
        """
        Find regions matching a specific color.
        
        Args:
            image: Input image
            target_color: RGB color to find
            tolerance: Color matching tolerance
            
        Returns:
            Binary mask with matching regions
        """
        if image is None:
            warning("Cannot find color regions in None image", LogCategory.CONTROL)
            return None
        
        # Convert RGB to BGR for OpenCV
        target_bgr = self.convert_rgb_to_bgr(target_color)
        
        # Create lower and upper bounds for color matching
        lower_bound = np.array([max(0, c - tolerance) for c in target_bgr])
        upper_bound = np.array([min(255, c + tolerance) for c in target_bgr])
        
        # Create mask where color matches
        mask = cv2.inRange(image, lower_bound, upper_bound)
        return mask
    
    def extract_text_regions(self, image: np.ndarray) -> List[Rect]:
        """
        Extract potential text regions from an image.
        This is a simplified approach that looks for areas with high edge density.
        
        Args:
            image: Input image
            
        Returns:
            List of potential text region rectangles
        """
        if image is None:
            warning("Cannot extract text regions from None image", LogCategory.CONTROL)
            return []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()
        
        # Apply bilateral filter to preserve edges while removing noise
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Detect edges
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on aspect ratio and size
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter out too small or too large regions
            if w < 10 or h < 10 or w > image.shape[1] * 0.8 or h > image.shape[0] * 0.8:
                continue
            
            # Filter based on aspect ratio (text is usually wider than tall)
            aspect_ratio = w / h
            if 0.1 <= aspect_ratio <= 10:
                text_regions.append((x, y, w, h))
        
        debug(f"Extracted {len(text_regions)} potential text regions", LogCategory.CONTROL)
        return text_regions
    
    def save_debug_image(self, image: np.ndarray, name: str) -> str:
        """
        Save an image for debugging purposes.
        
        Args:
            image: Image to save
            name: Base name for the image
            
        Returns:
            Path to the saved image
        """
        if image is None:
            warning("Cannot save None image", LogCategory.CONTROL)
            return None
        
        # Create debug directory
        debug_dir = os.path.join(get_config('system.data_dir', './data'), 'debug')
        os.makedirs(debug_dir, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{name}_{timestamp}.png"
        path = os.path.join(debug_dir, filename)
        
        # Save image
        cv2.imwrite(path, image)
        debug(f"Saved debug image: {path}", LogCategory.CONTROL)
        
        return path
    
    def draw_rects(self, image: np.ndarray, rects: List[Rect], 
                   color: RGBColor = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """
        Draw rectangles on an image for visualization.
        
        Args:
            image: Input image
            rects: List of rectangles (x, y, w, h)
            color: RGB color for rectangles
            thickness: Line thickness
            
        Returns:
            Image with rectangles drawn
        """
        if image is None:
            warning("Cannot draw on None image", LogCategory.CONTROL)
            return None
        
        # Make a copy to avoid modifying the original
        result = image.copy()
        
        # Convert RGB to BGR for OpenCV
        bgr_color = self.convert_rgb_to_bgr(color)
        
        # Draw each rectangle
        for rect in rects:
            x, y, w, h = rect
            cv2.rectangle(result, (x, y), (x + w, y + h), bgr_color, thickness)
        
        return result


# Create a singleton instance
image_processor = ImageProcessor()
