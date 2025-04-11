"""
Vision utilities for the WoW Bot.

This module provides helper functions for image processing
and vision-related tasks used by the detectors.
"""
import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Any, Union
import time

from src.utils.logging import debug, warning, LogCategory
from src.vision.models.detection_result import BoundingBox


def load_template(filepath: str) -> Tuple[Optional[np.ndarray], str]:
    """
    Load an image template from disk.
    
    Args:
        filepath: Path to the template image file
        
    Returns:
        Tuple of (template image as numpy array, template name)
    """
    if not os.path.exists(filepath):
        warning(f"Template file not found: {filepath}", LogCategory.VISION)
        return None, os.path.basename(filepath)
    
    try:
        # Load image and convert to BGR (OpenCV default)
        image = cv2.imread(filepath)
        if image is None:
            warning(f"Failed to load template: {filepath}", LogCategory.VISION)
            return None, os.path.basename(filepath)
            
        return image, os.path.basename(filepath)
    except Exception as e:
        warning(f"Error loading template {filepath}: {str(e)}", LogCategory.VISION)
        return None, os.path.basename(filepath)


def match_template(frame: np.ndarray, template: np.ndarray, 
                 threshold: float = 0.8, 
                 method: int = cv2.TM_CCOEFF_NORMED,
                 max_results: int = 5,
                 mask: Optional[np.ndarray] = None) -> List[Tuple[BoundingBox, float]]:
    """
    Find a template in an image using template matching.
    
    Args:
        frame: The image to search in
        template: The template to search for
        threshold: Minimum matching score (0.0-1.0)
        method: OpenCV template matching method
        max_results: Maximum number of results to return
        mask: Optional mask for template matching
        
    Returns:
        List of (bounding box, confidence) tuples for matches
    """
    # Check for empty inputs
    if frame is None or template is None:
        return []
        
    # Get dimensions
    frame_height, frame_width = frame.shape[:2]
    template_height, template_width = template.shape[:2]
    
    # Ensure template is smaller than frame
    if template_height > frame_height or template_width > frame_width:
        warning("Template is larger than frame", LogCategory.VISION)
        return []
    
    try:
        # Perform template matching
        result = cv2.matchTemplate(frame, template, method, mask=mask)
        
        # Find matches above threshold
        matches = []
        
        # Different handling based on method
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            # For these methods, smaller values are better
            match_indices = np.where(result <= 1.0 - threshold)
            # Convert score: lower is better -> higher is better
            result = 1.0 - result
        else:
            # For these methods, larger values are better
            match_indices = np.where(result >= threshold)
        
        # Extract coordinates and scores
        y_points, x_points = match_indices
        scores = result[match_indices]
        
        # Create (coordinate, score) pairs and sort by score
        coord_score_pairs = [(x, y, float(score)) for x, y, score in zip(x_points, y_points, scores)]
        coord_score_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Keep only the top matches
        for x, y, score in coord_score_pairs[:max_results]:
            bbox = BoundingBox(
                x=int(x),
                y=int(y),
                width=template_width,
                height=template_height
            )
            matches.append((bbox, score))
            
        return matches
    except Exception as e:
        warning(f"Template matching error: {str(e)}", LogCategory.VISION)
        return []


def detect_color_regions(frame: np.ndarray, 
                       lower_bound: Tuple[int, int, int],
                       upper_bound: Tuple[int, int, int],
                       min_area: int = 100,
                       max_regions: int = 10,
                       color_space: str = 'hsv') -> List[Tuple[BoundingBox, float]]:
    """
    Detect regions in an image based on color range.
    
    Args:
        frame: The image to process
        lower_bound: Lower color bound (in specified color space)
        upper_bound: Upper color bound (in specified color space)
        min_area: Minimum area in pixels for a region to be considered
        max_regions: Maximum number of regions to return
        color_space: Color space to use ('hsv', 'rgb', 'lab', etc.)
        
    Returns:
        List of (bounding box, confidence) tuples for detected regions
    """
    if frame is None:
        return []
        
    try:
        # Convert color space if needed
        if color_space.lower() == 'hsv':
            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        elif color_space.lower() == 'lab':
            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        elif color_space.lower() == 'rgb':
            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            converted = frame  # Use as is (BGR)
            
        # Create mask for the specified color range
        mask = cv2.inRange(converted, 
                           np.array(lower_bound, dtype=np.uint8), 
                           np.array(upper_bound, dtype=np.uint8))
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process contours
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                bbox = BoundingBox(x, y, w, h)
                
                # Use area as a proxy for confidence
                # Normalize to 0-1 range (larger area = higher confidence)
                max_possible_area = frame.shape[0] * frame.shape[1]
                confidence = min(1.0, area / max_possible_area * 10)  # Scale for better distribution
                
                regions.append((bbox, confidence))
        
        # Sort by confidence and limit number of results
        regions.sort(key=lambda r: r[1], reverse=True)
        return regions[:max_regions]
    except Exception as e:
        warning(f"Color detection error: {str(e)}", LogCategory.VISION)
        return []


def extract_text_region(frame: np.ndarray, bbox: BoundingBox, 
                      padding: int = 0) -> np.ndarray:
    """
    Extract a region from a frame for text recognition.
    
    This function applies preprocessing to improve OCR accuracy.
    
    Args:
        frame: The source frame
        bbox: Bounding box of the region to extract
        padding: Padding to add around the region
        
    Returns:
        Preprocessed image region optimized for OCR
    """
    # Extract region with padding
    padded_bbox = bbox.expand(padding)
    y_slice, x_slice = padded_bbox.to_slice_tuple()
    
    # Ensure within frame bounds
    height, width = frame.shape[:2]
    y_slice = slice(max(0, y_slice.start), min(height, y_slice.stop))
    x_slice = slice(max(0, x_slice.start), min(width, x_slice.stop))
    
    region = frame[y_slice, x_slice].copy()
    
    # Apply preprocessing for better OCR
    # Convert to grayscale
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    
    # Apply slight blur to remove noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    return thresh


def compute_frame_histogram(frame: np.ndarray, 
                          mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute color histogram for a frame.
    
    Args:
        frame: The input frame
        mask: Optional mask to only include certain areas
        
    Returns:
        Normalized histogram array
    """
    # Convert to HSV color space (better for color analysis)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Compute histogram
    hist = cv2.calcHist([hsv], [0, 1], mask, [30, 32], [0, 180, 0, 256])
    
    # Normalize histogram
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    
    return hist


def compare_histograms(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """
    Compare two histograms and return similarity score.
    
    Args:
        hist1: First histogram
        hist2: Second histogram
        
    Returns:
        Similarity score (0-1, where 1 is identical)
    """
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


def find_circles(frame: np.ndarray, 
               min_radius: int = 10, 
               max_radius: int = 100,
               param1: int = 50,
               param2: int = 30,
               max_circles: int = 10) -> List[Tuple[BoundingBox, float]]:
    """
    Detect circles in a frame using Hough Circle Transform.
    
    Useful for detecting circular UI elements like minimap, portraits, etc.
    
    Args:
        frame: Input frame
        min_radius: Minimum circle radius
        max_radius: Maximum circle radius
        param1: First method parameter (edge detection sensitivity)
        param2: Second method parameter (circle detection sensitivity)
        max_circles: Maximum number of circles to return
        
    Returns:
        List of (bounding box, confidence) tuples
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Detect circles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    results = []
    if circles is not None:
        # Convert to integer coordinates
        circles = np.round(circles[0, :]).astype(int)
        
        # Sort by param2 value (confidence proxy)
        for x, y, r in circles:
            # Create bounding box from circle
            bbox = BoundingBox(
                x=int(x - r),
                y=int(y - r),
                width=int(2 * r),
                height=int(2 * r)
            )
            
            # Use radius as a proxy for confidence
            max_possible_radius = min(frame.shape[0], frame.shape[1]) / 2
            confidence = min(1.0, r / max_possible_radius)
            
            results.append((bbox, confidence))
            
        # Sort by confidence and limit results
        results.sort(key=lambda r: r[1], reverse=True)
        results = results[:max_circles]
        
    return results


def draw_detections(frame: np.ndarray, 
                  detections: List[Tuple[BoundingBox, str, float]],
                  color: Tuple[int, int, int] = (0, 255, 0),
                  thickness: int = 2,
                  with_labels: bool = True,
                  with_confidence: bool = True) -> np.ndarray:
    """
    Draw detection boxes and labels on a frame.
    
    Args:
        frame: The frame to draw on
        detections: List of (bounding box, label, confidence) tuples
        color: BGR color tuple for the boxes
        thickness: Line thickness
        with_labels: Whether to draw labels
        with_confidence: Whether to include confidence in labels
        
    Returns:
        Frame with detections drawn on it
    """
    result = frame.copy()
    
    for bbox, label, confidence in detections:
        # Draw bounding box
        cv2.rectangle(
            result,
            (bbox.x, bbox.y),
            (bbox.x + bbox.width, bbox.y + bbox.height),
            color,
            thickness
        )
        
        # Draw label
        if with_labels:
            text = label
            if with_confidence:
                text += f" {confidence:.2f}"
                
            # Background for text
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(
                result,
                (bbox.x, bbox.y - text_size[1] - 5),
                (bbox.x + text_size[0], bbox.y),
                color,
                -1  # Filled rectangle
            )
            
            # Text
            cv2.putText(
                result,
                text,
                (bbox.x, bbox.y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),  # Black text
                1,
                cv2.LINE_AA
            )
    
    return result
