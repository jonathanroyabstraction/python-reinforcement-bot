"""
Base detector interface for the WoW Bot vision system.

This module provides the abstract base class for all detectors
in the vision system, ensuring consistent behavior and interfaces.
"""
from abc import ABC, abstractmethod
import time
from typing import Dict, List, Optional, Tuple, Any, Set
import cv2
import numpy as np

from src.utils.logging import debug, info, warning, LogCategory
from src.vision.models.detection_result import DetectionResult, DetectionResults, BoundingBox, DetectionType


class BaseDetector(ABC):
    """
    Abstract base class for all object detectors.
    
    All detector implementations must inherit from this class
    and implement the required methods.
    """
    def __init__(self, name: str, detection_type: DetectionType, 
               min_confidence: float = 0.5, enabled: bool = True):
        """
        Initialize the detector.
        
        Args:
            name: Unique name for this detector
            detection_type: DetectionType this detector can produce
            min_confidence: Minimum confidence threshold for detections
            enabled: Whether this detector is enabled by default
        """
        self.name = name
        self.detection_type = detection_type
        self.min_confidence = min_confidence
        self.enabled = enabled
        
        # Cache for detection results
        self._cache_enabled = False
        self._cache_timeout = 1.0  # seconds
        self._last_frame_id = -1
        self._cached_results = None
        self._last_cache_time = 0
        
        # Default regions of interest
        self._regions_of_interest = {}
        
        # Stats tracking
        self._detection_count = 0
        self._total_detection_time = 0
        self._last_detection_time = 0
        
        debug(f"Initialized {self.name} detector", LogCategory.VISION)
    
    @abstractmethod
    def detect(self, frame: np.ndarray, frame_id: int = 0, 
              regions: Optional[Dict[str, BoundingBox]] = None) -> DetectionResults:
        """
        Detect objects in a frame.
        
        This is the main detection method that must be implemented by all detectors.
        
        Args:
            frame: The image frame to process (numpy array in BGR format)
            frame_id: Unique identifier for the frame
            regions: Optional dict of named regions to limit detection to
            
        Returns:
            DetectionResults object containing all detections
        """
        pass
    
    def detect_with_cache(self, frame: np.ndarray, frame_id: int = 0,
                        regions: Optional[Dict[str, BoundingBox]] = None) -> DetectionResults:
        """
        Detect objects with caching support.
        
        This wrapper method handles caching of detection results to avoid
        redundant processing of the same frame multiple times.
        
        Args:
            frame: The image frame to process
            frame_id: Unique identifier for the frame
            regions: Optional dict of named regions to limit detection to
            
        Returns:
            DetectionResults object containing all detections
        """
        # If detector is disabled, return empty results
        if not self.enabled:
            return DetectionResults([], frame_id, time.time())
            
        # Check if we can use cached results
        current_time = time.time()
        cache_valid = (
            self._cache_enabled and
            frame_id == self._last_frame_id and
            current_time - self._last_cache_time < self._cache_timeout and
            self._cached_results is not None
        )
        
        if cache_valid:
            debug(f"Using cached results for {self.name} detector", LogCategory.VISION)
            return self._cached_results
            
        # Measure detection time
        start_time = time.time()
        
        # Perform detection
        results = self.detect(frame, frame_id, regions)
        
        # Update cache
        self._last_frame_id = frame_id
        self._cached_results = results
        self._last_cache_time = current_time
        
        # Update stats
        self._detection_count += 1
        self._last_detection_time = time.time() - start_time
        self._total_detection_time += self._last_detection_time
        
        debug(f"{self.name} detector found {len(results)} results in {self._last_detection_time:.3f}s", 
             LogCategory.VISION)
        
        return results
    
    def set_cache_params(self, enabled: bool, timeout: float = 1.0) -> None:
        """
        Configure the result caching behavior.
        
        Args:
            enabled: Whether to enable result caching
            timeout: How long cached results are valid (in seconds)
        """
        self._cache_enabled = enabled
        self._cache_timeout = timeout
    
    def set_regions_of_interest(self, regions: Dict[str, BoundingBox]) -> None:
        """
        Set regions of interest for this detector.
        
        Args:
            regions: Dict mapping region names to bounding boxes
        """
        self._regions_of_interest = regions
    
    def get_region(self, name: str) -> Optional[BoundingBox]:
        """
        Get a region of interest by name.
        
        Args:
            name: Name of the region
            
        Returns:
            BoundingBox for the region or None if not found
        """
        return self._regions_of_interest.get(name)
    
    def crop_to_region(self, frame: np.ndarray, region: BoundingBox) -> np.ndarray:
        """
        Crop a frame to a specific region.
        
        Args:
            frame: The image frame to crop
            region: The region to crop to
            
        Returns:
            Cropped image
        """
        # Ensure region is within frame bounds
        height, width = frame.shape[:2]
        x = max(0, min(region.x, width - 1))
        y = max(0, min(region.y, height - 1))
        w = min(region.width, width - x)
        h = min(region.height, height - y)
        
        return frame[y:y+h, x:x+w]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for this detector.
        
        Returns:
            Dict containing detector statistics
        """
        avg_time = 0
        if self._detection_count > 0:
            avg_time = self._total_detection_time / self._detection_count
            
        return {
            "name": self.name,
            "enabled": self.enabled,
            "detection_count": self._detection_count,
            "avg_detection_time": avg_time,
            "last_detection_time": self._last_detection_time
        }
    
    def reset_stats(self) -> None:
        """Reset the performance statistics."""
        self._detection_count = 0
        self._total_detection_time = 0
        self._last_detection_time = 0
    
    def create_detection(self, 
                       detection_type: DetectionType,
                       label: str,
                       bbox: BoundingBox,
                       confidence: float,
                       metadata: Optional[Dict[str, Any]] = None,
                       thumbnail: Optional[np.ndarray] = None) -> DetectionResult:
        """
        Create a detection result with this detector's information.
        
        Helper method to create properly formatted detection results.
        
        Args:
            detection_type: Type of the detection
            label: Label for the detection
            bbox: Bounding box
            confidence: Confidence score (0.0-1.0)
            metadata: Additional metadata for the detection
            thumbnail: Small image of the detected object (for visualization)
            
        Returns:
            DetectionResult instance
        """
        # Filter low confidence detections
        if confidence < self.min_confidence:
            return None
            
        # Create the detection result
        return DetectionResult(
            detector_name=self.name,
            detection_type=detection_type,
            label=label,
            bounding_box=bbox,
            confidence=confidence,
            metadata=metadata or {},
            thumbnail=thumbnail
        )

    def visualize(self, frame: np.ndarray, results: DetectionResults) -> Optional[np.ndarray]:
        """
        Create a visualization of detection results.
        
        Args:
            frame: The original frame
            results: Detection results to visualize
            
        Returns:
            Frame with visualized detections
        """
        if frame is None or not results.detections:
            return frame.copy() if frame is not None else None
            
        # Create visualization data
        vis_detections = [
            (d.bounding_box, d.label, d.confidence) 
            for d in results.detections 
            if d.detector_name == self.name
        ]
        
        # If no detections from this detector, return original frame
        if not vis_detections:
            return frame.copy()
            
        # Import here to avoid circular imports
        from src.vision.utils.vision_utils import draw_detections
        
        # Draw detections
        return draw_detections(
            frame.copy(),
            vis_detections,
            with_labels=True,
            with_confidence=True
        )
