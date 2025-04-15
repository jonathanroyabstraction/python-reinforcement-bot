"""
Detection Manager for the WoW Bot vision system.

This module provides a centralized manager for coordinating all detectors
and handling the detection pipeline.
"""
import os
from pipes import Template
import time
from typing import Dict, List, Set, Tuple, Optional, Any, Type, Union
import cv2
import numpy as np
import threading
from collections import defaultdict
import concurrent.futures

from src.utils.logging import debug, info, warning, error, LogCategory
from src.utils.config import ConfigManager
from src.vision.detectors.base_detector import BaseDetector
from src.vision.detectors.template_detector import TemplateDetector
from src.vision.detectors.text_detector import TextDetector
from src.vision.models.detection_result import DetectionResults, BoundingBox, DetectionType
from src.vision.utils.vision_utils import draw_detections


class DetectionManager:
    """
    Manages and coordinates the object detection process.
    
    This class serves as the central point for configuring detectors, 
    processing frames, and retrieving detection results.
    """
    def __init__(self, templates_dir: str = "data/templates",
                color_config_path: Optional[str] = None):
        """
        Initialize the detection manager.
        
        Args:
            templates_dir: Directory for template images
            color_config_path: Path to color detector configuration
        """
        self.detectors: Dict[str, BaseDetector] = {}
        self.region_definitions: Dict[str, BoundingBox] = {}
        self.frame_count = 0
        self.active_detectors: Set[str] = set()
        self.lock = threading.RLock()
        self.last_detection_time = 0.0
        self.last_detection_results = None  # type: Optional[DetectionResults]
        self.last_frame = None  # type: Optional[np.ndarray]
        self.config_manager = ConfigManager()
        
        # Performance metrics
        self.performance_metrics = {
            'last_detection_ms': 0,
            'avg_detection_ms': 0,
            'total_detections': 0,
            'detection_count': 0
        }

        # Detector for detecting if the game is displayed
        self.ingame_detector = TemplateDetector(
            name="ingame_detector",
            templates_dir='data/templates/ingame',
            detection_type=DetectionType.UI_ELEMENT,
            min_confidence=0.7,
            enabled=True
        )
        
        # Initialize common detectors
        self._init_default_detectors(templates_dir, color_config_path)
        
        info("Detection Manager initialized", LogCategory.VISION)
    
    def _init_default_detectors(self, templates_dir: str, 
                              color_config_path: Optional[str]) -> None:
        """
        Initialize the default detectors.
        
        Args:
            templates_dir: Directory for template images
            color_config_path: Path to color detector configuration
        """
        self.config_manager.load_config()
        screen_capture_config = self.config_manager.config.get("screen_capture", {})
        interface_regions = screen_capture_config.get("regions", {})

        # Player health detector
        player_health_text_detector = TextDetector(
            name="player_health_text_detector",
            detection_type=DetectionType.PLAYER_HEALTH_TEXT,
            min_confidence=0.4,
            enabled=True
        )
        
        player_health_text_config = interface_regions.get("player_health_text", {})
        player_health_text_detector.set_regions_of_interest({"player_health_text": BoundingBox(
            player_health_text_config.get("x"),
            player_health_text_config.get("y"),
            player_health_text_config.get("width"),
            player_health_text_config.get("height")
        )})
        self.add_detector(player_health_text_detector)

        # Target health detector
        target_health_text_config = interface_regions.get("target_health_text", {})
        target_health_text_detector = TextDetector(
            name="target_health_text_detector",
            detection_type=DetectionType.TARGET_HEALTH_TEXT,
            min_confidence=0.4,
            enabled=True
        )

        target_health_text_config = interface_regions.get("target_health_text", {})
        target_health_text_detector.set_regions_of_interest({"target_health_text": BoundingBox(
            target_health_text_config.get("x"),
            target_health_text_config.get("y"),
            target_health_text_config.get("width"),
            target_health_text_config.get("height")
        )})
        self.add_detector(target_health_text_detector)
        
        # Activate all detectors by default
        for detector_name in self.detectors:
            self.active_detectors.add(detector_name)
    
    def add_detector(self, detector: BaseDetector) -> None:
        """
        Add a detector to the manager.
        
        Args:
            detector: The detector to add
        """
        with self.lock:
            if detector.name in self.detectors:
                warning(f"Detector with name '{detector.name}' already exists", LogCategory.VISION)
                return
                
            self.detectors[detector.name] = detector
            debug(f"Added detector: {detector.name}", LogCategory.VISION)
        for detector_name in self.detectors:
            self.active_detectors.add(detector_name)
    
    def add_detector(self, detector: BaseDetector) -> None:
        """
        Add a detector to the manager.
        
        Args:
            detector: The detector to add
        """
        with self.lock:
            if detector.name in self.detectors:
                warning(f"Detector with name '{detector.name}' already exists", LogCategory.VISION)
                return
                
            self.detectors[detector.name] = detector
            debug(f"Added detector: {detector.name}", LogCategory.VISION)
    
    def remove_detector(self, detector_name: str) -> bool:
        """
        Remove a detector from the manager.
        
        Args:
            detector_name: Name of the detector to remove
            
        Returns:
            True if detector was removed
        """
        with self.lock:
            if detector_name in self.detectors:
                del self.detectors[detector_name]
                if detector_name in self.active_detectors:
                    self.active_detectors.remove(detector_name)
                debug(f"Removed detector: {detector_name}", LogCategory.VISION)
                return True
            return False
    
    def get_detector(self, detector_name: str) -> Optional[BaseDetector]:
        """
        Get a detector by name.
        
        Args:
            detector_name: Name of the detector
            
        Returns:
            The detector, or None if not found
        """
        with self.lock:
            return self.detectors.get(detector_name)
    
    def activate_detector(self, detector_name: str) -> bool:
        """
        Activate a detector.
        
        Args:
            detector_name: Name of the detector to activate
            
        Returns:
            True if detector was activated
        """
        with self.lock:
            if detector_name in self.detectors:
                self.active_detectors.add(detector_name)
                debug(f"Activated detector: {detector_name}", LogCategory.VISION)
                return True
            return False
    
    def deactivate_detector(self, detector_name: str) -> bool:
        """
        Deactivate a detector.
        
        Args:
            detector_name: Name of the detector to deactivate
            
        Returns:
            True if detector was deactivated
        """
        with self.lock:
            if detector_name in self.active_detectors:
                self.active_detectors.remove(detector_name)
                debug(f"Deactivated detector: {detector_name}", LogCategory.VISION)
                return True
            return False
    
    def add_region(self, region_name: str, bbox: BoundingBox) -> None:
        """
        Add a named region for targeted detection.
        
        Args:
            region_name: Name for the region
            bbox: Bounding box of the region
        """
        with self.lock:
            self.region_definitions[region_name] = bbox
            debug(f"Added region: {region_name} at {bbox}", LogCategory.VISION)
    
    def remove_region(self, region_name: str) -> bool:
        """
        Remove a named region.
        
        Args:
            region_name: Name of the region to remove
            
        Returns:
            True if region was removed
        """
        with self.lock:
            if region_name in self.region_definitions:
                del self.region_definitions[region_name]
                debug(f"Removed region: {region_name}", LogCategory.VISION)
                return True
            return False
    
    def detect(self, frame: np.ndarray, 
              detector_names: Optional[List[str]] = None,
              region_names: Optional[List[str]] = None,
              detection_types: Optional[Set[DetectionType]] = None,
              parallel: bool = False,
              visualization: bool = False) -> DetectionResults:
        """
        Process a frame and detect objects.
        
        Args:
            frame: The image frame to process
            detector_names: List of detector names to use, or None for all active
            region_names: List of region names to process, or None for the whole frame
            detection_types: Types of detections to include, or None for all
            parallel: Whether to run detectors in parallel
            visualization: Whether to create a visualization of the detections
            
        Returns:
            DetectionResults containing all detections
        """
            
        with self.lock:
            start_time = time.time()
            
            # Increment frame counter
            self.frame_count += 1
            frame_id = self.frame_count

            # Check if frame is in game, else skip detection
            if not self.is_in_game(frame):
                warning("Player not in game, skipping detection", LogCategory.VISION)
                return DetectionResults([], frame_id, start_time)

            # Determine which detectors to use
            if detector_names is None:
                detectors_to_use = [self.detectors[name] for name in self.active_detectors]
            else:
                detectors_to_use = [self.detectors[name] for name in detector_names 
                                   if name in self.detectors]
            
            # Filter by detection types if specified
            if detection_types:
                filtered_detectors = []
                for detector in detectors_to_use:
                    if detector.detection_type in detection_types:
                        filtered_detectors.append(detector)
                detectors_to_use = filtered_detectors
            
            # Determine which regions to process
            regions = None
            if region_names:
                regions = {name: self.region_definitions[name] 
                          for name in region_names 
                          if name in self.region_definitions}
                
                if not regions:
                    warning("No valid regions specified", LogCategory.VISION)
            
            # Collect all detection results
            all_results = DetectionResults([], frame_id, start_time)
            
            if parallel and len(detectors_to_use) > 1:
                # Process detectors in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(detectors_to_use)) as executor:
                    # For each detector, provide a copy of the frame to avoid cropping issues
                    future_to_detector = {
                        executor.submit(detector.detect, frame.copy(), frame_id, regions): detector
                        for detector in detectors_to_use
                    }
                    
                    for future in concurrent.futures.as_completed(future_to_detector):
                        detector = future_to_detector[future]
                        try:
                            results = future.result()
                            all_results.extend(results)
                        except Exception as e:
                            error(f"Error in detector {detector.name}: {str(e)}", LogCategory.VISION)
            else:
                # Process detectors sequentially
                for detector in detectors_to_use:
                    try:
                        results = detector.detect(frame, frame_id, regions)
                        all_results.extend(results)
                    except Exception as e:
                        error(f"Error in detector {detector.name}: {str(e)}", LogCategory.VISION)
            
            # Filter by detection types if needed
            if detection_types:
                all_results.detections = [d for d in all_results.detections 
                                         if d.detection_type in detection_types]
            
            # Remove duplicates
            all_results = all_results.remove_duplicates(iou_threshold=0.3)
            
            # Create visualization if requested
            if visualization and all_results.detections:
                visualization_frame = frame.copy()
                visualization_detections = [
                    (d.bounding_box, d.label, d.confidence) 
                    for d in all_results.detections
                ]
                
                all_results.visualization = draw_detections(
                    visualization_frame,
                    visualization_detections,
                    with_labels=True,
                    with_confidence=True
                )
            
            # Update performance metrics
            end_time = time.time()
            execution_time_ms = (end_time - start_time) * 1000
            
            self.performance_metrics['last_detection_ms'] = execution_time_ms
            self.performance_metrics['total_detections'] += execution_time_ms
            self.performance_metrics['detection_count'] += 1
            self.performance_metrics['avg_detection_ms'] = (
                self.performance_metrics['total_detections'] / 
                self.performance_metrics['detection_count']
            )
            
            # Store results
            self.last_detection_time = end_time
            self.last_detection_results = all_results
            
            debug(f"Detection completed in {execution_time_ms:.1f}ms with {len(all_results.detections)} objects", 
                 LogCategory.VISION)
            
            return all_results
    
    def detect_in_region(self, frame: np.ndarray, 
                        region_name: str,
                        detector_names: Optional[List[str]] = None,
                        detection_types: Optional[Set[DetectionType]] = None) -> DetectionResults:
        """
        Detect objects in a specific named region.
        
        Args:
            frame: The image frame to process
            region_name: Name of the region to process
            detector_names: List of detector names to use, or None for all active
            detection_types: Types of detections to include, or None for all
            
        Returns:
            DetectionResults for the specified region
        """
        if region_name not in self.region_definitions:
            warning(f"Region '{region_name}' not defined", LogCategory.VISION)
            return DetectionResults([], self.frame_count + 1, time.time())
            
        return self.detect(
            frame, 
            detector_names=detector_names,
            region_names=[region_name],
            detection_types=detection_types
        )
    
    def get_detection_types(self, frame: np.ndarray, 
                          detection_type: DetectionType,
                          detector_names: Optional[List[str]] = None,
                          min_confidence: float = 0.5) -> DetectionResults:
        """
        Get detections of a specific type.
        
        Args:
            frame: The image frame to process
            detection_type: Type of detection to retrieve
            detector_names: List of detector names to use, or None for all active
            min_confidence: Minimum confidence threshold
            
        Returns:
            DetectionResults containing detections of the specified type
        """
        results = self.detect(
            frame, 
            detector_names=detector_names,
            detection_types={detection_type}
        )
        
        # Filter by confidence
        filtered_detections = [d for d in results.detections if d.confidence >= min_confidence]
        results.detections = filtered_detections
        
        return results
    
    def get_last_results(self) -> Optional[DetectionResults]:
        """
        Get the results from the last detection.
        
        Returns:
            The last detection results, or None if no detection has been performed
        """
        with self.lock:
            return self.last_detection_results
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics for the detection system.
        
        Returns:
            Dictionary of performance metrics
        """
        with self.lock:
            return self.performance_metrics.copy()
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        with self.lock:
            self.performance_metrics = {
                'last_detection_ms': 0,
                'avg_detection_ms': 0,
                'total_detections': 0,
                'detection_count': 0
            }
    
    def visualize_detections(self, frame: np.ndarray = None, 
                           results: DetectionResults = None) -> Optional[np.ndarray]:
        """
        Create a visualization of detections.
        
        Args:
            frame: Frame to visualize on, or None to use last frame
            results: Detection results to visualize, or None to use last results
            
        Returns:
            Image with visualized detections, or None if no frame/results
        """
        with self.lock:
            # Use provided frame or last frame
            vis_frame = frame if frame is not None else self.last_frame
            if vis_frame is None:
                return None
                
            # Use provided results or last results
            vis_results = results if results is not None else self.last_detection_results
            if vis_results is None or not vis_results.detections:
                return vis_frame.copy()
                
            # Create visualization
            visualization_detections = [
                (d.bounding_box, d.label, d.confidence) 
                for d in vis_results.detections
            ]
            
            return draw_detections(
                vis_frame.copy(),
                visualization_detections,
                with_labels=True,
                with_confidence=True
            )

    def is_in_game(self, frame: np.ndarray) -> bool: 
        in_game_detection = self.ingame_detector.detect(frame, self.frame_count)

        return len(in_game_detection.detections) > 0 and in_game_detection.detections[0].confidence > 0.8
