"""
Template-based detector for the WoW Bot vision system.

This module provides a detector that uses template matching to identify
UI elements and other game features with known visual appearance.
"""
import os
import glob
from typing import Dict, List, Set, Tuple, Optional, Any
import time
import cv2
import numpy as np

from src.utils.logging import debug, info, warning, error, LogCategory
from src.vision.detectors.base_detector import BaseDetector
from src.vision.models.detection_result import DetectionResult, DetectionResults, BoundingBox, DetectionType
from src.vision.utils.vision_utils import load_template, match_template


class TemplateDetector(BaseDetector):
    """
    Detector that uses template matching to find UI elements.
    
    This detector loads template images from disk and uses OpenCV's
    template matching to find them in the game screen.
    """
    def __init__(self, detection_type: DetectionType,
                name: str = "TemplateDetector", 
               templates_dir: str = "data/templates",
               min_confidence: float = 0.7,
               enabled: bool = True):
        """
        Initialize the template detector.
        
        Args:
            name: Unique name for this detector
            templates_dir: Directory containing template images
            detection_type: Type of detection this detector can produce
            min_confidence: Minimum confidence threshold for detections
            enabled: Whether this detector is enabled by default
        """
        super().__init__(name, detection_type, min_confidence, enabled)
        
        self.templates_dir = templates_dir
        self.templates = {}  # type: Dict[str, Dict[str, Any]]
        self.template_categories = {}  # type: Dict[str, List[str]]
        
        # Load templates
        self._load_templates()
        
        # Enable caching by default
        self.set_cache_params(True, 0.5)  # Cache valid for 0.5 seconds
        
        info(f"Template detector initialized with {len(self.templates)} templates", 
             LogCategory.VISION)
    
    def _load_templates(self) -> None:
        """Load all template images from the templates directory."""
        if not os.path.exists(self.templates_dir):
            warning(f"Templates directory does not exist: {self.templates_dir}", LogCategory.VISION)
            return
            
        # Find template image files
        image_extensions = ["*.png", "*.jpg", "*.jpeg"]
        template_files = []
        for ext in image_extensions:
            pattern = os.path.join(self.templates_dir, "**", ext)
            template_files.extend(glob.glob(pattern, recursive=True))
            
        # Process each template
        for filepath in template_files:
            # Extract category from directory structure
            rel_path = os.path.relpath(filepath, self.templates_dir)
            parts = os.path.split(rel_path)
            
            if len(parts) > 1:
                category = os.path.dirname(rel_path).replace(os.path.sep, "_")
            else:
                category = "general"
                
            # Load the template image
            template, name = load_template(filepath)
            if template is None:
                continue
                
            # Store template info
            template_id = f"{category}_{name}"
            self.templates[template_id] = {
                "image": template,
                "name": name,
                "category": category,
                "path": filepath,
                "width": template.shape[1],
                "height": template.shape[0]
            }
            
            # Add to category mapping
            if category not in self.template_categories:
                self.template_categories[category] = []
            self.template_categories[category].append(template_id)
            
        debug(f"Loaded {len(self.templates)} templates in {len(self.template_categories)} categories", 
              LogCategory.VISION)
    
    def add_template(self, template_id: str, image: np.ndarray, 
                   category: str = "custom") -> bool:
        """
        Add a new template programmatically.
        
        Args:
            template_id: Unique identifier for the template
            image: Template image as numpy array
            category: Category for the template
            
        Returns:
            True if template was added successfully
        """
        if template_id in self.templates:
            warning(f"Template with ID {template_id} already exists", LogCategory.VISION)
            return False
            
        self.templates[template_id] = {
            "image": image,
            "name": template_id,
            "category": category,
            "path": None,
            "width": image.shape[1],
            "height": image.shape[0]
        }
        
        # Add to category mapping
        if category not in self.template_categories:
            self.template_categories[category] = []
        self.template_categories[category].append(template_id)
        
        debug(f"Added template {template_id} to category {category}", LogCategory.VISION)
        return True
    
    def remove_template(self, template_id: str) -> bool:
        """
        Remove a template from the detector.
        
        Args:
            template_id: ID of the template to remove
            
        Returns:
            True if template was removed
        """
        if template_id not in self.templates:
            return False
            
        template = self.templates.pop(template_id)
        category = template["category"]
        
        if category in self.template_categories:
            if template_id in self.template_categories[category]:
                self.template_categories[category].remove(template_id)
                
            # Remove category if empty
            if not self.template_categories[category]:
                del self.template_categories[category]
                
        debug(f"Removed template {template_id}", LogCategory.VISION)
        return True
    
    def detect(self, frame: np.ndarray, frame_id: int = 0,
              regions: Optional[Dict[str, BoundingBox]] = None) -> DetectionResults:
        """
        Detect templates in a frame.
        
        Args:
            frame: The image frame to process
            frame_id: Unique identifier for the frame
            regions: Optional dict of named regions to limit detection to
            
        Returns:
            DetectionResults containing all matched templates
        """
        results = DetectionResults([], frame_id, time.time())
        
        # Process each region if specified
        if regions:
            for region_name, bbox in regions.items():
                # Crop frame to region
                region_frame = self.crop_to_region(frame, bbox)
                
                # Match templates in this region
                region_results = self._match_templates(region_frame, frame_id)
                
                # Adjust bounding boxes to account for region offset
                for detection in region_results.detections:
                    detection.bounding_box.x += bbox.x
                    detection.bounding_box.y += bbox.y
                    detection.metadata["region"] = region_name
                    results.detections.append(detection)
        else:
            # Process entire frame
            full_results = self._match_templates(frame, frame_id)
            results.detections.extend(full_results.detections)
            
        return results
    
    def _match_templates(self, frame: np.ndarray, frame_id: int) -> DetectionResults:
        """
        Match all templates against a frame or region.
        
        Args:
            frame: Image to process
            frame_id: Frame identifier
            
        Returns:
            DetectionResults for this frame/region
        """
        results = DetectionResults([], frame_id, time.time())
        
        # Skip if frame is too small
        if frame.shape[0] < 10 or frame.shape[1] < 10:
            return results
            
        # Match each template
        for template_id, template_info in self.templates.items():
            template_img = template_info["image"]
            
            # Skip if template is larger than frame
            if (template_img.shape[0] > frame.shape[0] or 
                template_img.shape[1] > frame.shape[1]):
                continue
                
            # Perform template matching
            matches = match_template(
                frame, 
                template_img, 
                threshold=self.min_confidence,
                max_results=3  # Limit matches per template
            )
            
            # Create detection result for each match
            for bbox, confidence in matches:
                # Create detection
                detection = self.create_detection(
                    detection_type=self.detection_type,
                    label=template_info["name"],
                    bbox=bbox,
                    confidence=confidence,
                    metadata={
                        "template_id": template_id,
                        "category": template_info["category"]
                    }
                )
                
                if detection:
                    results.add(detection)
        
        # Remove duplicate detections
        results = results.remove_duplicates(iou_threshold=0.5, prefer_higher_confidence=True)
        
        return results
    
    def detect_specific_templates(self, frame: np.ndarray, 
                                template_ids: List[str],
                                frame_id: int = 0) -> DetectionResults:
        """
        Detect only specific templates in a frame.
        
        This is more efficient when you only need to find certain templates.
        
        Args:
            frame: The image frame to process
            template_ids: List of template IDs to detect
            frame_id: Unique identifier for the frame
            
        Returns:
            DetectionResults containing matches for specified templates
        """
        results = DetectionResults([], frame_id, time.time())
        
        # Skip if frame is too small
        if frame.shape[0] < 10 or frame.shape[1] < 10:
            return results
        
        # Match only specified templates
        for template_id in template_ids:
            if template_id not in self.templates:
                continue
                
            template_info = self.templates[template_id]
            template_img = template_info["image"]
            
            # Skip if template is larger than frame
            if (template_img.shape[0] > frame.shape[0] or 
                template_img.shape[1] > frame.shape[1]):
                continue
                
            # Perform template matching
            matches = match_template(
                frame, 
                template_img, 
                threshold=self.min_confidence,
                max_results=3  # Limit matches per template
            )
            
            # Create detection result for each match
            for bbox, confidence in matches:
                # Create detection
                detection_type = self._get_detection_type(template_info["category"])
                
                detection = self.create_detection(
                    detection_type=detection_type,
                    label=template_info["name"],
                    bbox=bbox,
                    confidence=confidence,
                    metadata={
                        "template_id": template_id,
                        "category": template_info["category"]
                    }
                )
                
                if detection:
                    results.add(detection)
        
        return results
    
    def detect_category(self, frame: np.ndarray, 
                      category: str,
                      frame_id: int = 0) -> DetectionResults:
        """
        Detect all templates in a specific category.
        
        Args:
            frame: The image frame to process
            category: Category of templates to detect
            frame_id: Unique identifier for the frame
            
        Returns:
            DetectionResults containing matches for the category
        """
        if category not in self.template_categories:
            return DetectionResults([], frame_id, time.time())
            
        template_ids = self.template_categories[category]
        return self.detect_specific_templates(frame, template_ids, frame_id)
