"""
Color-based detector for the WoW Bot vision system.

This module provides a detector that uses color thresholding to identify
game elements with distinctive colors like health bars, mana bars, etc.
"""
import os
import json
from typing import Dict, List, Set, Tuple, Optional, Any
import time
import cv2
import numpy as np

from src.utils.logging import debug, info, warning, error, LogCategory
from src.vision.detectors.base_detector import BaseDetector
from src.vision.models.detection_result import DetectionResult, DetectionResults, BoundingBox, DetectionType
from src.vision.utils.vision_utils import detect_color_regions


class ColorProfile:
    """Represents a color profile for detection."""
    def __init__(self, name: str, label: str, 
                lower_bound: Tuple[int, int, int],
                upper_bound: Tuple[int, int, int],
                detection_type: DetectionType,
                color_space: str = 'hsv',
                min_area: int = 100):
        """
        Initialize a color profile.
        
        Args:
            name: Unique name for this profile
            label: Human-readable label
            lower_bound: Lower color bound (in specified color space)
            upper_bound: Upper color bound (in specified color space)
            detection_type: Type of detection this profile produces
            color_space: Color space to use ('hsv', 'rgb', 'lab', etc.)
            min_area: Minimum area in pixels for a region to be considered
        """
        self.name = name
        self.label = label
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.detection_type = detection_type
        self.color_space = color_space.lower()
        self.min_area = min_area
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "label": self.label,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "detection_type": self.detection_type.name,
            "color_space": self.color_space,
            "min_area": self.min_area
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ColorProfile':
        """Create a color profile from a dictionary."""
        return cls(
            name=data["name"],
            label=data["label"],
            lower_bound=tuple(data["lower_bound"]),
            upper_bound=tuple(data["upper_bound"]),
            detection_type=DetectionType[data["detection_type"]],
            color_space=data.get("color_space", "hsv"),
            min_area=data.get("min_area", 100)
        )


class ColorDetector(BaseDetector):
    """
    Detector that uses color ranges to find game elements.
    
    This detector identifies areas with specific colors, such as
    health bars, mana bars, and other color-coded elements.
    """
    def __init__(self, detection_type: DetectionType,
                name: str = "ColorDetector",
               config_path: Optional[str] = None,
               min_confidence: float = 0.5,
               enabled: bool = True):
        """
        Initialize the color detector.
        
        Args:
            name: Unique name for this detector
            config_path: Path to color profiles configuration file
            detection_type: Type of detection this detector can produce
            min_confidence: Minimum confidence threshold for detections
            enabled: Whether this detector is enabled by default
        """
            
        super().__init__(name, detection_type, min_confidence, enabled)
        
        self.config_path = config_path
        self.color_profiles = {}  # type: Dict[str, ColorProfile]
        
        # Add default color profiles for WoW
        self._add_default_profiles()
        
        # Load profiles from config if provided
        if config_path and os.path.exists(config_path):
            self._load_profiles(config_path)
        
        # Enable caching by default
        self.set_cache_params(True, 0.2)  # Cache valid for 0.2 seconds
        
        info(f"Color detector initialized with {len(self.color_profiles)} profiles", 
             LogCategory.VISION)
    
    def _add_default_profiles(self) -> None:
        """Add default color profiles for common WoW elements."""
        # Health bar (red)
        self.add_profile(ColorProfile(
            name="health_bar",
            label="Health Bar",
            lower_bound=(0, 100, 100),  # HSV: Red with high saturation
            upper_bound=(10, 255, 255),
            detection_type=DetectionType.HEALTH_BAR,
            min_area=200
        ))
        
        # Mana bar (blue)
        self.add_profile(ColorProfile(
            name="mana_bar",
            label="Mana Bar",
            lower_bound=(100, 100, 100),  # HSV: Blue with high saturation
            upper_bound=(140, 255, 255),
            detection_type=DetectionType.MANA_BAR,
            min_area=200
        ))
        
        # Focus bar (orange/yellow for Hunter)
        self.add_profile(ColorProfile(
            name="focus_bar",
            label="Focus Bar",
            lower_bound=(20, 150, 150),  # HSV: Orange-yellow
            upper_bound=(40, 255, 255),
            detection_type=DetectionType.RESOURCE_BAR,
            min_area=200
        ))
        
        # Enemy nameplate (red)
        self.add_profile(ColorProfile(
            name="enemy_nameplate",
            label="Enemy Nameplate",
            lower_bound=(0, 150, 100),  # HSV: Red
            upper_bound=(10, 255, 255),
            detection_type=DetectionType.ENEMY,
            min_area=50
        ))
        
        # Friendly nameplate (green)
        self.add_profile(ColorProfile(
            name="friendly_nameplate",
            label="Friendly Nameplate",
            lower_bound=(40, 100, 100),  # HSV: Green
            upper_bound=(80, 255, 255),
            detection_type=DetectionType.NPC,
            min_area=50
        ))
        
        # Quest item/NPC highlight (yellow glow)
        self.add_profile(ColorProfile(
            name="quest_highlight",
            label="Quest Highlight",
            lower_bound=(25, 180, 180),  # HSV: Bright yellow
            upper_bound=(35, 255, 255),
            detection_type=DetectionType.QUEST_OBJECTIVE,
            min_area=100
        ))
        
        # Interactive object highlight (sparkle)
        self.add_profile(ColorProfile(
            name="interactive_highlight",
            label="Interactive Highlight",
            lower_bound=(0, 0, 200),  # HSV: Very bright (white/sparkle)
            upper_bound=(180, 30, 255),
            detection_type=DetectionType.INTERACTIVE,
            min_area=30
        ))
    
    def _load_profiles(self, config_path: str) -> None:
        """
        Load color profiles from a configuration file.
        
        Args:
            config_path: Path to JSON configuration file
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            for profile_data in config.get("color_profiles", []):
                profile = ColorProfile.from_dict(profile_data)
                self.color_profiles[profile.name] = profile
                
            debug(f"Loaded {len(self.color_profiles)} color profiles from {config_path}", 
                 LogCategory.VISION)
        except Exception as e:
            error(f"Failed to load color profiles: {str(e)}", LogCategory.VISION)
    
    def save_profiles(self, config_path: Optional[str] = None) -> bool:
        """
        Save color profiles to a configuration file.
        
        Args:
            config_path: Path to save to, or None to use the default path
            
        Returns:
            True if profiles were saved successfully
        """
        path = config_path or self.config_path
        if not path:
            warning("No config path specified for saving color profiles", LogCategory.VISION)
            return False
            
        try:
            profiles_data = [p.to_dict() for p in self.color_profiles.values()]
            config = {"color_profiles": profiles_data}
            
            with open(path, 'w') as f:
                json.dump(config, f, indent=2)
                
            debug(f"Saved {len(self.color_profiles)} color profiles to {path}", 
                 LogCategory.VISION)
            return True
        except Exception as e:
            error(f"Failed to save color profiles: {str(e)}", LogCategory.VISION)
            return False
    
    def add_profile(self, profile: ColorProfile) -> None:
        """
        Add a color profile to the detector.
        
        Args:
            profile: The color profile to add
        """
        self.color_profiles[profile.name] = profile
        debug(f"Added color profile: {profile.name}", LogCategory.VISION)
    
    def remove_profile(self, profile_name: str) -> bool:
        """
        Remove a color profile from the detector.
        
        Args:
            profile_name: Name of the profile to remove
            
        Returns:
            True if profile was removed
        """
        if profile_name in self.color_profiles:
            del self.color_profiles[profile_name]
            debug(f"Removed color profile: {profile_name}", LogCategory.VISION)
            return True
        return False
    
    def detect(self, frame: np.ndarray, frame_id: int = 0,
              regions: Optional[Dict[str, BoundingBox]] = None) -> DetectionResults:
        """
        Detect areas with specific colors in a frame.
        
        Args:
            frame: The image frame to process
            frame_id: Unique identifier for the frame
            regions: Optional dict of named regions to limit detection to
            
        Returns:
            DetectionResults containing all detected color regions
        """
        results = DetectionResults([], frame_id, time.time())
        
        # Process each region if specified
        if regions:
            for region_name, bbox in regions.items():
                # Crop frame to region
                region_frame = self.crop_to_region(frame, bbox)
                
                # Detect colors in this region
                region_results = self._detect_colors(region_frame, frame_id)
                
                # Adjust bounding boxes to account for region offset
                for detection in region_results.detections:
                    detection.bounding_box.x += bbox.x
                    detection.bounding_box.y += bbox.y
                    detection.metadata["region"] = region_name
                    results.detections.append(detection)
        else:
            # Process entire frame
            full_results = self._detect_colors(frame, frame_id)
            results.detections.extend(full_results.detections)
            
        return results
    
    def _detect_colors(self, frame: np.ndarray, frame_id: int) -> DetectionResults:
        """
        Detect colors in a frame or region.
        
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
            
        # Process each color profile
        for profile_name, profile in self.color_profiles.items():
            # Detect regions matching this color profile
            color_regions = detect_color_regions(
                frame,
                profile.lower_bound,
                profile.upper_bound,
                min_area=profile.min_area,
                max_regions=10,
                color_space=profile.color_space
            )
            
            # Create detection for each region
            for bbox, confidence in color_regions:
                if confidence >= self.min_confidence:
                    detection = self.create_detection(
                        detection_type=profile.detection_type,
                        label=profile.label,
                        bbox=bbox,
                        confidence=confidence,
                        metadata={
                            "profile_name": profile_name,
                            "color_space": profile.color_space
                        }
                    )
                    
                    if detection:
                        results.add(detection)
        
        # Remove duplicate detections
        results = results.remove_duplicates(iou_threshold=0.3)
        
        return results
    
    def detect_specific_colors(self, frame: np.ndarray,
                             profile_names: List[str],
                             frame_id: int = 0) -> DetectionResults:
        """
        Detect only specific color profiles in a frame.
        
        Args:
            frame: The image frame to process
            profile_names: List of color profile names to detect
            frame_id: Unique identifier for the frame
            
        Returns:
            DetectionResults containing matches for specified profiles
        """
        results = DetectionResults([], frame_id, time.time())
        
        # Skip if frame is too small
        if frame.shape[0] < 10 or frame.shape[1] < 10:
            return results
            
        # Process each specified color profile
        for profile_name in profile_names:
            if profile_name not in self.color_profiles:
                continue
                
            profile = self.color_profiles[profile_name]
            
            # Detect regions matching this color profile
            color_regions = detect_color_regions(
                frame,
                profile.lower_bound,
                profile.upper_bound,
                min_area=profile.min_area,
                max_regions=10,
                color_space=profile.color_space
            )
            
            # Create detection for each region
            for bbox, confidence in color_regions:
                if confidence >= self.min_confidence:
                    detection = self.create_detection(
                        detection_type=profile.detection_type,
                        label=profile.label,
                        bbox=bbox,
                        confidence=confidence,
                        metadata={
                            "profile_name": profile_name,
                            "color_space": profile.color_space
                        }
                    )
                    
                    if detection:
                        results.add(detection)
        
        return results
