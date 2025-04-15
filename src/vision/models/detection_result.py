"""
Detection result models for the WoW Bot vision system.

This module provides classes for representing detection results
from various detectors in the vision system.
"""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any
import time
import numpy as np


class DetectionType(Enum):
    """Types of detections that can be made."""
    # UI Elements
    UI_ELEMENT = auto()   # Generic UI element
    BUTTON = auto()       # Clickable button
    ICON = auto()         # Icon/image in UI
    TOOLTIP = auto()      # Floating information panel
    DIALOG = auto()       # Dialog/window
    
    # Game Entities
    ENEMY = auto()        # Hostile entities
    NPC = auto()          # Non-hostile NPCs
    PLAYER = auto()       # Other players
    INTERACTIVE = auto()  # Interactive objects
    ITEM = auto()         # Items/loot
    
    # Status Effects
    BUFF = auto()         # Positive effects
    DEBUFF = auto()       # Negative effects
    
    # Combat
    ABILITY = auto()      # Abilities/spells
    COOLDOWN = auto()     # Cooldown indicator
    COMBAT_TEXT = auto()  # Combat text (damage, healing, etc.)
    
    # Resources
    HEALTH_BAR = auto()   # Health indicators
    MANA_BAR = auto()     # Mana indicators
    RESOURCE_BAR = auto() # Class-specific resources
    
    # Navigation
    MINIMAP = auto()      # Minimap elements
    WAYPOINT = auto()     # Navigation waypoints
    
    # Quests
    QUEST_OBJECTIVE = auto() # Quest goals/targets
    QUEST_TEXT = auto()      # Quest descriptions
    
    # Text Types
    TEXT = auto()         # Generic text
    PLAYER_HEALTH_TEXT = auto()  # Player health text
    TARGET_HEALTH_TEXT = auto()  # Target health text
    UI_TEXT = auto()      # Text in UI elements
    CHAT_TEXT = auto()    # Chat messages
    NAME_TEXT = auto()    # Entity names
    DIALOG_TEXT = auto()  # Dialog text
    ERROR_TEXT = auto()   # Error messages
    
    # Miscellaneous
    UNKNOWN = auto()      # Unidentified elements


@dataclass
class BoundingBox:
    """Represents a bounding box for a detected object."""
    x: int  # Top-left x coordinate
    y: int  # Top-left y coordinate
    width: int
    height: int
    
    @property
    def area(self) -> int:
        """Calculate the area of the bounding box."""
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get the center point of the bounding box."""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def bottom_right(self) -> Tuple[int, int]:
        """Get the bottom-right corner coordinates."""
        return (self.x + self.width, self.y + self.height)
    
    def contains_point(self, point: Tuple[int, int]) -> bool:
        """Check if the bounding box contains a point."""
        x, y = point
        return (self.x <= x < self.x + self.width and 
                self.y <= y < self.y + self.height)
    
    def intersection(self, other: 'BoundingBox') -> Optional['BoundingBox']:
        """Calculate intersection with another bounding box."""
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)
        
        if x1 < x2 and y1 < y2:
            return BoundingBox(x1, y1, x2 - x1, y2 - y1)
        return None
    
    def iou(self, other: 'BoundingBox') -> float:
        """Calculate Intersection over Union with another bounding box."""
        intersection = self.intersection(other)
        if not intersection:
            return 0.0
            
        intersection_area = intersection.area
        union_area = self.area + other.area - intersection_area
        
        if union_area == 0:
            return 0.0
        return intersection_area / union_area
    
    def expand(self, padding: int) -> 'BoundingBox':
        """Create a new expanded bounding box with padding."""
        return BoundingBox(
            max(0, self.x - padding),
            max(0, self.y - padding),
            self.width + 2 * padding,
            self.height + 2 * padding
        )
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Convert to (x, y, width, height) tuple."""
        return (self.x, self.y, self.width, self.height)
    
    def to_slice_tuple(self) -> Tuple[slice, slice]:
        """Convert to tuple of slices for numpy array indexing."""
        return (slice(self.y, self.y + self.height), 
                slice(self.x, self.x + self.width))
    
    @classmethod
    def from_points(cls, p1: Tuple[int, int], p2: Tuple[int, int]) -> 'BoundingBox':
        """Create a bounding box from two points (any corners)."""
        x1, y1 = p1
        x2, y2 = p2
        
        x = min(x1, x2)
        y = min(y1, y2)
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        return cls(x, y, width, height)
    
    @classmethod
    def from_cv_rect(cls, rect: Tuple[int, int, int, int]) -> 'BoundingBox':
        """Create a bounding box from OpenCV rectangle (x, y, w, h)."""
        x, y, w, h = rect
        return cls(x, y, w, h)


@dataclass
class DetectionResult:
    """
    Represents a single detection result from the vision system.
    
    This class stores information about detected game elements including
    position, type, confidence, and additional metadata.
    """
    # Basic detection information
    detector_name: str
    detection_type: DetectionType
    label: str
    bounding_box: BoundingBox
    confidence: float
    
    # Detection metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Detection image data (for visualization)
    thumbnail: Optional[np.ndarray] = None
    
    # System metadata
    timestamp: float = field(default_factory=time.time)
    
    def distance_to(self, other: 'DetectionResult') -> float:
        """Calculate Euclidean distance to another detection (between centers)."""
        x1, y1 = self.bounding_box.center
        x2, y2 = other.bounding_box.center
        
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    
    def is_similar_to(self, other: 'DetectionResult', 
                    iou_threshold: float = 0.5, 
                    same_type: bool = True) -> bool:
        """
        Check if this detection is similar to another.
        
        Args:
            other: The detection to compare with
            iou_threshold: Minimum Intersection over Union value to be considered similar
            same_type: Whether detections must be of the same type to be similar
            
        Returns:
            True if detections are similar, False otherwise
        """
        if same_type and self.detection_type != other.detection_type:
            return False
            
        iou = self.bounding_box.iou(other.bounding_box)
        return iou >= iou_threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert detection result to a dictionary."""
        return {
            "detector_name": self.detector_name,
            "detection_type": self.detection_type.name,
            "label": self.label,
            "bounding_box": {
                "x": self.bounding_box.x,
                "y": self.bounding_box.y,
                "width": self.bounding_box.width,
                "height": self.bounding_box.height
            },
            "confidence": self.confidence,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DetectionResult':
        """Create a detection result from a dictionary."""
        bbox_data = data.get("bounding_box", {})
        bbox = BoundingBox(
            x=bbox_data.get("x", 0),
            y=bbox_data.get("y", 0),
            width=bbox_data.get("width", 0),
            height=bbox_data.get("height", 0)
        )
        
        return cls(
            detector_name=data.get("detector_name", "unknown"),
            detection_type=DetectionType[data.get("detection_type", "UNKNOWN")],
            label=data.get("label", ""),
            bounding_box=bbox,
            confidence=data.get("confidence", 0.0),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", time.time())
        )


@dataclass
class DetectionResults:
    """
    Collection of detection results from a single processing pass.
    
    This class aggregates multiple detection results and provides
    methods for filtering, sorting, and accessing them.
    """
    detections: List[DetectionResult] = field(default_factory=list)
    frame_id: int = 0
    timestamp: float = field(default_factory=time.time)
    visualization: Optional[np.ndarray] = None
    
    def add(self, detection: DetectionResult) -> None:
        """Add a detection to the collection."""
        self.detections.append(detection)
        
    def extend(self, other_results: 'DetectionResults') -> None:
        """Add all detections from another DetectionResults object.
        
        Args:
            other_results: DetectionResults object to merge with this one
        """
        if other_results and hasattr(other_results, 'detections'):
            self.detections.extend(other_results.detections)
    
    def get_by_type(self, detection_type: DetectionType) -> List[DetectionResult]:
        """Get all detections of a specific type."""
        return [d for d in self.detections if d.detection_type == detection_type]
    
    def get_by_label(self, label: str) -> List[DetectionResult]:
        """Get all detections with a specific label."""
        return [d for d in self.detections if d.label == label]
    
    def get_by_detector(self, detector_name: str) -> List[DetectionResult]:
        """Get all detections from a specific detector."""
        return [d for d in self.detections if d.detector_name == detector_name]
    
    def get_highest_confidence(self, 
                              detection_type: Optional[DetectionType] = None,
                              label: Optional[str] = None) -> Optional[DetectionResult]:
        """Get the detection with highest confidence, optionally filtered by type/label."""
        filtered = self.detections
        
        if detection_type:
            filtered = [d for d in filtered if d.detection_type == detection_type]
            
        if label:
            filtered = [d for d in filtered if d.label == label]
            
        if not filtered:
            return None
            
        return max(filtered, key=lambda d: d.confidence)
    
    def filter(self, 
              min_confidence: float = 0.0,
              detection_types: Optional[List[DetectionType]] = None,
              labels: Optional[List[str]] = None,
              region: Optional[BoundingBox] = None) -> 'DetectionResults':
        """
        Filter detections based on multiple criteria.
        
        Args:
            min_confidence: Minimum confidence threshold
            detection_types: List of detection types to include
            labels: List of labels to include
            region: Bounding box region to filter by (detections must intersect)
            
        Returns:
            A new DetectionResults object with the filtered detections
        """
        filtered = [d for d in self.detections if d.confidence >= min_confidence]
        
        if detection_types:
            filtered = [d for d in filtered if d.detection_type in detection_types]
            
        if labels:
            filtered = [d for d in filtered if d.label in labels]
            
        if region:
            filtered = [d for d in filtered if d.bounding_box.intersection(region)]
            
        results = DetectionResults(filtered, self.frame_id, self.timestamp)
        return results
    
    def remove_duplicates(self, 
                         iou_threshold: float = 0.5, 
                         prefer_higher_confidence: bool = True) -> 'DetectionResults':
        """
        Remove duplicate detections based on IoU threshold.
        
        Args:
            iou_threshold: Minimum IoU value to consider detections as duplicates
            prefer_higher_confidence: Whether to keep the higher confidence detection
                                    when duplicates are found
            
        Returns:
            A new DetectionResults object with duplicates removed
        """
        if not self.detections:
            return DetectionResults([], self.frame_id, self.timestamp)
            
        # Sort by confidence (descending) if we prefer higher confidence
        if prefer_higher_confidence:
            detections_sorted = sorted(self.detections, 
                                      key=lambda d: d.confidence, reverse=True)
        else:
            detections_sorted = self.detections.copy()
            
        unique_detections = []
        for detection in detections_sorted:
            # Check if this detection is a duplicate of any in our unique list
            is_duplicate = any(
                detection.is_similar_to(unique, iou_threshold)
                for unique in unique_detections
            )
            
            if not is_duplicate:
                unique_detections.append(detection)
                
        results = DetectionResults(unique_detections, self.frame_id, self.timestamp)
        return results
    
    def to_dict(self) -> Dict:
        """Convert detection results to a dictionary."""
        return {
            "detections": [d.to_dict() for d in self.detections],
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "count": len(self.detections)
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DetectionResults':
        """Create detection results from a dictionary."""
        detections = [
            DetectionResult.from_dict(d) for d in data.get("detections", [])
        ]
        
        return cls(
            detections=detections,
            frame_id=data.get("frame_id", 0),
            timestamp=data.get("timestamp", time.time())
        )
    
    def __len__(self) -> int:
        """Get the number of detections."""
        return len(self.detections)
