"""
    Abstract base class for detection handlers.
    
    Detection handlers can register for detections in frame and receive details of detected objects
    They register for different type of detections and update the game state accordingly
    """

from abc import ABC, abstractmethod
from typing import List, Any, Callable

from src.decision.state.game_state import GameState
from src.vision.models.detection_result import DetectionResult, DetectionType

class DetectionHandler(ABC):

    def __init__(self, detection_types: List[DetectionType], 
               game_state: GameState):
        """
        Initialize the detector.
        
        Args:
            name: Unique name for this detector
            detection_type: DetectionType this detector can produce
            min_confidence: Minimum confidence threshold for detections
            enabled: Whether this detector is enabled by default
        """
        self.detection_types = detection_types
        self.game_state = game_state
        self.detections: List[[DetectionType, DetectionResult]] = []

    @abstractmethod
    def process_detections(self) -> None:
        """
        Process all detections and update game state.
        """

    def add_detection(self, detection_type: DetectionType, detection: DetectionResult) -> None:
        """
        Called when an observed state change occurs.
        
        Args:
            detection_type: Type of detection that occurred
            detection: Detection result
        """
        self.detections.append((detection_type, detection))

    def reset_detections(self) -> None:
        self.detections = []
    
    def get_detection_types(self) -> List[DetectionType]:
        return self.detection_types

    def detection_type_is_supported(self, detection_type: DetectionType) -> bool:
        return detection_type in self.detection_types

    def update_game_state_value(self, on_update_function: Callable[[Any], None], new_value: Any, current_value: Any):
        if new_value != current_value:
            on_update_function(new_value)