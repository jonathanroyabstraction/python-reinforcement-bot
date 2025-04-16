"""
Core orchestrator for the WoW Bot.

This module manage the communication between the vision system, the game state and the control system.
In the future, it will also handle the communication between the decision system and the control system.
"""

from typing import Any, Callable, override
from src.decision.state.game_state import GameState, GameStateChangeType, GameStateObserver
from src.utils.logging import LogCategory, debug
from src.vision.models.detection_result import DetectionResults, DetectionType
from src.control.input_simulator import InputSimulator
from src.decision.state.detection_handlers.player_detection_handler import PlayerDetectionHandler
from src.decision.state.detection_handlers.target_detection_handler import TargetDetectionHandler

class CoreManager:
    def __init__(self):
        self.game_state = GameState()
        self.control_system = InputSimulator()
        self.detection_handlers = [
            PlayerDetectionHandler(self.game_state),
            TargetDetectionHandler(self.game_state)
        ]

        debug(f"CoreManager initialized with {len(self.detection_handlers)} detection handlers,", LogCategory.SYSTEM)

    def process_vision_detections(self, detection_results: DetectionResults):
        """
        Process detected objects and update game state.
        
        Args:
            detection_results: Results of object detection
        """
        # If detection results is None, game screen has never been found, skip
        if not detection_results:
            return

        for detection in detection_results.detections:
            for detection_handler in self.detection_handlers:
                if detection_handler.detection_type_is_supported(detection.detection_type):
                    detection_handler.add_detection(detection.detection_type, detection)

        for detection_handler in self.detection_handlers:
            detection_handler.process_detections()