
from src.decision.state.detection_handlers.detection_handler import DetectionHandler
from src.decision.state.target_state import TargetState
from src.vision.models.detection_result import DetectionResult, DetectionType

from src.decision.state.game_state import GameState

class TargetDetectionHandler(DetectionHandler):
    """
    Detection handler for target-related detections.
    """
    def __init__(self, game_state: GameState):
        super().__init__([
            DetectionType.TARGET_HEALTH_TEXT, 
        ], game_state)

    def process_detections(self) -> None:
        """
        Handle target related detection
        
        Args:
            detection_type: Type of detection
            detection: Detection result
        """

        # If no detections and target is set, clear target
        if len(self.detections) < 1 and self.game_state.target != None:
            self.game_state.set_target(None)

        # If detection and target is not set, set target
        if len(self.detections) > 0 and self.game_state.target == None:
            self.game_state.set_target(TargetState())

        for detection_type, detection in self.detections:
            match detection_type:
                case DetectionType.TARGET_HEALTH_TEXT:
                    # Update target health in game state
                    self.update_game_state_value(self.game_state.update_target_health, int(detection.label), self.game_state.target.current_health)
                case _:
                    raise ValueError(f"Unsupported detection type: {detection_type}")
        
        self.reset_detections()
    
        