
from src.decision.state.detection_handlers.detection_handler import DetectionHandler
from src.utils.logging import LogCategory, debug
from src.vision.models.detection_result import DetectionResult, DetectionType

from src.decision.state.game_state import GameState

class PlayerDetectionHandler(DetectionHandler):
    """
    Detection handler for player-related detections.
    """
    def __init__(self, game_state: GameState):
        super().__init__([
            DetectionType.PLAYER_HEALTH_TEXT, 
            DetectionType.PLAYER_RESOURCE_TEXT
        ], game_state)

    def process_detections(self) -> None:
        """
        Handle player related detection
        
        Args:
            detection_type: Type of detection
            detection: Detection result
        """
        
        for detection_type, detection in self.detections:
            match detection_type:
                case DetectionType.PLAYER_HEALTH_TEXT:
                    # Update player health in game state
                    self.update_game_state_value(self.game_state.update_player_health, int(detection.label), self.game_state.player.current_health)
                case DetectionType.PLAYER_RESOURCE_TEXT:
                    # Update player resource in game state
                    self.update_game_state_value(self.game_state.update_player_resource, int(detection.label), self.game_state.player.current_resource)
                case _:
                    raise ValueError(f"Unsupported detection type: {detection_type}")

        self.reset_detections()
    
        