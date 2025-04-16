import os
import time
import numpy as np
import cv2
import threading
from typing import Callable, Dict, List, Any

from src.utils.logging import info, debug, warning, LogCategory
from src.vision.capture.screen_capture import ScreenCapture
from src.vision.models.detection_result import DetectionResult, DetectionResults, DetectionType
from src.vision.detection_manager import DetectionManager
from src.decision.state.game_state import GameState

class DetectionObserver:
    """
    Observer that updates game state based on detection results.
    
    This class demonstrates how detection results can be used to
    update the game state, including player health, target info, etc.
    """
    def __init__(self, game_state: GameState = GameState()):
        """
        Initialize the detection observer.
        
        Args:
            game_state: The game state to update
        """
        self.game_state = game_state
        self.last_update_time = 0.0
    
    def update_from_detections(self, detections: List[Dict[str, Any]]) -> None:
        """
        Update game state based on detection results.
        
        Args:
            detections: List of detection dictionaries
        """
        current_time = time.time()

        print(f"\nProcessing {len(detections)} detections...")
        
        # Process player health text
        health_detections = [d for d in detections 
            if d['detection_type'] == DetectionType.PLAYER_HEALTH_TEXT]

        # Process target health text
        target_detections = [d for d in detections 
            if d['detection_type'] == DetectionType.TARGET_HEALTH_TEXT]
        
        if health_detections:
            # Update player health in game state
            self.game_state.player.update_health(health_detections[0]['label'])
            info(f"Updated player health to {self.game_state.player.current_health}%", LogCategory.VISION)

        if target_detections:
            # Update target health in game state
            self.game_state.target.update_health(target_detections[0]['label'])
            info(f"Updated target health to {self.game_state.target.current_health}%", LogCategory.VISION)
        
        # Update last update time
        self.last_update_time = current_time


class ScreenProcessor:
    """
    Class that processes WoW screenshots and send screenshot to detection manager.
    
    This demonstrates the integration between the vision system and
    the game state system.
    """
    def __init__(self, templates_dir: str = "data/templates", 
                screenshot_dir: str = "data/screenshots"):
        """
        Initialize the screen processor.
        
        Args:
            templates_dir: Directory containing template images
            screenshot_dir: Directory for saving screenshots
        """
        self.templates_dir = templates_dir
        self.screenshot_dir = screenshot_dir
        self.detection_manager = DetectionManager(templates_dir=templates_dir)
        self.capture = ScreenCapture()
        
        # Make sure directories exist
        os.makedirs(templates_dir, exist_ok=True)
        os.makedirs(screenshot_dir, exist_ok=True)
        
        # Flag for running the processing loop
        self.running = False
        self.processing_thread = None
    
    def save_frame(self, frame: np.ndarray, prefix: str = "screenshot") -> str:
        """
        Save a screenshot to disk.
        
        Args:
            frame: The screenshot to save
            prefix: Prefix for the filename
            
        Returns:
            Path to the saved screenshot
        """
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{prefix}_{timestamp}.png"
        filepath = os.path.join(self.screenshot_dir, filename)
        
        cv2.imwrite(filepath, frame)
        debug(f"Saved screenshot to {filepath}", LogCategory.VISION)
        
        return filepath
    
    def process_frame(self, frame: np.ndarray,
                     save: bool = False,
                     visualize: bool = True) -> DetectionResults:
        """
        Process a WoW screenshot and update game state.
        
        Args:
            frame: The screenshot to process
            save: Whether to save the screenshot
            visualize: Whether to show visualization
            
        Returns:
            Dictionary of detection results
        """
        if save:
            self.save_frame(frame)
        
        # Perform detection with all active detectors
        results = self.detection_manager.detect(
            frame, 
            visualization=visualize
        )
        
        # Show visualization if requested
        if visualize and results.visualization is not None:
            try:
                cv2.imshow("WoW Detection Results", results.visualization)
                # Wait for key with timeout (1 second)
                cv2.waitKey(1000)  # 1000ms = 1 second
            except Exception as e:
                warning(f"Visualization error: {str(e)}", LogCategory.VISION)
            finally:
                # Make sure to close windows properly
                cv2.destroyWindow("WoW Detection Results")
        
        return results
    
    def start_processing(self, 
                        detections_callback: Callable[[DetectionResults], None],
                        interval: float = 1,
                        save_frames: bool = False,
                        visualize: bool = False) -> None:
        """
        Start continuous screenshot processing.
        
        Args:
            capture_func: Function that returns a screenshot
            interval: Time between processing frames (seconds)
            visualize: Whether to show visualization
        """
        self.running = True
        
        def processing_loop():
            while self.running:
                #try:
                    # Get full game frame
                    frame = self.capture.capture()
                    
                    if frame is not None:
                        # Process the screenshot
                        frame_process_results = self.process_frame(
                            frame,
                            save=save_frames,
                            visualize=visualize
                        )
                        
                        # Call detections callback
                        detections_callback(frame_process_results)
                    
                    # Wait for the next interval
                    time.sleep(interval)
                #except Exception as e:
                #    warning(f"Error in processing loop: {str(e)}", LogCategory.VISION)
        
        # Start processing in a separate thread
        #self.processing_thread = threading.Thread(target=processing_loop)
        #self.processing_thread.daemon = True
        #self.processing_thread.start()

        processing_loop()
        
        info("Started continuous screenshot processing", LogCategory.VISION)
    
    def stop_processing(self) -> None:
        """Stop continuous screenshot processing."""
        self.running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
            self.processing_thread = None
        
        info("Stopped continuous screenshot processing", LogCategory.VISION)