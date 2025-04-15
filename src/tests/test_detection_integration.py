"""
Integration test for the WoW Bot vision system.

This script demonstrates how to integrate the object detection system
with the game state system to process screenshots and update the game state.
"""
import os
import sys
import time
import argparse
import numpy as np
import cv2
import threading
from typing import Dict, List, Optional, Set, Tuple, Any

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.logging import configure, info, debug, warning, error, LogCategory
from src.utils.config import ConfigManager, get_config, set_config
from src.vision.models.detection_result import DetectionType, BoundingBox
from src.vision.detection_manager import DetectionManager
from src.decision.state.game_state import GameState
from src.decision.state.player_state import PlayerState
from src.decision.state.target_state import TargetState
from src.decision.state.ability_state import AbilityState


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


class WoWScreenProcessor:
    """
    Class that processes WoW screenshots and updates game state.
    
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
        self.game_state = GameState()
        self.detection_manager = DetectionManager(templates_dir=templates_dir)
        self.observer = DetectionObserver(self.game_state)
        
        # Make sure directories exist
        os.makedirs(templates_dir, exist_ok=True)
        os.makedirs(screenshot_dir, exist_ok=True)
        
        # Flag for running the processing loop
        self.running = False
        self.processing_thread = None
    
    def save_screenshot(self, frame: np.ndarray, prefix: str = "screenshot") -> str:
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
    
    def process_screenshot(self, frame: np.ndarray, 
                          save: bool = False,
                          visualize: bool = True) -> Dict[str, Any]:
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
            self.save_screenshot(frame)
        
        # Perform detection with all active detectors
        results = self.detection_manager.detect(
            frame, 
            visualization=visualize
        )
        
        # Convert to list of dictionaries for easier processing
        detections_list = []
        for detection in results.detections:
            detections_list.append({
                'label': detection.label,
                'detection_type': detection.detection_type,
                'bounding_box': detection.bounding_box,
                'confidence': detection.confidence,
                'metadata': detection.metadata,
                'region': detection.metadata.get('region', None)
            })
        
        # Update game state based on detections
        self.observer.update_from_detections(detections_list)
        
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
        
        return {
            'frame_id': results.frame_id,
            'timestamp': results.timestamp,
            'detections': detections_list,
            'detection_count': len(results.detections)
        }
    
    def start_processing(self, 
                        capture_func,
                        interval: float = 0.5,
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
                try:
                    # Get screenshot from capture function
                    frame = capture_func()
                    
                    if frame is not None:
                        # Process the screenshot
                        self.process_screenshot(
                            frame,
                            save=False,  # Don't save every frame
                            visualize=visualize
                        )
                    
                    # Wait for the next interval
                    time.sleep(interval)
                except Exception as e:
                    warning(f"Error in processing loop: {str(e)}", LogCategory.VISION)
        
        # Start processing in a separate thread
        self.processing_thread = threading.Thread(target=processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        info("Started continuous screenshot processing", LogCategory.VISION)
    
    def stop_processing(self) -> None:
        """Stop continuous screenshot processing."""
        self.running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
            self.processing_thread = None
        
        info("Stopped continuous screenshot processing", LogCategory.VISION)
    
    def get_game_state(self) -> GameState:
        """
        Get the current game state.
        
        Returns:
            The current game state
        """
        return self.game_state


def test_with_sample_image(processor: WoWScreenProcessor, 
                          image_path: str) -> None:
    """
    Test the processor with a sample image.
    
    Args:
        processor: The WoW screen processor
        image_path: Path to sample image
    """
    print(f"\nProcessing sample image: {image_path}")
    
    # Load sample image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    try:
        # Process the image
        start_time = time.time()
        results = processor.process_screenshot(
            image,
            save=False,
            visualize=True
        )
        elapsed = (time.time() - start_time) * 1000
    finally:
        # Ensure windows are closed even if an exception occurs
        cv2.destroyAllWindows()
    
    # Print detection summary
    print(f"Processing completed in {elapsed:.2f}ms")
    print(f"Detected {len(results['detections'])} objects:")
    
    # Group by detection type
    type_counts = {}
    for detection in results['detections']:
        type_name = detection['detection_type'].name
        if type_name not in type_counts:
            type_counts[type_name] = 0
        type_counts[type_name] += 1
    
    for type_name, count in type_counts.items():
        print(f"  • {type_name}: {count}")
    
    # Print current game state
    player = processor.game_state.player
    target = processor.game_state.target
    
    print("\nGame State:")
    print(f"  • Player Health: {player.health_percent()}%")
    print(f"  • Player Resource: {player.resource_percent()}%")
    
    if target:
        print(f"  • Target Health: {target.health_percent()}%")
    else:
        print("  • No target selected")
    
    cv2.destroyAllWindows()


def test_with_generated_image(processor: WoWScreenProcessor) -> None:
    """
    Test the processor with a generated test image.
    
    Args:
        processor: The WoW screen processor
    """
    print("\nTesting with generated test image")
    
    # Create a test image
    width, height = 1920, 1080
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw UI elements
    # Player health bar - red rectangle
    cv2.rectangle(image, (10, 10), (10 + 250, 10 + 20), (0, 0, 255), -1)
    
    # Player mana bar - blue rectangle
    cv2.rectangle(image, (10, 40), (10 + 200, 40 + 20), (255, 0, 0), -1)
    
    # Target health bar - red rectangle
    cv2.rectangle(image, (10, 120), (10 + 180, 120 + 20), (0, 0, 255), -1)
    
    # Action bar abilities - yellow circles
    for i in range(10):
        center_x = 550 + i * 80
        center_y = 940
        cv2.circle(image, (center_x, center_y), 30, (0, 255, 255), -1)
    
    try:
        print("Press any key to continue (or window will close in 5 seconds)...")
        cv2.imshow("Test Image", image)
        cv2.waitKey(5000)  # 5000ms = 5 seconds
        
        # Process the image
        results = processor.process_screenshot(
            image,
            save=False,
            visualize=True
        )
    finally:
        # Ensure windows are closed even if an exception occurs
        cv2.destroyAllWindows()


def test_sample_detection_cycle(processor: WoWScreenProcessor, 
                              num_cycles: int = 5,
                              cycle_duration: float = 1.0) -> None:
    """
    Run a sample detection cycle with gradually changing UI elements.
    
    Args:
        processor: The WoW screen processor
        num_cycles: Number of cycles to run
        cycle_duration: Duration of each cycle in seconds
    """
    print(f"\nRunning {num_cycles} detection cycles...")
    
    # Create a base image
    width, height = 1920, 1080
    
    try:
        for cycle in range(num_cycles):
            # Create a new image for this cycle
            image = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Vary health and mana levels based on cycle
            health_percent = max(10, 100 - (cycle * 20))  # Decrease health
            mana_percent = min(100, 50 + (cycle * 10))   # Increase mana
            
            # Calculate bar widths
            health_width = int(250 * (health_percent / 100))
            mana_width = int(200 * (mana_percent / 100))
            
            # Player health bar - red rectangle
            cv2.rectangle(image, (10, 10), (10 + health_width, 10 + 20), (0, 0, 255), -1)
            
            # Player mana bar - blue rectangle
            cv2.rectangle(image, (10, 40), (10 + mana_width, 40 + 20), (255, 0, 0), -1)
            
            # Process the image
            print(f"Cycle {cycle+1}: Health={health_percent}%, Mana={mana_percent}%")
            results = processor.process_screenshot(
                image,
                save=False,
                visualize=True
            )
            
            # Print current state
            player_state = processor.get_game_state().player_state
            print(f"Game State: Health={player_state.current_health}, Mana={player_state.current_resource}")
            
            # Wait for the next cycle
            time.sleep(cycle_duration)
    finally:
        # Ensure windows are closed when done or if an error occurs
        cv2.destroyAllWindows()


def main():
    """Run the WoW bot vision integration tests."""
    configure()
    
    parser = argparse.ArgumentParser(description="Test WoW Bot vision integration")
    parser.add_argument("--sample", type=str, help="Path to sample image")
    parser.add_argument("--cycle", action="store_true", help="Run detection cycle test")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    try:
        # Create a screen processor
        processor = WoWScreenProcessor()
        
        # If no specific test is requested, run with generated image
        if not (args.sample or args.cycle or args.all):
            test_with_generated_image(processor)
        else:
            if args.sample or args.all:
                test_with_sample_image(processor, args.sample or "data/screenshots/sample.png")
            
            if args.cycle or args.all:
                test_sample_detection_cycle(processor)
    finally:
        # Make absolutely sure we destroy all OpenCV windows
        cv2.destroyAllWindows()
    
    print("\nTests completed")


if __name__ == "__main__":
    main()
