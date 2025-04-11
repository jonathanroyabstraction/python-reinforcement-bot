"""
Test script for the vision system.

This script demonstrates how to use the object detection system
to identify game elements from screenshots.
"""
import os
import sys
import time
import argparse
import numpy as np
import cv2
from typing import Set, List, Dict, Optional, Tuple

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.logging import configure, debug, info, warning, error, LogCategory
from src.vision.models.detection_result import DetectionType, BoundingBox
from src.vision.detectors.template_detector import TemplateDetector
from src.vision.detectors.color_detector import ColorDetector, ColorProfile
from src.vision.detectors.text_detector import TextDetector
from src.vision.detection_manager import DetectionManager


def create_test_image(width: int = 800, height: int = 600) -> np.ndarray:
    """Create a test image with simulated UI elements."""
    # Create base image (dark gray background)
    image = np.ones((height, width, 3), dtype=np.uint8) * 40
    
    # Add player health bar (red)
    cv2.rectangle(image, (50, 50), (250, 70), (0, 0, 200), -1)  # BGR: Red
    cv2.rectangle(image, (50, 50), (250, 70), (0, 0, 100), 2)   # Border
    
    # Add player mana bar (blue)
    cv2.rectangle(image, (50, 80), (200, 100), (200, 0, 0), -1)  # BGR: Blue
    cv2.rectangle(image, (50, 80), (200, 100), (100, 0, 0), 2)   # Border
    
    # Add resource bar (yellow/orange for Hunter focus)
    cv2.rectangle(image, (50, 110), (180, 130), (0, 180, 220), -1)  # BGR: Orange
    cv2.rectangle(image, (50, 110), (180, 130), (0, 100, 150), 2)   # Border
    
    # Add enemy health bar
    cv2.rectangle(image, (400, 50), (550, 65), (0, 0, 180), -1)  # BGR: Red
    cv2.rectangle(image, (400, 50), (550, 65), (0, 0, 100), 2)   # Border
    
    # Add minimap (circle)
    cv2.circle(image, (700, 100), 80, (70, 70, 70), -1)  # Gray circle
    cv2.circle(image, (700, 100), 80, (150, 150, 150), 2)  # Border
    
    # Add ability icons (colored squares)
    icon_positions = [(100, 500), (170, 500), (240, 500), (310, 500)]
    icon_colors = [(0, 0, 200), (0, 200, 0), (200, 0, 0), (150, 150, 0)]
    
    for pos, color in zip(icon_positions, icon_colors):
        x, y = pos
        cv2.rectangle(image, (x, y), (x + 50, y + 50), color, -1)
        cv2.rectangle(image, (x, y), (x + 50, y + 50), (200, 200, 200), 2)
    
    # Add enemy nameplate with red color
    cv2.rectangle(image, (400, 30), (550, 45), (0, 0, 180), -1)  # BGR: Red
    
    # Add friendly NPC nameplate with green color
    cv2.rectangle(image, (600, 400), (750, 415), (0, 180, 0), -1)  # BGR: Green
    
    # Add some text (this won't be readable in OpenCV, but represents UI text)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "Player: L60 Hunter", (50, 30), font, 0.7, (200, 200, 200), 2)
    cv2.putText(image, "Enemy: Wolf", (400, 30), font, 0.7, (200, 200, 200), 2)
    cv2.putText(image, "NPC: Questgiver", (600, 400), font, 0.7, (200, 200, 200), 2)
    
    # Add quest highlight (yellow)
    cv2.rectangle(image, (600, 430), (650, 480), (0, 230, 230), -1)  # BGR: Yellow
    cv2.rectangle(image, (600, 430), (650, 480), (0, 150, 150), 2)   # Border
    
    # Add sparkle effect (white dots)
    for _ in range(10):
        x = np.random.randint(600, 650)
        y = np.random.randint(430, 480)
        cv2.circle(image, (x, y), 2, (255, 255, 255), -1)
    
    return image


def test_template_detector() -> None:
    """Test the template detection functionality."""
    print("\n=== Testing Template Detector ===")
    
    # Create test image
    test_image = create_test_image(800, 600)
    
    # Create a small template from the test image (e.g., player health bar)
    health_bar_template = test_image[50:70, 50:250].copy()
    mana_bar_template = test_image[80:100, 50:200].copy()
    
    # Create template detector
    detector = TemplateDetector(name="test_template_detector")
    
    # Add templates programmatically
    detector.add_template("health_bar", health_bar_template, "ui")
    detector.add_template("mana_bar", mana_bar_template, "ui")
    
    # Perform detection
    start_time = time.time()
    results = detector.detect(test_image)
    elapsed = (time.time() - start_time) * 1000
    
    # Print results
    print(f"Template detection completed in {elapsed:.2f}ms")
    print(f"Found {len(results.detections)} templates:")
    
    for detection in results.detections:
        print(f"  • {detection.label} at {detection.bounding_box} with confidence {detection.confidence:.2f}")
    
    try:
        # Visualize results
        visualized = detector.visualize(test_image, results)
        
        if visualized is not None:
            # Show visualization
            cv2.imshow("Template Detection Results", visualized)
            
            # Wait for a key press with a timeout (5 seconds)
            print("Press any key to continue (or window will close in 5 seconds)...")
            key = cv2.waitKey(5000)  # 5000ms = 5 seconds
    finally:
        # Ensure windows are closed even if an exception occurs
        cv2.destroyAllWindows()


def test_color_detector() -> None:
    """Test the color detection functionality."""
    print("\n=== Testing Color Detector ===")
    
    # Create test image
    test_image = create_test_image(800, 600)
    
    # Create color detector
    detector = ColorDetector(name="test_color_detector")
    
    # Add custom color profile
    detector.add_profile(ColorProfile(
        name="yellow_highlight",
        label="Quest Highlight",
        lower_bound=(15, 150, 150),  # HSV: Yellow (lower)
        upper_bound=(35, 255, 255),  # HSV: Yellow (upper)
        detection_type=DetectionType.QUEST_OBJECTIVE,
        min_area=100
    ))
    
    # Perform detection
    start_time = time.time()
    results = detector.detect(test_image)
    elapsed = (time.time() - start_time) * 1000
    
    # Print results
    print(f"Color detection completed in {elapsed:.2f}ms")
    print(f"Found {len(results.detections)} color regions:")
    
    # Group by type
    type_counts = {}
    for detection in results.detections:
        type_name = detection.detection_type.name
        if type_name not in type_counts:
            type_counts[type_name] = 0
        type_counts[type_name] += 1
    
    for type_name, count in type_counts.items():
        print(f"  • {type_name}: {count} regions")
    
    try:
        # Visualize results
        visualized = detector.visualize(test_image, results)
        
        if visualized is not None:
            # Show visualization
            cv2.imshow("Color Detection Results", visualized)
            
            # Wait for a key press with a timeout (5 seconds)
            print("Press any key to continue (or window will close in 5 seconds)...")
            key = cv2.waitKey(5000)  # 5000ms = 5 seconds
    finally:
        # Ensure windows are closed even if an exception occurs
        cv2.destroyAllWindows()


def test_detection_manager() -> None:
    """Test the detection manager with multiple detector types."""
    print("\n=== Testing Detection Manager ===")
    
    # Create test image
    test_image = create_test_image(800, 600)
    
    # Create detection manager
    manager = DetectionManager()
    
    # Create and add a custom template detector
    template_detector = TemplateDetector(name="custom_template_detector")
    
    # Extract templates from test image
    health_bar_template = test_image[50:70, 50:250].copy()
    mana_bar_template = test_image[80:100, 50:200].copy()
    
    # Add templates
    template_detector.add_template("health_bar", health_bar_template, "ui")
    template_detector.add_template("mana_bar", mana_bar_template, "ui")
    
    # Add detector to manager
    manager.add_detector(template_detector)
    
    # Explicitly activate our custom detector and deactivate default detectors
    manager.activate_detector("custom_template_detector")
    
    # Define regions of interest
    manager.add_region("player_ui", BoundingBox(40, 40, 220, 100))
    manager.add_region("enemy_ui", BoundingBox(390, 40, 170, 35))
    manager.add_region("ability_bar", BoundingBox(90, 490, 280, 70))
    
    # Perform detection with our specific detector and region
    print("Running detection with custom template detector...")
    start_time = time.time()
    results = manager.detect(
        test_image, 
        detector_names=["custom_template_detector"],
        region_names=["player_ui"],  # This is where our health/mana bars are
        visualization=True
    )
    elapsed = (time.time() - start_time) * 1000
    
    # Print results
    print(f"Detection completed in {elapsed:.2f}ms")
    print(f"Found {len(results.detections)} objects:")
    
    # Group by type
    type_counts = {}
    for detection in results.detections:
        type_name = detection.detection_type.name
        if type_name not in type_counts:
            type_counts[type_name] = 0
        type_counts[type_name] += 1
    
    for type_name, count in type_counts.items():
        print(f"  • {type_name}: {count} regions")
    
    try:
        # Show visualization
        if results.visualization is not None:
            cv2.imshow("Detection Manager Results", results.visualization)
            
            # Wait for a key press with a timeout (5 seconds)
            print("Press any key to continue (or window will close in 5 seconds)...")
            key = cv2.waitKey(5000)  # 5000ms = 5 seconds
    finally:
        # Ensure windows are closed even if an exception occurs
        cv2.destroyAllWindows()
    
    # Get performance metrics
    metrics = manager.get_performance_metrics()
    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  • {key}: {value:.2f}")
        else:
            print(f"  • {key}: {value}")


def main():
    """Run the vision system tests."""
    configure()
    
    parser = argparse.ArgumentParser(description="Test the vision system")
    parser.add_argument("--template", action="store_true", help="Run template detector test")
    parser.add_argument("--color", action="store_true", help="Run color detector test")
    parser.add_argument("--manager", action="store_true", help="Run detection manager test")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--no-gui", action="store_true", help="Run tests without GUI") 
    
    args = parser.parse_args()
    
    # If no args specified, run all tests
    all_tests = not (args.template or args.color or args.manager)
    
    # Ensure all OpenCV windows are properly closed before and after tests
    try:
        cv2.destroyAllWindows()
    
        # Run the selected tests with proper exception handling
        try:
            if args.template or args.all or all_tests:
                test_template_detector()
        
            if args.color or args.all or all_tests:
                test_color_detector()
        
            if args.manager or args.all or all_tests:
                test_detection_manager()
        except Exception as e:
            error(f"Error running tests: {str(e)}", LogCategory.VISION)
            import traceback
            traceback.print_exc()
    finally:
        # Make absolutely sure all windows are closed
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
