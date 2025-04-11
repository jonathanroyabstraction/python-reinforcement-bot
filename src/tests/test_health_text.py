"""
Test script for health text detection in WoW UI.
"""
import os
import sys
import cv2
import numpy as np
import time

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.logging import configure, info, debug, warning, error, LogCategory
from src.vision.models.detection_result import DetectionType, BoundingBox
from src.vision.detectors.text_detector import TextDetector
from src.vision.detection_manager import DetectionManager

def test_health_text_detection(image_path, region=None):
    """
    Test health text detection with improved OCR.
    
    Args:
        image_path: Path to screenshot
        region: BoundingBox for health text region or None to use default
    """
    print(f"Testing health text detection on: {image_path}")
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Create text detector specifically for health percentages
    detector = TextDetector(
        detection_type=DetectionType.HEALTH_TEXT,
        name="health_text_detector",
        min_confidence=0.4,
        enabled=True
    )
    
    region = BoundingBox(718, 753, 24, 14)  # Using the coordinates you specified
    
    # Set region for detector
    detector.set_regions_of_interest({"health_percentage": region})
    
    try:
        # Perform detection
        start_time = time.time()
        results = detector.detect(image)
        elapsed = (time.time() - start_time) * 1000
        
        # Print results
        print(f"Detection completed in {elapsed:.2f}ms")
        print(f"Found {len(results.detections)} text elements")
        
        for detection in results.detections:
            bbox = detection.bounding_box
            text = detection.metadata.get('text', '')
            conf = detection.confidence
            
            print(f"Text: '{text}' at BoundingBox(x={bbox.x}, y={bbox.y}, "
                  f"width={bbox.width}, height={bbox.height}) "
                  f"with confidence {conf:.2f}")
            
            # Extract percentage if present
            import re
            match = re.search(r'(\d+)%', text)
            if match:
                percentage = int(match.group(1))
                print(f"Health percentage: {percentage}%")
        
    finally:
        # Make sure to close any open windows
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Configure logging
    configure()
    
    # Run the test with the specified screenshot
    test_health_text_detection("src/tests/data/screenshots/basic-ui-no-target-full-life.png")
    test_health_text_detection("src/tests/data/screenshots/basic-ui-target-low-health.png")
    test_health_text_detection("src/tests/data/screenshots/basic-ui-target-and-player-low-health.png")

    print("\nTests completed!")
    
