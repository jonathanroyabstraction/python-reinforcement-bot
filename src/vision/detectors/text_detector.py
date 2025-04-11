"""
Text recognition detector for the WoW Bot vision system.

This module provides a detector that uses OCR to read text 
from UI elements and other in-game text.
"""
import os
import time
from typing import Dict, List, Set, Optional, Tuple, Any
import cv2
import numpy as np

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

from src.utils.logging import debug, info, warning, error, LogCategory
from src.vision.detectors.base_detector import BaseDetector
from src.vision.models.detection_result import DetectionResult, DetectionResults, BoundingBox, DetectionType
from src.vision.utils.vision_utils import extract_text_region


class TextDetector(BaseDetector):
    """
    Detector that uses OCR to read text from the game screen.
    
    This detector uses Tesseract OCR to recognize text in UI elements
    and other areas of the game screen.
    """
    def __init__(self, detection_type: DetectionType,
                name: str = "TextDetector",
               tesseract_cmd: Optional[str] = None,
               min_confidence: float = 0.4,
               enabled: bool = True):
        """
        Initialize the text detector.
        
        Args:
            name: Unique name for this detector
            tesseract_cmd: Path to Tesseract executable
            detection_type: Type of detection this detector can produce
            min_confidence: Minimum confidence threshold for detections
            enabled: Whether this detector is enabled by default
        """
        super().__init__(name, detection_type, min_confidence, enabled)
        
        # Check if Tesseract is available
        if not TESSERACT_AVAILABLE:
            warning("Pytesseract is not installed. Text detection will be disabled.", 
                   LogCategory.VISION)
            self.enabled = False
            return
            
        # Set Tesseract command path if provided
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            
        # Check if Tesseract is properly installed
        try:
            pytesseract.get_tesseract_version()
            info("Tesseract OCR initialized successfully", LogCategory.VISION)
        except Exception as e:
            warning(f"Tesseract OCR initialization failed: {str(e)}", LogCategory.VISION)
            warning("Make sure Tesseract is installed and available in PATH", LogCategory.VISION)
            self.enabled = False
            return
            
        # Enable caching by default (OCR is computationally expensive)
        self.set_cache_params(True, 1.0)  # Cache valid for 1 second
    
    def detect(self, frame: np.ndarray, frame_id: int = 0,
              regions: Optional[Dict[str, BoundingBox]] = None) -> DetectionResults:
        """
        Detect and read text in a frame.
        
        This method is intended for targeted text detection in specific regions.
        For general text detection, use detect_in_regions.
        
        Args:
            frame: The image frame to process
            frame_id: Unique identifier for the frame
            regions: Optional dict of named regions to limit detection to
            
        Returns:
            DetectionResults containing all detected text
        """
        if not self.enabled or not TESSERACT_AVAILABLE:
            return DetectionResults([], frame_id, time.time())
            
        results = DetectionResults([], frame_id, time.time())
        
        # Process each region if specified
        if self._regions_of_interest:
            for region_name, bbox in self._regions_of_interest.items():
                # Crop frame to region
                region_frame = self.crop_to_region(frame, bbox)

                # Read text in this region
                text, confidence = self._perform_ocr(region_frame)

                print("Text detection region:", region_name, bbox, text, confidence)
                
                if text and confidence >= self.min_confidence:
                    # Create detection
                    detection_type = self._get_detection_type(region_name)
                    
                    detection = self.create_detection(
                        detection_type=self.detection_type,
                        label=text,  
                        bbox=bbox,
                        confidence=confidence,
                        metadata={
                            "text": text,  
                            "region": region_name
                        }
                    )

                    print("Text detection result:", detection)
                    
                    if detection:
                        results.add(detection)
        else:
            # Without regions, perform OCR on entire frame
            # This is not recommended for performance reasons
            warning("Text detection on entire frame is inefficient. Consider specifying regions.", 
                   LogCategory.VISION)
                   
            text, confidence = self._perform_ocr(frame)
            
            if text and confidence >= self.min_confidence:
                # Create a bounding box for the entire frame
                bbox = BoundingBox(0, 0, frame.shape[1], frame.shape[0])
                
                detection = self.create_detection(
                    detection_type=DetectionType.TEXT,
                    label=text[:40] + ("..." if len(text) > 40 else ""),
                    bbox=bbox,
                    confidence=confidence,
                    metadata={"full_text": text}
                )
                
                if detection:
                    results.add(detection)
        
        return results
    
    def detect_in_regions(self, frame: np.ndarray, 
                        text_regions: List[Tuple[BoundingBox, str]], 
                        frame_id: int = 0) -> DetectionResults:
        """
        Detect text in specific regions of a frame.
        
        Args:
            frame: The image frame to process
            text_regions: List of (bounding box, region name) tuples
            frame_id: Unique identifier for the frame
            
        Returns:
            DetectionResults containing all detected text
        """
        if not self.enabled or not TESSERACT_AVAILABLE:
            return DetectionResults([], frame_id, time.time())
            
        results = DetectionResults([], frame_id, time.time())
        
        # Process each text region
        for bbox, region_name in text_regions:
            # Crop frame to region
            region_frame = self.crop_to_region(frame, bbox)
            
            # Skip if region is too small
            if region_frame.shape[0] < 10 or region_frame.shape[1] < 10:
                continue
                
            # Preprocess region for better OCR
            processed_region = extract_text_region(frame, bbox)
            
            # Read text in this region
            text, confidence = self._perform_ocr(processed_region)
            
            if text and confidence >= self.min_confidence:
                # Create detection
                detection_type = self._get_detection_type(region_name)
                
                detection = self.create_detection(
                    detection_type=detection_type,
                    label=text[:40] + ("..." if len(text) > 40 else ""),  # Truncate for label
                    bbox=bbox,
                    confidence=confidence,
                    metadata={
                        "full_text": text,
                        "region": region_name
                    }
                )
                
                if detection:
                    results.add(detection)
        
        return results
    
    def _perform_ocr(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Perform OCR on an image.
        
        Args:
            image: The image to process
            
        Returns:
            Tuple of (recognized text, confidence)
        """
        if not TESSERACT_AVAILABLE:
            return "", 0.0
            
        try:
            # Ensure grayscale for OCR
            if len(image.shape) == 3 and image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Apply preprocessing to enhance text
            # 1. Resize the image (make it larger)
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            
            # 2. Apply thresholding to make text more distinct
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            
            # 3. Denoise to remove speckles
            denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
            
            # Set OCR config for digits and percentage signs
            # --psm 7: Treat the image as a single text line
            # --oem 1: Use LSTM OCR Engine
            # -c tessedit_char_whitelist="0123456789%": Only recognize these characters
            config = '--psm 7 --oem 1 -c tessedit_char_whitelist="0123456789%"'
            
            # Perform OCR with improved configuration
            text = pytesseract.image_to_string(denoised, config=config).strip()
            
            # If we're expecting percentages, verify the result looks like a percentage
            import re
            if re.search(r'\d+%', text):
                confidence = 0.9  # High confidence if it matches expected format
            else:
                # Try with a different PSM mode if first attempt failed
                config = '--psm 8 --oem 1 -c tessedit_char_whitelist="0123456789%"'
                text = pytesseract.image_to_string(denoised, config=config).strip()
                confidence = 0.7
                
            # If still no match, try with the raw threshold image
            if not re.search(r'\d+%', text):
                config = '--psm 10 --oem 1 -c tessedit_char_whitelist="0123456789%"'
                text = pytesseract.image_to_string(thresh, config=config).strip()
                confidence = 0.5
            
            print(f"OCR result: '{text}' with confidence {confidence}")
            return text, confidence
        except Exception as e:
            warning(f"OCR error: {str(e)}", LogCategory.VISION)
            return "", 0.0
    
    def _get_detection_type(self, region_name: str) -> DetectionType:
        """Map region name to detection type."""
        # Map common region names to detection types
        region_map = {
            "quest": DetectionType.QUEST_TEXT,
            "chat": DetectionType.CHAT_TEXT,
            "ui": DetectionType.UI_TEXT,
            "tooltip": DetectionType.UI_TEXT,
            "combat_log": DetectionType.COMBAT_TEXT,
            "name": DetectionType.NAME_TEXT,
            "objective": DetectionType.QUEST_TEXT,
            "dialog": DetectionType.DIALOG_TEXT,
            "error": DetectionType.ERROR_TEXT
        }
        
        # Check for region name in map
        for key, value in region_map.items():
            if key in region_name.lower():
                return value
                
        # Default to general text
        return DetectionType.TEXT
    
    def read_text(self, frame: np.ndarray, bbox: BoundingBox, 
                preprocessing: bool = True) -> Tuple[str, float]:
        """
        Read text from a specific area of a frame.
        
        This is a convenience method for directly extracting text without creating detections.
        
        Args:
            frame: Source image frame
            bbox: Bounding box of the text region
            preprocessing: Whether to apply text preprocessing
            
        Returns:
            Tuple of (recognized text, confidence)
        """
        if not self.enabled or not TESSERACT_AVAILABLE:
            return "", 0.0
            
        # Crop region
        region = self.crop_to_region(frame, bbox)
        
        # Apply preprocessing if requested
        if preprocessing:
            region = extract_text_region(frame, bbox)
        
        # Perform OCR
        return self._perform_ocr(region)
