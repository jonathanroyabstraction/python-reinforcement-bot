"""
Detectors module for the WoW Bot vision system.

This module provides detector classes for identifying game elements using
various detection methods like template matching and color detection.
"""

from src.vision.detectors.base_detector import BaseDetector
from src.vision.detectors.template_detector import TemplateDetector
from src.vision.detectors.color_detector import ColorDetector
from src.vision.detectors.text_detector import TextDetector

__all__ = [
    'BaseDetector',
    'TemplateDetector',
    'ColorDetector',
    'TextDetector'
]
