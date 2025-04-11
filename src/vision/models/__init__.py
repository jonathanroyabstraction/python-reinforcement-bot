"""
Models module for the WoW Bot vision system.

This module provides data models for representing detection results,
bounding boxes, and detection types.
"""

from src.vision.models.detection_result import (
    DetectionResult, 
    DetectionResults, 
    BoundingBox, 
    DetectionType
)

__all__ = [
    'DetectionResult',
    'DetectionResults',
    'BoundingBox',
    'DetectionType'
]
