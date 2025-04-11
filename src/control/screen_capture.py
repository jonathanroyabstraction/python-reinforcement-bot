"""
Screen capture module for the WoW Bot.

This module provides functionality to capture the screen or specific regions
efficiently using the MSS library, with support for caching, FPS throttling,
and performance monitoring.
"""
import os
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple, Union

import cv2
import mss
import numpy as np
from PIL import Image

from src.utils.config import get_config
from src.utils.logging import LogCategory, debug, info, warning

# Type aliases for clarity
ScreenRegion = Dict[str, int]  # x, y, width, height
RGBColor = Tuple[int, int, int]


@dataclass
class CaptureStats:
    """Statistics for screen capture performance monitoring."""
    last_capture_time: float = 0.0
    avg_capture_time: float = 0.0
    total_captures: int = 0
    cached_captures: int = 0
    capture_times: List[float] = None
    
    def __post_init__(self):
        """Initialize the capture times list."""
        if self.capture_times is None:
            self.capture_times = []
    
    def update(self, capture_time: float, cached: bool = False) -> None:
        """
        Update capture statistics.
        
        Args:
            capture_time: Time taken for capture in milliseconds
            cached: Whether the capture was served from cache
        """
        self.last_capture_time = capture_time
        self.capture_times.append(capture_time)
        
        # Keep only the last 100 captures for stats
        if len(self.capture_times) > 100:
            self.capture_times.pop(0)
        
        self.avg_capture_time = sum(self.capture_times) / len(self.capture_times)
        self.total_captures += 1
        
        if cached:
            self.cached_captures += 1
    
    @property
    def cache_hit_ratio(self) -> float:
        """Calculate the cache hit ratio."""
        if self.total_captures == 0:
            return 0.0
        return self.cached_captures / self.total_captures
    
    def get_summary(self) -> Dict[str, Union[float, int]]:
        """Get a summary of capture statistics."""
        return {
            'last_ms': self.last_capture_time,
            'avg_ms': self.avg_capture_time,
            'total': self.total_captures,
            'cached': self.cached_captures,
            'cache_hit_ratio': self.cache_hit_ratio
        }


class ScreenCapture:
    """
    Screen capture implementation using MSS library.
    
    Features:
    - Capture full screen or specific regions
    - Configurable FPS throttling
    - Frame caching to avoid redundant captures
    - Performance monitoring
    """
    
    def __init__(self, cache_timeout_ms: int = 50):
        """
        Initialize the screen capture.
        
        Args:
            cache_timeout_ms: How long to keep cached frames (in milliseconds)
        """
        # Initialize screen capture
        self._sct = mss.mss()
        self._lock = Lock()
        
        # Capture configuration
        self._cache_timeout_ms = cache_timeout_ms
        self._min_frame_time_ms = 1000.0 / get_config('screen_capture.fps', 30)
        
        # Get screen resolution from config or detect automatically
        self._width = get_config('screen_capture.resolution.width', None)
        self._height = get_config('screen_capture.resolution.height', None)
        
        if self._width is None or self._height is None:
            # Auto-detect screen resolution
            monitor = self._sct.monitors[0]  # Primary monitor
            self._width = monitor['width']
            self._height = monitor['height']
        
        # Prepare regions from configuration
        self._regions = self._load_regions_from_config()
        
        # Cache for captured frames
        self._frame_cache = {}
        self._last_capture_time = 0.0
        
        # Performance monitoring
        self._stats = CaptureStats()
        
        # Log initialization
        info(f"Screen capture initialized ({self._width}x{self._height})", LogCategory.CONTROL)
        debug(f"Loaded regions: {list(self._regions.keys())}", LogCategory.CONTROL)
        debug(f"Cache timeout: {self._cache_timeout_ms}ms, Min frame time: {self._min_frame_time_ms}ms", 
              LogCategory.CONTROL)
    
    def _load_regions_from_config(self) -> Dict[str, ScreenRegion]:
        """
        Load screen regions from configuration.
        
        Returns:
            Dictionary of named regions with their coordinates
        """
        regions = {}
        
        # Add full screen region
        regions['full'] = {
            'left': 0,
            'top': 0,
            'width': self._width,
            'height': self._height
        }
        
        # Load regions from configuration
        config_regions = get_config('screen_capture.regions', {})
        
        for name, region in config_regions.items():
            # Convert to mss format (using 'left' and 'top' instead of 'x' and 'y')
            regions[name] = {
                'left': region.get('x', 0),
                'top': region.get('y', 0),
                'width': region.get('width', 100),
                'height': region.get('height', 100)
            }
        
        return regions
    
    def get_region(self, region_name: str) -> Optional[ScreenRegion]:
        """
        Get the coordinates for a named region.
        
        Args:
            region_name: Name of the region to get
            
        Returns:
            Region coordinates or None if not found
        """
        return self._regions.get(region_name)
    
    def add_region(self, name: str, x: int, y: int, width: int, height: int) -> None:
        """
        Add or update a custom region.
        
        Args:
            name: Name for this region
            x: X coordinate (left)
            y: Y coordinate (top)
            width: Width of the region
            height: Height of the region
        """
        self._regions[name] = {
            'left': x,
            'top': y,
            'width': width,
            'height': height
        }
        debug(f"Added region '{name}': {self._regions[name]}", LogCategory.CONTROL)
    
    def capture(self, region_name: str = 'full', force_update: bool = False) -> np.ndarray:
        """
        Capture a screenshot of the specified region.
        
        Args:
            region_name: Name of the region to capture
            force_update: Force a new capture even if a recent one is cached
            
        Returns:
            Screenshot as a numpy array (OpenCV format, BGR)
            
        Raises:
            ValueError: If the region name is not recognized
        """
        with self._lock:
            # Check if we need to throttle based on FPS settings
            current_time = time.time() * 1000  # ms
            elapsed = current_time - self._last_capture_time
            
            # Get region information
            region = self._regions.get(region_name)
            if not region:
                warning(f"Region '{region_name}' not found", LogCategory.CONTROL)
                raise ValueError(f"Unknown screen region: {region_name}")
            
            # Check if we can use a cached frame
            cache_key = region_name
            if (not force_update and 
                cache_key in self._frame_cache and 
                elapsed < self._cache_timeout_ms):
                
                # Use cached frame
                self._stats.update(0.0, cached=True)
                debug(f"Using cached frame for region '{region_name}'", LogCategory.CONTROL)
                return self._frame_cache[cache_key].copy()
            
            # FPS throttling
            if elapsed < self._min_frame_time_ms and not force_update:
                # Sleep to maintain FPS
                sleep_time = (self._min_frame_time_ms - elapsed) / 1000.0
                time.sleep(sleep_time)
            
            # Start timing the capture
            start_time = time.time()
            
            # Capture the screenshot
            screenshot = self._sct.grab(region)
            
            # Convert to numpy array (OpenCV format)
            # mss returns BGRA, convert to BGR for OpenCV
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # Update cache
            self._frame_cache[cache_key] = frame.copy()
            
            # Update timing stats
            capture_time = (time.time() - start_time) * 1000  # ms
            self._stats.update(capture_time)
            self._last_capture_time = time.time() * 1000  # ms
            
            debug(f"Captured region '{region_name}' in {capture_time:.2f}ms", LogCategory.CONTROL)
            
            return frame
    
    def capture_to_file(self, filename: str, region_name: str = 'full') -> str:
        """
        Capture a screenshot and save it to a file.
        
        Args:
            filename: Name of the file to save (without extension)
            region_name: Name of the region to capture
            
        Returns:
            Full path to the saved file
        """
        # Capture the region
        frame = self.capture(region_name, force_update=True)
        
        # Create directory if needed
        save_dir = get_config('system.data_dir', './data') + '/screenshots'
        os.makedirs(save_dir, exist_ok=True)
        
        # Add timestamp to filename
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        full_filename = f"{filename}_{timestamp}.png"
        full_path = os.path.join(save_dir, full_filename)
        
        # Save the image
        cv2.imwrite(full_path, frame)
        info(f"Saved screenshot to {full_path}", LogCategory.CONTROL)
        
        return full_path
    
    def get_performance_stats(self) -> Dict[str, Union[float, int]]:
        """
        Get performance statistics for the screen capture.
        
        Returns:
            Dictionary with performance metrics
        """
        return self._stats.get_summary()
    
    def set_capture_rate(self, fps: int) -> None:
        """
        Update the maximum capture rate.
        
        Args:
            fps: Frames per second
        """
        self._min_frame_time_ms = 1000.0 / max(1, fps)
        info(f"Screen capture rate set to {fps} FPS (min frame time: {self._min_frame_time_ms:.2f}ms)",
             LogCategory.CONTROL)
    
    def set_cache_timeout(self, timeout_ms: int) -> None:
        """
        Update the cache timeout.
        
        Args:
            timeout_ms: Cache timeout in milliseconds
        """
        self._cache_timeout_ms = timeout_ms
        info(f"Screen capture cache timeout set to {timeout_ms}ms", LogCategory.CONTROL)


# Create a singleton instance
screen_capture = ScreenCapture()
