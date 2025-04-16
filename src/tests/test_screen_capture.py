#!/usr/bin/env python3
"""
Test script for screen capture and image processing functionality.

This script demonstrates the basic capabilities of the screen capture
and image processing modules by capturing the screen, applying various
image transformations, and saving the results.
"""
import time
import os
import argparse
from pathlib import Path

import cv2
import numpy as np

# Add the project root to path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the bot modules
from src.utils.config import ConfigManager, get_config
from src.utils.logging import BotLogger, LogCategory, info, debug, warning
from src.control.image_processing import image_processor


def init_bot():
    """Initialize the bot's configuration and logging."""
    # Initialize configuration
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(src_dir, 'configs/config.yml')
    config_manager = ConfigManager(config_path)
    
    # Initialize logging
    logger = BotLogger(config_path)
    
    # Log system info
    info("Screen Capture Test Started", LogCategory.SYSTEM)
    debug(f"Using configuration: {config_path}", LogCategory.SYSTEM)
    debug(f"Debug mode: {'Enabled' if get_config('system.debug_mode', False) else 'Disabled'}", 
          LogCategory.SYSTEM)


def test_screen_capture():
    """Test basic screen capture functionality."""
    info("Testing screen capture...", LogCategory.CONTROL)
    
    # Capture full screen
    start_time = time.time()
    full_screen = screen_capture.capture('full')
    elapsed = (time.time() - start_time) * 1000
    
    info(f"Full screen capture completed in {elapsed:.2f}ms", LogCategory.CONTROL)
    info(f"Screen dimensions: {full_screen.shape[1]}x{full_screen.shape[0]}", LogCategory.CONTROL)
    
    # Save the captured screen
    save_path = screen_capture.capture_to_file('full_screen')
    info(f"Saved full screen to {save_path}", LogCategory.CONTROL)
    
    # Test performance with multiple captures
    info("Testing capture performance (10 frames)...", LogCategory.CONTROL)
    start_time = time.time()
    for _ in range(10):
        screen_capture.capture('full')
    elapsed = (time.time() - start_time) * 1000
    info(f"10 captures completed in {elapsed:.2f}ms (avg: {elapsed/10:.2f}ms/frame)", 
         LogCategory.CONTROL)
    
    # Get performance stats
    stats = screen_capture.get_performance_stats()
    info(f"Capture statistics: {stats}", LogCategory.CONTROL)
    
    return full_screen


def test_image_processing(image):
    """Test image processing capabilities."""
    info("Testing image processing...", LogCategory.CONTROL)
    
    # Create output directory
    output_dir = os.path.join(get_config('system.data_dir', './data'), 'debug')
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Test resizing
    resized = image_processor.resize(image, width=800)
    cv2.imwrite(os.path.join(output_dir, f"resized_{timestamp}.png"), resized)
    info(f"Resized image to {resized.shape[1]}x{resized.shape[0]}", LogCategory.CONTROL)
    
    # Test grayscale conversion and normalization
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized = image_processor.normalize(gray)
    cv2.imwrite(os.path.join(output_dir, f"normalized_{timestamp}.png"), normalized)
    info("Applied normalization", LogCategory.CONTROL)
    
    # Test edge detection
    edges = image_processor.detect_edges(image)
    cv2.imwrite(os.path.join(output_dir, f"edges_{timestamp}.png"), edges)
    info("Applied edge detection", LogCategory.CONTROL)
    
    # Test color analysis
    dominant_colors = image_processor.get_dominant_colors(image, 5)
    info("Dominant colors:", LogCategory.CONTROL)
    for color, percentage in dominant_colors:
        info(f"  BGR {color}: {percentage*100:.2f}%", LogCategory.CONTROL)
    
    # Test text region extraction
    text_regions = image_processor.extract_text_regions(image)
    result_image = image_processor.draw_rects(image.copy(), text_regions)
    cv2.imwrite(os.path.join(output_dir, f"text_regions_{timestamp}.png"), result_image)
    info(f"Detected {len(text_regions)} potential text regions", LogCategory.CONTROL)
    
    return output_dir


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Test screen capture and image processing')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()
    
    # Initialize bot systems
    init_bot()
    
    try:
        # Test screen capture
        captured_image = test_screen_capture()
        
        # Test image processing
        output_dir = test_image_processing(captured_image)
        
        info(f"All tests completed successfully. Output saved to {output_dir}", LogCategory.SYSTEM)
        
    except Exception as e:
        warning(f"Test failed: {str(e)}", LogCategory.SYSTEM)
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
