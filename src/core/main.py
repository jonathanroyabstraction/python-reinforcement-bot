#!/usr/bin/env python3
"""
Main entry point for the WoW Bot application.
"""
import argparse
import os
import sys
from typing import Any, Dict, List

from src.core.core_manager import CoreManager
from src.vision.models.detection_result import DetectionResults

# Simple imports that work reliably
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.vision.main import ScreenProcessor
from src.utils.config import ConfigManager, get_config, set_config
from src.utils.logging import configure as configure_logging, info, error, debug, warning, critical, exception, LogCategory

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Automation Tool - Deep learning based WoW bot")
    parser.add_argument(
        "--config", 
        type=str, 
        default="src/configs/config.yml",  # This is relative to project root
        help="Path to configuration file"
    )

    parser.add_argument(
        "--debug",
        action="store_true", 
        help="Enable debug mode"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="DEBUG",
        help="Set the logging level"
    )

    return parser.parse_args()

def init_config(config_path, debug_mode, log_level):
    """Initialize the configuration system."""
    # Create config manager with the specified config file
    config_manager = ConfigManager(config_path)
    
    # Override config with command line arguments if provided
    if debug_mode:
        set_config('system.debug_mode', True)
    
    if log_level:
        set_config('system.log_level', log_level)
    
    # Return the config manager
    return config_manager

def init_logging():
    """Initialize the logging system."""
    # Configure logging based on current configuration
    configure_logging()
    
    # Log startup information
    info("WoW Bot starting up", LogCategory.SYSTEM)
    info("Configuration loaded from {}".format(get_config('_config_path', 'default')), LogCategory.CONFIG)
    debug("Debug mode: {}".format(get_config('system.debug_mode', False)), LogCategory.SYSTEM)
    debug("Log level: {}".format(get_config('system.log_level', 'INFO')), LogCategory.SYSTEM)

def init_game_state():
    """Initialize the game state."""
    return

def detections_callback(detection_results: DetectionResults):
    """Callback function for detections."""

    debug(f"Detected {len(detection_results.detections)} objects from vision system", LogCategory.SYSTEM)
    return
    

def main():
    """Main function to start the WoW Bot."""
    args = parse_args()
    
    print("WoW Bot - Starting up...")
    print("Using configuration: {}".format(args.config))
    print("Debug mode:", "Enabled" if args.debug else "Disabled")
    
    try:
        # Initialize configuration
        config_manager = init_config(args.config, args.debug, args.log_level)
        
        # Initialize logging
        init_logging()

        # Initialize screen processor
        screen_processor = ScreenProcessor()
        core_manager = CoreManager()

        # Start processing screenshots
        screen_processor.start_processing(core_manager.process_vision_detections, interval=5, save_frames=False)
        
        info("WoW Bot is ready. Press Ctrl+C to exit.", LogCategory.SYSTEM)
        
            
    except KeyboardInterrupt:
        info("Received shutdown signal", LogCategory.SYSTEM)
    except Exception as e:
        exception("Unhandled exception: {}".format(e), LogCategory.ERROR)
    finally:
        # Clean up resources
        info("Shutting down WoW Bot", LogCategory.SYSTEM)

if __name__ == "__main__":
    main()
