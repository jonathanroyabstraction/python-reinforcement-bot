#!/usr/bin/env python3
"""
Main entry point for the WoW Bot application.
"""
import argparse
import os
import sys
import time
from pathlib import Path

# Simple imports that work reliably
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

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
        
        # TODO: Initialize modules
        # TODO: Start the bot
        
        info("WoW Bot is ready. Press Ctrl+C to exit.", LogCategory.SYSTEM)
        print("WoW Bot is ready. Press Ctrl+C to exit.")
        
        # Main loop will be implemented here
        while True:
            # Placeholder for main loop
            time.sleep(1)
            
    except KeyboardInterrupt:
        info("Received shutdown signal", LogCategory.SYSTEM)
        print("\nWoW Bot - Shutting down...")
    except Exception as e:
        exception("Unhandled exception: {}".format(e), LogCategory.ERROR)
        print("Error: {}".format(e))
    finally:
        # Clean up resources
        info("Shutting down WoW Bot", LogCategory.SYSTEM)
        print("WoW Bot - Shutdown complete.")

if __name__ == "__main__":
    main()
