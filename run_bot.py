#!/usr/bin/env python3
"""
Launcher script for the WoW Bot.
Running from the project root ensures reliable imports.
"""
import sys
import os
from pathlib import Path

# No path manipulation needed when running from project root
from src.utils.config import ConfigManager, get_config, set_config
from src.utils.logging import configure as configure_logging, info, warning, error, LogCategory

def main():
    """
    Main entry point for the WoW Bot.
    Initializes configuration and logging, then starts the bot.
    """
    print("WoW Bot - Starting up...")
    
    try:
        # Initialize configuration with default path
        config_path = "configs/config.yml"
        print(f"Using configuration: {config_path}")
        
        config_manager = ConfigManager(config_path)
        
        # Initialize logging
        configure_logging()
        
        # Log startup information
        info("WoW Bot starting up", LogCategory.SYSTEM)
        info(f"Configuration loaded", LogCategory.CONFIG)
        
        print("WoW Bot is ready. Press Ctrl+C to exit.")
        info("WoW Bot is ready. Press Ctrl+C to exit.", LogCategory.SYSTEM)
        
        # Main loop placeholder
        while True:
            import time
            time.sleep(1)
            
    except KeyboardInterrupt:
        info("Received shutdown signal", LogCategory.SYSTEM)
        print("\nWoW Bot - Shutting down...")
    except Exception as e:
        error(f"Unhandled exception: {e}", LogCategory.ERROR)
        print(f"Error: {e}")
    finally:
        # Clean up resources
        info("Shutting down WoW Bot", LogCategory.SYSTEM)
        print("WoW Bot - Shutdown complete.")

if __name__ == "__main__":
    main()
