#!/usr/bin/env python3
"""
Test script for the input simulation system.

This script demonstrates the capabilities of the input simulator
by performing various keyboard and mouse actions in a controlled manner.
"""
import time
import os
import argparse
from pathlib import Path

# Add the project root to path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the bot modules
from src.utils.config import ConfigManager, get_config
from src.utils.logging import BotLogger, LogCategory, info, debug, warning
from src.control.input_simulator import InputSimulator, InputMode, input_simulator


def init_bot():
    """Initialize the bot's configuration and logging."""
    # Initialize configuration
    # Get the absolute path to the config file from the test location
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(src_dir, 'configs/config.yml')
    config_manager = ConfigManager(config_path)
    
    # Initialize logging
    logger = BotLogger(config_path)
    
    # Log system info
    info("Input Simulator Test Started", LogCategory.SYSTEM)
    debug(f"Using configuration: {config_path}", LogCategory.SYSTEM)
    debug(f"Debug mode: {'Enabled' if get_config('system.debug_mode', False) else 'Disabled'}", 
          LogCategory.SYSTEM)


def test_keyboard_input(mode=InputMode.TESTING):
    """Test keyboard input functionality."""
    info("Testing keyboard input...", LogCategory.CONTROL)
    
    # Create a new simulator in testing mode
    sim = InputSimulator(mode=mode)
    sim.start()
    
    try:
        # Test single key press
        info("Queueing key press actions...", LogCategory.CONTROL)
        sim.key_tap('a')
        time.sleep(0.5)
        
        # Test key combination
        sim.key_combination(['ctrl', 'c'])
        time.sleep(0.5)
        
        # Test typing
        sim.type_text("Hello World!")
        time.sleep(2)
        
        # Test game-specific actions
        sim.press_ability_key(1)
        time.sleep(0.5)
        
        sim.cast_spell_by_name("Fireball")
        time.sleep(1)
        
        # Wait for completion
        info("Waiting for all actions to complete...", LogCategory.CONTROL)
        time.sleep(3)
        
    finally:
        sim.stop()
    
    info("Keyboard input test completed", LogCategory.CONTROL)


def test_mouse_input(mode=InputMode.TESTING):
    """Test mouse input functionality."""
    info("Testing mouse input...", LogCategory.CONTROL)
    
    # Create a new simulator in testing mode
    sim = InputSimulator(mode=mode)
    sim.start()
    
    try:
        # Test mouse movement
        info("Testing mouse movement...", LogCategory.CONTROL)
        # Move to center of screen
        screen_width = get_config('screen_capture.resolution.width', 1920)
        screen_height = get_config('screen_capture.resolution.height', 1080)
        
        sim.mouse_move(screen_width // 2, screen_height // 2, duration=0.5)
        time.sleep(1)
        
        # Test relative movement
        sim.mouse_move_relative(100, 0)
        time.sleep(0.5)
        
        # Test clicking
        sim.mouse_click()
        time.sleep(0.5)
        
        # Test click at position
        sim.mouse_click_at(screen_width // 4, screen_height // 4)
        time.sleep(0.5)
        
        # Test scrolling
        sim.mouse_scroll(0, -5)  # Scroll down
        time.sleep(0.5)
        
        # Test game-specific mouse actions
        sim.camera_rotation(50)
        time.sleep(0.5)
        
        # Wait for completion
        info("Waiting for all actions to complete...", LogCategory.CONTROL)
        time.sleep(3)
        
    finally:
        sim.stop()
    
    info("Mouse input test completed", LogCategory.CONTROL)


def test_complex_sequence(mode=InputMode.TESTING):
    """Test a complex sequence of inputs."""
    info("Testing complex input sequence...", LogCategory.CONTROL)
    
    # Create a new simulator in testing mode
    sim = InputSimulator(mode=mode)
    sim.start()
    
    try:
        # Simulate logging into the game
        info("Simulating game login sequence...", LogCategory.CONTROL)
        
        # Press enter to get to the login screen
        sim.key_tap('enter')
        time.sleep(1)
        
        # Type character name (just a test)
        sim.type_text("MyCharacter")
        time.sleep(0.5)
        
        # Press enter to login
        sim.key_tap('enter')
        time.sleep(2)
        
        # Open character menu
        sim.open_game_menu('character')
        time.sleep(1)
        
        # Close menu with escape
        sim.key_tap('escape')
        time.sleep(0.5)
        
        # Move forward a bit
        sim.use_movement_key('forward', 2.0)
        time.sleep(2.5)
        
        # Jump
        sim.jump()
        time.sleep(0.5)
        
        # Target an enemy
        sim.target_by_tab()
        time.sleep(0.5)
        
        # Cast a spell
        sim.press_ability_key(2)
        time.sleep(0.5)
        
        # Wait for completion
        info("Waiting for all actions to complete...", LogCategory.CONTROL)
        time.sleep(5)
        
    finally:
        sim.stop()
    
    info("Complex sequence test completed", LogCategory.CONTROL)


def test_emergency_stop():
    """Test the emergency stop functionality."""
    info("Testing emergency stop...", LogCategory.CONTROL)
    
    # Create a new simulator in testing mode
    sim = InputSimulator(mode=InputMode.TESTING)
    sim.start()
    
    try:
        # Queue a lot of actions
        for i in range(10):
            sim.key_tap(str(i))
            sim.delay(0.5)
        
        # Simulate emergency stop
        info("Triggering emergency stop...", LogCategory.CONTROL)
        sim.emergency_stop()
        
        # Check if queue is cleared
        time.sleep(1)
        
    finally:
        sim.stop()
    
    info("Emergency stop test completed", LogCategory.CONTROL)


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Test input simulation')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--real', action='store_true', 
                      help='WARNING: Execute actual inputs instead of just testing mode')
    args = parser.parse_args()
    
    # Initialize bot systems
    init_bot()
    
    # Determine mode
    mode = InputMode.NORMAL if args.real else InputMode.TESTING
    
    if mode == InputMode.NORMAL:
        warning("REAL INPUT MODE ENABLED - actual inputs will be sent to your system!", 
                LogCategory.SYSTEM)
        warning("You have 5 seconds to cancel (Ctrl+C) if this was not intended...", 
                LogCategory.SYSTEM)
        time.sleep(5)
    
    try:
        # Run tests
        test_keyboard_input(mode)
        test_mouse_input(mode)
        test_complex_sequence(mode)
        test_emergency_stop()
        
        info("All tests completed successfully", LogCategory.SYSTEM)
        
    except KeyboardInterrupt:
        warning("Tests aborted by user", LogCategory.SYSTEM)
        # Make sure to stop all simulators
        input_simulator.emergency_stop()
        return 1
    except Exception as e:
        warning(f"Test failed: {str(e)}", LogCategory.SYSTEM)
        # Make sure to stop all simulators
        input_simulator.emergency_stop()
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
