"""Control module for the WoW Bot.

This module provides functionality for interacting with the game environment,
including screen capture, image processing, and input simulation.
"""

from .input_simulator import InputSimulator, InputMode, InputAction, InputEventType, input_simulator

__all__ = [
    'InputSimulator',
    'InputMode',
    'InputAction',
    'InputEventType',
    'input_simulator'
]