"""
Game state representation module for the WoW Bot.

This module provides classes and utilities for tracking, updating, and persisting
the current state of the game, including player, target, and environment information.
"""

from .game_state import GameState, GameStateObserver, GameStateSnapshot
from .player_state import PlayerState, PlayerClass, PlayerSpec
from .target_state import TargetState, TargetType
from .ability_state import AbilityState, Cooldown, Resource
from .environment_state import EnvironmentState, Entity, Obstacle

__all__ = [
    'GameState',
    'GameStateObserver',
    'GameStateSnapshot',
    'PlayerState',
    'PlayerClass',
    'PlayerSpec',
    'TargetState',
    'TargetType',
    'AbilityState',
    'Cooldown',
    'Resource',
    'EnvironmentState',
    'Entity',
    'Obstacle'
]
