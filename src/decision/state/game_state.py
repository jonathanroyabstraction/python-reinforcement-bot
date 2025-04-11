"""
Game state module for tracking the overall state of the game.
"""
import json
import os
import threading
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Callable, Any

from src.utils.logging import LogCategory, debug, info, warning, error
from .player_state import PlayerState, Position
from .target_state import TargetState
from .ability_state import AbilityState
from .environment_state import EnvironmentState


class GameStateChangeType(Enum):
    """Types of game state changes for notifications."""
    PLAYER_HEALTH = auto()
    PLAYER_RESOURCE = auto()
    PLAYER_POSITION = auto()
    PLAYER_COMBAT = auto()
    PLAYER_BUFF = auto()
    PLAYER_DEBUFF = auto()
    TARGET_CHANGED = auto()
    TARGET_HEALTH = auto()
    TARGET_POSITION = auto()
    TARGET_CAST = auto()
    ABILITY_COOLDOWN = auto()
    ABILITY_RESOURCE = auto()
    ENVIRONMENT_ENTITY = auto()
    ENVIRONMENT_ZONE = auto()
    FULL_UPDATE = auto()


@dataclass
class GameStateSnapshot:
    """
    Immutable snapshot of game state at a specific point in time.
    Used for state comparison and history.
    """
    timestamp: float
    player: PlayerState
    target: TargetState
    abilities: AbilityState
    environment: EnvironmentState
    
    @classmethod
    def from_game_state(cls, game_state: 'GameState') -> 'GameStateSnapshot':
        """Create a snapshot from the current game state."""
        return cls(
            timestamp=time.time(),
            player=deepcopy(game_state.player),
            target=deepcopy(game_state.target),
            abilities=deepcopy(game_state.abilities),
            environment=deepcopy(game_state.environment)
        )
    
    def to_dict(self) -> Dict:
        """Convert snapshot to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "player": self.player.to_dict(),
            "target": self.target.to_dict(),
            "abilities": self.abilities.to_dict(),
            "environment": self.environment.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GameStateSnapshot':
        """Create a snapshot from a dictionary."""
        return cls(
            timestamp=data.get("timestamp", time.time()),
            player=PlayerState.from_dict(data.get("player", {})),
            target=TargetState.from_dict(data.get("target", {})),
            abilities=AbilityState.from_dict(data.get("abilities", {})),
            environment=EnvironmentState.from_dict(data.get("environment", {}))
        )


class GameStateObserver(ABC):
    """
    Abstract base class for game state observers.
    
    Observers can register for specific state change events and receive
    notifications when those events occur.
    """
    @abstractmethod
    def on_state_changed(self, change_type: GameStateChangeType, 
                        old_state: Optional[Any] = None,
                        new_state: Optional[Any] = None) -> None:
        """
        Called when an observed state change occurs.
        
        Args:
            change_type: Type of state change that occurred
            old_state: Previous state value (if applicable)
            new_state: New state value (if applicable)
        """
        pass


class GameState:
    """
    Main game state tracking class.
    
    This class integrates all the state components (player, target, abilities,
    environment) and provides methods for state management, comparison,
    and persistence.
    """
    def __init__(self, history_size: int = 10, state_dir: str = None):
        """
        Initialize the game state.
        
        Args:
            history_size: Number of historical states to keep
            state_dir: Directory to store persisted state files
        """
        # Initialize state components
        self.player = PlayerState()
        self.target = TargetState()
        self.abilities = AbilityState()
        self.environment = EnvironmentState()
        
        # History tracking
        self.history_size = history_size
        self.history: List[GameStateSnapshot] = []
        
        # Observer pattern implementation
        self._observers: Dict[GameStateChangeType, List[GameStateObserver]] = {
            change_type: [] for change_type in GameStateChangeType
        }
        
        # State persistence
        self.state_dir = state_dir or os.path.join(os.getcwd(), 'state')
        os.makedirs(self.state_dir, exist_ok=True)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Last update timestamp
        self.last_updated = time.time()
        
        info("Game state initialized", LogCategory.SYSTEM)
    
    def register_observer(self, observer: GameStateObserver, 
                         change_types: List[GameStateChangeType] = None) -> None:
        """
        Register an observer for state change notifications.
        
        Args:
            observer: The observer to register
            change_types: List of change types to observe, or None for all changes
        """
        with self._lock:
            if change_types is None:
                # Register for all change types
                change_types = list(GameStateChangeType)
                
            for change_type in change_types:
                if observer not in self._observers[change_type]:
                    self._observers[change_type].append(observer)
                    
        debug(f"Observer registered for {len(change_types)} change types", LogCategory.SYSTEM)
    
    def unregister_observer(self, observer: GameStateObserver,
                           change_types: List[GameStateChangeType] = None) -> None:
        """
        Unregister an observer for state change notifications.
        
        Args:
            observer: The observer to unregister
            change_types: List of change types to stop observing, or None for all
        """
        with self._lock:
            if change_types is None:
                # Unregister from all change types
                change_types = list(GameStateChangeType)
                
            for change_type in change_types:
                if observer in self._observers[change_type]:
                    self._observers[change_type].remove(observer)
                    
        debug(f"Observer unregistered from {len(change_types)} change types", LogCategory.SYSTEM)
    
    def notify_observers(self, change_type: GameStateChangeType,
                        old_state: Optional[Any] = None,
                        new_state: Optional[Any] = None) -> None:
        """
        Notify observers of a state change.
        
        Args:
            change_type: Type of state change that occurred
            old_state: Previous state value (if applicable)
            new_state: New state value (if applicable)
        """
        # Create a copy of observers to avoid modification during iteration
        observers = []
        with self._lock:
            observers = list(self._observers[change_type])
            
        for observer in observers:
            try:
                observer.on_state_changed(change_type, old_state, new_state)
            except Exception as e:
                error(f"Error in observer notification: {str(e)}", LogCategory.SYSTEM)
    
    def add_to_history(self) -> None:
        """Add the current state to history."""
        with self._lock:
            # Create a snapshot and add to history
            snapshot = GameStateSnapshot.from_game_state(self)
            self.history.append(snapshot)
            
            # Trim history if needed
            while len(self.history) > self.history_size:
                self.history.pop(0)
                
        debug(f"Added state to history (size: {len(self.history)})", LogCategory.SYSTEM)
    
    def get_historical_state(self, index: int = -1) -> Optional[GameStateSnapshot]:
        """
        Get a historical state snapshot.
        
        Args:
            index: Index in history (-1 for most recent)
            
        Returns:
            Historical state snapshot or None if index is out of bounds
        """
        with self._lock:
            if not self.history or index >= len(self.history) or abs(index) > len(self.history):
                return None
            return self.history[index]
    
    def compare_with_history(self, index: int = -1) -> Dict[str, Any]:
        """
        Compare current state with a historical state.
        
        Args:
            index: Index in history (-1 for most recent)
            
        Returns:
            Dictionary of differences between states
        """
        historical = self.get_historical_state(index)
        if not historical:
            return {"error": "No historical state available for comparison"}
            
        current = GameStateSnapshot.from_game_state(self)
        
        differences = {
            "timestamp_diff": current.timestamp - historical.timestamp,
            "player": {},
            "target": {},
            "abilities": {},
            "environment": {}
        }
        
        # Compare player state
        player_diff = {}
        for attr in ["current_health", "health_max", "current_resource", "mana_max", 
                    "position", "in_combat", "is_mounted", "is_dead"]:
            if getattr(current.player, attr) != getattr(historical.player, attr):
                if attr == "position":
                    player_diff["position"] = {
                        "from": {
                            "x": historical.player.position.x,
                            "y": historical.player.position.y,
                            "z": historical.player.position.z
                        },
                        "to": {
                            "x": current.player.position.x,
                            "y": current.player.position.y,
                            "z": current.player.position.z
                        }
                    }
                else:
                    player_diff[attr] = {
                        "from": getattr(historical.player, attr),
                        "to": getattr(current.player, attr)
                    }
                    
        if player_diff:
            differences["player"] = player_diff
        
        # Compare target state
        if historical.target.name != current.target.name:
            differences["target"]["changed"] = {
                "from": historical.target.name,
                "to": current.target.name
            }
        else:
            target_diff = {}
            for attr in ["current_health", "health_max", "distance", "is_casting"]:
                if getattr(current.target, attr) != getattr(historical.target, attr):
                    target_diff[attr] = {
                        "from": getattr(historical.target, attr),
                        "to": getattr(current.target, attr)
                    }
            if target_diff:
                differences["target"] = target_diff
        
        # For abilities and environment, just note if there were changes
        if current.abilities.last_updated != historical.abilities.last_updated:
            differences["abilities"]["changed"] = True
            
        if current.environment.last_updated != historical.environment.last_updated:
            differences["environment"]["changed"] = True
            
        return differences
    
    def save_state(self, filename: str = None) -> str:
        """
        Save the current state to a file.
        
        Args:
            filename: Name of file to save to, or None for automatic name
            
        Returns:
            Path to the saved state file
        """
        if filename is None:
            # Generate a filename based on timestamp
            timestamp = int(time.time())
            filename = f"game_state_{timestamp}.json"
            
        filepath = os.path.join(self.state_dir, filename)
        
        snapshot = GameStateSnapshot.from_game_state(self)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(snapshot.to_dict(), f, indent=2)
                
            info(f"Game state saved to {filepath}", LogCategory.SYSTEM)
            return filepath
        except Exception as e:
            error(f"Failed to save game state: {str(e)}", LogCategory.SYSTEM)
            return ""
    
    def load_state(self, filename: str) -> bool:
        """
        Load game state from a file.
        
        Args:
            filename: Name of file to load from
            
        Returns:
            True if state was loaded successfully, False otherwise
        """
        filepath = os.path.join(self.state_dir, filename)
        
        if not os.path.exists(filepath):
            warning(f"State file {filepath} does not exist", LogCategory.SYSTEM)
            return False
            
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            snapshot = GameStateSnapshot.from_dict(data)
            
            with self._lock:
                self.player = snapshot.player
                self.target = snapshot.target
                self.abilities = snapshot.abilities
                self.environment = snapshot.environment
                self.last_updated = time.time()
                
            # Notify observers about full update
            self.notify_observers(GameStateChangeType.FULL_UPDATE)
            
            info(f"Game state loaded from {filepath}", LogCategory.SYSTEM)
            return True
        except Exception as e:
            error(f"Failed to load game state: {str(e)}", LogCategory.SYSTEM)
            return False
    
    def validate_state(self) -> List[str]:
        """
        Validate the current state and return any issues found.
        
        Returns:
            List of validation issues (empty if state is valid)
        """
        issues = []
        
        # Check player state
        if self.player.current_health < 0 or self.player.current_health > 100:
            issues.append("Invalid player current health")
        if self.player.current_resource < 0 or self.player.current_resource > 100:
            issues.append("Invalid player current mana")
            
        # Check target state if valid
        if self.target.name:
            if self.target.health_max <= 0:
                issues.append("Invalid target health maximum")
            if self.target.current_health < 0 or self.target.current_health > self.target.health_max:
                issues.append("Invalid target current health")
                
        # Add more validation as needed
            
        return issues
    
    # Player state update methods
    def update_player_health(self, current: int) -> None:
        """Update player health values."""
        old_health = self.player.current_health
        
        with self._lock:
            self.player.update_health(current)
            self.last_updated = time.time()
            
        new_health = self.player.current_health
        self.notify_observers(GameStateChangeType.PLAYER_HEALTH, old_health, new_health)
    
    def update_player_resource(self, current: int) -> None:
        """Update player resource values."""
        old_mana = self.player.current_resource
        
        with self._lock:
            self.player.update_resource(current)
            self.last_updated = time.time()
            
        new_resource = self.player.current_resource
        self.notify_observers(GameStateChangeType.PLAYER_RESOURCE, old_mana, new_mana)
    
    def update_player_position(self, x: float, y: float, z: float = None) -> None:
        """Update the player's position."""
        old_position = Position(
            self.player.position.x, 
            self.player.position.y,
            self.player.position.z
        )
        
        with self._lock:
            self.player.update_position(x, y, z)
            self.last_updated = time.time()
            
        new_position = self.player.position
        self.notify_observers(GameStateChangeType.PLAYER_POSITION, old_position, new_position)
    
    def update_player_combat_status(self, in_combat: bool) -> None:
        """Update the player's combat status."""
        old_status = self.player.in_combat
        
        with self._lock:
            self.player.update_combat_status(in_combat)
            self.last_updated = time.time()
            
        self.notify_observers(GameStateChangeType.PLAYER_COMBAT, old_status, in_combat)
    
    def add_player_buff(self, name: str, duration: float) -> None:
        """Add or refresh a buff on the player."""
        with self._lock:
            self.player.add_buff(name, duration)
            self.last_updated = time.time()
            
        self.notify_observers(GameStateChangeType.PLAYER_BUFF, None, (name, duration))
    
    def add_player_debuff(self, name: str, duration: float) -> None:
        """Add or refresh a debuff on the player."""
        with self._lock:
            self.player.add_debuff(name, duration)
            self.last_updated = time.time()
            
        self.notify_observers(GameStateChangeType.PLAYER_DEBUFF, None, (name, duration))
    
    # Target state update methods
    def set_target(self, target: TargetState) -> None:
        """Set a new target."""
        old_target = self.target
        
        with self._lock:
            self.target = target
            self.last_updated = time.time()
            
        self.notify_observers(GameStateChangeType.TARGET_CHANGED, old_target, target)
    
    def clear_target(self) -> None:
        """Clear the current target."""
        old_target = self.target
        
        with self._lock:
            self.target = TargetState()
            self.last_updated = time.time()
            
        self.notify_observers(GameStateChangeType.TARGET_CHANGED, old_target, self.target)
    
    def update_target_health(self, current: int, maximum: int = None) -> None:
        """Update target health values."""
        if not self.target.name:
            return
            
        old_health = (self.target.current_health, self.target.health_max)
        
        with self._lock:
            self.target.update_health(current, maximum)
            self.last_updated = time.time()
            
        new_health = (self.target.current_health, self.target.health_max)
        self.notify_observers(GameStateChangeType.TARGET_HEALTH, old_health, new_health)
    
    def update_target_position(self, position: Position) -> None:
        """Update the target's position."""
        if not self.target.name:
            return
            
        old_position = Position(
            self.target.position.x,
            self.target.position.y,
            self.target.position.z
        )
        
        with self._lock:
            self.target.update_position(position, self.player.position)
            self.last_updated = time.time()
            
        self.notify_observers(GameStateChangeType.TARGET_POSITION, old_position, position)
    
    def update_target_cast(self, is_casting: bool, name: str = "", time_remaining: float = 0.0) -> None:
        """Update target casting information."""
        if not self.target.name:
            return
            
        old_cast = (self.target.is_casting, self.target.cast_name, self.target.cast_time_remaining)
        
        with self._lock:
            self.target.update_cast_info(is_casting, name, time_remaining)
            self.last_updated = time.time()
            
        new_cast = (is_casting, name, time_remaining)
        self.notify_observers(GameStateChangeType.TARGET_CAST, old_cast, new_cast)
    
    # Ability state update methods
    def update_ability_cooldown(self, ability_name: str, duration: float) -> None:
        """Start or refresh a cooldown for an ability."""
        with self._lock:
            self.abilities.update_cooldown(ability_name, duration)
            self.last_updated = time.time()
            
        self.notify_observers(GameStateChangeType.ABILITY_COOLDOWN, None, (ability_name, duration))
    
    def update_resource(self, resource_type, current: int, maximum: int = None) -> None:
        """Update a resource value."""
        old_value = (self.abilities.get_resource(resource_type),
                   self.abilities.get_resource_max(resource_type))
        
        with self._lock:
            self.abilities.update_resource(resource_type, current, maximum)
            self.last_updated = time.time()
            
        new_value = (current, maximum if maximum is not None else 
                   self.abilities.get_resource_max(resource_type))
        self.notify_observers(GameStateChangeType.ABILITY_RESOURCE, old_value, new_value)
    
    # Environment state update methods
    def add_entity(self, entity) -> None:
        """Add or update an entity in the environment."""
        with self._lock:
            self.environment.add_entity(entity)
            self.last_updated = time.time()
            
        self.notify_observers(GameStateChangeType.ENVIRONMENT_ENTITY, None, entity)
    
    def update_zone_info(self, zone: str, subzone: str = None, 
                       is_instance: bool = None, instance_name: str = None) -> None:
        """Update zone information."""
        old_zone = (self.environment.zone_name, self.environment.subzone_name,
                  self.environment.is_instance, self.environment.instance_name)
        
        with self._lock:
            self.environment.update_zone_info(zone, subzone, is_instance, instance_name)
            self.last_updated = time.time()
            
        new_zone = (zone, subzone if subzone is not None else self.environment.subzone_name,
                  is_instance if is_instance is not None else self.environment.is_instance,
                  instance_name if instance_name is not None else self.environment.instance_name)
        self.notify_observers(GameStateChangeType.ENVIRONMENT_ZONE, old_zone, new_zone)
    
    # Screen capture integration
    def update_from_screen_capture(self, screen_data: Dict) -> None:
        """
        Update game state based on information extracted from screen capture.
        
        Args:
            screen_data: Dictionary containing parsed information from screen
        """
        with self._lock:
            # Process player information
            player_data = screen_data.get("player", {})
            if "health" in player_data:
                self.update_player_health(
                    player_data["health"].get("current", self.player.current_health)                
                    )
                
            if "mana" in player_data:
                self.update_player_mana(
                    player_data["mana"].get("current", self.player.current_resource)
                )
                
            if "position" in player_data:
                pos = player_data["position"]
                self.update_player_position(
                    pos.get("x", self.player.position.x),
                    pos.get("y", self.player.position.y),
                    pos.get("z", self.player.position.z)
                )
                
            if "combat" in player_data:
                self.update_player_combat_status(player_data["combat"])
                
            # Process target information
            target_data = screen_data.get("target", {})
            if target_data.get("name"):
                # Create or update target
                new_target = TargetState(
                    name=target_data.get("name", ""),
                    guid=target_data.get("guid", ""),
                    target_type=target_data.get("type", self.target.target_type),
                    level=target_data.get("level", 1),
                    is_elite=target_data.get("is_elite", False),
                    is_rare=target_data.get("is_rare", False)
                )
                
                if "health" in target_data:
                    new_target.current_health = target_data["health"].get("current", 0)
                    new_target.health_max = target_data["health"].get("max", 100)
                    
                if "position" in target_data:
                    pos = target_data["position"]
                    new_target.position = Position(
                        x=pos.get("x", 0.0),
                        y=pos.get("y", 0.0),
                        z=pos.get("z", 0.0)
                    )
                    new_target.distance = new_target.position.distance_to(self.player.position)
                    
                if "casting" in target_data:
                    cast_info = target_data["casting"]
                    new_target.is_casting = cast_info.get("is_casting", False)
                    new_target.cast_name = cast_info.get("name", "")
                    new_target.cast_time_remaining = cast_info.get("time_remaining", 0.0)
                    
                self.set_target(new_target)
            elif self.target.name and not target_data.get("exists", True):
                # Clear target if it no longer exists
                self.clear_target()
                
            # Process ability information
            ability_data = screen_data.get("abilities", {})
            if "cooldowns" in ability_data:
                for ability_name, cooldown in ability_data["cooldowns"].items():
                    self.update_ability_cooldown(ability_name, cooldown)
                    
            if "resources" in ability_data:
                for resource_name, data in ability_data["resources"].items():
                    try:
                        # Use Resource directly instead of AbilityState.Resource
                        from src.decision.state.ability_state import Resource
                        resource_type = getattr(Resource, resource_name.upper())
                        self.update_resource(
                            resource_type, 
                            data.get("current", 0),
                            data.get("max", None)
                        )
                    except (AttributeError, ValueError):
                        warning(f"Unknown resource type: {resource_name}", LogCategory.SYSTEM)
                        
            # Process environment information
            env_data = screen_data.get("environment", {})
            if "entities" in env_data:
                for entity_data in env_data["entities"]:
                    if "id" in entity_data and "name" in entity_data:
                        # Create position for entity
                        pos_data = entity_data.get("position", {})
                        position = Position(
                            x=pos_data.get("x", 0.0),
                            y=pos_data.get("y", 0.0),
                            z=pos_data.get("z", 0.0)
                        )
                        
                        from src.decision.state.environment_state import Entity
                        entity = Entity(
                            id=entity_data["id"],
                            name=entity_data["name"],
                            entity_type=entity_data.get("type", "unknown"),
                            position=position,
                            is_hostile=entity_data.get("is_hostile", False),
                            is_friendly=entity_data.get("is_friendly", False),
                            is_neutral=entity_data.get("is_neutral", True),
                            is_elite=entity_data.get("is_elite", False),
                            level=entity_data.get("level", 1),
                            health_percent=entity_data.get("health_percent", 100.0)
                        )
                        
                        entity.distance = entity.position.distance_to(self.player.position)
                        self.add_entity(entity)
                        
            if "zone" in env_data:
                zone_data = env_data["zone"]
                self.update_zone_info(
                    zone=zone_data.get("name", ""),
                    subzone=zone_data.get("subzone", None),
                    is_instance=zone_data.get("is_instance", None),
                    instance_name=zone_data.get("instance_name", None)
                )
                
            # After updating, add to history
            self.add_to_history()
            
            info("Game state updated from screen capture", LogCategory.SYSTEM)
    
    def to_dict(self) -> Dict:
        """Convert game state to a dictionary for serialization."""
        with self._lock:
            return {
                "player": self.player.to_dict(),
                "target": self.target.to_dict(),
                "abilities": self.abilities.to_dict(),
                "environment": self.environment.to_dict(),
                "last_updated": self.last_updated
            }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GameState':
        """Create a GameState object from a dictionary."""
        game_state = cls()
        
        game_state.player = PlayerState.from_dict(data.get("player", {}))
        game_state.target = TargetState.from_dict(data.get("target", {}))
        game_state.abilities = AbilityState.from_dict(data.get("abilities", {}))
        game_state.environment = EnvironmentState.from_dict(data.get("environment", {}))
        game_state.last_updated = data.get("last_updated", time.time())
        
        return game_state
