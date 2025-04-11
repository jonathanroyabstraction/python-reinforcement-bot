"""
Target state module for tracking target information.
"""
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from src.utils.logging import LogCategory, debug, info
from .player_state import Position


class TargetType(Enum):
    """Types of targets in the game."""
    NONE = auto()
    FRIENDLY = auto()
    NEUTRAL = auto()
    HOSTILE = auto()
    QUEST = auto()
    BOSS = auto()


@dataclass
class TargetState:
    """
    Represents the state of the current target.
    
    This includes target health, type, position, and other relevant information.
    """
    # Basic information
    name: str = ""
    guid: str = ""  # Unique game ID for the target
    target_type: TargetType = TargetType.NONE
    level: int = 1
    is_elite: bool = False
    is_rare: bool = False
    
    # Health and status
    current_health: int = 0
    health_max: int = 100
    
    # Position and distance
    position: Position = field(default_factory=Position)
    distance: float = 0.0  # Distance to player
    
    # Combat information
    is_casting: bool = False
    cast_name: str = ""
    cast_time_remaining: float = 0.0
    
    # Buffs and debuffs applied to the target
    buffs: Dict[str, float] = field(default_factory=dict)  # name -> expiration time
    debuffs: Dict[str, float] = field(default_factory=dict)  # name -> expiration time
    
    # Last update timestamp
    last_updated: float = field(default_factory=time.time)
    
    def health_percent(self) -> float:
        """Get target health as a percentage."""
        if self.health_max == 0:
            return 0.0
        return (self.current_health / self.health_max) * 100.0
    
    def is_valid(self) -> bool:
        """Check if the target is valid (has a name and health)."""
        return bool(self.name and self.health_max > 0)
    
    def is_in_range(self, range_yards: float) -> bool:
        """Check if the target is within specified range."""
        return self.distance <= range_yards
    
    def update_health(self, current: int, maximum: int = None) -> None:
        """Update target health values."""
        self.current_health = current
        if maximum is not None:
            self.health_max = maximum
        self.last_updated = time.time()
        
        debug(f"Target health: {self.current_health}/{self.health_max} ({self.health_percent():.1f}%)",
              LogCategory.COMBAT)
    
    def update_position(self, new_position: Position, player_position: Position) -> None:
        """Update the target's position and calculate distance to player."""
        self.position = new_position
        self.distance = self.position.distance_to(player_position)
        self.last_updated = time.time()
        
        debug(f"Target position updated: ({new_position.x}, {new_position.y}, {new_position.z})", 
              LogCategory.SYSTEM)
        debug(f"Distance to target: {self.distance:.1f} yards", LogCategory.SYSTEM)
    
    def update_cast_info(self, is_casting: bool, name: str = "", time_remaining: float = 0.0) -> None:
        """Update information about what the target is casting."""
        self.is_casting = is_casting
        self.cast_name = name if is_casting else ""
        self.cast_time_remaining = time_remaining
        self.last_updated = time.time()
        
        if is_casting:
            debug(f"Target casting: {name} ({time_remaining:.1f}s remaining)", LogCategory.COMBAT)
        elif self.cast_name:  # Was casting but stopped
            debug(f"Target stopped casting {self.cast_name}", LogCategory.COMBAT)
    
    def add_buff(self, name: str, duration: float) -> None:
        """Add or refresh a buff on the target."""
        expiration = time.time() + duration
        self.buffs[name] = expiration
        self.last_updated = time.time()
        
        debug(f"Target gained buff: {name} ({duration:.1f}s)", LogCategory.COMBAT)
    
    def add_debuff(self, name: str, duration: float) -> None:
        """Add or refresh a debuff on the target."""
        expiration = time.time() + duration
        self.debuffs[name] = expiration
        self.last_updated = time.time()
        
        debug(f"Target gained debuff: {name} ({duration:.1f}s)", LogCategory.COMBAT)
    
    def update_buffs(self) -> None:
        """Update buff durations and remove expired buffs."""
        current_time = time.time()
        expired = []
        
        for name, expiration in self.buffs.items():
            if current_time > expiration:
                expired.append(name)
                debug(f"Target buff expired: {name}", LogCategory.COMBAT)
        
        for name in expired:
            self.buffs.pop(name, None)
            
        if expired:
            self.last_updated = current_time
    
    def update_debuffs(self) -> None:
        """Update debuff durations and remove expired debuffs."""
        current_time = time.time()
        expired = []
        
        for name, expiration in self.debuffs.items():
            if current_time > expiration:
                expired.append(name)
                debug(f"Target debuff expired: {name}", LogCategory.COMBAT)
        
        for name in expired:
            self.debuffs.pop(name, None)
            
        if expired:
            self.last_updated = current_time
    
    def has_buff(self, name: str) -> bool:
        """Check if target has a specific buff."""
        self.update_buffs()  # Clean up expired buffs first
        return name in self.buffs
    
    def has_debuff(self, name: str) -> bool:
        """Check if target has a specific debuff."""
        self.update_debuffs()  # Clean up expired debuffs first
        return name in self.debuffs
    
    def to_dict(self) -> Dict:
        """Convert target state to a dictionary for serialization."""
        return {
            "name": self.name,
            "guid": self.guid,
            "target_type": self.target_type.name,
            "level": self.level,
            "is_elite": self.is_elite,
            "is_rare": self.is_rare,
            "current_health": self.current_health,
            "health_max": self.health_max,
            "position": {
                "x": self.position.x,
                "y": self.position.y,
                "z": self.position.z
            },
            "distance": self.distance,
            "is_casting": self.is_casting,
            "cast_name": self.cast_name,
            "cast_time_remaining": self.cast_time_remaining,
            "buffs": self.buffs,
            "debuffs": self.debuffs,
            "last_updated": self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TargetState':
        """Create a TargetState object from a dictionary."""
        target = cls(
            name=data.get("name", ""),
            guid=data.get("guid", ""),
            target_type=TargetType[data.get("target_type", "NONE")],
            level=data.get("level", 1),
            is_elite=data.get("is_elite", False),
            is_rare=data.get("is_rare", False),
            current_health=data.get("current_health", 0),
            health_max=data.get("health_max", 100),
            distance=data.get("distance", 0.0),
            is_casting=data.get("is_casting", False),
            cast_name=data.get("cast_name", ""),
            cast_time_remaining=data.get("cast_time_remaining", 0.0)
        )
        
        # Set position
        pos_data = data.get("position", {})
        target.position = Position(
            x=pos_data.get("x", 0.0),
            y=pos_data.get("y", 0.0),
            z=pos_data.get("z", 0.0)
        )
        
        # Set buffs and debuffs
        target.buffs = data.get("buffs", {})
        target.debuffs = data.get("debuffs", {})
        
        # Set last updated
        target.last_updated = data.get("last_updated", time.time())
        
        return target
