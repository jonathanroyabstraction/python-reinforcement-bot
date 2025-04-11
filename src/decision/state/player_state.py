"""
Player state module for tracking player character information.
"""
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from src.utils.logging import LogCategory, debug, info


class PlayerClass(Enum):
    """World of Warcraft player classes."""
    HUNTER = auto()
    WARRIOR = auto()
    PALADIN = auto()
    ROGUE = auto()
    PRIEST = auto()
    SHAMAN = auto()
    MAGE = auto()
    WARLOCK = auto()
    DRUID = auto()
    DEATH_KNIGHT = auto()
    MONK = auto()
    DEMON_HUNTER = auto()
    EVOKER = auto()


class PlayerSpec(Enum):
    """Specializations for each class."""
    # Hunter specs
    BEAST_MASTERY = auto()
    MARKSMANSHIP = auto()
    SURVIVAL = auto()
    
    # Other classes' specs would be defined here
    # This will be expanded as support for other classes is added


@dataclass
class Position:
    """Represents a 3D position in the game world."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0  # Height/elevation
    
    def distance_to(self, other: 'Position') -> float:
        """Calculate Euclidean distance to another position."""
        return ((self.x - other.x) ** 2 + 
                (self.y - other.y) ** 2 + 
                (self.z - other.z) ** 2) ** 0.5

    def direction_to(self, other: 'Position') -> Tuple[float, float]:
        """Calculate direction vector to another position."""
        dx = other.x - self.x
        dy = other.y - self.y
        length = (dx ** 2 + dy ** 2) ** 0.5
        
        if length > 0:
            return (dx / length, dy / length)
        return (0.0, 0.0)


@dataclass
class PlayerState:
    """
    Represents the state of the player character.
    
    This includes health, resources, position, combat status, and other
    relevant information about the player character.
    """
    # Basic information
    name: str = ""
    level: int = 1
    player_class: PlayerClass = PlayerClass.HUNTER  # Default to Hunter as specified
    specialization: PlayerSpec = PlayerSpec.BEAST_MASTERY  # Default spec
    
    # Health and resources in %
    current_health: int = 100
    current_resource: int = 100
    
    # Position and movement
    position: Position = field(default_factory=Position)
    facing_direction: float = 0.0  # Radians, 0 = North, π/2 = East
    movement_speed: float = 0.0  # Current movement speed
    
    # Status flags
    in_combat: bool = False
    is_mounted: bool = False
    is_resting: bool = False
    is_dead: bool = False
    
    # Buffs and debuffs
    buffs: Dict[str, float] = field(default_factory=dict)  # name -> expiration time
    debuffs: Dict[str, float] = field(default_factory=dict)  # name -> expiration time
    
    # Equipment and stats
    equipment_level: int = 0
    primary_stat: int = 0
    stamina: int = 0
    critical_strike: float = 0.0
    haste: float = 0.0
    mastery: float = 0.0
    versatility: float = 0.0
    
    # Last update timestamp
    last_updated: float = field(default_factory=time.time)
    
    def health_percent(self) -> float:
        """Get current health as a percentage."""
        return self.current_health
    
    def resource_percent(self) -> float:
        """Get current resource as a percentage."""
        return self.current_resource
    
    def is_low_health(self, threshold: float = 30.0) -> bool:
        """Check if health is below a threshold percentage."""
        return self.health_percent() < threshold
    
    def update_position(self, x: float, y: float, z: float = None) -> None:
        """Update the player's position."""
        old_pos = Position(self.position.x, self.position.y, self.position.z)
        
        self.position.x = x
        self.position.y = y
        if z is not None:
            self.position.z = z
            
        # Calculate movement speed based on position change
        self.movement_speed = old_pos.distance_to(self.position) / 0.1  # Assuming update every 100ms
        self.last_updated = time.time()
        
        debug(f"Player position updated: ({x}, {y}, {self.position.z})", LogCategory.SYSTEM)
    
    def update_combat_status(self, in_combat: bool) -> None:
        """Update the player's combat status."""
        if self.in_combat != in_combat:
            self.in_combat = in_combat
            self.last_updated = time.time()
            
            status = "entered combat" if in_combat else "left combat"
            info(f"Player {status}", LogCategory.COMBAT)
    
    def update_health(self, current: int) -> None:
        """Update player health values."""
        self.current_health = current
        self.last_updated = time.time()
        
        debug(f"Player health: {self.health_percent()} %", 
              LogCategory.COMBAT)
    
    def update_resource(self, current: int, maximum: int = None) -> None:
        """Update player resource values."""
        self.current_resource = current
        if maximum is not None:
            self.current_resource_max = maximum
        self.last_updated = time.time()
        
        debug(f"Player resource: {self.resource_percent()}%", 
              LogCategory.COMBAT)
    
    def add_buff(self, name: str, duration: float) -> None:
        """Add or refresh a buff on the player."""
        expiration = time.time() + duration
        self.buffs[name] = expiration
        self.last_updated = time.time()
        
        debug(f"Player gained buff: {name} ({duration:.1f}s)", LogCategory.COMBAT)
    
    def add_debuff(self, name: str, duration: float) -> None:
        """Add or refresh a debuff on the player."""
        expiration = time.time() + duration
        self.debuffs[name] = expiration
        self.last_updated = time.time()
        
        debug(f"Player gained debuff: {name} ({duration:.1f}s)", LogCategory.COMBAT)
    
    def update_buffs(self) -> None:
        """Update buff durations and remove expired buffs."""
        current_time = time.time()
        expired = []
        
        for name, expiration in self.buffs.items():
            if current_time > expiration:
                expired.append(name)
                debug(f"Buff expired: {name}", LogCategory.COMBAT)
        
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
                debug(f"Debuff expired: {name}", LogCategory.COMBAT)
        
        for name in expired:
            self.debuffs.pop(name, None)
            
        if expired:
            self.last_updated = current_time
    
    def has_buff(self, name: str) -> bool:
        """Check if player has a specific buff."""
        self.update_buffs()  # Clean up expired buffs first
        return name in self.buffs
    
    def has_debuff(self, name: str) -> bool:
        """Check if player has a specific debuff."""
        self.update_debuffs()  # Clean up expired debuffs first
        return name in self.debuffs
    
    def to_dict(self) -> Dict:
        """Convert player state to a dictionary for serialization."""
        return {
            "name": self.name,
            "level": self.level,
            "player_class": self.player_class.name,
            "specialization": self.specialization.name,
            "current_health": self.current_health,
            "health_max": self.health_max,
            "current_resource": self.current_resource,
            "mana_max": self.mana_max,
            "position": {
                "x": self.position.x,
                "y": self.position.y,
                "z": self.position.z
            },
            "facing_direction": self.facing_direction,
            "movement_speed": self.movement_speed,
            "in_combat": self.in_combat,
            "is_mounted": self.is_mounted,
            "is_resting": self.is_resting,
            "is_dead": self.is_dead,
            "buffs": self.buffs,
            "debuffs": self.debuffs,
            "equipment_level": self.equipment_level,
            "last_updated": self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PlayerState':
        """Create a PlayerState object from a dictionary."""
        player = cls(
            name=data.get("name", ""),
            level=data.get("level", 1),
            player_class=PlayerClass[data.get("player_class", "HUNTER")],
            specialization=PlayerSpec[data.get("specialization", "BEAST_MASTERY")],
            current_health=data.get("current_health", 100),
            health_max=data.get("health_max", 100),
            current_resource=data.get("current_resource", 100),
            mana_max=data.get("mana_max", 100),
            facing_direction=data.get("facing_direction", 0.0),
            movement_speed=data.get("movement_speed", 0.0),
            in_combat=data.get("in_combat", False),
            is_mounted=data.get("is_mounted", False),
            is_resting=data.get("is_resting", False),
            is_dead=data.get("is_dead", False),
            equipment_level=data.get("equipment_level", 0)
        )
        
        # Set position
        pos_data = data.get("position", {})
        player.position = Position(
            x=pos_data.get("x", 0.0),
            y=pos_data.get("y", 0.0),
            z=pos_data.get("z", 0.0)
        )
        
        # Set buffs and debuffs
        player.buffs = data.get("buffs", {})
        player.debuffs = data.get("debuffs", {})
        
        # Set last updated
        player.last_updated = data.get("last_updated", time.time())
        
        return player
