"""
Ability state module for tracking cooldowns and resources.
"""
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set

from src.utils.logging import LogCategory, debug, info


@dataclass
class Cooldown:
    """Represents the cooldown state of an ability."""
    name: str
    duration: float  # Total cooldown duration in seconds
    started_at: float  # Time when the cooldown started
    
    @property
    def remaining(self) -> float:
        """Get the remaining cooldown time in seconds."""
        elapsed = time.time() - self.started_at
        remaining = max(0.0, self.duration - elapsed)
        return remaining
    
    @property
    def is_ready(self) -> bool:
        """Check if the ability is ready to use (not on cooldown)."""
        return self.remaining <= 0.0
    
    @property
    def percent_ready(self) -> float:
        """Get the percentage of cooldown completion (0-100)."""
        if self.duration <= 0:
            return 100.0
        return min(100.0, (1.0 - (self.remaining / self.duration)) * 100.0)
    
    def to_dict(self) -> Dict:
        """Convert cooldown to a dictionary for serialization."""
        return {
            "name": self.name,
            "duration": self.duration,
            "started_at": self.started_at,
            "remaining": self.remaining,
            "is_ready": self.is_ready
        }


class Resource(Enum):
    """Types of resources used for abilities."""
    MANA = auto()
    RAGE = auto()
    ENERGY = auto()
    FOCUS = auto()
    RUNIC_POWER = auto()
    RUNES = auto()
    COMBO_POINTS = auto()
    SOUL_SHARDS = auto()
    HOLY_POWER = auto()
    FURY = auto()
    MAELSTROM = auto()
    PAIN = auto()
    INSANITY = auto()
    CHI = auto()
    ARCANE_CHARGES = auto()
    ESSENCE = auto()


@dataclass
class AbilityState:
    """
    Tracks the state of abilities, including cooldowns and resources.
    
    This class maintains information about ability availability, cooldown
    tracking, and resource management for decision making.
    """
    # Cooldown tracking
    cooldowns: Dict[str, Cooldown] = field(default_factory=dict)
    
    # Global cooldown tracking
    gcd_duration: float = 1.5  # Default GCD duration in seconds
    gcd_started_at: float = 0.0  # Time when the GCD was last triggered
    
    # Resource tracking
    resources: Dict[Resource, int] = field(default_factory=dict)
    resource_max: Dict[Resource, int] = field(default_factory=dict)
    
    # Available abilities
    known_abilities: Set[str] = field(default_factory=set)
    
    # Last update timestamp
    last_updated: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Initialize default resources based on class."""
        # Default all resources to 0/100
        for resource in Resource:
            self.resources.setdefault(resource, 0)
            self.resource_max.setdefault(resource, 100)
    
    def update_cooldown(self, ability_name: str, duration: float) -> None:
        """Start or refresh a cooldown for an ability."""
        self.cooldowns[ability_name] = Cooldown(
            name=ability_name,
            duration=duration,
            started_at=time.time()
        )
        self.last_updated = time.time()
        
        debug(f"Ability on cooldown: {ability_name} ({duration:.1f}s)", LogCategory.COMBAT)
    
    def trigger_gcd(self) -> None:
        """Trigger the global cooldown."""
        self.gcd_started_at = time.time()
        self.last_updated = time.time()
        
        debug(f"GCD triggered ({self.gcd_duration:.1f}s)", LogCategory.COMBAT)
    
    def get_ability_cooldown(self, ability_name: str) -> Optional[Cooldown]:
        """Get the cooldown object for a specific ability."""
        return self.cooldowns.get(ability_name)
    
    def is_ability_ready(self, ability_name: str) -> bool:
        """Check if an ability is ready to use (not on cooldown)."""
        # First check if the ability is known
        if ability_name not in self.known_abilities:
            return False
            
        # Check if ability has a specific cooldown
        cooldown = self.cooldowns.get(ability_name)
        if cooldown and not cooldown.is_ready:
            return False
            
        # Check global cooldown
        if not self.is_gcd_ready():
            return False
            
        return True
    
    def is_gcd_ready(self) -> bool:
        """Check if the global cooldown is ready."""
        elapsed = time.time() - self.gcd_started_at
        return elapsed >= self.gcd_duration
    
    def gcd_remaining(self) -> float:
        """Get the remaining time on the global cooldown."""
        elapsed = time.time() - self.gcd_started_at
        return max(0.0, self.gcd_duration - elapsed)
    
    def update_resource(self, resource_type: Resource, current: int, maximum: int = None) -> None:
        """Update a resource value."""
        self.resources[resource_type] = current
        if maximum is not None:
            self.resource_max[resource_type] = maximum
        self.last_updated = time.time()
        
        debug(f"Resource updated: {resource_type.name} {current}/{self.resource_max[resource_type]}", 
              LogCategory.COMBAT)
    
    def get_resource(self, resource_type: Resource) -> int:
        """Get the current value of a resource."""
        return self.resources.get(resource_type, 0)
    
    def get_resource_max(self, resource_type: Resource) -> int:
        """Get the maximum value of a resource."""
        return self.resource_max.get(resource_type, 0)
    
    def get_resource_percent(self, resource_type: Resource) -> float:
        """Get the percentage of a resource (0-100)."""
        current = self.resources.get(resource_type, 0)
        maximum = self.resource_max.get(resource_type, 100)
        
        if maximum <= 0:
            return 0.0
        return (current / maximum) * 100.0
    
    def has_resource(self, resource_type: Resource, amount: int) -> bool:
        """Check if there's enough of a resource available."""
        return self.resources.get(resource_type, 0) >= amount
    
    def update_known_abilities(self, abilities: List[str]) -> None:
        """Update the set of known abilities."""
        self.known_abilities = set(abilities)
        self.last_updated = time.time()
        
        debug(f"Known abilities updated: {len(self.known_abilities)} abilities", LogCategory.SYSTEM)
    
    def add_known_ability(self, ability_name: str) -> None:
        """Add a single ability to the known abilities set."""
        if ability_name not in self.known_abilities:
            self.known_abilities.add(ability_name)
            self.last_updated = time.time()
            debug(f"New ability learned: {ability_name}", LogCategory.SYSTEM)
    
    def cleanup_cooldowns(self) -> None:
        """Remove any expired cooldowns from the tracking dict."""
        expired = []
        
        for ability_name, cooldown in self.cooldowns.items():
            if cooldown.is_ready:
                expired.append(ability_name)
                debug(f"Ability ready: {ability_name}", LogCategory.COMBAT)
        
        for ability_name in expired:
            self.cooldowns.pop(ability_name, None)
            
        if expired:
            self.last_updated = time.time()
    
    def to_dict(self) -> Dict:
        """Convert ability state to a dictionary for serialization."""
        # Clean up expired cooldowns before serializing
        self.cleanup_cooldowns()
        
        return {
            "cooldowns": {name: cd.to_dict() for name, cd in self.cooldowns.items()},
            "gcd_duration": self.gcd_duration,
            "gcd_started_at": self.gcd_started_at,
            "gcd_remaining": self.gcd_remaining(),
            "resources": {res.name: val for res, val in self.resources.items()},
            "resource_max": {res.name: val for res, val in self.resource_max.items()},
            "known_abilities": list(self.known_abilities),
            "last_updated": self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AbilityState':
        """Create an AbilityState object from a dictionary."""
        ability_state = cls(
            gcd_duration=data.get("gcd_duration", 1.5),
            gcd_started_at=data.get("gcd_started_at", 0.0)
        )
        
        # Set cooldowns
        cooldowns_data = data.get("cooldowns", {})
        for name, cd_data in cooldowns_data.items():
            ability_state.cooldowns[name] = Cooldown(
                name=name,
                duration=cd_data.get("duration", 0.0),
                started_at=cd_data.get("started_at", 0.0)
            )
        
        # Set resources
        resources_data = data.get("resources", {})
        for res_name, value in resources_data.items():
            try:
                resource = Resource[res_name]
                ability_state.resources[resource] = value
            except KeyError:
                continue
                
        # Set resource maximums
        resource_max_data = data.get("resource_max", {})
        for res_name, value in resource_max_data.items():
            try:
                resource = Resource[res_name]
                ability_state.resource_max[resource] = value
            except KeyError:
                continue
        
        # Set known abilities
        ability_state.known_abilities = set(data.get("known_abilities", []))
        
        # Set last updated
        ability_state.last_updated = data.get("last_updated", time.time())
        
        return ability_state
