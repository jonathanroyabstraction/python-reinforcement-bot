"""
Environment state module for tracking world entities and objects.
"""
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

from src.utils.logging import LogCategory, debug, info
from .player_state import Position


@dataclass
class Entity:
    """Represents an entity in the game world (NPC, monster, player)."""
    id: str  # Unique identifier
    name: str
    entity_type: str  # NPC, monster, player, etc.
    position: Position
    is_hostile: bool = False
    is_friendly: bool = False
    is_neutral: bool = True
    is_elite: bool = False
    level: int = 1
    health_percent: float = 100.0
    distance: float = 0.0  # Distance to player
    last_updated: float = field(default_factory=time.time)
    
    def update_position(self, new_position: Position, player_position: Position) -> None:
        """Update entity position and distance to player."""
        self.position = new_position
        self.distance = self.position.distance_to(player_position)
        self.last_updated = time.time()
    
    def is_in_range(self, range_yards: float) -> bool:
        """Check if the entity is within the specified range."""
        return self.distance <= range_yards


@dataclass
class Obstacle:
    """Represents an obstacle or terrain feature in the game world."""
    position: Position
    radius: float = 1.0  # Size/radius of the obstacle
    height: float = 0.0  # Height of the obstacle (0 for flat terrain)
    obstacle_type: str = "generic"  # Type of obstacle (wall, water, cliff, etc.)
    is_passable: bool = False  # Whether the player can move through this obstacle
    last_updated: float = field(default_factory=time.time)
    
    def intersects_path(self, start: Position, end: Position) -> bool:
        """
        Check if this obstacle intersects a path between start and end positions.
        Simple 2D circle intersection check for path planning.
        """
        # Vector from start to end
        path_x = end.x - start.x
        path_y = end.y - start.y
        path_length = (path_x**2 + path_y**2)**0.5
        
        if path_length == 0:
            # Start and end are the same point
            return self.position.distance_to(start) <= self.radius
        
        # Normalize path vector
        path_x /= path_length
        path_y /= path_length
        
        # Vector from start to obstacle
        to_obstacle_x = self.position.x - start.x
        to_obstacle_y = self.position.y - start.y
        
        # Project obstacle vector onto path vector
        projection = to_obstacle_x * path_x + to_obstacle_y * path_y
        
        # Find closest point on path to obstacle
        if projection < 0:
            closest_x, closest_y = start.x, start.y
        elif projection > path_length:
            closest_x, closest_y = end.x, end.y
        else:
            closest_x = start.x + projection * path_x
            closest_y = start.y + projection * path_y
        
        # Distance from closest point to obstacle
        distance = ((self.position.x - closest_x)**2 + (self.position.y - closest_y)**2)**0.5
        
        # Check if path intersects obstacle
        return distance <= self.radius


@dataclass
class EnvironmentState:
    """
    Represents the state of the game environment.
    
    This includes nearby entities, obstacles, and other environmental factors.
    """
    # Entities in the environment
    entities: Dict[str, Entity] = field(default_factory=dict)
    obstacles: List[Obstacle] = field(default_factory=list)
    
    # Zone information
    zone_name: str = ""
    subzone_name: str = ""
    is_instance: bool = False
    instance_name: str = ""
    
    # Environmental factors
    is_indoors: bool = False
    is_swimming: bool = False
    is_flying: bool = False
    
    # Timestamps
    last_updated: float = field(default_factory=time.time)
    last_scanned: float = 0.0  # Last time a full environment scan was performed
    
    def add_entity(self, entity: Entity) -> None:
        """Add or update an entity in the environment."""
        self.entities[entity.id] = entity
        self.last_updated = time.time()
        
        debug(f"Entity added/updated: {entity.name} ({entity.id})", LogCategory.SYSTEM)
    
    def remove_entity(self, entity_id: str) -> None:
        """Remove an entity from the environment."""
        if entity_id in self.entities:
            entity = self.entities.pop(entity_id)
            self.last_updated = time.time()
            
            debug(f"Entity removed: {entity.name} ({entity_id})", LogCategory.SYSTEM)
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by its ID."""
        return self.entities.get(entity_id)
    
    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        """Get an entity by its name (returns first match)."""
        for entity in self.entities.values():
            if entity.name == name:
                return entity
        return None
    
    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Get all entities of a specific type."""
        return [e for e in self.entities.values() if e.entity_type == entity_type]
    
    def get_nearest_entity(self, position: Position, 
                          filter_fn=None) -> Optional[Entity]:
        """
        Get the nearest entity to a position.
        
        Args:
            position: The reference position
            filter_fn: Optional function to filter entities
        
        Returns:
            The nearest entity that passes the filter, or None if no entities found
        """
        nearest = None
        min_distance = float('inf')
        
        for entity in self.entities.values():
            if filter_fn and not filter_fn(entity):
                continue
                
            distance = entity.position.distance_to(position)
            if distance < min_distance:
                min_distance = distance
                nearest = entity
                
        return nearest
    
    def get_entities_in_range(self, position: Position, range_yards: float,
                             filter_fn=None) -> List[Entity]:
        """
        Get all entities within a specified range of a position.
        
        Args:
            position: The reference position
            range_yards: The range in yards
            filter_fn: Optional function to filter entities
            
        Returns:
            List of entities within the specified range that pass the filter
        """
        in_range = []
        
        for entity in self.entities.values():
            if filter_fn and not filter_fn(entity):
                continue
                
            if entity.position.distance_to(position) <= range_yards:
                in_range.append(entity)
                
        return in_range
    
    def add_obstacle(self, obstacle: Obstacle) -> None:
        """Add an obstacle to the environment."""
        self.obstacles.append(obstacle)
        self.last_updated = time.time()
        
        debug(f"Obstacle added at ({obstacle.position.x}, {obstacle.position.y})", 
              LogCategory.SYSTEM)
    
    def clear_obstacles(self) -> None:
        """Clear all obstacles from the environment."""
        count = len(self.obstacles)
        self.obstacles = []
        self.last_updated = time.time()
        
        debug(f"Cleared {count} obstacles from environment", LogCategory.SYSTEM)
    
    def is_path_clear(self, start: Position, end: Position) -> bool:
        """Check if there's a clear path between two positions."""
        for obstacle in self.obstacles:
            if obstacle.intersects_path(start, end):
                return False
        return True
    
    def update_zone_info(self, zone: str, subzone: str = None, 
                       is_instance: bool = None, instance_name: str = None) -> None:
        """Update zone information."""
        self.zone_name = zone
        
        if subzone is not None:
            self.subzone_name = subzone
            
        if is_instance is not None:
            self.is_instance = is_instance
            
        if instance_name is not None:
            self.instance_name = instance_name
            
        self.last_updated = time.time()
        
        if self.is_instance:
            info(f"Zone updated: {zone} - {instance_name} (Instance)", LogCategory.SYSTEM)
        else:
            info(f"Zone updated: {zone}{f' - {subzone}' if subzone else ''}", LogCategory.SYSTEM)
    
    def cleanup_entities(self, max_age: float = 30.0) -> None:
        """Remove entities that haven't been updated for a certain time."""
        current_time = time.time()
        to_remove = []
        
        for entity_id, entity in self.entities.items():
            if current_time - entity.last_updated > max_age:
                to_remove.append(entity_id)
                
        for entity_id in to_remove:
            self.remove_entity(entity_id)
            
        if to_remove:
            debug(f"Cleaned up {len(to_remove)} stale entities", LogCategory.SYSTEM)
    
    def to_dict(self) -> Dict:
        """Convert environment state to a dictionary for serialization."""
        return {
            "entities": {
                entity_id: {
                    "id": entity.id,
                    "name": entity.name,
                    "entity_type": entity.entity_type,
                    "position": {
                        "x": entity.position.x,
                        "y": entity.position.y,
                        "z": entity.position.z
                    },
                    "is_hostile": entity.is_hostile,
                    "is_friendly": entity.is_friendly,
                    "is_neutral": entity.is_neutral,
                    "is_elite": entity.is_elite,
                    "level": entity.level,
                    "health_percent": entity.health_percent,
                    "distance": entity.distance,
                    "last_updated": entity.last_updated
                } for entity_id, entity in self.entities.items()
            },
            "obstacles": [
                {
                    "position": {
                        "x": obstacle.position.x,
                        "y": obstacle.position.y,
                        "z": obstacle.position.z
                    },
                    "radius": obstacle.radius,
                    "height": obstacle.height,
                    "obstacle_type": obstacle.obstacle_type,
                    "is_passable": obstacle.is_passable,
                    "last_updated": obstacle.last_updated
                } for obstacle in self.obstacles
            ],
            "zone_name": self.zone_name,
            "subzone_name": self.subzone_name,
            "is_instance": self.is_instance,
            "instance_name": self.instance_name,
            "is_indoors": self.is_indoors,
            "is_swimming": self.is_swimming,
            "is_flying": self.is_flying,
            "last_updated": self.last_updated,
            "last_scanned": self.last_scanned
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EnvironmentState':
        """Create an EnvironmentState object from a dictionary."""
        environment = cls(
            zone_name=data.get("zone_name", ""),
            subzone_name=data.get("subzone_name", ""),
            is_instance=data.get("is_instance", False),
            instance_name=data.get("instance_name", ""),
            is_indoors=data.get("is_indoors", False),
            is_swimming=data.get("is_swimming", False),
            is_flying=data.get("is_flying", False),
            last_updated=data.get("last_updated", time.time()),
            last_scanned=data.get("last_scanned", 0.0)
        )
        
        # Set entities
        entities_data = data.get("entities", {})
        for entity_id, entity_data in entities_data.items():
            position = Position(
                x=entity_data.get("position", {}).get("x", 0.0),
                y=entity_data.get("position", {}).get("y", 0.0),
                z=entity_data.get("position", {}).get("z", 0.0)
            )
            
            entity = Entity(
                id=entity_data.get("id", entity_id),
                name=entity_data.get("name", ""),
                entity_type=entity_data.get("entity_type", ""),
                position=position,
                is_hostile=entity_data.get("is_hostile", False),
                is_friendly=entity_data.get("is_friendly", False),
                is_neutral=entity_data.get("is_neutral", True),
                is_elite=entity_data.get("is_elite", False),
                level=entity_data.get("level", 1),
                health_percent=entity_data.get("health_percent", 100.0),
                distance=entity_data.get("distance", 0.0),
                last_updated=entity_data.get("last_updated", time.time())
            )
            
            environment.entities[entity_id] = entity
        
        # Set obstacles
        obstacles_data = data.get("obstacles", [])
        for obstacle_data in obstacles_data:
            position = Position(
                x=obstacle_data.get("position", {}).get("x", 0.0),
                y=obstacle_data.get("position", {}).get("y", 0.0),
                z=obstacle_data.get("position", {}).get("z", 0.0)
            )
            
            obstacle = Obstacle(
                position=position,
                radius=obstacle_data.get("radius", 1.0),
                height=obstacle_data.get("height", 0.0),
                obstacle_type=obstacle_data.get("obstacle_type", "generic"),
                is_passable=obstacle_data.get("is_passable", False),
                last_updated=obstacle_data.get("last_updated", time.time())
            )
            
            environment.obstacles.append(obstacle)
            
        return environment
