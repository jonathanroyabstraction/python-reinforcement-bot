"""
Tests for the Game State system.

This module tests the functionality of the GameState class and its components,
including PlayerState, TargetState, AbilityState, and EnvironmentState.
"""
import os
import sys
import unittest
import time
import json
import tempfile

# Handle mock import compatibility with Python 2 and 3
try:
    from unittest.mock import MagicMock, patch  # Python 3
except ImportError:
    from mock import MagicMock, patch  # Python 2 (requires mock package)

# Add the project root to the Python path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from src.decision.state.game_state import GameState, GameStateChangeType, GameStateObserver, GameStateSnapshot
from src.decision.state.player_state import PlayerState, PlayerClass, PlayerSpec, Position
from src.decision.state.target_state import TargetState, TargetType
from src.decision.state.ability_state import AbilityState, Cooldown, Resource
from src.decision.state.environment_state import EnvironmentState, Obstacle
from src.decision.state.environment_state import Entity  # Import Entity directly


class TestGameStateComponents(unittest.TestCase):
    """Test individual components of the game state system."""
    
    def test_position(self):
        """Test Position class functionality."""
        pos1 = Position(10.0, 20.0, 30.0)
        pos2 = Position(13.0, 24.0, 30.0)
        
        # Test distance calculation
        self.assertAlmostEqual(pos1.distance_to(pos2), 5.0)
        
        # Test direction calculation
        direction = pos1.direction_to(pos2)
        self.assertAlmostEqual(direction[0], 0.6, places=1)  # dx normalized
        self.assertAlmostEqual(direction[1], 0.8, places=1)  # dy normalized
    
    def test_player_state(self):
        """Test PlayerState class functionality."""
        player = PlayerState(
            name="TestPlayer",
            level=60,
            player_class=PlayerClass.HUNTER,
            specialization=PlayerSpec.BEAST_MASTERY,
            current_health=80,
            health_max=100,
            current_resource=60,
            mana_max=100
        )
        
        # Test health and mana percentage calculations
        self.assertEqual(player.health_percent(), 80.0)
        self.assertEqual(player.mana_percent(), 60.0)
        
        # Test is_low_health method
        self.assertFalse(player.is_low_health(threshold=70.0))
        self.assertTrue(player.is_low_health(threshold=90.0))
        
        # Test update methods
        player.update_health(90, 120)
        self.assertEqual(player.current_health, 90)
        
        player.update_position(100.0, 200.0, 50.0)
        self.assertEqual(player.position.x, 100.0)
        self.assertEqual(player.position.y, 200.0)
        self.assertEqual(player.position.z, 50.0)
        
        # Test buff and debuff tracking
        player.add_buff("TestBuff", 10.0)
        player.add_debuff("TestDebuff", 5.0)
        
        self.assertTrue(player.has_buff("TestBuff"))
        self.assertTrue(player.has_debuff("TestDebuff"))
        
        # Test serialization
        player_dict = player.to_dict()
        self.assertEqual(player_dict["name"], "TestPlayer")
        self.assertEqual(player_dict["player_class"], "HUNTER")
        
        # Test deserialization
        new_player = PlayerState.from_dict(player_dict)
        self.assertEqual(new_player.name, "TestPlayer")
        self.assertEqual(new_player.player_class, PlayerClass.HUNTER)
    
    def test_target_state(self):
        """Test TargetState class functionality."""
        target = TargetState(
            name="TestTarget",
            guid="mob-12345",
            target_type=TargetType.HOSTILE,
            level=58,
            is_elite=True,
            current_health=50,
            health_max=100
        )
        
        # Test health percentage calculation
        self.assertEqual(target.health_percent(), 50.0)
        
        # Test update methods
        player_pos = Position(0.0, 0.0, 0.0)
        target_pos = Position(30.0, 40.0, 0.0)
        target.update_position(target_pos, player_pos)
        
        self.assertEqual(target.distance, 50.0)
        self.assertTrue(target.is_in_range(60.0))
        self.assertFalse(target.is_in_range(40.0))
        
        # Test casting updates
        target.update_cast_info(True, "Fireball", 2.5)
        self.assertTrue(target.is_casting)
        self.assertEqual(target.cast_name, "Fireball")
        self.assertEqual(target.cast_time_remaining, 2.5)
        
        # Test serialization and deserialization
        target_dict = target.to_dict()
        new_target = TargetState.from_dict(target_dict)
        
        self.assertEqual(new_target.name, "TestTarget")
        self.assertEqual(new_target.target_type, TargetType.HOSTILE)
        self.assertTrue(new_target.is_elite)
    
    def test_ability_state(self):
        """Test AbilityState class functionality."""
        abilities = AbilityState()
        
        # Test cooldown tracking
        abilities.update_cooldown("Aimed Shot", 10.0)
        cooldown = abilities.get_ability_cooldown("Aimed Shot")
        
        self.assertIsNotNone(cooldown)
        self.assertEqual(cooldown.name, "Aimed Shot")
        self.assertEqual(cooldown.duration, 10.0)
        self.assertFalse(cooldown.is_ready)
        
        # Test GCD tracking
        abilities.trigger_gcd()
        self.assertFalse(abilities.is_gcd_ready())
        self.assertTrue(abilities.gcd_remaining() > 0)
        
        # Test resource tracking
        abilities.update_resource(Resource.FOCUS, 80, 100)
        self.assertEqual(abilities.get_resource(Resource.FOCUS), 80)
        self.assertEqual(abilities.get_resource_max(Resource.FOCUS), 100)
        self.assertEqual(abilities.get_resource_percent(Resource.FOCUS), 80.0)
        
        # Test ability knowledge tracking
        abilities.update_known_abilities(["Aimed Shot", "Multi-Shot", "Arcane Shot"])
        abilities.add_known_ability("Kill Command")
        
        self.assertTrue("Aimed Shot" in abilities.known_abilities)
        self.assertTrue("Kill Command" in abilities.known_abilities)
        
        # Test is_ability_ready with an ability on cooldown
        abilities.add_known_ability("Rapid Fire")
        
        # Mock the GCD to ensure it's ready
        with patch.object(AbilityState, 'is_gcd_ready', return_value=True):
            self.assertTrue(abilities.is_ability_ready("Rapid Fire"))  # Not on cooldown
            self.assertFalse(abilities.is_ability_ready("Aimed Shot"))  # On cooldown
        
        # Test cleanup
        time.sleep(0.1)  # Wait a moment
        
        # Create a fresh cooldown for testing cleanup
        test_cooldown = Cooldown(name="Test Ability", duration=1.0, started_at=time.time() - 2.0)
        abilities.cooldowns["Test Ability"] = test_cooldown
        
        # Now clean up - since started_at is 2 seconds ago and duration is 1 second, it should be removed
        abilities.cleanup_cooldowns()
        self.assertIsNone(abilities.get_ability_cooldown("Test Ability"))
    
    def test_environment_state(self):
        """Test EnvironmentState class functionality."""
        env = EnvironmentState()
        player_pos = Position(0.0, 0.0, 0.0)
        
        # Test entity tracking
        entity1 = Entity(
            id="npc-1",
            name="Friendly NPC", 
            entity_type="npc",
            position=Position(10.0, 0.0, 0.0),
            is_friendly=True
        )
        entity1.update_position(entity1.position, player_pos)
        
        entity2 = Entity(
            id="mob-1",
            name="Hostile Mob", 
            entity_type="mob",
            position=Position(0.0, 20.0, 0.0),
            is_hostile=True,
            level=60
        )
        entity2.update_position(entity2.position, player_pos)
        
        # Add entities
        env.add_entity(entity1)
        env.add_entity(entity2)
        
        # Test retrieval
        self.assertEqual(len(env.entities), 2)
        self.assertEqual(env.get_entity("npc-1").name, "Friendly NPC")
        
        # Test nearest entity
        nearest = env.get_nearest_entity(player_pos)
        self.assertEqual(nearest.id, "npc-1")  # 10 units away vs 20
        
        # Test entities in range
        in_range = env.get_entities_in_range(player_pos, 15.0)
        self.assertEqual(len(in_range), 1)
        self.assertEqual(in_range[0].id, "npc-1")
        
        # Test filtering
        hostile_only = lambda e: e.is_hostile
        hostiles = env.get_entities_in_range(player_pos, 30.0, filter_fn=hostile_only)
        self.assertEqual(len(hostiles), 1)
        self.assertEqual(hostiles[0].id, "mob-1")
        
        # Test zone updates
        env.update_zone_info("Elwynn Forest", "Goldshire", False)
        self.assertEqual(env.zone_name, "Elwynn Forest")
        self.assertEqual(env.subzone_name, "Goldshire")
        
        # Test obstacle path checking
        obstacle = Obstacle(
            position=Position(5.0, 0.0, 0.0),
            radius=2.0
        )
        env.add_obstacle(obstacle)
        
        # Path should intersect with obstacle
        start = Position(0.0, 0.0, 0.0)
        end = Position(10.0, 0.0, 0.0)
        self.assertFalse(env.is_path_clear(start, end))
        
        # Path should not intersect
        end2 = Position(0.0, 10.0, 0.0)
        self.assertTrue(env.is_path_clear(start, end2))
        
        # Test cleanup
        env.cleanup_entities(max_age=0.1)  # Should be kept since just added
        self.assertEqual(len(env.entities), 2)


class MockObserver(GameStateObserver):
    """Mock observer for testing the observer pattern."""
    
    def __init__(self):
        self.notifications = []
    
    def on_state_changed(self, change_type, old_state=None, new_state=None):
        """Record notification."""
        self.notifications.append((change_type, old_state, new_state))


class TestGameState(unittest.TestCase):
    """Test the main GameState class functionality."""
    
    def setUp(self):
        """Set up a test game state."""
        # Create a temporary directory for state files
        self.temp_dir = tempfile.mkdtemp()
        self.game_state = GameState(state_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test resources."""
        # Remove temporary directory
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_observer_pattern(self):
        """Test the observer notification system."""
        observer = MockObserver()
        
        # Register observer for specific change types
        self.game_state.register_observer(
            observer, 
            [GameStateChangeType.PLAYER_HEALTH, GameStateChangeType.TARGET_CHANGED]
        )
        
        # Trigger notifications
        self.game_state.update_player_health(80, 100)
        
        new_target = TargetState(name="Test Target", current_health=50, health_max=100)
        self.game_state.set_target(new_target)
        
        # Update that shouldn't notify this observer
        self.game_state.update_player_mana(90, 100)
        
        # Verify notifications
        self.assertEqual(len(observer.notifications), 2)
        self.assertEqual(observer.notifications[0][0], GameStateChangeType.PLAYER_HEALTH)
        self.assertEqual(observer.notifications[1][0], GameStateChangeType.TARGET_CHANGED)
        
        # Test unregistering
        self.game_state.unregister_observer(observer)
        self.game_state.update_player_health(70, 100)
        self.assertEqual(len(observer.notifications), 2)  # No new notifications
    
    def test_history_tracking(self):
        """Test game state history functionality."""
        # Initial state
        self.game_state.update_player_health(100, 100)
        self.game_state.add_to_history()
        
        # Update state
        self.game_state.update_player_health(80, 100)
        self.game_state.add_to_history()
        
        # Another update
        self.game_state.update_player_health(60, 100)
        self.game_state.add_to_history()
        
        # Check history
        self.assertEqual(len(self.game_state.history), 3)
        
        # Get historical state
        historical = self.game_state.get_historical_state(1)  # Second state
        self.assertEqual(historical.player.current_health, 80)
        
        # Compare current with historical
        diffs = self.game_state.compare_with_history(1)
        self.assertIn("player", diffs)
        self.assertIn("current_health", diffs["player"])
        self.assertEqual(diffs["player"]["current_health"]["from"], 80)
        self.assertEqual(diffs["player"]["current_health"]["to"], 60)
    
    def test_serialization(self):
        """Test state serialization and persistence."""
        # Setup state with some data
        self.game_state.update_player_health(75, 100)
        self.game_state.update_player_mana(80, 100)
        self.game_state.player.name = "TestChar"
        self.game_state.player.level = 60
        
        new_target = TargetState(
            name="Boss Enemy",
            current_health=500000,
            health_max=1000000,
            target_type=TargetType.BOSS
        )
        self.game_state.set_target(new_target)
        
        # Save state
        filepath = self.game_state.save_state("test_state.json")
        self.assertTrue(os.path.exists(filepath))
        
        # Create a new game state and load the saved state
        new_game_state = GameState(state_dir=self.temp_dir)
        self.assertTrue(new_game_state.load_state("test_state.json"))
        
        # Verify loaded state
        self.assertEqual(new_game_state.player.name, "TestChar")
        self.assertEqual(new_game_state.player.level, 60)
        self.assertEqual(new_game_state.player.current_health, 75)
        self.assertEqual(new_game_state.target.name, "Boss Enemy")
        self.assertEqual(new_game_state.target.target_type, TargetType.BOSS)
    
    def test_screen_capture_integration(self):
        """Test updating game state from screen capture data."""
        # Mock screen capture data
        screen_data = {
            "player": {
                "health": {"current": 85, "max": 100},
                "mana": {"current": 70, "max": 100},
                "position": {"x": 123.4, "y": 456.7, "z": 78.9},
                "combat": True
            },
            "target": {
                "name": "Kobold Miner",
                "guid": "mob-112233",
                "level": 8,
                "health": {"current": 42, "max": 84},
                "position": {"x": 125.4, "y": 460.7, "z": 78.9},
                "type": "HOSTILE",
                "casting": {
                    "is_casting": True,
                    "name": "Mining",
                    "time_remaining": 1.5
                }
            },
            "abilities": {
                "cooldowns": {
                    "Aimed Shot": 8.5,
                    "Rapid Fire": 90.0
                },
                "resources": {
                    "FOCUS": {"current": 88, "max": 100}
                }
            },
            "environment": {
                "zone": {
                    "name": "Elwynn Forest",
                    "subzone": "Fargodeep Mine",
                    "is_instance": False
                },
                "entities": [
                    {
                        "id": "mob-112244",
                        "name": "Kobold Laborer",
                        "type": "mob",
                        "is_hostile": True,
                        "level": 7,
                        "health_percent": 100.0,
                        "position": {"x": 130.0, "y": 455.0, "z": 78.9}
                    }
                ]
            }
        }
        
        # Update state from screen data
        self.game_state.update_from_screen_capture(screen_data)
        
        # Verify updates were applied
        self.assertEqual(self.game_state.player.current_health, 85)
        self.assertEqual(self.game_state.player.current_resource, 70)
        self.assertEqual(self.game_state.player.position.x, 123.4)
        self.assertTrue(self.game_state.player.in_combat)
        
        self.assertEqual(self.game_state.target.name, "Kobold Miner")
        self.assertEqual(self.game_state.target.current_health, 42)
        self.assertTrue(self.game_state.target.is_casting)
        self.assertEqual(self.game_state.target.cast_name, "Mining")
        
        cooldown = self.game_state.abilities.get_ability_cooldown("Aimed Shot")
        self.assertIsNotNone(cooldown)
        self.assertEqual(cooldown.duration, 8.5)
        
        self.assertEqual(self.game_state.abilities.get_resource(Resource.FOCUS), 88)
        
        self.assertEqual(self.game_state.environment.zone_name, "Elwynn Forest")
        self.assertEqual(self.game_state.environment.subzone_name, "Fargodeep Mine")
        
        # Verify that an entity was added
        self.assertEqual(len(self.game_state.environment.entities), 1)
        entity = self.game_state.environment.get_entity("mob-112244")
        self.assertEqual(entity.name, "Kobold Laborer")
        self.assertEqual(entity.level, 7)
    
    def test_state_validation(self):
        """Test state validation functionality."""
        # Create an invalid state
        self.game_state.player.current_health = -10
        self.game_state.player.mana_max = -5
        
        # Validate state
        issues = self.game_state.validate_state()
        
        # Should have identified the issues
        self.assertGreaterEqual(len(issues), 2)
        self.assertTrue(any("health" in issue.lower() for issue in issues))
        self.assertTrue(any("mana" in issue.lower() for issue in issues))


if __name__ == "__main__":
    unittest.main()
