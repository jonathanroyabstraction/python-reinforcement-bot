"""
Default configuration values for the WoW Bot.

This module defines the default values for all configuration settings,
ensuring that the application has sensible defaults even if some
settings are missing from the configuration files.
"""
from typing import Dict, Any

# Default system configuration
DEFAULT_SYSTEM = {
    "debug_mode": False,
    "log_level": "INFO",
    "log_dir": "../data/logs",
    "data_dir": "../data"
}

# Default screen capture configuration
DEFAULT_SCREEN_CAPTURE = {
    "fps": 20,
    "resolution": {
        "width": 1920,
        "height": 1080
    },
    "regions": {
        "game": {
            "x": 0,
            "y": 0,
            "width": 1920,
            "height": 1080
        },
        "minimap": {
            "x": 1670,
            "y": 50,
            "width": 200,
            "height": 200
        },
        "character_frame": {
            "x": 10,
            "y": 10,
            "width": 300,
            "height": 100
        },
        "action_bars": {
            "x": 500,
            "y": 800,
            "width": 900,
            "height": 200
        }
    }
}

# Default input configuration
DEFAULT_INPUT = {
    "delay": {
        "min_ms": 50,
        "max_ms": 150
    },
    "movement": {
        "sensitivity": 1.0
    },
    "safety": {
        "emergency_key": "escape",
        "max_actions_per_second": 10
    }
}

# Default combat configuration
DEFAULT_COMBAT = {
    "rotation": {
        "hunter": {
            "beast_mastery": {
                "cooldown_usage": "balanced",
                "pet_management": True,
                "priority": ["Kill Command", "Bestial Wrath", "Barbed Shot", "Cobra Shot"]
            }
        }
    }
}

# Default navigation configuration
DEFAULT_NAVIGATION = {
    "movement_speed": 1.0,
    "obstacle_detection": True,
    "stuck_timeout_seconds": 5.0
}

# Default error recovery configuration
DEFAULT_ERROR_RECOVERY = {
    "tier1_max_attempts": 3,
    "tier2_max_attempts": 2,
    "tier3_max_attempts": 1,
    "return_to_safe_zone_timeout": 60
}

# Combined default configuration
DEFAULT_CONFIG = {
    "system": DEFAULT_SYSTEM,
    "screen_capture": DEFAULT_SCREEN_CAPTURE,
    "input": DEFAULT_INPUT,
    "combat": DEFAULT_COMBAT,
    "navigation": DEFAULT_NAVIGATION,
    "error_recovery": DEFAULT_ERROR_RECOVERY
}

def get_default_config() -> Dict[str, Any]:
    """
    Get the complete default configuration.
    
    Returns:
        The default configuration dictionary
    """
    return DEFAULT_CONFIG.copy()

def get_default_for_section(section: str) -> Dict[str, Any]:
    """
    Get default values for a specific configuration section.
    
    Args:
        section: The section name (e.g., 'system', 'screen_capture')
        
    Returns:
        The default configuration for the specified section
        
    Raises:
        ValueError: If the section is not recognized
    """
    defaults = {
        "system": DEFAULT_SYSTEM,
        "screen_capture": DEFAULT_SCREEN_CAPTURE,
        "input": DEFAULT_INPUT,
        "combat": DEFAULT_COMBAT,
        "navigation": DEFAULT_NAVIGATION,
        "error_recovery": DEFAULT_ERROR_RECOVERY
    }
    
    if section not in defaults:
        raise ValueError(f"Unknown configuration section: {section}")
    
    return defaults[section].copy()

def get_default_value(key_path: str) -> Any:
    """
    Get a default value for a specific configuration key.
    
    Args:
        key_path: Dot-notation path to the configuration key (e.g., 'system.log_level')
        
    Returns:
        The default value for the specified key
        
    Raises:
        ValueError: If the key is not found in the default configuration
    """
    config = get_default_config()
    keys = key_path.split('.')
    
    # Navigate through the nested dictionary
    for key in keys:
        if isinstance(config, dict) and key in config:
            config = config[key]
        else:
            raise ValueError(f"No default value found for key: {key_path}")
    
    return config
