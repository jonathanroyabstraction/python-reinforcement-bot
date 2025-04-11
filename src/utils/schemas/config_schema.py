"""
JSON Schema definitions for configuration validation.

This module defines the expected structure of the configuration
in JSON Schema format (https://json-schema.org/).
"""
from typing import Dict, Any

# Schema for screen capture configuration
SCREEN_CAPTURE_SCHEMA = {
    "type": "object",
    "properties": {
        "fps": {"type": "integer", "minimum": 1, "maximum": 60},
        "resolution": {
            "type": "object",
            "properties": {
                "width": {"type": "integer", "minimum": 640, "maximum": 7680},
                "height": {"type": "integer", "minimum": 480, "maximum": 4320}
            },
            "required": ["width", "height"]
        },
        "regions": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "minimum": 0},
                    "y": {"type": "integer", "minimum": 0},
                    "width": {"type": "integer", "minimum": 1},
                    "height": {"type": "integer", "minimum": 1}
                },
                "required": ["x", "y", "width", "height"]
            }
        }
    },
    "required": ["fps", "resolution"]
}

# Schema for input configuration
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "delay": {
            "type": "object",
            "properties": {
                "min_ms": {"type": "integer", "minimum": 0},
                "max_ms": {"type": "integer", "minimum": 0}
            },
            "required": ["min_ms", "max_ms"]
        },
        "movement": {
            "type": "object",
            "properties": {
                "sensitivity": {"type": "number", "minimum": 0.1, "maximum": 10.0}
            }
        },
        "safety": {
            "type": "object",
            "properties": {
                "emergency_key": {"type": "string"},
                "max_actions_per_second": {"type": "integer", "minimum": 1, "maximum": 20}
            }
        }
    }
}

# Schema for system configuration
SYSTEM_SCHEMA = {
    "type": "object",
    "properties": {
        "debug_mode": {"type": "boolean"},
        "log_level": {
            "type": "string",
            "enum": ["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        },
        "log_dir": {"type": "string"},
        "data_dir": {"type": "string"}
    },
    "required": ["log_level"]
}

# Schema for combat configuration
COMBAT_SCHEMA = {
    "type": "object",
    "properties": {
        "rotation": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "cooldown_usage": {"type": "string", "enum": ["conservative", "balanced", "aggressive"]},
                        "pet_management": {"type": "boolean"},
                        "priority": {"type": "array", "items": {"type": "string"}}
                    }
                }
            }
        }
    }
}

# Schema for navigation configuration
NAVIGATION_SCHEMA = {
    "type": "object",
    "properties": {
        "movement_speed": {"type": "number", "minimum": 0.1, "maximum": 10.0},
        "obstacle_detection": {"type": "boolean"},
        "stuck_timeout_seconds": {"type": "number", "minimum": 0.5}
    }
}

# Schema for error recovery configuration
ERROR_RECOVERY_SCHEMA = {
    "type": "object",
    "properties": {
        "tier1_max_attempts": {"type": "integer", "minimum": 1},
        "tier2_max_attempts": {"type": "integer", "minimum": 1},
        "tier3_max_attempts": {"type": "integer", "minimum": 1},
        "return_to_safe_zone_timeout": {"type": "number", "minimum": 1.0}
    }
}

# Combined schema for the entire configuration
FULL_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "system": SYSTEM_SCHEMA,
        "screen_capture": SCREEN_CAPTURE_SCHEMA,
        "input": INPUT_SCHEMA,
        "combat": COMBAT_SCHEMA,
        "navigation": NAVIGATION_SCHEMA,
        "error_recovery": ERROR_RECOVERY_SCHEMA
    },
    "required": ["system", "screen_capture", "input"]
}

def get_schema_for_section(section: str) -> Dict[str, Any]:
    """
    Get the schema for a specific configuration section.
    
    Args:
        section: The section name (e.g., 'screen_capture', 'input')
        
    Returns:
        The JSON Schema for the specified section
        
    Raises:
        ValueError: If the section is not recognized
    """
    schemas = {
        "system": SYSTEM_SCHEMA,
        "screen_capture": SCREEN_CAPTURE_SCHEMA,
        "input": INPUT_SCHEMA,
        "combat": COMBAT_SCHEMA,
        "navigation": NAVIGATION_SCHEMA,
        "error_recovery": ERROR_RECOVERY_SCHEMA,
        "full": FULL_CONFIG_SCHEMA
    }
    
    if section not in schemas:
        raise ValueError(f"Unknown configuration section: {section}")
    
    return schemas[section]
