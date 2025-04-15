"""
Configuration management module for the WoW Bot.

This module provides functionality to load, validate, and access configuration
settings from YAML files with support for environment-specific overrides.
"""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import yaml
from jsonschema import validate as json_validate, ValidationError as JsonSchemaError
from yaml.parser import ParserError

from src.utils.env import get_current_env
from src.utils.events import publish, CONFIG_CHANGED


class ConfigError(Exception):
    """Exception raised for configuration errors."""
    pass


class ConfigManager:
    """
    Configuration manager that handles loading and accessing configuration from YAML files.
    
    Features:
    - Load configuration from YAML files
    - Support for environment-specific overrides
    - Default values for missing settings
    - Configuration validation
    - Runtime configuration updates
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the main configuration file. If None, will look for
                        config.yml in the default location.
        """
        # Start with programmatic defaults
        self._config_path = config_path or self._get_default_config_path()
        self._config: Dict[str, Any] = {}
        self._change_listeners: Set[str] = set()  # Tracks keys with active listeners
        self._transaction_active = False  # For batch updates
        self._pending_changes: Set[str] = set()  # Changes during transaction
        self.load_config()
    
    def _get_default_config_path(self) -> Path:
        """Get the default configuration file path."""
        # Look for config.yml in the configs directory
        src_dir = Path(__file__).parent.parent
        return src_dir / "configs" / "config.yml"
    
    def load_config(self) -> None:
        """
        Load configuration from the config file.
        
        Raises:
            ConfigError: If the configuration file cannot be loaded or parsed.
        """
        try:
            # Then try to load default configuration file
            default_config_path = Path(self._config_path).parent / "default_config.yml"
            if default_config_path.exists():
                with open(default_config_path, 'r') as f:
                    file_defaults = yaml.safe_load(f) or {}
                    self._update_nested_dict(self._config, file_defaults)
            
            # Then load the main configuration, overriding defaults
            config_path = Path(self._config_path)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    main_config = yaml.safe_load(f) or {}
                    self._update_nested_dict(self._config, main_config)
            elif not default_config_path.exists():
                raise ConfigError(f"Configuration file not found: {config_path}")
            
            # Load environment-specific configuration
            env = get_current_env()
            env_config_path = config_path.parent / f"config.{env}.yml"
            if env_config_path.exists():
                with open(env_config_path, 'r') as f:
                    env_config = yaml.safe_load(f) or {}
                    self._update_nested_dict(self._config, env_config)
                    
            # Load local overrides if they exist (for development)
            local_config_path = config_path.parent / "config.local.yml"
            if local_config_path.exists():
                with open(local_config_path, 'r') as f:
                    local_config = yaml.safe_load(f) or {}
                    self._update_nested_dict(self._config, local_config)
        
        except ParserError as e:
            raise ConfigError(f"Error parsing configuration file: {e}")
        except Exception as e:
            raise ConfigError(f"Error loading configuration: {e}")
    
    def _update_nested_dict(self, d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a nested dictionary with values from another dictionary.
        
        Args:
            d: The dictionary to update
            u: The dictionary with updates
            
        Returns:
            The updated dictionary
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: The configuration key, can use dot notation for nested keys (e.g., 'system.debug_mode')
            default: Default value to return if key is not found. If None, will try to use programmatic default.
            
        Returns:
            The configuration value or default if not found
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                # If no explicit default is provided, try to use programmatic default
                if default is None:
                    try:
                        return get_default_value(key)
                    except ValueError:
                        return None
                return default
        
        return value
    
    def set(self, key: str, value: Any, notify: bool = True) -> None:
        """
        Set a configuration value at runtime.
        
        Args:
            key: The configuration key, can use dot notation for nested keys
            value: The value to set
            notify: Whether to notify listeners about the change
        """
        keys = key.split('.')
        config = self._config
        
        # Get the old value for comparison
        old_value = self.get(key)
        
        # Navigate to the right level in the nested dict
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            elif not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        
        # Notify listeners if the value has changed and we're not in a transaction
        if notify and old_value != value:
            if self._transaction_active:
                self._pending_changes.add(key)
            else:
                self._notify_change(key, old_value, value)
    
    def save_config(self) -> None:
        """
        Save the current configuration back to the config file.
        
        Raises:
            ConfigError: If the configuration cannot be saved.
        """
        try:
            config_path = Path(self._config_path)
            
            # Make sure the directory exists
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False)
                
            # Publish event for config saved
            publish(CONFIG_CHANGED, {
                'action': 'saved',
                'path': str(config_path)
            })
                
        except Exception as e:
            raise ConfigError(f"Error saving configuration: {e}")
            
    def _notify_change(self, key: str, old_value: Any, new_value: Any) -> None:
        """
        Notify listeners about a configuration change.
        
        Args:
            key: The configuration key that changed
            old_value: The previous value
            new_value: The new value
        """
        # Publish event for the specific key change
        publish(CONFIG_CHANGED, {
            'action': 'updated',
            'key': key,
            'old_value': old_value,
            'new_value': new_value
        })
        
        # Also publish events for parent keys
        parts = key.split('.')
        for i in range(1, len(parts)):
            parent_key = '.'.join(parts[:-i])
            if parent_key:
                publish(CONFIG_CHANGED, {
                    'action': 'child_updated',
                    'key': parent_key,
                    'child_key': key
                })
                
    def begin_transaction(self) -> None:
        """
        Begin a configuration transaction.
        
        This allows multiple configuration changes to be made without triggering
        notifications for each change. Notifications will be sent when the transaction
        is committed.
        """
        self._transaction_active = True
        self._pending_changes.clear()
        
    def commit_transaction(self) -> None:
        """
        Commit a configuration transaction and send notifications.
        """
        if not self._transaction_active:
            return
            
        self._transaction_active = False
        
        # Send notifications for all changed keys
        for key in self._pending_changes:
            value = self.get(key)
            self._notify_change(key, None, value)  # We don't have the old value anymore
            
        # Clear pending changes
        self._pending_changes.clear()
        
        # Publish a general notification for the transaction
        if self._pending_changes:
            publish(CONFIG_CHANGED, {
                'action': 'transaction',
                'keys': list(self._pending_changes)
            })
            
    def rollback_transaction(self) -> None:
        """
        Rollback a configuration transaction without applying changes.
        """
        self._transaction_active = False
        self._pending_changes.clear()
        
        # Reload config to restore previous state
        self.load_config()
    
    def validate(self, schema: Dict[str, Any]) -> bool:
        """
        Validate the configuration against a JSON schema.
        
        Args:
            schema: A dictionary describing the expected configuration structure in JSON Schema format
            
        Returns:
            True if the configuration is valid
            
        Raises:
            ConfigError: If the configuration is invalid
        """
        try:
            json_validate(instance=self._config, schema=schema)
            return True
        except JsonSchemaError as e:
            path = '.'.join(str(p) for p in e.path)
            raise ConfigError(f"Configuration validation error at '{path}': {e.message}")
            
    def validate_field(self, field_path: str, schema: Dict[str, Any]) -> bool:
        """
        Validate a specific field in the configuration.
        
        Args:
            field_path: Dot-notation path to the field to validate
            schema: JSON Schema for the field
            
        Returns:
            True if the field is valid
            
        Raises:
            ConfigError: If the field is invalid or doesn't exist
        """
        value = self.get(field_path)
        if value is None:
            raise ConfigError(f"Field '{field_path}' does not exist in configuration")
            
        try:
            json_validate(instance=value, schema=schema)
            return True
        except JsonSchemaError as e:
            path = field_path + '.' + '.'.join(str(p) for p in e.path) if e.path else field_path
            raise ConfigError(f"Configuration validation error at '{path}': {e.message}")
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the entire configuration dictionary."""
        return self._config

# Create a singleton instance of the ConfigManager
config_manager = ConfigManager()

# Convenience functions to access the singleton
def get_config(key: str, default: Any = None) -> Any:
    """Get a configuration value by key."""
    return config_manager.get(key, default)

def set_config(key: str, value: Any, notify: bool = True) -> None:
    """Set a configuration value."""
    config_manager.set(key, value, notify)

def reload_config() -> None:
    """Reload the configuration from disk."""
    config_manager.load_config()
    
def get_default(key: str) -> Any:
    """Get the default value for a configuration key."""
    try:
        return get_default_value(key)
    except ValueError:
        return None
        
def save_config() -> None:
    """Save the current configuration to disk."""
    config_manager.save_config()
    
def begin_transaction() -> None:
    """Begin a configuration transaction for batch updates."""
    config_manager.begin_transaction()
    
def commit_transaction() -> None:
    """Commit a configuration transaction."""
    config_manager.commit_transaction()
    
def rollback_transaction() -> None:
    """Rollback a configuration transaction."""
    config_manager.rollback_transaction()
