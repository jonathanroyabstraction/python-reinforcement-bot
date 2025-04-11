"""
Environment variable handling for the WoW Bot.

This module provides utilities for working with environment variables
and loading values from .env files.
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv


def load_env_file(env_file: Optional[str] = None) -> None:
    """
    Load environment variables from a .env file.
    
    Args:
        env_file: Path to the .env file. If None, looks for .env in the project root.
    """
    if env_file is None:
        # Look for .env in the project root
        project_root = Path(__file__).parent.parent.parent
        env_file = project_root / '.env'
    
    load_dotenv(env_file)


def get_env(name: str, default: Any = None) -> Any:
    """
    Get an environment variable.
    
    Args:
        name: Name of the environment variable
        default: Default value if the environment variable is not set
        
    Returns:
        The value of the environment variable or the default value
    """
    return os.environ.get(name, default)


def get_current_env() -> str:
    """
    Get the current environment (development, production, etc.).
    
    Returns:
        The current environment name (defaults to 'development')
    """
    return get_env('WOW_BOT_ENV', 'development')


def is_development() -> bool:
    """Check if the current environment is development."""
    return get_current_env() == 'development'


def is_production() -> bool:
    """Check if the current environment is production."""
    return get_current_env() == 'production'


def is_test() -> bool:
    """Check if the current environment is test."""
    return get_current_env() == 'test'


# Load environment variables when the module is imported
load_env_file()
