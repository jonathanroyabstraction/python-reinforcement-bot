"""
Logging system for the WoW Bot.

This module provides a comprehensive logging system with:
1. Multiple log levels (TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL)
2. Multiple categories (Control, Combat, Navigation, ML, System)
3. Configurable output (console and file)
4. Log rotation and retention
"""
import logging
import os
import sys
from datetime import datetime
from enum import Enum
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional, Union

from src.utils.config import get_config

# Create a TRACE level (more detailed than DEBUG)
TRACE_LEVEL = 5
logging.addLevelName(TRACE_LEVEL, "TRACE")


class LogCategory(Enum):
    """Categories for logging to allow filtering and organization."""
    SYSTEM = "SYSTEM"       # Core system logs
    CONTROL = "CONTROL"     # Input/output and screen capture
    COMBAT = "COMBAT"       # Combat-related activities
    NAVIGATION = "NAV"      # Movement and pathing
    ML = "ML"               # Machine learning
    VISION = "VISION"       # Vision and recognition
    API = "API"             # API and external communication
    UI = "UI"               # User interface
    CONFIG = "CONFIG"       # Configuration-related logs
    ERROR = "ERROR"         # Error handling and recovery


class ColorFormatter(logging.Formatter):
    """
    Formatter that adds colors to log messages based on their level.
    Only applies colors when outputting to a terminal.
    """
    # ANSI color codes
    COLORS = {
        'TRACE': '\033[35m',     # Magenta
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[41m',  # Red background
        'RESET': '\033[0m',      # Reset
        # Category colors
        'SYSTEM': '\033[37m',    # White
        'CONTROL': '\033[36m',   # Cyan
        'COMBAT': '\033[31m',    # Red
        'NAV': '\033[32m',       # Green
        'VISION': '\033[35m',    # Magenta
        'API': '\033[34m',       # Blue
        'UI': '\033[33m',        # Yellow
        'CONFIG': '\033[37m',    # White
        'ERROR': '\033[31m',     # Red
    }

    def __init__(self, fmt: str = None, datefmt: str = None, use_colors: bool = True):
        """
        Initialize the formatter.
        
        Args:
            fmt: Format string
            datefmt: Date format string
            use_colors: Whether to use colors
        """
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record with colors if enabled.
        
        Args:
            record: The log record to format
            
        Returns:
            The formatted log message
        """
        # Make a copy of the record to avoid modifying the original
        record_copy = logging.makeLogRecord(record.__dict__)
        
        # Extract the category from the record
        category = getattr(record_copy, 'category', LogCategory.SYSTEM.value)
        
        # Add the category to the record if not already present
        if not hasattr(record_copy, 'category_str'):
            record_copy.category_str = f"[{category}]"
        
        # Add colors if enabled
        if self.use_colors:
            level_color = self.COLORS.get(record_copy.levelname, self.COLORS['RESET'])
            category_color = self.COLORS.get(category, self.COLORS['RESET'])
            reset = self.COLORS['RESET']
            
            record_copy.levelname = f"{level_color}{record_copy.levelname}{reset}"
            if hasattr(record_copy, 'category_str'):
                record_copy.category_str = f"{category_color}{record_copy.category_str}{reset}"
        
        return super().format(record_copy)


class BotLogger:
    """
    Logger for the WoW Bot that supports multiple levels and categories.
    """
    # Singleton instance
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super(BotLogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the logger.
        
        Args:
            config_path: Path to the configuration file
        """
        # Only initialize once (singleton pattern)
        if self._initialized:
            return
            
        self._initialized = True
        self._loggers: Dict[str, logging.Logger] = {}
        self._default_category = LogCategory.SYSTEM

        # Configure the logger from configuration
        self.configure(config_path)
    
    def configure(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """
        Configure the logger based on the configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        # Get logging configuration
        log_level = get_config('system.log_level', 'INFO').upper()
        log_dir = get_config('system.log_dir', './logs')
        console_logging = get_config('system.logging.console', True)
        file_logging = get_config('system.logging.file', True)
        max_log_size_mb = get_config('system.logging.max_size_mb', 10)
        backup_count = get_config('system.logging.backup_count', 5)
        
        # Make sure log directory exists
        if file_logging and log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Convert string log level to int
        numeric_level = getattr(logging, log_level, None)
        if not isinstance(numeric_level, int):
            if log_level == 'TRACE':
                numeric_level = TRACE_LEVEL
            else:
                numeric_level = logging.INFO
                print(f"Invalid log level: {log_level}, defaulting to INFO")
        
        # Create the root logger
        root_logger = logging.getLogger('wow_bot')
        root_logger.setLevel(numeric_level)
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Add handlers based on configuration
        handlers = []
        
        # Console handler
        if console_logging:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(numeric_level)
            console_format = "%(asctime)s - %(levelname)-8s %(category_str)s - %(message)s"
            console_formatter = ColorFormatter(console_format, datefmt="%H:%M:%S")
            console_handler.setFormatter(console_formatter)
            handlers.append(console_handler)
        
        # File handler
        if file_logging:
            # Ensure log directory exists
            os.makedirs(log_dir, exist_ok=True)
            
            # Log file paths
            log_file = os.path.join(log_dir, 'wow_bot.log')
            error_log_file = os.path.join(log_dir, 'error.log')
            
            # Regular log file (rotating by size)
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_log_size_mb * 1024 * 1024,
                backupCount=backup_count
            )
            file_handler.setLevel(numeric_level)
            
            # Error log file (rotating daily)
            error_file_handler = TimedRotatingFileHandler(
                error_log_file,
                when='midnight',
                interval=1,
                backupCount=backup_count * 2
            )
            error_file_handler.setLevel(logging.ERROR)
            
            # Set formatters
            file_format = "%(asctime)s - %(levelname)-8s %(category_str)s - %(message)s"
            file_formatter = logging.Formatter(file_format, datefmt="%Y-%m-%d %H:%M:%S")
            
            file_handler.setFormatter(file_formatter)
            error_file_handler.setFormatter(file_formatter)
            
            handlers.append(file_handler)
            handlers.append(error_file_handler)
        
        # Add all handlers to the root logger
        for handler in handlers:
            root_logger.addHandler(handler)
            
        # Store the root logger
        self._loggers['root'] = root_logger
        
        # Configure category loggers
        for category in LogCategory:
            self._create_category_logger(category)
    
    def _create_category_logger(self, category: LogCategory) -> logging.Logger:
        """
        Create a logger for a specific category.
        
        Args:
            category: The log category
            
        Returns:
            The logger instance
        """
        category_name = category.value
        
        # Create a new logger that inherits from the root logger
        logger = logging.getLogger(f'wow_bot.{category_name.lower()}')
        
        # Create a custom adapter that adds category_str to the record
        class CategoryAdapter(logging.LoggerAdapter):
            def process(self, msg, kwargs):
                # Add the category to the kwargs
                kwargs['extra'] = kwargs.get('extra', {})
                kwargs['extra']['category'] = self.extra['category']
                kwargs['extra']['category_str'] = f"[{self.extra['category']}]"
                return msg, kwargs
        
        # Set the category attribute for the formatter
        logger = CategoryAdapter(logger, {'category': category_name})
        
        # Store the logger
        self._loggers[category_name] = logger
        
        return logger
    
    def get_logger(self, category: Optional[Union[str, LogCategory]] = None) -> logging.Logger:
        """
        Get a logger for the specified category.
        
        Args:
            category: The category to get a logger for (string or enum)
            
        Returns:
            A logger instance
        """
        # Convert string to enum if needed
        if isinstance(category, str):
            try:
                category = LogCategory[category.upper()]
            except KeyError:
                category = self._default_category
        
        # Use default category if None
        if category is None:
            category = self._default_category
            
        # Get the category name
        category_name = category.value if isinstance(category, LogCategory) else category
        
        # Create the category logger if it doesn't exist
        if category_name not in self._loggers:
            self._create_category_logger(LogCategory(category_name))
            
        return self._loggers[category_name]
    
    def trace(self, msg: str, category: Optional[Union[str, LogCategory]] = None, *args, **kwargs):
        """Log a message at TRACE level."""
        logger = self.get_logger(category)
        logger.log(TRACE_LEVEL, msg, *args, **kwargs)
    
    def debug(self, msg: str, category: Optional[Union[str, LogCategory]] = None, *args, **kwargs):
        """Log a message at DEBUG level."""
        logger = self.get_logger(category)
        logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, category: Optional[Union[str, LogCategory]] = None, *args, **kwargs):
        """Log a message at INFO level."""
        logger = self.get_logger(category)
        logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, category: Optional[Union[str, LogCategory]] = None, *args, **kwargs):
        """Log a message at WARNING level."""
        logger = self.get_logger(category)
        logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, category: Optional[Union[str, LogCategory]] = None, *args, **kwargs):
        """Log a message at ERROR level."""
        logger = self.get_logger(category)
        logger.error(msg, *args, **kwargs)
    
    def critical(self, msg: str, category: Optional[Union[str, LogCategory]] = None, *args, **kwargs):
        """Log a message at CRITICAL level."""
        logger = self.get_logger(category)
        logger.critical(msg, *args, **kwargs)
    
    def exception(self, msg: str, category: Optional[Union[str, LogCategory]] = None, *args, **kwargs):
        """Log an exception at ERROR level, including the stack trace."""
        logger = self.get_logger(category)
        logger.exception(msg, *args, **kwargs)


# Create a singleton instance
bot_logger = BotLogger()

# Convenience functions
def get_logger(category: Optional[Union[str, LogCategory]] = None) -> logging.Logger:
    """Get a logger for the specified category."""
    return bot_logger.get_logger(category)

def trace(msg: str, category: Optional[Union[str, LogCategory]] = None, *args, **kwargs):
    """Log a message at TRACE level."""
    bot_logger.trace(msg, category, *args, **kwargs)

def debug(msg: str, category: Optional[Union[str, LogCategory]] = None, *args, **kwargs):
    """Log a message at DEBUG level."""
    bot_logger.debug(msg, category, *args, **kwargs)

def info(msg: str, category: Optional[Union[str, LogCategory]] = None, *args, **kwargs):
    """Log a message at INFO level."""
    bot_logger.info(msg, category, *args, **kwargs)

def warning(msg: str, category: Optional[Union[str, LogCategory]] = None, *args, **kwargs):
    """Log a message at WARNING level."""
    bot_logger.warning(msg, category, *args, **kwargs)

def error(msg: str, category: Optional[Union[str, LogCategory]] = None, *args, **kwargs):
    """Log a message at ERROR level."""
    bot_logger.error(msg, category, *args, **kwargs)

def critical(msg: str, category: Optional[Union[str, LogCategory]] = None, *args, **kwargs):
    """Log a message at CRITICAL level."""
    bot_logger.critical(msg, category, *args, **kwargs)

def exception(msg: str, category: Optional[Union[str, LogCategory]] = None, *args, **kwargs):
    """Log an exception at ERROR level, including the stack trace."""
    bot_logger.exception(msg, category, *args, **kwargs)

def configure(config_path: Optional[Union[str, Path]] = None) -> None:
    """Reconfigure the logger."""
    bot_logger.configure(config_path)
