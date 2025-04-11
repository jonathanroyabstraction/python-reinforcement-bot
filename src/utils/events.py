"""
Event handling system for the WoW Bot.

This module provides a simple event system that allows components
to subscribe to and publish events, enabling decoupled communication.
"""
from typing import Any, Callable, Dict, List, Set

# Type definition for event handlers
EventHandler = Callable[[Dict[str, Any]], None]


class EventManager:
    """
    Event manager for publishing and subscribing to events.
    
    This is a simple implementation of the Observer pattern,
    allowing components to communicate without direct dependencies.
    """
    
    def __init__(self):
        """Initialize the event manager."""
        self._subscribers: Dict[str, Set[EventHandler]] = {}
    
    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """
        Subscribe to an event type.
        
        Args:
            event_type: The type of event to subscribe to
            handler: The function to call when the event is published
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = set()
        self._subscribers[event_type].add(handler)
    
    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """
        Unsubscribe from an event type.
        
        Args:
            event_type: The type of event to unsubscribe from
            handler: The handler to remove
        """
        if event_type in self._subscribers and handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)
            if not self._subscribers[event_type]:
                del self._subscribers[event_type]
    
    def publish(self, event_type: str, data: Dict[str, Any] = None) -> None:
        """
        Publish an event to all subscribers.
        
        Args:
            event_type: The type of event to publish
            data: Data associated with the event
        """
        if data is None:
            data = {}
        
        # Include the event type in the data
        event_data = {'event_type': event_type, **data}
        
        if event_type in self._subscribers:
            for handler in list(self._subscribers[event_type]):
                try:
                    handler(event_data)
                except Exception as e:
                    # Log error but don't stop event propagation
                    print(f"Error in event handler for {event_type}: {e}")


# Create singleton instance
event_manager = EventManager()

# Define common event types
CONFIG_CHANGED = 'config_changed'
SYSTEM_SHUTDOWN = 'system_shutdown'
ERROR_OCCURRED = 'error_occurred'
TASK_COMPLETED = 'task_completed'
TASK_FAILED = 'task_failed'
STATE_CHANGED = 'state_changed'

# Convenience functions
def subscribe(event_type: str, handler: EventHandler) -> None:
    """Subscribe to an event."""
    event_manager.subscribe(event_type, handler)

def unsubscribe(event_type: str, handler: EventHandler) -> None:
    """Unsubscribe from an event."""
    event_manager.unsubscribe(event_type, handler)

def publish(event_type: str, data: Dict[str, Any] = None) -> None:
    """Publish an event."""
    event_manager.publish(event_type, data)
