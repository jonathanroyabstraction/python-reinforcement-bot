"""
Input simulation module for the WoW Bot.

This module provides functionality to control keyboard and mouse inputs
for interacting with the World of Warcraft game, with support for
human-like delays, safety features, and action sequences.
"""
import random
import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum, auto
from queue import PriorityQueue, Queue
from threading import Event, Lock, Thread
from typing import Callable, Dict, List, Optional, Tuple, Union

import pynput
from pynput.keyboard import Controller as KeyboardController
from pynput.keyboard import Key, KeyCode
from pynput.mouse import Button, Controller as MouseController

from src.utils.config import get_config
from src.utils.logging import LogCategory, debug, info, warning, error

# Type aliases for clarity
KeyType = Union[Key, KeyCode, str]
MousePosition = Tuple[int, int]


class InputEventType(Enum):
    """Types of input events for the simulator."""
    KEY_PRESS = auto()
    KEY_RELEASE = auto()
    KEY_TAP = auto()  # Press and release
    MOUSE_MOVE = auto()
    MOUSE_CLICK = auto()
    MOUSE_SCROLL = auto()
    DELAY = auto()
    CUSTOM = auto()  # For custom actions


@dataclass
class InputAction:
    """Represents a single input action to be performed."""
    event_type: InputEventType
    params: Dict
    timestamp: float = 0.0  # When this action was executed
    priority: int = 1  # Lower = higher priority
    id: str = None  # Unique ID for the action
    
    def __post_init__(self):
        """Initialize with a unique ID if none provided."""
        if self.id is None:
            self.id = str(uuid.uuid4())

    def __lt__(self, other):
        """Less than comparison for priority queue."""
        if not isinstance(other, InputAction):
            return NotImplemented
        return self.priority < other.priority
        
    def __eq__(self, other):
        """Equality comparison for priority queue."""
        if not isinstance(other, InputAction):
            return NotImplemented
        return self.id == other.id


class InputMode(Enum):
    """Operating modes for the input simulator."""
    NORMAL = auto()  # Normal operation, inputs applied to system
    TESTING = auto()  # Testing mode, no actual inputs, just logging
    DISABLED = auto()  # No inputs allowed


class InputSimulator:
    """
    Simulates keyboard and mouse inputs for controlling WoW.
    
    Features:
    - Human-like randomized delays between actions
    - Safety mechanisms to prevent excessive inputs
    - Input sequence recording and playback
    - Thread-safe operation
    - Testing mode for visualization without execution
    """
    
    def __init__(self, mode: InputMode = InputMode.NORMAL):
        """
        Initialize the input simulator.
        
        Args:
            mode: Operating mode (NORMAL, TESTING, DISABLED)
        """
        # Initialize controllers
        self._keyboard = KeyboardController()
        self._mouse = MouseController()
        
        # Operating mode
        self._mode = mode
        
        # Safety and control mechanisms
        self._stop_event = Event()
        self._lock = Lock()
        self._action_queue = PriorityQueue()
        self._active = False
        self._last_action_time = 0.0
        self._action_count = 0
        self._action_count_reset_time = time.time()
        
        # Load configuration
        self._min_delay_ms = get_config('input.delay.min_ms', 50)
        self._max_delay_ms = get_config('input.delay.max_ms', 150)
        self._max_actions_per_second = get_config('input.safety.max_actions_per_second', 10)
        self._emergency_key = self._parse_key(get_config('input.safety.emergency_key', 'escape'))
        
        # Track pressed keys for safety
        self._pressed_keys = set()
        
        # Worker thread for processing the action queue
        self._worker_thread = None
        
        # Initialize logging
        info(f"Input simulator initialized in {mode.name} mode", LogCategory.CONTROL)
        debug(f"Input delay range: {self._min_delay_ms}-{self._max_delay_ms}ms", LogCategory.CONTROL)
        debug(f"Max actions per second: {self._max_actions_per_second}", LogCategory.CONTROL)
        debug(f"Emergency key: {self._emergency_key}", LogCategory.CONTROL)
        
        # Register listeners for the emergency key if safety is enabled
        if get_config('input.safety.enable_emergency_key', True):
            self._setup_emergency_key_listener()
    
    def _setup_emergency_key_listener(self):
        """Set up a listener for the emergency stop key."""
        def on_press(key):
            if key == self._emergency_key:
                self.emergency_stop()
                return False  # Stop listener
            return True
        
        # Start the keyboard listener in a non-blocking way
        self._key_listener = pynput.keyboard.Listener(on_press=on_press)
        self._key_listener.start()
        debug("Emergency key listener activated", LogCategory.CONTROL)
    
    def _parse_key(self, key_name: str) -> KeyType:
        """
        Convert a key name string to a pynput Key or KeyCode.
        
        Args:
            key_name: Name of the key (e.g., 'a', 'escape', 'ctrl')
            
        Returns:
            pynput Key or KeyCode
        """
        # Check if it's a special key
        key_name = key_name.lower()
        if hasattr(Key, key_name):
            return getattr(Key, key_name)
        
        # Handle modifier keys
        if key_name in ['ctrl', 'control']:
            return Key.ctrl
        elif key_name in ['alt', 'option']:
            return Key.alt
        elif key_name in ['shift']:
            return Key.shift
        elif key_name in ['cmd', 'command']:
            return Key.cmd
        
        # Handle function keys
        if key_name.startswith('f') and key_name[1:].isdigit():
            fn_num = int(key_name[1:])
            if 1 <= fn_num <= 20:
                return getattr(Key, key_name)
        
        # Single character key
        if len(key_name) == 1:
            return key_name
        
        # Default to a KeyCode for other keys
        return KeyCode.from_char(key_name[0])
    
    def _get_human_delay(self) -> float:
        """
        Generate a human-like delay between actions.
        
        Returns:
            Delay in seconds
        """
        delay_ms = random.uniform(self._min_delay_ms, self._max_delay_ms)
        return delay_ms / 1000.0
    
    def _check_rate_limit(self) -> bool:
        """
        Check if we're exceeding the maximum action rate.
        
        Returns:
            True if the action is allowed, False if rate limited
        """
        current_time = time.time()
        
        # Reset counter after 1 second
        if current_time - self._action_count_reset_time >= 1.0:
            self._action_count = 0
            self._action_count_reset_time = current_time
        
        # Check if we're over the limit
        if self._action_count >= self._max_actions_per_second:
            warning(f"Rate limit exceeded: {self._action_count} actions per second", LogCategory.CONTROL)
            return False
        
        # Increment the counter
        self._action_count += 1
        return True

    def _execute_action(self, action: InputAction) -> None:
        """
        Execute a single input action based on its type.
        
        Args:
            action: Action to execute
        """
        # Skip execution in testing or disabled mode
        if self._mode == InputMode.DISABLED:
            info(f"Action skipped (DISABLED mode): {action.event_type.name}", LogCategory.CONTROL)
            return
        
        # Log the action
        if self._mode == InputMode.TESTING:
            info(f"Test action (no execution): {action.event_type.name} - {action.params}", 
                 LogCategory.CONTROL)
            return
        
        # Execute based on action type
        try:
            if action.event_type == InputEventType.KEY_PRESS:
                key = action.params.get('key')
                self._keyboard.press(key)
                self._pressed_keys.add(key)
                debug(f"Key press: {key}", LogCategory.CONTROL)
                
            elif action.event_type == InputEventType.KEY_RELEASE:
                key = action.params.get('key')
                self._keyboard.release(key)
                if key in self._pressed_keys:
                    self._pressed_keys.remove(key)
                debug(f"Key release: {key}", LogCategory.CONTROL)
                
            elif action.event_type == InputEventType.KEY_TAP:
                key = action.params.get('key')
                self._keyboard.press(key)
                time.sleep(0.05)  # Short delay between press and release
                self._keyboard.release(key)
                debug(f"Key tap: {key}", LogCategory.CONTROL)
                
            elif action.event_type == InputEventType.MOUSE_MOVE:
                x, y = action.params.get('position')
                duration = action.params.get('duration', 0.0)
                
                if duration > 0:
                    # Smooth movement over duration
                    start_x, start_y = self._mouse.position
                    steps = int(duration * 60)  # 60 updates per second
                    for i in range(1, steps + 1):
                        progress = i / steps
                        current_x = start_x + (x - start_x) * progress
                        current_y = start_y + (y - start_y) * progress
                        self._mouse.position = (current_x, current_y)
                        time.sleep(duration / steps)
                else:
                    # Instant movement
                    self._mouse.position = (x, y)
                    
                debug(f"Mouse move to: ({x}, {y})", LogCategory.CONTROL)
                
            elif action.event_type == InputEventType.MOUSE_CLICK:
                button = action.params.get('button', Button.left)
                count = action.params.get('count', 1)
                
                for _ in range(count):
                    self._mouse.press(button)
                    time.sleep(0.05)  # Short delay between press and release
                    self._mouse.release(button)
                    if count > 1:
                        time.sleep(0.1)  # Delay between multiple clicks
                        
                debug(f"Mouse click: {button} (count: {count})", LogCategory.CONTROL)
                
            elif action.event_type == InputEventType.MOUSE_SCROLL:
                dx = action.params.get('dx', 0)
                dy = action.params.get('dy', 0)
                self._mouse.scroll(dx, dy)
                debug(f"Mouse scroll: dx={dx}, dy={dy}", LogCategory.CONTROL)
                
            elif action.event_type == InputEventType.DELAY:
                duration = action.params.get('duration', 0.1)
                time.sleep(duration)
                debug(f"Delay: {duration}s", LogCategory.CONTROL)
                
            elif action.event_type == InputEventType.CUSTOM:
                callback = action.params.get('callback')
                if callback and callable(callback):
                    callback()
                    debug(f"Custom action executed", LogCategory.CONTROL)
                    
        except Exception as e:
            error(f"Error executing action {action.event_type.name}: {str(e)}", LogCategory.CONTROL)
            # Release all pressed keys for safety
            self.release_all_keys()
            
        # Update timestamp
        action.timestamp = time.time()
        
    def _worker(self) -> None:
        """
        Worker thread function to process the action queue.
        """
        while not self._stop_event.is_set() and self._active:
            try:
                # Get the next action, with a timeout to allow checking the stop event
                try:
                    # Get an action from the queue
                    action = self._action_queue.get(timeout=0.1)
                except Exception:
                    continue
                
                # Check rate limiting
                if not self._check_rate_limit():
                    # Re-add the action with a delay
                    time.sleep(0.1)
                    self._action_queue.put(action)
                    continue
                    
                # Execute the action
                self._execute_action(action)
                
                # Add a human-like delay before the next action
                time.sleep(self._get_human_delay())
                
                # Mark the task as done
                self._action_queue.task_done()
                    
            except Exception as e:
                error(f"Error in input worker thread: {str(e)}", LogCategory.CONTROL)
                
        # Release all keys when the worker stops
        self.release_all_keys()
        debug("Input worker thread stopped", LogCategory.CONTROL)

    # === Public API Methods ===
    
    def start(self) -> None:
        """Start processing the input queue."""
        with self._lock:
            if not self._active:
                self._active = True
                self._stop_event.clear()
                self._worker_thread = Thread(target=self._worker, daemon=True)
                self._worker_thread.start()
                info("Input simulator started", LogCategory.CONTROL)
    
    def stop(self) -> None:
        """Stop processing the input queue."""
        with self._lock:
            if self._active:
                self._active = False
                self._stop_event.set()
                
                if self._worker_thread and self._worker_thread.is_alive():
                    self._worker_thread.join(timeout=1.0)
                
                # Release all keys for safety
                self.release_all_keys()
                
                info("Input simulator stopped", LogCategory.CONTROL)
    
    def emergency_stop(self) -> None:
        """Emergency stop - clear the queue and stop all inputs."""
        warning("EMERGENCY STOP triggered", LogCategory.CONTROL)
        
        # Stop the queue processing
        self.stop()
        
        # Clear the action queue
        with self._lock:
            while not self._action_queue.empty():
                try:
                    self._action_queue.get_nowait()
                    self._action_queue.task_done()
                except Exception:
                    pass
        
        # Release all held keys
        self.release_all_keys()
    
    def set_mode(self, mode: InputMode) -> None:
        """Set the operating mode of the input simulator."""
        if self._mode != mode:
            info(f"Input simulator mode changed from {self._mode.name} to {mode.name}", LogCategory.CONTROL)
            self._mode = mode
    
    def get_mode(self) -> InputMode:
        """Get the current operating mode."""
        return self._mode
    
    def release_all_keys(self) -> None:
        """Release all currently pressed keys for safety."""
        if self._pressed_keys:
            for key in list(self._pressed_keys):
                try:
                    self._keyboard.release(key)
                except Exception as e:
                    warning(f"Error releasing key {key}: {str(e)}", LogCategory.CONTROL)
            
            self._pressed_keys.clear()
            info("All keys released", LogCategory.CONTROL)
    
    def is_active(self) -> bool:
        """Check if the input simulator is active."""
        return self._active
    
    def add_action(self, action: InputAction) -> None:
        """Add an action to the queue."""
        if self._mode == InputMode.DISABLED:
            debug(f"Action {action.event_type.name} ignored (DISABLED mode)", LogCategory.CONTROL)
            return
            
        with self._lock:
            self._action_queue.put(action)
            debug(f"Added {action.event_type.name} action to queue", LogCategory.CONTROL)
    
    # === Keyboard Input Methods ===
    
    def key_press(self, key: Union[str, KeyType], priority: int = 1) -> None:
        """Queue a key press action."""
        if isinstance(key, str) and len(key) == 1:
            key_obj = key
        else:
            key_obj = self._parse_key(key) if isinstance(key, str) else key
            
        action = InputAction(
            event_type=InputEventType.KEY_PRESS,
            params={'key': key_obj},
            priority=priority
        )
        self.add_action(action)
    
    def key_release(self, key: Union[str, KeyType], priority: int = 1) -> None:
        """Queue a key release action."""
        if isinstance(key, str) and len(key) == 1:
            key_obj = key
        else:
            key_obj = self._parse_key(key) if isinstance(key, str) else key
            
        action = InputAction(
            event_type=InputEventType.KEY_RELEASE,
            params={'key': key_obj},
            priority=priority
        )
        self.add_action(action)
    
    def key_tap(self, key: Union[str, KeyType], priority: int = 1) -> None:
        """Queue a key tap (press and release) action."""
        if isinstance(key, str) and len(key) == 1:
            key_obj = key
        else:
            key_obj = self._parse_key(key) if isinstance(key, str) else key
            
        action = InputAction(
            event_type=InputEventType.KEY_TAP,
            params={'key': key_obj},
            priority=priority
        )
        self.add_action(action)
    
    def key_combination(self, keys: List[Union[str, KeyType]], priority: int = 1) -> None:
        """Queue a key combination (hold multiple keys, then release)."""
        # Parse all keys first
        key_objs = []
        for key in keys:
            if isinstance(key, str) and len(key) == 1:
                key_objs.append(key)
            else:
                key_objs.append(self._parse_key(key) if isinstance(key, str) else key)
        
        # Add press actions for all keys
        for i, key_obj in enumerate(key_objs):
            self.key_press(key_obj, priority=priority)
            
        # Add a small delay
        self.delay(0.1, priority=priority)
        
        # Add release actions in reverse order
        for key_obj in reversed(key_objs):
            self.key_release(key_obj, priority=priority)
    
    def type_text(self, text: str, interval: float = None, priority: int = 1) -> None:
        """Queue typing a sequence of text."""
        if not text:
            return
            
        if interval is None:
            # Use random intervals for natural typing
            min_interval = get_config('input.typing.min_interval', 0.05) 
            max_interval = get_config('input.typing.max_interval', 0.15)
        else:
            min_interval = max_interval = interval
            
        for char in text:
            self.key_tap(char, priority=priority)
            delay = random.uniform(min_interval, max_interval)
            self.delay(delay, priority=priority)
    
    # === Mouse Input Methods ===
    
    def mouse_move(self, x: int, y: int, duration: float = 0.0, priority: int = 1) -> None:
        """Queue a mouse movement action."""
        action = InputAction(
            event_type=InputEventType.MOUSE_MOVE,
            params={'position': (x, y), 'duration': duration},
            priority=priority
        )
        self.add_action(action)
    
    def mouse_move_relative(self, dx: int, dy: int, priority: int = 1) -> None:
        """Queue a relative mouse movement action."""
        current_pos = self._mouse.position
        x = current_pos[0] + dx
        y = current_pos[1] + dy
        self.mouse_move(x, y, priority=priority)
    
    def mouse_click(self, button: Button = Button.left, count: int = 1, priority: int = 1) -> None:
        """Queue a mouse click action."""
        action = InputAction(
            event_type=InputEventType.MOUSE_CLICK,
            params={'button': button, 'count': count},
            priority=priority
        )
        self.add_action(action)
    
    def mouse_click_at(self, x: int, y: int, button: Button = Button.left, 
                     move_duration: float = 0.2, priority: int = 1) -> None:
        """Queue a mouse move followed by a click action."""
        # Move to position
        self.mouse_move(x, y, duration=move_duration, priority=priority)
        
        # Small delay before clicking
        self.delay(0.05, priority=priority)
        
        # Click
        self.mouse_click(button, priority=priority)
    
    def mouse_scroll(self, dx: int = 0, dy: int = 0, priority: int = 1) -> None:
        """Queue a mouse scroll action."""
        action = InputAction(
            event_type=InputEventType.MOUSE_SCROLL,
            params={'dx': dx, 'dy': dy},
            priority=priority
        )
        self.add_action(action)
    
    # === Utility Methods ===
    
    def delay(self, duration: float, priority: int = 1) -> None:
        """Queue a delay action."""
        action = InputAction(
            event_type=InputEventType.DELAY,
            params={'duration': duration},
            priority=priority
        )
        self.add_action(action)
    
    def custom_action(self, callback: Callable, priority: int = 1) -> None:
        """Queue a custom action using a callback function."""
        action = InputAction(
            event_type=InputEventType.CUSTOM,
            params={'callback': callback},
            priority=priority
        )
        self.add_action(action)
    
    # === Game-Specific Actions ===
    
    def press_ability_key(self, key: Union[str, int], priority: int = 1) -> None:
        """Press a game ability key (1-9, or specific key)."""
        if isinstance(key, int) and 1 <= key <= 9:
            key = str(key)
        self.key_tap(key, priority=priority)
    
    def target_by_tab(self, priority: int = 1) -> None:
        """Use tab targeting to find a target."""
        self.key_tap(Key.tab, priority=priority)
    
    def interact_with_target(self, priority: int = 1) -> None:
        """Interact with the current target (default bind is usually 'T')."""
        interact_key = get_config('wow.keybinds.interact', 't')
        self.key_tap(interact_key, priority=priority)
    
    def cast_spell_by_name(self, spell_name: str, priority: int = 1) -> None:
        """Cast a spell by name using the /cast command."""
        command = f"/cast {spell_name}"
        self.type_text(command, priority=priority)
        self.key_tap(Key.enter, priority=priority)
    
    def use_movement_key(self, direction: str, duration: float, priority: int = 1) -> None:
        """Hold a movement key for a duration."""
        # Map direction to key
        direction = direction.lower()
        if direction == 'forward':
            key = get_config('wow.keybinds.movement.forward', 'w')
        elif direction == 'backward':
            key = get_config('wow.keybinds.movement.backward', 's')
        elif direction == 'left':
            key = get_config('wow.keybinds.movement.strafe_left', 'a')
        elif direction == 'right':
            key = get_config('wow.keybinds.movement.strafe_right', 'd')
        elif direction == 'jump':
            key = get_config('wow.keybinds.movement.jump', 'space')
        else:
            warning(f"Unknown movement direction: {direction}", LogCategory.CONTROL)
            return
        
        # Press key, wait, then release
        self.key_press(key, priority=priority)
        self.delay(duration, priority=priority)
        self.key_release(key, priority=priority)
    
    def camera_rotation(self, dx: int, speed: float = 0.5, priority: int = 1) -> None:
        """Rotate the camera using right mouse button."""
        # Start rotation (right mouse button down)
        self.mouse_click(Button.right, priority=priority)
        
        # Move mouse to rotate
        self.mouse_move_relative(dx, 0, priority=priority)
        
        # End rotation (right mouse button up)
        self.mouse_click(Button.right, priority=priority)
    
    def jump(self, priority: int = 1) -> None:
        """Make the character jump."""
        jump_key = get_config('wow.keybinds.movement.jump', 'space')
        self.key_tap(jump_key, priority=priority)
    
    def open_game_menu(self, menu_name: str, priority: int = 1) -> None:
        """Open a game menu by its keybind."""
        menu_name = menu_name.lower()
        menu_keys = {
            'character': get_config('wow.keybinds.menus.character', 'c'),
            'spellbook': get_config('wow.keybinds.menus.spellbook', 'p'),
            'talents': get_config('wow.keybinds.menus.talents', 'n'),
            'quest_log': get_config('wow.keybinds.menus.quest_log', 'l'),
            'map': get_config('wow.keybinds.menus.map', 'm'),
            'bags': get_config('wow.keybinds.menus.bags', 'b')
        }
        
        key = menu_keys.get(menu_name)
        if key:
            self.key_tap(key, priority=priority)
        else:
            warning(f"Unknown game menu: {menu_name}", LogCategory.CONTROL)
    
    def toggle_auto_run(self, priority: int = 1) -> None:
        """Toggle auto-run."""
        auto_run_key = get_config('wow.keybinds.movement.auto_run', 'num_lock')
        self.key_tap(auto_run_key, priority=priority)


# Create a singleton instance
input_simulator = InputSimulator()
