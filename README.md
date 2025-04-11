# World of Warcraft Bot

## Overview

This is an educational project designed to automate gameplay activities in World of Warcraft using computer vision, machine learning, and input simulation. The project is specifically optimized for macOS with M1 processors.

⚠️ **Educational Purpose Only**: This project is intended for educational purposes to explore techniques in computer vision, machine learning, and automation. Please be aware that using automation tools with World of Warcraft may violate the game's Terms of Service.

## Features

- **Combat Automation**: Autonomous execution of combat rotations for Beast Mastery Hunter
- **Navigation**: Basic movement and pathfinding in the game world
- **Error Recovery**: Tiered system for handling and recovering from errors
- **Command Interface**: Simple CLI for controlling the bot

## Architecture

The project follows a modular architecture with three main components:

1. **Control Module**: Handles screen capture, image processing, and input simulation
2. **Decision Module**: Manages game state, combat logic, and navigation
3. **Learning Module**: Provides machine learning capabilities for object detection and decision making

## Setup Instructions

### System Requirements

- macOS 12.0+ (Monterey or newer)
- Python 3.10+ (native ARM64 build)
- Node.js 18+ (for TypeScript components)
- World of Warcraft client

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/wow-bot.git
   cd wow-bot
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install Node.js dependencies (for CLI):
   ```bash
   cd cli
   npm install
   npm run build
   cd ..
   ```

5. Configure the bot:
   ```bash
   cp src/configs/default_config.yml src/configs/config.yml
   # Edit config.yml with your specific settings
   ```

### Running the Bot

1. Ensure WoW is running in windowed mode at the specified resolution in your config
2. Activate the virtual environment if not already active:
   ```bash
   source venv/bin/activate
   ```
3. Run the bot:
   ```bash
   python -m src.core.main
   ```

## Project Structure

```
wow-bot/
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore file
├── src/                # Main package
│   ├── core/               # Core functionality
│   ├── control/            # Screen capture and input simulation
│   ├── decision/           # Game state and decision making
│   ├── learning/           # Machine learning models
│   ├── cli/                # Command-line interface
│   ├── utils/              # Utility functions
│   ├── configs/            # Configuration files
│   └── data/               # Data storage
│       ├── logs/           # Log files
│       ├── screenshots/    # Captured screenshots
│       └── models/         # ML model files
└── tests/                  # Unit and integration tests
```

## Development

### Testing

Run tests using pytest:
```bash
pytest
```

### Code Style

This project uses Black for code formatting and isort for import sorting:
```bash
black src
isort src
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project is for educational purposes only
- Special thanks to the open-source computer vision and machine learning communities
