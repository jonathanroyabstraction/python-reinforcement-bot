# World of Warcraft Bot - Comprehensive Specification

## Project Overview

This document outlines the complete specification for an educational World of Warcraft bot project designed to automate gameplay activities using computer vision, machine learning, and input simulation. The system focuses on a modular architecture that can be extended to support various in-game activities while maintaining performance on macOS.

## 1. Core Functionality Requirements

### 1.1 Gameplay Capabilities
- **Combat Automation**: Autonomous execution of combat rotations
- **Questing**: Navigation and completion of quest objectives
- **Dungeon Running**: Simple dungeon navigation and completion
- **Character Focus**: Initial focus on DPS roles, specifically Beast Mastery Hunter
- **Explicitly Excluded**: Farming resources (not part of initial scope)

### 1.2 Bot Autonomy
- **Initial Phase**: Command-driven where the bot is told what to do
- **Later Phase**: Priority-based autonomous operation following a list of tasks
- **Basic Commands**:
  * "Go kill nearest monster"
  * "Do all the quests you cross" 
  * "Queue to X dungeon and do it"
  * "Stop"

### 1.3 Error Handling and Recovery
- **Tiered Recovery System**:
  1. **Tier 1 (Minor)**: Reorient and retry
  2. **Tier 2 (Medium)**: Temporarily abandon task and try simpler task
  3. **Tier 3 (Major)**: Return to safe zone
  4. **Tier 4 (Critical)**: Use hearthstone as last resort
- **Logging**: All errors and recovery attempts logged for analysis

## 2. Technical Architecture

### 2.1 High-Level Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface Layer                      │
│                                                                 │
│  CLI Interface (initial)      Web Dashboard (future)            │
│  [TypeScript/Node.js]         [TypeScript/React]                │
└──────────────────────────────┬──────────────────────────────────┘
                              │
                              │ REST API/WebSockets
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Bot Orchestration Layer                     │
│                         [Python/FastAPI]                         │
│                                                                 │
│  Command Processing        Task Scheduling       State Management│
└──────────────┬─────────────────────┬───────────────────┬────────┘
               │                     │                   │
    ┌──────────▼──────────┐ ┌────────▼─────────┐ ┌──────▼─────────┐
    │   Control Module    │ │  Decision Module │ │ Learning Module │
    │     [Python]        │ │    [Python]      │ │    [Python]     │
    │                     │ │                  │ │                 │
    │ • Screen Capture    │ │ • Task Selection │ │ • Model Training│
    │   (mss)             │ │ • Path Planning  │ │   (PyTorch)     │
    │ • Input Simulation  │ │ • Combat Logic   │ │ • Reinforcement │
    │   (pynput, pyobjc)  │ │ • Error Handling │ │   Learning      │
    │ • Game State        │ │ • Quest Tracking │ │ • Data Pipeline │
    │   Recognition       │ │                  │ │ • Model         │
    │ • UI Element        │◄┼─────────────────►│ │  Versioning     │
    │   Detection         │ │                  │ │                 │
    └─────────┬───────────┘ └─────────┬────────┘ └────────┬────────┘
              │                       │                    │
              │                       │                    │
              ▼                       ▼                    ▼
    ┌───────────────────────────────────────────────────────────────┐
    │                      Data Storage Layer                        │
    │                                                               │
    │  Game State DB       Training Data Storage     Model Registry  │
    │  (SQLite/MongoDB)    (File System/S3)         (MLflow)         │
    └───────────────────────────────────────────────────────────────┘
```

### 2.2 Technology Stack
- **Core Bot Framework**: Python
  * Computer vision: OpenCV, PyTorch
  * Screen capture: mss
  * Input simulation: pynput, pyobjc
  * API framework: FastAPI

- **Web Interface & API Layer**: TypeScript/Node.js
  * Frontend: React (future)
  * API: Express.js/Nest.js

- **Data Storage**:
  * SQLite/MongoDB for game state
  * File system for training data
  * MLflow for model registry

### 2.3 Game Interaction Method
- **Primary**: Screen capture and pixel analysis
  * Uses computer vision for UI element detection
  * OCR for text recognition

- **Secondary**: Addon/API integration
  * WoW's official API through custom addons
  * Access to combat logs and structured data

### 2.4 Module Responsibilities

#### Control Module
- Screen capture and processing
- Mouse and keyboard input simulation
- UI element detection
- Game state recognition
- Low-level macOS integration

#### Decision Module
- Task selection and prioritization
- Path planning and navigation
- Combat rotation logic
- Error detection and recovery strategies
- Quest tracking and objective identification

#### Learning Module
- Model training infrastructure
- Class-specific models for Hunter (initially)
- Reinforcement learning systems
- Data processing pipeline
- Model versioning and storage

## 3. MVP Requirements and Timeline

### 3.1 MVP Scope
- **Combat Capabilities**:
  * Ability to fight autonomously as Beast Mastery Hunter
  * Monster recognition
  * DPS rotation execution
  * Looting ability

- **Survival and Navigation**:
  * Survive for 1 hour in monster-populated zone
  * Fluid movement capabilities
  * Stuck state detection
  * Basic error recovery

- **Interface**:
  * Basic CLI with command set
  * Configuration system
  * Basic analytics dashboard
  * Health monitoring (bot and system)

- **Data Collection**:
  * Structured storage for screenshots and observations
  * Basic preprocessing pipeline

### 3.2 Development Phases

#### Phase 1: Foundation
- Basic control system setup
- Screen capture and processing
- Input simulation
- Initial data collection framework

#### Phase 2: Core Mechanics
- Monster detection
- Basic navigation
- Combat rotation implementation
- Loot functionality

#### Phase 3: Intelligence
- Error detection and recovery
- Stuck state detection
- Basic decision making
- Task prioritization

#### Phase 4: Interface & Training
- CLI implementation
- Model training pipeline
- Testing and refinement
- Documentation

### 3.3 Additional MVP Components
- Configuration system for key settings
- Basic analytics dashboard
- Data collection pipeline
- Installation/setup documentation
- Health monitoring system

## 4. Testing Strategy

### 4.1 Component-Level Testing

#### Control Module Testing
- Unit tests for input simulation functions
- Image recognition accuracy tests
- Latency testing for screen capture

#### Decision Module Testing
- Unit tests for combat rotation logic
- Pathfinding algorithm validation
- Mock state transitions for error recovery

#### Learning Module Testing
- Model accuracy metrics on test data
- Inference speed benchmarks
- Model drift detection

### 4.2 Integration Testing
- Module interaction tests
- System resource tests
- End-to-end data flow testing

### 4.3 Scenario-Based Testing
- Controlled environment tests in isolated areas
- Navigation through known terrain with obstacles
- Deliberate error state induction and recovery
- Timed survival runs in monster-populated zones

### 4.4 Evaluation Metrics
- Combat performance (DPS, target acquisition)
- Navigation performance (collision frequency, path optimization)
- Error recovery success rate and time
- Overall survival time in challenging environments

### 4.5 Testing Environment
- Controlled test environment (preferably on test server if available)
- Low-level zones for initial testing
- Progressively more complex zones for advanced testing
- Long-duration stability testing

## 5. Logging and Debugging

### 5.1 Logging Architecture
- **Multi-level Logging**: Trace, Debug, Info, Warning, Error, Critical
- **Specialized Categories**: Control, Combat, Navigation, ML, System
- **Storage and Rotation**: Time-based rotation with compression

### 5.2 Debugging Tools
- Visual debugging interface with overlays
- State inspection tools
- Performance profiling
- Runtime state inspector

### 5.3 Error Detection and Recovery
- Anomaly detection for unusual behavior
- Self-diagnostic capabilities
- Recovery instrumentation and metrics

### 5.4 Development Aids
- Replay system for testing scenarios
- Mock environment for testing without WoW running
- Remote debugging capabilities

## 6. Performance Requirements

### 6.1 System Requirements
- **Target Hardware**: Mac with M1 processor
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 10GB minimum free space
- **Network**: Stable internet connection (min 5Mbps/1Mbps)

### 6.2 Resource Utilization Targets
- **CPU Usage**: Max 30% during normal operation
- **Memory**: Max 2GB RAM footprint
- **GPU**: Limited to 20% of resources
- **Disk I/O**: Minimize continuous writes

### 6.3 Response Time Requirements
- Screen capture & analysis: < 50ms per frame
- Combat decision making: < 100ms
- Input simulation: < 20ms latency
- Stuck detection: < 2 seconds
- Error recovery initiation: < 1 second

### 6.4 Optimization Strategies
- Throttle screen capture (max 20 FPS)
- Region-of-interest processing
- Model quantization
- Dynamic resource allocation
- Real-time performance monitoring

## 7. Extensibility Framework

### 7.1 Plugin Architecture
- Core plugin system with registry
- Standardized interfaces for extensions
- Hot-swapping capability
- Extension points for class modules, activities, UI detectors

### 7.2 API Design
- Stable core API with backward compatibility
- Event system (publish-subscribe)
- Configuration and state management APIs
- Developer tools including template generators

### 7.3 Data Exchange Standards
- Structured formats for state, commands, telemetry
- Plugin-specific storage with isolation
- Migration tools for data structure evolution

### 7.4 Future-proofing Strategies
- Compatibility layers for game versions
- Support for multiple ML model versions
- Expansion planning for multi-account support

## 8. Data Collection for Training

### 8.1 Environment Data
- Screenshots at regular intervals
- Minimap state and positions
- Combat logs (available through WoW API)
- Quest state information

### 8.2 Action Data
- Bot decisions and context
- Ability usage sequences and timing
- Movement paths and navigation decisions
- Success/failure of individual actions

### 8.3 Performance Metrics
- DPS output during combat
- Quest completion time
- Survival rate
- Experience gained per hour

### 8.4 Error States
- Screenshots of error conditions
- Sequences leading to failures
- Recovery attempts and outcomes

### 8.5 User Interactions
- Commands issued
- Manual overrides
- Settings adjustments

## 9. Setup and Installation

### 9.1 System Prerequisites
- Mac with M1 processor
- macOS 12.0+ (Monterey or newer)
- Python 3.10+ (ARM64 build)
- Node.js 18+ (for TypeScript components)
- World of Warcraft client with active subscription
- XCode Command Line Tools

### 9.2 Development Environment Setup
- Python virtual environment with dependencies
- Node.js environment for TypeScript components
- VS Code with recommended extensions

### 9.3 WoW Client Configuration
- Resolution: 1920x1080 (windowed mode)
- UI Scale: 1.0
- Nameplates: Enabled for enemies
- Standard keybind configuration

### 9.4 First-Run Calibration
- Screen region calibration process
- Model initialization
- Pre-trained base model download

### 9.5 Troubleshooting Guidelines
- Graphics recognition issues
- Control and input problems
- Environment and dependency troubleshooting

## 10. Security Considerations

### 10.1 Educational Purpose Statement
- Clear documentation that the project is for educational purposes only
- Acknowledgment of Blizzard's Terms of Service
- Recommendation to use only on private servers if testing with actual game

### 10.2 Data Security
- No storage of account credentials
- Anonymization of any collected data
- Local-only operation (no remote control)

### 10.3 System Protection
- Resource throttling to prevent system damage
- Safe shutdown procedures
- Protection against accidental macro execution

---

## Appendix A: Glossary

- **Bot**: The automated system controlling a World of Warcraft character
- **DPS**: Damage Per Second, a measure of combat effectiveness
- **MVP**: Minimum Viable Product, the initial feature set
- **Control Module**: Component responsible for interacting with the game
- **Decision Module**: Component responsible for game strategy
- **Learning Module**: Component responsible for ML model training
- **Computer Vision**: Technology for interpreting visual information
- **OCR**: Optical Character Recognition for reading text from images
- **ML**: Machine Learning
