# TownSim Python

A town/city generator and growth simulator built with Python and PyQt6.

## Overview

TownSim Python generates realistic terrain and creates logical city layouts that evolve over time through sophisticated simulation algorithms. The application features a modern GUI interface with zoom/pan capabilities, event tracking, and multiple export options.

## Features

### Core Functionality
- **Terrain Generation**: User-configurable terrain with multiple parameters (land/water ratio, rivers, elevation, ground types, vegetation)
- **Logical City Layout**: Smart building placement based on terrain constraints and urban planning principles
- **Time-based Simulation**: Growth simulation with configurable time scales (1-100 years per frame)
- **Event Tracking**: Complete history of all changes with ID-based feature storage
- **Road-led Expansion**: Realistic growth patterns where infrastructure leads development

### User Interface
- **PyQt6 GUI**: Professional interface with zoom/pan map view
- **Real-time Controls**: Interactive parameter adjustment and simulation controls
- **Animation System**: Time-lapse playback of city growth
- **User Approval Workflow**: Review and approve terrain and initial city before simulation

### Data Management
- **Dual Save System**: SQLite databases for saves, JSON for sharing
- **Multiple Export Formats**: Animations, snapshots, JSON data, proprietary format
- **Complete Feature Tracking**: Every change recorded with year, type, and full feature data

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone https://github.com/WolfyD/townsim_py.git
cd townsim_py
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python main.py
```

## Usage

### GUI Mode (Default)
```bash
python main.py
```

### CLI Mode
```bash
python main.py --cli
```

### Command Line Options
- `--cli`: Run in command-line interface mode
- `--debug`: Enable debug logging
- `--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}`: Set logging level
- `--load <filepath>`: Load a saved simulation on startup

## Project Structure

```
townsim_py/
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
├── context.md             # Project context and specifications
├── README.md              # This file
├── logs/                  # Log files (auto-created)
├── tests/                 # Test files
└── src/
    └── townsim/
        ├── __init__.py
        └── modules/
            ├── ui/           # User interface components
            ├── simulation/   # Core simulation logic
            ├── terrain/      # Terrain generation
            ├── events/       # Event tracking system
            ├── buildings/    # Building placement logic
            ├── data/         # Data management and persistence
            └── utils/        # Utility functions and logging
```

## Development

### Architecture
The project follows a modular architecture with clear separation of concerns:

- **UI Module**: All interface components and user interaction
- **Simulation Module**: Core simulation logic and time progression
- **Terrain Module**: Terrain generation algorithms and management
- **Events Module**: Event tracking and feature management
- **Buildings Module**: Building placement logic and constraints
- **Data Module**: Save/load functionality and data persistence
- **Utils Module**: Logging, configuration, and helper functions

### Logging
- Automatic daily log rotation
- Configurable log levels
- Structured logging with timestamps
- Logs stored in `logs/` directory

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/townsim
```

## Contributing

### Code Style
- Follow PEP8 guidelines
- Use type hints for all public functions
- Add docstrings to all classes and public methods
- Keep functions small and single-purpose

### Development Workflow
1. Create feature branch
2. Write tests for new functionality
3. Implement features following the cursor rules
4. Run tests and ensure they pass
5. Submit pull request

## License

[Add your license here]

## Roadmap

### Version 0.1.0 (Current)
- [x] Basic project structure
- [x] GUI framework setup
- [x] Logging system
- [ ] Terrain generation
- [ ] Event tracking system
- [ ] Building placement logic
- [ ] Save/load functionality

### Version 0.2.0 (Planned)
- [ ] Advanced terrain features
- [ ] Random events system
- [ ] Animation export
- [ ] Performance optimizations

### Version 0.3.0 (Future)
- [ ] Custom textures and themes
- [ ] Advanced simulation parameters
- [ ] Multi-simulation comparison
- [ ] Plugin system

## Support

For questions, bug reports, or feature requests, please [create an issue](link-to-issues).

## Acknowledgments

Built with:
- Python 3.8+
- PyQt6 for GUI
- NumPy for numerical operations
- Pillow for image processing
- noise for terrain generation
- SciPy for advanced mathematics 