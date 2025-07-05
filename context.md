# TownSim Python - Project Context

## Project Overview
A Python-based town/city generator and growth simulator with GUI interface that creates realistic settlements and simulates their development over time.

## Core Workflow
1. **Terrain Generation** → User-configurable terrain generation
2. **User Approval** → User can accept or regenerate terrain
3. **Initial City Layout** → Generate year 0 city on approved terrain
4. **User Approval** → User can accept or regenerate initial city
5. **Time-based Simulation** → Growth simulation with events and animations

## Technical Requirements

### GUI Framework
- **PyQt6** - chosen for graphics performance, animation support, zoom/pan functionality
- Primary interface with GUI, CLI support as secondary

### Performance Requirements
- Quality over speed, but keep generation under 30-60 seconds
- Smooth animations for time-lapse playback
- Efficient rendering for large-scale maps

### Grid System
- Start with large grid (e.g., 1000x1000) to accommodate growth
- Zoom functionality for detail viewing
- Scalable from small village to massive urban expanse

## Feature Specifications

### Terrain Generation
**Terrain Types (Basic Set):**
- Sand, Grass, Mud, Rock, Sea, Coast, Snow
- Extensible for future terrain types

**User-Configurable Parameters:**
- Land/water ratio
- Rivers placement
- Elevation variations
- Ground type distribution
- Vegetation density and types

### City Layout & Building System
**Building Categories:**
- Residential, Commercial, Industrial, Civic, Infrastructure, Farms

**Placement Logic (Multi-Layer Constraint System):**
- **Layer 1: Terrain Constraints**
  - Water accessibility → Docks only near water
  - Fertile land → Farms on grass/suitable terrain
  - Resource proximity → Industrial near resources/transport
  - Elevation → Buildings avoid extreme slopes
- **Layer 2: Infrastructure Dependencies**
  - Road networks → Buildings connect to existing roads
  - Utilities → Power, water supply proximity
  - Transportation → Access to main routes
- **Layer 3: Zoning Logic**
  - Residential zones → Safe, accessible, away from industry
  - Commercial hubs → Central locations, high traffic
  - Industrial districts → Near resources, transport, away from residential
- **Layer 4: Growth Patterns**
  - **Road-led expansion** → New roads built first, then buildings follow
  - Organic expansion → Buildings cluster around existing structures
  - Economic drivers → Growth follows trade routes, resources
  - Population pressure → Density increases with population

### Event-Driven Simulation System
**Event Tracking:**
- **Every change** gets tracked with year and Add/Remove status
- **Complete Feature Data Storage:**
  - Features stored in array with unique IDs
  - Events reference feature IDs (not duplicate data)
  - Each feature contains: location, type, properties, materials, style, etc.
- Examples:
  - Road added: `{year: 30, type: "Add", feature_id: "road_001"}`
  - Building demolished + road built: `{year: 45, type: "Remove", feature_id: "building_023"}` + `{year: 45, type: "Add", feature_id: "road_047"}`
- Track: roads, buildings, bridges, terrain changes, paving, demolitions, style changes

**Feature Data Structure:**
```json
{
  "features": [
    {
      "id": "road_001",
      "type": "road",
      "location": {"x": 100, "y": 200},
      "properties": {"material": "cobblestone", "width": 2},
      "style": "medieval"
    }
  ],
  "events": [
    {"year": 30, "type": "Add", "feature_id": "road_001"}
  ]
}
```

**Growth Factors:**
- Population growth
- Resource availability
- Trade route development
- **Random Events System:**
  - User-configurable event list
  - Events: resource fluctuation, population influx, famine, war, floods, etc.
  - Events displayed to user for removal/modification
  - Events influence simulation direction

### Time Scale System
**User-Configurable Time Scale:**
- Range: 1 year per frame (1ypf) to 100 years per frame (100ypf)
- Buildings/features appear based on creation year
- Timeline playback regardless of viewing scale

### Visual System
**Style:** Top-down 2D view
- Simple, clean representation
- Texture support for future enhancement
- Zoom/pan functionality
- Animation support for time-lapse

## Data Management

### Save System
**Two Save Formats:**
1. **SQLite Database** - General save files with full simulation state
2. **Exportable JSON** - Sharing format for simulations

**Data Structure:**
- Track creation year for every feature (relative to simulation start = year 0)
- Complete event history for accurate timeline reconstruction
- User settings and parameters
- Terrain data and generation settings

### Export Capabilities
- **Animations** - Time-lapse videos of city growth
- **Snapshots** - Images from specific time periods
- **JSON Data** - Structured data about layout and events
- **Proprietary Format** - Full simulation save for reloading

## Architecture Requirements

### Modular Design
**Separate Classes/Modules:**
- **UI Module** - All interface components
- **Simulation Module** - Core simulation logic
- **Terrain Module** - Terrain generation and management
- **Import/Export Module** - File handling and data conversion
- **Event System** - Event tracking and management
- **Building System** - Structure placement and management
- **Random Events** - Event generation and management

### Code Quality
- Follow cursor rules for clean, modular Python code
- Easy customization (custom textures, recoloring, new features)
- Robust error handling with rotating daily logs
- Test-driven development support
- Type hints for all public functions
- Comprehensive docstrings

## User Interface Features

### Main Interface
- Terrain generation parameter controls
- Simulation speed controls
- Time scale selection (1ypf to 100ypf)
- Zoom/pan map view
- Animation playback controls

### User Controls
- **Terrain Parameters** - All generation settings
- **Random Events** - View/edit event list
- **Time Controls** - Speed, scale, jump to specific years
- **Export Options** - Choose output formats
- **Save/Load** - Manage simulation files

## Success Criteria
- Generate varied, realistic terrain based on user parameters
- Create logical city layouts that make sense for the terrain
- Smooth animation playback at various time scales
- Modular codebase allowing easy feature additions
- Professional, intuitive GUI interface
- Robust save/load system with multiple export formats
- Comprehensive event tracking for accurate simulation playback

## Future Extensibility
- New terrain types
- Custom building textures
- Additional random events
- New building types
- Enhanced visual effects
- Advanced terrain features
- Multiplayer or comparison modes 