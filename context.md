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
**Scale:** Settlement-appropriate areas (not continental scale)

**Settlement Scale Reference:**
- **Village**: <1,000 people, 5–15 km²
- **Town**: 1,000–20,000 people, 15–50 km²
- **City**: 20,000–500,000 people, 50–250 km²
- **Urban/Metro Area**: >500,000 people, 250–2,000+ km²

**Tile Scale:** Each tile represents approximately 6-8 meters, allowing for:
- Individual buildings (1-4 tiles)
- Roads and paths (1-2 tiles wide)
- Parks and plazas (10-50 tiles)
- Realistic urban density and layout

**Terrain Types (Town Scale):**
- **Grass**: General grassland/fields
- **Mud**: Wetlands near water bodies
- **Sand**: Dry areas, riverbanks, beaches
- **Rock**: Steep slopes, exposed ridges
- **Coast**: Edges of water bodies (lakes, rivers)
- **Sea**: Lakes, ponds, rivers, streams
- **Snow**: High elevation areas (if cold climate)
- Extensible for future terrain types

**User-Configurable Parameters:**

**Core Terrain Settings:**
- **Map Size**: Settlement-appropriate sizes (powers of 2 for noise optimization)
  - Village: 512×512–768×768 (5-15 km²)
  - Town: 768×768–1024×1024 (15-50 km²)  
  - City: 1024×1024–2048×2048 (50-250 km²)
  - Metro: 2048×2048–4096×4096+ (250-2000+ km²)
- **Elevation Variance**: 0.0-1.0 (flat plains to rolling hills - appropriate for settlement scale)
- **Water Bodies**: 0-10 (number of lakes, ponds, major streams - scales with settlement size)
- **Noise Scale**: 0.1-10.0 (size of terrain features)
- **Random Seed**: Integer for reproducible generation

**Climate Settings (Local Scale):**
- **Base Temperature**: 0.0-1.0 (cold/temperate/warm climate)
- **Moisture Level**: 0.0-1.0 (general wetness/dryness)
- **Wind Direction**: 0-360° (compass control with arrow - North=0°, East=90°, etc.)
- **Wind Strength**: 0.0-1.0 (how much wind affects local weather patterns)

**Generation Algorithm:**

**Three-Layer Noise System:**
```python
# Layer 1: Elevation (height map for hills/valleys)
elevation_map = perlin_noise(scale=noise_scale, octaves=4)

# Layer 2: Moisture (affected by water proximity + wind)
moisture_map = perlin_noise(scale=noise_scale*0.7, octaves=3)

# Layer 3: Temperature (elevation + coastal + wind effects)
temperature_map = base_temperature - (elevation * lapse_rate) + wind_effects
```

**Wind System:**
- **Wind Compass UI**: Circular compass control allowing user to set prevailing wind direction
- **Orographic Effects**: Windward slopes get more moisture, leeward slopes create rain shadows
- **Temperature Moderation**: Wind brings maritime influence from water bodies
- **Realistic Scale**: Wind effects appropriate for town-scale geography

**Water Flow System:**
- **Downhill Flow**: Rivers and streams follow elevation gradients downhill
- **Lake Connections**: Streams flow into and out of lakes realistically
- **Drainage Networks**: Connected waterways that make hydrological sense
- **Coastal Features**: Realistic shorelines and wetlands around water bodies

**Terrain Assignment Logic:**
```python
def determine_terrain_type(elevation, moisture, temperature, distance_to_water):
    if elevation < water_level:
        return TERRAIN_SEA
    elif distance_to_water < 0.05:  # Near water
        return TERRAIN_COAST
    elif temperature < 0.2:  # Cold areas
        return TERRAIN_SNOW
    elif moisture < 0.3:  # Dry areas
        if temperature > 0.7:
            return TERRAIN_SAND  # Hot + dry
        else:
            return TERRAIN_ROCK  # Cold + dry = rocky
    elif moisture > 0.8:  # Very wet areas
        return TERRAIN_MUD  # Wetlands
    else:
        return TERRAIN_GRASS  # Default grassland
```

**Generation Phases:**
1. **Basic Elevation**: Generate hills, valleys, and ridges using Perlin noise
2. **Water Placement**: Position lakes, ponds, and water sources
3. **Wind Effects**: Apply orographic and temperature effects based on wind direction
4. **River Generation**: Create realistic downhill water flow networks
5. **Terrain Assignment**: Combine all factors to determine final terrain types
6. **Post-Processing**: Smooth transitions and validate realistic placement

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