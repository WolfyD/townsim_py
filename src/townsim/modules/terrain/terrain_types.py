"""
Terrain Types and Related Classes

Defines the different terrain types and their properties for town-scale generation.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple


class TerrainType(Enum):
    """Terrain types appropriate for town/city scale."""
    
    GRASS = "grass"          # General grassland/fields
    MUD = "mud"              # Wetlands near water bodies
    SAND = "sand"            # Dry areas, riverbanks, beaches
    ROCK = "rock"            # Steep slopes, exposed ridges
    COAST = "coast"          # Edges of water bodies
    SEA = "sea"              # Lakes, ponds, rivers, streams
    SNOW = "snow"            # High elevation areas (cold climate)


@dataclass
class TerrainProperties:
    """Properties for each terrain type."""
    
    name: str
    color: Tuple[int, int, int]  # RGB color for visualization
    buildable: bool              # Can buildings be placed here?
    traversable: bool            # Can units move through?
    fertile: bool                # Good for farms?
    elevation_preference: float  # Preferred elevation (0.0-1.0)
    moisture_preference: float   # Preferred moisture (0.0-1.0)
    temperature_preference: float # Preferred temperature (0.0-1.0)


# Terrain type properties lookup
TERRAIN_PROPERTIES = {
    TerrainType.GRASS: TerrainProperties(
        name="Grassland",
        color=(80, 150, 80),     # Green
        buildable=True,
        traversable=True,
        fertile=True,
        elevation_preference=0.4,
        moisture_preference=0.6,
        temperature_preference=0.5
    ),
    TerrainType.MUD: TerrainProperties(
        name="Wetlands",
        color=(120, 100, 60),    # Brown-green
        buildable=False,
        traversable=False,
        fertile=False,
        elevation_preference=0.2,
        moisture_preference=0.9,
        temperature_preference=0.5
    ),
    TerrainType.SAND: TerrainProperties(
        name="Sand",
        color=(200, 180, 120),   # Tan
        buildable=True,
        traversable=True,
        fertile=False,
        elevation_preference=0.3,
        moisture_preference=0.2,
        temperature_preference=0.7
    ),
    TerrainType.ROCK: TerrainProperties(
        name="Rocky",
        color=(120, 120, 120),   # Gray
        buildable=False,
        traversable=True,
        fertile=False,
        elevation_preference=0.8,
        moisture_preference=0.3,
        temperature_preference=0.4
    ),
    TerrainType.COAST: TerrainProperties(
        name="Coast",
        color=(180, 180, 100),   # Sandy yellow
        buildable=True,
        traversable=True,
        fertile=False,
        elevation_preference=0.1,
        moisture_preference=0.8,
        temperature_preference=0.6
    ),
    TerrainType.SEA: TerrainProperties(
        name="Water",
        color=(60, 120, 180),    # Blue
        buildable=False,
        traversable=False,
        fertile=False,
        elevation_preference=0.0,
        moisture_preference=1.0,
        temperature_preference=0.5
    ),
    TerrainType.SNOW: TerrainProperties(
        name="Snow",
        color=(240, 240, 240),   # White
        buildable=False,
        traversable=True,
        fertile=False,
        elevation_preference=0.9,
        moisture_preference=0.7,
        temperature_preference=0.1
    )
}


def get_terrain_color(terrain_type: TerrainType) -> Tuple[int, int, int]:
    """Get the RGB color for a terrain type."""
    return TERRAIN_PROPERTIES[terrain_type].color


def is_buildable(terrain_type: TerrainType) -> bool:
    """Check if buildings can be placed on this terrain type."""
    return TERRAIN_PROPERTIES[terrain_type].buildable


def is_traversable(terrain_type: TerrainType) -> bool:
    """Check if this terrain type can be moved through."""
    return TERRAIN_PROPERTIES[terrain_type].traversable


def is_fertile(terrain_type: TerrainType) -> bool:
    """Check if this terrain type is good for farming."""
    return TERRAIN_PROPERTIES[terrain_type].fertile 