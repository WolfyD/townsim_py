"""
Terrain Generation Parameters

Defines the user-configurable parameters for terrain generation.
"""

from dataclasses import dataclass
from typing import Optional
import random


@dataclass
class TerrainParameters:
    """User-configurable parameters for terrain generation."""
    
    # Core terrain settings
    map_size: int = 512                    # Grid size (power of 2) - 512 = Small Village (~5-10 km²)
    elevation_variance: float = 0.4        # 0.0-1.0 (flat to hilly)
    water_bodies: int = 2                  # Number of lakes/ponds (scales with settlement)
    noise_scale: float = 1.0               # Size of terrain features
    random_seed: Optional[int] = None      # For reproducible generation
    
    # Tile scale: Each tile ≈ 6-8 meters (allows realistic building placement)
    
    # Climate settings (local scale)
    base_temperature: float = 0.6          # 0.0-1.0 (cold to warm)
    moisture_level: float = 0.5            # 0.0-1.0 (dry to wet)
    wind_direction: float = 45.0           # 0-360° (North=0°, East=90°)
    wind_strength: float = 0.5             # 0.0-1.0 (wind effect strength)
    
    # Advanced settings
    octaves: int = 4                       # Noise complexity
    persistence: float = 0.5               # Noise amplitude falloff
    lacunarity: float = 2.0                # Noise frequency multiplier
    
    # Water level settings
    water_level: float = 0.3               # Below this = water
    coast_width: float = 0.05              # Width of coastal areas
    
    # Temperature variation
    elevation_lapse_rate: float = 0.3      # Temperature drop per elevation
    coastal_moderation: float = 0.1        # How much water moderates temp
    
    def __post_init__(self):
        """Validate and set defaults after initialization."""
        # Ensure map_size is power of 2
        if self.map_size & (self.map_size - 1) != 0:
            # Round up to nearest power of 2
            self.map_size = 2 ** (self.map_size.bit_length())
        
        # Clamp values to valid ranges
        self.elevation_variance = max(0.0, min(1.0, self.elevation_variance))
        self.base_temperature = max(0.0, min(1.0, self.base_temperature))
        self.moisture_level = max(0.0, min(1.0, self.moisture_level))
        self.wind_direction = self.wind_direction % 360.0
        self.wind_strength = max(0.0, min(1.0, self.wind_strength))
        self.water_bodies = max(0, min(10, self.water_bodies))
        self.water_level = max(0.0, min(1.0, self.water_level))
        self.coast_width = max(0.01, min(0.2, self.coast_width))
        
        # Set random seed if not provided
        if self.random_seed is None:
            self.random_seed = random.randint(0, 999999)
    
    def get_wind_vector(self) -> tuple[float, float]:
        """
        Get wind direction as a unit vector.
        
        Returns:
            (x, y) components of wind direction vector
        """
        import math
        # Convert degrees to radians, with North=0° -> (0, -1)
        angle_rad = math.radians(self.wind_direction - 90)
        return (math.cos(angle_rad), math.sin(angle_rad))
    
    def copy(self) -> 'TerrainParameters':
        """Create a copy of these parameters."""
        return TerrainParameters(
            map_size=self.map_size,
            elevation_variance=self.elevation_variance,
            water_bodies=self.water_bodies,
            noise_scale=self.noise_scale,
            random_seed=self.random_seed,
            base_temperature=self.base_temperature,
            moisture_level=self.moisture_level,
            wind_direction=self.wind_direction,
            wind_strength=self.wind_strength,
            octaves=self.octaves,
            persistence=self.persistence,
            lacunarity=self.lacunarity,
            water_level=self.water_level,
            coast_width=self.coast_width,
            elevation_lapse_rate=self.elevation_lapse_rate,
            coastal_moderation=self.coastal_moderation
        )
    
    def to_dict(self) -> dict:
        """Convert parameters to dictionary for serialization."""
        return {
            'map_size': self.map_size,
            'elevation_variance': self.elevation_variance,
            'water_bodies': self.water_bodies,
            'noise_scale': self.noise_scale,
            'random_seed': self.random_seed,
            'base_temperature': self.base_temperature,
            'moisture_level': self.moisture_level,
            'wind_direction': self.wind_direction,
            'wind_strength': self.wind_strength,
            'octaves': self.octaves,
            'persistence': self.persistence,
            'lacunarity': self.lacunarity,
            'water_level': self.water_level,
            'coast_width': self.coast_width,
            'elevation_lapse_rate': self.elevation_lapse_rate,
            'coastal_moderation': self.coastal_moderation
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TerrainParameters':
        """Create parameters from dictionary."""
        return cls(**data)


# Preset parameter configurations
PRESET_PARAMETERS = {
    # Settlement Type Presets
    "Small Village": TerrainParameters(
        map_size=512,
        elevation_variance=0.3,
        base_temperature=0.6,
        moisture_level=0.6,
        water_bodies=1,
        wind_direction=225.0  # SW wind
    ),
    
    "Large Village": TerrainParameters(
        map_size=768,
        elevation_variance=0.4,
        base_temperature=0.6,
        moisture_level=0.6,
        water_bodies=2,
        wind_direction=225.0  # SW wind
    ),
    
    "Small Town": TerrainParameters(
        map_size=1024,
        elevation_variance=0.4,
        base_temperature=0.6,
        moisture_level=0.6,
        water_bodies=3,
        wind_direction=225.0  # SW wind
    ),
    
    "Large Town": TerrainParameters(
        map_size=1024,
        elevation_variance=0.5,
        base_temperature=0.6,
        moisture_level=0.6,
        water_bodies=4,
        wind_direction=225.0  # SW wind
    ),
    
    "Small City": TerrainParameters(
        map_size=2048,
        elevation_variance=0.5,
        base_temperature=0.6,
        moisture_level=0.6,
        water_bodies=5,
        wind_direction=225.0  # SW wind
    ),
    
    # Terrain Type Presets
    "Temperate Plains": TerrainParameters(
        elevation_variance=0.3,
        base_temperature=0.6,
        moisture_level=0.6,
        water_bodies=2,
        wind_direction=225.0  # SW wind
    ),
    
    "Hilly Countryside": TerrainParameters(
        elevation_variance=0.7,
        base_temperature=0.5,
        moisture_level=0.5,
        water_bodies=3,
        wind_direction=270.0  # W wind
    ),
    
    "Coastal Area": TerrainParameters(
        elevation_variance=0.4,
        base_temperature=0.7,
        moisture_level=0.8,
        water_bodies=1,
        wind_direction=90.0,  # E wind (from sea)
        wind_strength=0.8
    ),
    
    "Arid Valley": TerrainParameters(
        elevation_variance=0.5,
        base_temperature=0.8,
        moisture_level=0.2,
        water_bodies=1,
        wind_direction=180.0,  # S wind
        wind_strength=0.3
    ),
    
    "Mountain Foothills": TerrainParameters(
        elevation_variance=0.9,
        base_temperature=0.3,
        moisture_level=0.7,
        water_bodies=4,
        wind_direction=315.0,  # NW wind
        wind_strength=0.6
    )
} 