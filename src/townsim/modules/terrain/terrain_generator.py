"""
Terrain Generator

Main class for generating terrain based on user parameters.
"""

import math
import random
from typing import List, Tuple, Optional
import numpy as np
from dataclasses import dataclass

from .terrain_types import TerrainType, TERRAIN_PROPERTIES
from .terrain_parameters import TerrainParameters
from ..utils.logging_setup import get_logger

try:
    import noise
except ImportError:
    noise = None


@dataclass
class TerrainTile:
    """Represents a single terrain tile."""
    
    x: int
    y: int
    terrain_type: TerrainType
    elevation: float
    moisture: float
    temperature: float
    distance_to_water: float


class TerrainMap:
    """Represents the generated terrain map."""
    
    def __init__(self, width: int, height: int, parameters: TerrainParameters):
        self.width = width
        self.height = height
        self.parameters = parameters
        self.tiles = np.full((height, width), TerrainType.GRASS, dtype=object)
        self.elevation_map = np.zeros((height, width), dtype=float)
        self.moisture_map = np.zeros((height, width), dtype=float)
        self.temperature_map = np.zeros((height, width), dtype=float)
        self.water_distance_map = np.zeros((height, width), dtype=float)
    
    def get_tile(self, x: int, y: int) -> Optional[TerrainTile]:
        """Get a terrain tile at the specified coordinates."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return TerrainTile(
                x=x,
                y=y,
                terrain_type=self.tiles[y, x],
                elevation=self.elevation_map[y, x],
                moisture=self.moisture_map[y, x],
                temperature=self.temperature_map[y, x],
                distance_to_water=self.water_distance_map[y, x]
            )
        return None
    
    def set_tile(self, x: int, y: int, terrain_type: TerrainType) -> None:
        """Set a terrain tile at the specified coordinates."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.tiles[y, x] = terrain_type
    
    def get_terrain_array(self) -> np.ndarray:
        """Get the terrain type array for visualization."""
        return self.tiles.copy()


class TerrainGenerator:
    """Generates terrain based on parameters."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Check if noise library is available
        if noise is None:
            self.logger.warning("noise library not available - using fallback noise generation")
    
    def generate_terrain(self, parameters: TerrainParameters) -> TerrainMap:
        """
        Generate terrain based on the provided parameters.
        
        Args:
            parameters: Terrain generation parameters
            
        Returns:
            Generated terrain map
        """
        self.logger.info(f"Generating terrain with size {parameters.map_size}x{parameters.map_size}")
        
        # Set random seed for reproducibility
        random.seed(parameters.random_seed)
        np.random.seed(parameters.random_seed)
        
        # Create terrain map
        terrain_map = TerrainMap(parameters.map_size, parameters.map_size, parameters)
        
        # Generation phases
        self._generate_elevation(terrain_map, parameters)
        self._place_water_bodies(terrain_map, parameters)
        self._calculate_water_distances(terrain_map)
        self._generate_moisture(terrain_map, parameters)
        self._generate_temperature(terrain_map, parameters)
        self._apply_wind_effects(terrain_map, parameters)
        self._assign_terrain_types(terrain_map, parameters)
        self._generate_rivers(terrain_map, parameters)
        self._post_process_terrain(terrain_map, parameters)
        
        self.logger.info("Terrain generation completed")
        return terrain_map
    
    def _generate_elevation(self, terrain_map: TerrainMap, parameters: TerrainParameters) -> None:
        """Generate elevation map using noise."""
        self.logger.debug("Generating elevation map")
        
        size = parameters.map_size
        scale = parameters.noise_scale
        
        for y in range(size):
            for x in range(size):
                # Normalize coordinates to 0-1 range
                nx = x / size
                ny = y / size
                
                # Generate noise value
                if noise is not None:
                    elevation = noise.pnoise2(
                        nx * scale,
                        ny * scale,
                        octaves=parameters.octaves,
                        persistence=parameters.persistence,
                        lacunarity=parameters.lacunarity,
                        repeatx=size,
                        repeaty=size,
                        base=parameters.random_seed
                    )
                else:
                    # Fallback noise using numpy
                    elevation = self._fallback_noise(nx * scale, ny * scale)
                
                # Scale elevation by variance parameter
                elevation = elevation * parameters.elevation_variance
                
                # Normalize to 0-1 range
                elevation = (elevation + 1.0) / 2.0
                
                terrain_map.elevation_map[y, x] = elevation
    
    def _fallback_noise(self, x: float, y: float) -> float:
        """Fallback noise function when noise library is not available."""
        # Simple pseudo-random noise based on coordinates
        return (math.sin(x * 12.9898 + y * 78.233) * 43758.5453) % 1.0 * 2.0 - 1.0
    
    def _place_water_bodies(self, terrain_map: TerrainMap, parameters: TerrainParameters) -> None:
        """Place water bodies (lakes, ponds) on the terrain."""
        self.logger.debug(f"Placing {parameters.water_bodies} water bodies")
        
        size = parameters.map_size
        
        for _ in range(parameters.water_bodies):
            # Choose random location
            center_x = random.randint(size // 4, 3 * size // 4)
            center_y = random.randint(size // 4, 3 * size // 4)
            
            # Choose size based on map size
            radius = random.randint(size // 20, size // 10)
            
            # Create water body
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    x = center_x + dx
                    y = center_y + dy
                    
                    if 0 <= x < size and 0 <= y < size:
                        distance = math.sqrt(dx * dx + dy * dy)
                        if distance <= radius:
                            # Make water level lower than current elevation
                            terrain_map.elevation_map[y, x] = min(
                                terrain_map.elevation_map[y, x],
                                parameters.water_level * 0.8
                            )
    
    def _calculate_water_distances(self, terrain_map: TerrainMap) -> None:
        """Calculate distance to nearest water for each point."""
        self.logger.debug("Calculating water distances")
        
        size = terrain_map.width
        water_points = []
        
        # Find all water tiles
        for y in range(size):
            for x in range(size):
                if terrain_map.elevation_map[y, x] < terrain_map.parameters.water_level:
                    water_points.append((x, y))
        
        # Calculate distances
        for y in range(size):
            for x in range(size):
                if water_points:
                    min_distance = min(
                        math.sqrt((x - wx) ** 2 + (y - wy) ** 2)
                        for wx, wy in water_points
                    )
                    terrain_map.water_distance_map[y, x] = min_distance / size
                else:
                    terrain_map.water_distance_map[y, x] = 1.0
    
    def _generate_moisture(self, terrain_map: TerrainMap, parameters: TerrainParameters) -> None:
        """Generate moisture map based on parameters and water proximity."""
        self.logger.debug("Generating moisture map")
        
        size = parameters.map_size
        
        for y in range(size):
            for x in range(size):
                # Base moisture from parameters
                moisture = parameters.moisture_level
                
                # Modify based on water proximity
                water_distance = terrain_map.water_distance_map[y, x]
                moisture_bonus = max(0, 0.5 - water_distance * 2.0)
                moisture += moisture_bonus
                
                # Add some noise variation
                nx = x / size
                ny = y / size
                
                if noise is not None:
                    moisture_noise = noise.pnoise2(
                        nx * parameters.noise_scale * 0.7,
                        ny * parameters.noise_scale * 0.7,
                        octaves=3,
                        persistence=0.5,
                        lacunarity=2.0,
                        base=parameters.random_seed + 1000
                    )
                else:
                    moisture_noise = self._fallback_noise(nx * parameters.noise_scale * 0.7, ny * parameters.noise_scale * 0.7)
                
                moisture += moisture_noise * 0.2
                
                # Clamp to 0-1 range
                moisture = max(0.0, min(1.0, moisture))
                
                terrain_map.moisture_map[y, x] = moisture
    
    def _generate_temperature(self, terrain_map: TerrainMap, parameters: TerrainParameters) -> None:
        """Generate temperature map based on elevation and water proximity."""
        self.logger.debug("Generating temperature map")
        
        size = parameters.map_size
        
        for y in range(size):
            for x in range(size):
                # Base temperature from parameters
                temperature = parameters.base_temperature
                
                # Elevation effect (higher = cooler)
                elevation = terrain_map.elevation_map[y, x]
                temperature -= elevation * parameters.elevation_lapse_rate
                
                # Water moderation effect
                water_distance = terrain_map.water_distance_map[y, x]
                if water_distance < 0.1:
                    temperature += parameters.coastal_moderation
                
                # Clamp to 0-1 range
                temperature = max(0.0, min(1.0, temperature))
                
                terrain_map.temperature_map[y, x] = temperature
    
    def _apply_wind_effects(self, terrain_map: TerrainMap, parameters: TerrainParameters) -> None:
        """Apply wind effects to moisture and temperature."""
        self.logger.debug("Applying wind effects")
        
        if parameters.wind_strength < 0.1:
            return  # Skip if wind is too weak
        
        wind_x, wind_y = parameters.get_wind_vector()
        size = parameters.map_size
        
        # Apply orographic effects
        for y in range(size):
            for x in range(size):
                elevation = terrain_map.elevation_map[y, x]
                
                # Calculate slope in wind direction
                wind_slope = 0.0
                if 0 < x < size - 1 and 0 < y < size - 1:
                    # Elevation gradient in wind direction
                    dx_elev = terrain_map.elevation_map[y, x + 1] - terrain_map.elevation_map[y, x - 1]
                    dy_elev = terrain_map.elevation_map[y + 1, x] - terrain_map.elevation_map[y - 1, x]
                    
                    wind_slope = (dx_elev * wind_x + dy_elev * wind_y) * parameters.wind_strength
                
                # Windward slopes get more moisture
                if wind_slope > 0:  # Windward
                    terrain_map.moisture_map[y, x] += wind_slope * 0.3
                else:  # Leeward (rain shadow)
                    terrain_map.moisture_map[y, x] += wind_slope * 0.2
                
                # Clamp moisture
                terrain_map.moisture_map[y, x] = max(0.0, min(1.0, terrain_map.moisture_map[y, x]))
    
    def _assign_terrain_types(self, terrain_map: TerrainMap, parameters: TerrainParameters) -> None:
        """Assign terrain types based on elevation, moisture, and temperature."""
        self.logger.debug("Assigning terrain types")
        
        size = parameters.map_size
        
        for y in range(size):
            for x in range(size):
                elevation = terrain_map.elevation_map[y, x]
                moisture = terrain_map.moisture_map[y, x]
                temperature = terrain_map.temperature_map[y, x]
                water_distance = terrain_map.water_distance_map[y, x]
                
                # Determine terrain type
                terrain_type = self._determine_terrain_type(
                    elevation, moisture, temperature, water_distance, parameters
                )
                
                terrain_map.tiles[y, x] = terrain_type
    
    def _determine_terrain_type(self, elevation: float, moisture: float, temperature: float, 
                               water_distance: float, parameters: TerrainParameters) -> TerrainType:
        """Determine terrain type based on environmental factors."""
        
        # Water areas
        if elevation < parameters.water_level:
            return TerrainType.SEA
        
        # Coastal areas
        if water_distance < parameters.coast_width:
            return TerrainType.COAST
        
        # Snow at high elevations and low temperatures
        if temperature < 0.2 and elevation > 0.7:
            return TerrainType.SNOW
        
        # Dry areas
        if moisture < 0.3:
            if temperature > 0.7:
                return TerrainType.SAND  # Hot and dry
            else:
                return TerrainType.ROCK  # Cold and dry
        
        # Very wet areas
        if moisture > 0.8 and elevation < 0.4:
            return TerrainType.MUD  # Wetlands
        
        # Default to grass
        return TerrainType.GRASS
    
    def _generate_rivers(self, terrain_map: TerrainMap, parameters: TerrainParameters) -> None:
        """Generate rivers flowing from high to low elevation."""
        self.logger.debug("Generating rivers")
        
        # TODO: Implement river generation algorithm
        # This would trace paths from high elevation to water bodies
        pass
    
    def _post_process_terrain(self, terrain_map: TerrainMap, parameters: TerrainParameters) -> None:
        """Post-process terrain to smooth transitions and fix issues."""
        self.logger.debug("Post-processing terrain")
        
        # TODO: Implement terrain smoothing and validation
        # This would ensure realistic terrain transitions
        pass 