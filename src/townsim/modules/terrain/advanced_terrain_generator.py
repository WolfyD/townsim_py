"""
Advanced Terrain Generator

Implements the 7-step realistic terrain generation approach:
1. Determine coastal/island/landlocked
2. Generate base land height with gradients  
3. Apply multi-octave Perlin noise for terrain fluctuations
4. Apply terrain smoothing based on user settings
5. Add water bodies in concave areas
6. Generate rivers based on elevation changes
7. Assign terrain types

Based on techniques from Red Blob Games: https://www.redblobgames.com/maps/terrain-from-noise/
"""

import math
import random
from typing import List, Tuple, Optional, Dict
import numpy as np
from enum import Enum
from dataclasses import dataclass

from .terrain_types import TerrainType, TERRAIN_PROPERTIES
from .terrain_parameters import TerrainParameters
from ..utils.logging_setup import get_logger

try:
    import noise
    NOISE_AVAILABLE = True
except ImportError:
    NOISE_AVAILABLE = False


class CoastalType(Enum):
    """Types of coastal configuration."""
    RANDOM = "random"
    LANDLOCKED = "landlocked" 
    COASTAL = "coastal"
    ISLAND = "island"


@dataclass
class AdvancedTerrainParameters(TerrainParameters):
    """Extended parameters for advanced terrain generation."""
    
    # Step 1: Coastal configuration
    coastal_type: CoastalType = CoastalType.RANDOM
    
    # Step 2: Base elevation
    coastal_gradient_min: float = 0.10  # 10% of map
    coastal_gradient_max: float = 0.30  # 30% of map
    
    # Step 4: Terrain smoothing
    terrain_smoothness: float = 0.5  # 0.0 = rough, 1.0 = smooth
    
    # Step 5: Water body constraints
    min_water_body_size: int = 5  # minimum units
    max_water_body_ratio_edge: float = 0.10  # 10% if on edge
    max_water_body_ratio_center: float = 0.06  # 6% if in center
    
    # Advanced noise parameters (Red Blob Games approach)
    noise_octaves: int = 4
    noise_amplitudes: List[float] = None  # Will default to [1, 0.5, 0.25, 0.125]
    elevation_exponent: float = 1.2  # For valley/peak redistribution


class AdvancedTerrainGenerator:
    """Advanced terrain generator using geological principles."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        if not NOISE_AVAILABLE:
            self.logger.warning("noise library not available - using fallback")
    
    def generate_terrain(self, params: AdvancedTerrainParameters) -> 'TerrainMap':
        """
        Generate terrain using the 7-step advanced approach.
        
        Args:
            params: Advanced terrain generation parameters
            
        Returns:
            Generated terrain map
        """
        self.logger.info(f"Generating advanced terrain {params.map_size}x{params.map_size}")
        start_time = time.time()
        
        # Set random seed
        random.seed(params.random_seed)
        np.random.seed(params.random_seed)
        
        # Initialize default amplitudes if not provided
        if params.noise_amplitudes is None:
            params.noise_amplitudes = [1.0, 0.5, 0.25, 0.125]
        
        size = params.map_size
        
        # Step 1: Determine coastal configuration
        coastal_config = self._determine_coastal_config(params)
        self.logger.debug(f"Coastal config: {coastal_config}")
        
        # Step 2: Generate base land height with gradients
        base_elevation = self._generate_base_elevation(size, coastal_config, params)
        
        # Step 3: Apply multi-octave Perlin noise (Red Blob Games technique)
        noise_elevation = self._generate_noise_elevation(size, params)
        
        # Combine base + noise
        elevation_map = self._combine_elevations(base_elevation, noise_elevation, params)
        
        # Step 4: Apply terrain smoothing
        if params.terrain_smoothness > 0.1:
            elevation_map = self._apply_smoothing(elevation_map, params.terrain_smoothness)
        
        # Step 5: Add water bodies in concave areas
        water_mask = self._generate_water_bodies(elevation_map, params)
        
        # Step 6: Generate rivers based on elevation changes
        river_mask = self._generate_rivers(elevation_map, params)
        
        # Combine water features
        final_elevation = self._apply_water_features(elevation_map, water_mask, river_mask)
        
        # Step 7: Assign terrain types
        terrain_types = self._assign_terrain_types(final_elevation, water_mask, river_mask, params)
        
        # Create terrain map
        from .terrain_generator import TerrainMap
        terrain_map = TerrainMap(size, size, params)
        terrain_map.elevation_map = final_elevation
        terrain_map.tiles = terrain_types
        
        elapsed = time.time() - start_time
        self.logger.info(f"Advanced terrain generation completed in {elapsed:.2f}s")
        
        return terrain_map
    
    def _determine_coastal_config(self, params: AdvancedTerrainParameters) -> Dict:
        """Step 1: Determine coastal/island/landlocked configuration."""
        
        if params.coastal_type == CoastalType.RANDOM:
            # Randomly choose configuration
            config_type = random.choice([CoastalType.LANDLOCKED, CoastalType.COASTAL, CoastalType.ISLAND])
        else:
            config_type = params.coastal_type
        
        config = {"type": config_type}
        
        if config_type == CoastalType.COASTAL:
            # Randomly select which edges are coastal (1-3 edges)
            edges = ["north", "south", "east", "west"]
            num_coastal_edges = random.randint(1, 3)
            config["coastal_edges"] = random.sample(edges, num_coastal_edges)
            
        elif config_type == CoastalType.ISLAND:
            # Island surrounded by water
            config["coastal_edges"] = ["north", "south", "east", "west"]
            config["island_center"] = (
                random.uniform(0.3, 0.7),  # x center
                random.uniform(0.3, 0.7)   # y center
            )
            config["island_size"] = random.uniform(0.4, 0.7)  # radius as fraction of map
        
        return config
    
    def _generate_base_elevation(self, size: int, coastal_config: Dict, 
                                params: AdvancedTerrainParameters) -> np.ndarray:
        """Step 2: Generate base land height with coastal gradients."""
        
        # Create coordinate grids
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        X, Y = np.meshgrid(x, y)
        
        if coastal_config["type"] == CoastalType.LANDLOCKED:
            # Uniform base elevation
            base_elevation = np.full((size, size), 0.6)
            
        elif coastal_config["type"] == CoastalType.COASTAL:
            # Start with base elevation
            base_elevation = np.full((size, size), 0.6)
            
            # Apply gradients for each coastal edge
            gradient_distance = random.uniform(
                params.coastal_gradient_min, 
                params.coastal_gradient_max
            )
            
            for edge in coastal_config["coastal_edges"]:
                if edge == "north":
                    # Gradient from top (y=0) going down
                    gradient = np.maximum(0, (gradient_distance - Y) / gradient_distance)
                    base_elevation *= (0.3 + 0.7 * gradient)
                elif edge == "south":
                    # Gradient from bottom (y=1) going up
                    gradient = np.maximum(0, (gradient_distance - (1 - Y)) / gradient_distance)
                    base_elevation *= (0.3 + 0.7 * gradient)
                elif edge == "west":
                    # Gradient from left (x=0) going right
                    gradient = np.maximum(0, (gradient_distance - X) / gradient_distance)
                    base_elevation *= (0.3 + 0.7 * gradient)
                elif edge == "east":
                    # Gradient from right (x=1) going left
                    gradient = np.maximum(0, (gradient_distance - (1 - X)) / gradient_distance)
                    base_elevation *= (0.3 + 0.7 * gradient)
                    
        elif coastal_config["type"] == CoastalType.ISLAND:
            # Island with radial gradient
            center_x, center_y = coastal_config["island_center"]
            island_radius = coastal_config["island_size"]
            
            # Distance from island center
            distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            
            # Radial gradient: high in center, low at edges
            gradient = np.maximum(0, (island_radius - distance) / island_radius)
            base_elevation = 0.2 + 0.6 * gradient  # 0.2 to 0.8 range
        
        return base_elevation
    
    def _generate_noise_elevation(self, size: int, params: AdvancedTerrainParameters) -> np.ndarray:
        """Step 3: Generate multi-octave Perlin noise (Red Blob Games technique)."""
        
        # Create coordinate grid
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        X, Y = np.meshgrid(x, y)
        
        # Scale coordinates by noise scale
        X_scaled = X * params.noise_scale
        Y_scaled = Y * params.noise_scale
        
        # Generate multi-octave noise
        noise_map = np.zeros((size, size))
        amplitude_sum = 0
        
        for octave, amplitude in enumerate(params.noise_amplitudes):
            frequency = 2 ** octave
            
            if NOISE_AVAILABLE:
                # Use proper Perlin noise
                octave_noise = np.zeros((size, size))
                for i in range(size):
                    for j in range(size):
                        octave_noise[i, j] = noise.pnoise2(
                            X_scaled[i, j] * frequency,
                            Y_scaled[i, j] * frequency,
                            octaves=1,
                            persistence=0.5,
                            lacunarity=2.0,
                            repeatx=size,
                            repeaty=size,
                            base=params.random_seed + octave * 100
                        )
            else:
                # Fallback noise
                octave_noise = self._fallback_noise_2d(X_scaled * frequency, Y_scaled * frequency)
            
            noise_map += amplitude * octave_noise
            amplitude_sum += amplitude
        
        # Normalize to 0-1 range
        noise_map = (noise_map + amplitude_sum) / (2 * amplitude_sum)
        
        # Apply elevation redistribution (Red Blob Games technique)
        if params.elevation_exponent != 1.0:
            noise_map = np.power(noise_map, params.elevation_exponent)
        
        return noise_map
    
    def _fallback_noise_2d(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Fallback 2D noise function when noise library unavailable."""
        return (np.sin(X * 12.9898 + Y * 78.233) * 43758.5453) % 1.0 * 2.0 - 1.0
    
    def _combine_elevations(self, base: np.ndarray, noise: np.ndarray, 
                           params: AdvancedTerrainParameters) -> np.ndarray:
        """Combine base elevation with noise elevation."""
        # Weight the combination based on elevation variance
        noise_weight = params.elevation_variance
        base_weight = 1.0 - noise_weight
        
        return base_weight * base + noise_weight * noise
    
    def _apply_smoothing(self, elevation: np.ndarray, smoothness: float) -> np.ndarray:
        """Step 4: Apply terrain smoothing using Gaussian filter."""
        try:
            from scipy import ndimage
            # Convert smoothness (0-1) to sigma for Gaussian filter
            # Higher smoothness = more blurring
            sigma = smoothness * 3.0
            return ndimage.gaussian_filter(elevation, sigma=sigma, mode='reflect')
        except ImportError:
            self.logger.warning("scipy not available - using simple smoothing")
            # Fallback: simple averaging filter
            return self._simple_smoothing(elevation, smoothness)
    
    def _simple_smoothing(self, elevation: np.ndarray, smoothness: float) -> np.ndarray:
        """Fallback smoothing without scipy."""
        if smoothness < 0.1:
            return elevation
        
        # Simple 3x3 averaging filter
        kernel_size = int(smoothness * 5) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        result = elevation.copy()
        pad_size = kernel_size // 2
        
        # Simple box filter
        for i in range(pad_size, elevation.shape[0] - pad_size):
            for j in range(pad_size, elevation.shape[1] - pad_size):
                window = elevation[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1]
                result[i, j] = np.mean(window)
        
        return result
    
    def _generate_water_bodies(self, elevation: np.ndarray, 
                              params: AdvancedTerrainParameters) -> np.ndarray:
        """Step 5: Generate water bodies in concave areas."""
        # This is a placeholder - will implement sophisticated concave detection
        water_mask = np.zeros_like(elevation, dtype=bool)
        
        # For now, just create some random water bodies
        # TODO: Implement proper concave area detection
        
        return water_mask
    
    def _generate_rivers(self, elevation: np.ndarray, 
                        params: AdvancedTerrainParameters) -> np.ndarray:
        """Step 6: Generate rivers based on elevation changes."""
        # This is a placeholder - will implement river flow simulation
        river_mask = np.zeros_like(elevation, dtype=bool)
        
        # TODO: Implement elevation-based river generation
        
        return river_mask
    
    def _apply_water_features(self, elevation: np.ndarray, water_mask: np.ndarray, 
                             river_mask: np.ndarray) -> np.ndarray:
        """Apply water features to elevation map."""
        result = elevation.copy()
        
        # Lower elevation for water areas
        result[water_mask] = np.minimum(result[water_mask], 0.25)
        result[river_mask] = np.minimum(result[river_mask], 0.3)
        
        return result
    
    def _assign_terrain_types(self, elevation: np.ndarray, water_mask: np.ndarray,
                             river_mask: np.ndarray, params: AdvancedTerrainParameters) -> np.ndarray:
        """Step 7: Assign terrain types based on elevation and features."""
        size = elevation.shape[0]
        terrain_types = np.full((size, size), TerrainType.GRASS, dtype=object)
        
        # Water areas
        terrain_types[water_mask] = TerrainType.SEA
        terrain_types[river_mask] = TerrainType.SEA
        
        # Elevation-based assignment
        terrain_types[elevation < 0.3] = TerrainType.SEA
        terrain_types[(elevation >= 0.3) & (elevation < 0.35)] = TerrainType.COAST
        terrain_types[elevation > 0.8] = TerrainType.ROCK
        
        # TODO: Add moisture and temperature considerations
        
        return terrain_types


# Add time import
import time 