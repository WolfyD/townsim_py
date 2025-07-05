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
import time
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
    max_lakes: int = 2  # maximum number of lakes
    
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
        try:
            self.logger.info(f"Generating advanced terrain {params.map_size}x{params.map_size}")
            start_time = time.time()
            
            # Set random seed
            random.seed(params.random_seed)
            np.random.seed(params.random_seed)
            
            # Initialize default amplitudes if not provided
            if params.noise_amplitudes is None:
                params.noise_amplitudes = [1.0, 0.5, 0.25, 0.125]
            
            self.logger.debug(f"Using noise amplitudes: {params.noise_amplitudes}")
            
            size = params.map_size
            
            # Step 1: Determine coastal configuration
            coastal_config = self._determine_coastal_config(params)
            self.logger.debug(f"Coastal config: {coastal_config}")
            
            # Step 2: Generate base land height with gradients
            self.logger.debug("Starting base elevation generation...")
            base_elevation = self._generate_base_elevation(size, coastal_config, params)
            self.logger.debug("Base elevation generation completed")
            
            # Step 3: Apply multi-octave Perlin noise (Red Blob Games technique)
            self.logger.debug("Starting noise elevation generation...")
            noise_elevation = self._generate_noise_elevation(size, params)
            self.logger.debug("Noise elevation generation completed")
            
            # Combine base + noise
            self.logger.debug("Combining elevations...")
            elevation_map = self._combine_elevations(base_elevation, noise_elevation, params)
            self.logger.debug("Elevation combination completed")
            
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
            
        except Exception as e:
            import traceback
            self.logger.error(f"Failed to generate terrain: {e}")
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            
            # Return a basic fallback terrain 
            self.logger.warning("Returning basic fallback terrain due to error")
            return self._generate_fallback_terrain(params)
    
    def _determine_coastal_config(self, params: AdvancedTerrainParameters) -> Dict:
        """Step 1: Determine coastal/island/landlocked configuration."""
        
        if params.coastal_type == CoastalType.RANDOM:
            # Randomly choose configuration
            config_type = random.choice([CoastalType.LANDLOCKED, CoastalType.COASTAL, CoastalType.ISLAND])
        else:
            config_type = params.coastal_type
        
        config = {"type": config_type}
        
        if config_type == CoastalType.COASTAL:
            # Generate a random angled coastline (not just cardinal directions)
            # Random angle from 0 to 360 degrees
            coast_angle = random.uniform(0, 360)
            
            # Random position along the angle (how far the coastline cuts into the map)
            coast_depth = random.uniform(0.15, 0.35)  # 15-35% of map for 10-20% water coverage
            
            config["coast_angle"] = coast_angle
            config["coast_depth"] = coast_depth
            
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
        """Step 2: Generate base land height with proper coastal boundaries."""
        
        # Create coordinate grids
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        X, Y = np.meshgrid(x, y)
        
        if coastal_config["type"] == CoastalType.LANDLOCKED:
            # Uniform base elevation for landlocked areas
            base_elevation = np.full((size, size), 0.6)
            
        elif coastal_config["type"] == CoastalType.COASTAL:
            # Start with land elevation
            base_elevation = np.full((size, size), 0.6)
            
            # Create angled coastline with varied edges
            coast_angle = coastal_config["coast_angle"]
            coast_depth = coastal_config["coast_depth"]
            
            # Create the angled coastline boundary
            base_elevation = self._create_angled_coastline(
                base_elevation, coast_angle, coast_depth, X, Y, params
            )
                    
        elif coastal_config["type"] == CoastalType.ISLAND:
            # Island with varied, natural boundary
            center_x, center_y = coastal_config["island_center"]
            island_radius = coastal_config["island_size"]
            
            # Create varied island boundary with noise
            base_elevation = self._create_natural_island(
                size, center_x, center_y, island_radius, X, Y, params
            )
        
        return base_elevation
    
    def _create_angled_coastline(self, elevation: np.ndarray, coast_angle: float, 
                                coast_depth: float, X: np.ndarray, Y: np.ndarray, 
                                params: AdvancedTerrainParameters) -> np.ndarray:
        """Create a single angled coastline with natural variation."""
        result = elevation.copy()
        
        # Convert angle to radians
        angle_rad = np.radians(coast_angle)
        
        # Create the base coastline using the angle
        # Line equation: ax + by + c = 0
        # For angle θ, normal vector is (cos(θ), sin(θ))
        normal_x = np.cos(angle_rad)
        normal_y = np.sin(angle_rad)
        
        # Position the line so it cuts through the map at the desired depth
        # Center the line and offset it based on coast_depth
        center_x, center_y = 0.5, 0.5
        offset = coast_depth - 0.5  # Offset from center
        
        # Distance from each point to the line
        distance_to_line = normal_x * (X - center_x) + normal_y * (Y - center_y) - offset
        
        # Add noise to make the coastline more natural
        coastline_noise = self._generate_coastline_noise(X, Y, params)
        
        # Create water mask: points on the "water side" of the line
        water_mask = distance_to_line + coastline_noise < 0
        
        # Apply sea level to water areas
        result[water_mask] = 0.15  # Sea level
        
        # Apply additional smoothing to coastline transitions for flowing curves
        result = self._smooth_coastline_transitions(result, water_mask)
        
        return result
    
    def _generate_coastline_noise(self, X: np.ndarray, Y: np.ndarray, 
                                 params: AdvancedTerrainParameters) -> np.ndarray:
        """Generate smooth flowing coastline curves using mathematical sine waves."""
        # Use smooth mathematical functions instead of noise for flowing curves
        coastline_variation = np.zeros_like(X)
        
        # Create multiple sine wave patterns for natural flowing curves
        # Pattern 1: Large scale meandering
        wave1 = np.sin(X * math.pi * 2 * 0.8) * 0.012  # 1.2% amplitude
        
        # Pattern 2: Medium scale curves  
        wave2 = np.sin(Y * math.pi * 2 * 1.2) * 0.008  # 0.8% amplitude
        
        # Pattern 3: Diagonal flowing pattern
        diagonal = (X + Y) * 0.5
        wave3 = np.sin(diagonal * math.pi * 2 * 1.0) * 0.005  # 0.5% amplitude
        
        # Combine waves for natural flowing coastline
        coastline_variation = wave1 + wave2 + wave3
        
        # Add very gentle randomization based on seed
        np.random.seed(params.random_seed)
        random_offset = np.random.uniform(-0.003, 0.003)  # Very small random offset
        coastline_variation += random_offset
        
        return coastline_variation
    
    def _smooth_coastline_transitions(self, elevation: np.ndarray, water_mask: np.ndarray) -> np.ndarray:
        """Apply additional smoothing to coastline transitions for flowing curves."""
        result = elevation.copy()
        
        # Find coastline boundary pixels (land pixels adjacent to water)
        from scipy.ndimage import binary_dilation
        
        try:
            # Create a boundary zone around water areas
            water_expanded = binary_dilation(water_mask, iterations=3)
            coastline_zone = water_expanded & ~water_mask
            
            # Apply gentle smoothing only to the coastline transition zone
            if np.any(coastline_zone):
                # Use a small Gaussian filter for very gentle smoothing
                from scipy.ndimage import gaussian_filter
                
                # Create a smoothed version of the elevation
                smoothed = gaussian_filter(elevation, sigma=1.0, mode='reflect')
                
                # Apply smoothing only in the coastline zone
                result[coastline_zone] = smoothed[coastline_zone]
                
        except ImportError:
            # Fallback: simple averaging for coastline areas
            self.logger.debug("scipy not available - using simple coastline smoothing")
            
            # Find pixels near water manually
            size = elevation.shape[0]
            for y in range(1, size - 1):
                for x in range(1, size - 1):
                    if not water_mask[y, x]:  # Land pixel
                        # Check if adjacent to water
                        adjacent_to_water = (
                            water_mask[y-1, x] or water_mask[y+1, x] or 
                            water_mask[y, x-1] or water_mask[y, x+1]
                        )
                        
                        if adjacent_to_water:
                            # Apply gentle smoothing to this coastline pixel
                            neighbors = [
                                elevation[y-1, x], elevation[y+1, x],
                                elevation[y, x-1], elevation[y, x+1],
                                elevation[y, x]
                            ]
                            result[y, x] = np.mean(neighbors) * 0.3 + elevation[y, x] * 0.7
        
        return result
    
    def _create_natural_island(self, size: int, center_x: float, center_y: float, 
                              island_radius: float, X: np.ndarray, Y: np.ndarray,
                              params: AdvancedTerrainParameters) -> np.ndarray:
        """Create an island with natural, organic shape - not circular."""
        # Start with sea level everywhere
        base_elevation = np.full((size, size), 0.15)
        
        # Create organic island shape using multiple techniques
        # 1. Multiple center points for irregular shape
        num_centers = random.randint(3, 6)  # 3-6 influence points
        centers = []
        for _ in range(num_centers):
            # Centers clustered around main center
            offset_x = random.uniform(-0.15, 0.15)
            offset_y = random.uniform(-0.15, 0.15)
            weight = random.uniform(0.3, 1.0)
            centers.append((center_x + offset_x, center_y + offset_y, weight))
        
        # 2. Create organic distance field
        organic_distance = np.zeros_like(X)
        for i, (cx, cy, weight) in enumerate(centers):
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
            organic_distance += weight * dist
        
        # Normalize organic distance
        organic_distance = organic_distance / len(centers)
        
        # 3. Add large-scale shape variation
        shape_noise = self._generate_island_shape_noise(X, Y, center_x, center_y, params)
        
        # 4. Add medium-scale edge variation  
        edge_noise = self._generate_island_edge_noise(X, Y, center_x, center_y, params)
        
        # Combine all influences for final island boundary
        final_radius = island_radius + shape_noise + edge_noise
        
        # Create island mask with organic shape
        island_mask = organic_distance <= final_radius
        
        # Elevated land for island
        base_elevation[island_mask] = 0.6
        
        # Create smooth transition zones
        edge_buffer = 0.05  # 5% buffer for gradual transition
        transition_mask = (organic_distance <= final_radius + edge_buffer) & (organic_distance > final_radius - edge_buffer)
        
        if np.any(transition_mask):
            # Smooth transition from sea to land
            distance_masked = organic_distance[transition_mask]
            radius_masked = final_radius[transition_mask]
            transition_factor = (radius_masked + edge_buffer - distance_masked) / (2 * edge_buffer)
            base_elevation[transition_mask] = 0.15 + 0.45 * transition_factor
        
        # Apply additional smoothing to island coastline transitions for flowing curves
        sea_mask = base_elevation <= 0.2
        base_elevation = self._smooth_coastline_transitions(base_elevation, sea_mask)
        
        return base_elevation
    
    def _generate_island_shape_noise(self, X: np.ndarray, Y: np.ndarray, 
                                    center_x: float, center_y: float,
                                    params: AdvancedTerrainParameters) -> np.ndarray:
        """Generate smooth flowing shape variation using mathematical curves."""
        # Use smooth mathematical functions for island shape variation
        rel_x = X - center_x
        rel_y = Y - center_y
        
        # Convert to polar-like coordinates for island shaping
        angle = np.arctan2(rel_y, rel_x)
        
        # Create smooth flowing shape variation using trigonometric functions
        # Multiple wave patterns for organic but smooth shape
        shape_wave1 = np.sin(angle * 3) * 0.04      # 4% variation with 3 lobes
        shape_wave2 = np.cos(angle * 5) * 0.025     # 2.5% variation with 5 lobes  
        shape_wave3 = np.sin(angle * 2) * 0.015     # 1.5% variation with 2 lobes
        
        # Combine for organic but smooth island shape
        shape_variation = shape_wave1 + shape_wave2 + shape_wave3
        
        # Add seed-based rotation for variety
        np.random.seed(params.random_seed + 1000)
        rotation_offset = np.random.uniform(0, 2 * math.pi)
        rotated_angle = angle + rotation_offset
        shape_variation += np.sin(rotated_angle * 4) * 0.02
        
        return shape_variation
    
    def _generate_island_edge_noise(self, X: np.ndarray, Y: np.ndarray, 
                                   center_x: float, center_y: float,
                                   params: AdvancedTerrainParameters) -> np.ndarray:
        """Generate smooth flowing edge variation using mathematical curves."""
        # Use smooth mathematical functions for flowing island edges
        rel_x = X - center_x
        rel_y = Y - center_y
        
        # Convert to polar coordinates for smooth edge variation
        angle = np.arctan2(rel_y, rel_x)
        distance = np.sqrt(rel_x**2 + rel_y**2)
        
        # Create fine-scale smooth edge variation
        edge_wave1 = np.sin(angle * 8) * 0.015      # 1.5% variation with 8 ripples
        edge_wave2 = np.cos(angle * 12) * 0.008     # 0.8% variation with 12 ripples
        edge_wave3 = np.sin(angle * 6) * 0.005      # 0.5% variation with 6 ripples
        
        # Add distance-based modulation for more organic feel
        distance_factor = np.exp(-distance * 2)  # Fade effect from center
        edge_variation = (edge_wave1 + edge_wave2 + edge_wave3) * (0.5 + distance_factor * 0.5)
        
        # Add seed-based phase shift for variety
        np.random.seed(params.random_seed + 2000)
        phase_shift = np.random.uniform(0, 2 * math.pi)
        edge_variation += np.sin(angle * 10 + phase_shift) * 0.003
        
        return edge_variation
    
    def _generate_noise_elevation(self, size: int, params: AdvancedTerrainParameters) -> np.ndarray:
        """Step 3: Generate multi-octave Perlin noise (Red Blob Games technique)."""
        
        try:
            self.logger.debug(f"Creating coordinate grid for {size}x{size} map...")
            
            # Create coordinate grid
            x = np.linspace(0, 1, size)
            y = np.linspace(0, 1, size)
            X, Y = np.meshgrid(x, y)
            
            # Scale coordinates by noise scale
            X_scaled = X * params.noise_scale
            Y_scaled = Y * params.noise_scale
            
            self.logger.debug(f"Grid created. Total pixels: {size * size}")
            
            # For large maps, use fast fallback method to avoid performance issues
            use_fallback = size > 300  # Use fallback for maps larger than 300x300
            
            if use_fallback:
                self.logger.debug("Using fast fallback noise for large map")
                return self._generate_fast_noise_elevation(X_scaled, Y_scaled, params)
            
            self.logger.debug("Using high-quality Perlin noise for small map")
            
            # Generate multi-octave noise
            noise_map = np.zeros((size, size))
            amplitude_sum = 0
            
            for octave, amplitude in enumerate(params.noise_amplitudes):
                self.logger.debug(f"Processing octave {octave + 1}/{len(params.noise_amplitudes)} (amplitude: {amplitude})")
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
                
                self.logger.debug(f"Octave {octave + 1} completed")
            
            # Normalize to 0-1 range
            self.logger.debug("Normalizing noise map...")
            noise_map = (noise_map + amplitude_sum) / (2 * amplitude_sum)
            
            # Apply elevation redistribution (Red Blob Games technique)
            if params.elevation_exponent != 1.0:
                self.logger.debug(f"Applying elevation redistribution (exponent: {params.elevation_exponent})")
                noise_map = np.power(noise_map, params.elevation_exponent)
            
            self.logger.debug("Noise elevation generation completed successfully")
            return noise_map
            
        except Exception as e:
            self.logger.error(f"Error in noise elevation generation: {e}")
            self.logger.debug("Falling back to simple fallback noise")
            # Emergency fallback
            return self._generate_simple_fallback(size, params)
    
    def _generate_fast_noise_elevation(self, X_scaled: np.ndarray, Y_scaled: np.ndarray, 
                                      params: AdvancedTerrainParameters) -> np.ndarray:
        """Generate noise elevation using fast methods for large maps."""
        self.logger.debug("Starting fast noise generation...")
        
        noise_map = np.zeros_like(X_scaled)
        amplitude_sum = 0
        
        for octave, amplitude in enumerate(params.noise_amplitudes):
            self.logger.debug(f"Fast octave {octave + 1}/{len(params.noise_amplitudes)}")
            frequency = 2 ** octave
            
            # Use fallback noise for each octave
            octave_noise = self._fallback_noise_2d(X_scaled * frequency, Y_scaled * frequency)
            noise_map += amplitude * octave_noise
            amplitude_sum += amplitude
        
        # Normalize to 0-1 range
        noise_map = (noise_map + amplitude_sum) / (2 * amplitude_sum)
        
        # Apply elevation redistribution
        if params.elevation_exponent != 1.0:
            noise_map = np.power(noise_map, params.elevation_exponent)
        
        self.logger.debug("Fast noise generation completed")
        return noise_map
    
    def _generate_simple_fallback(self, size: int, params: AdvancedTerrainParameters) -> np.ndarray:
        """Emergency fallback for noise generation."""
        self.logger.debug("Using emergency simple fallback")
        
        # Create simple noise pattern
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        X, Y = np.meshgrid(x, y)
        
        # Simple sine-based noise
        noise_map = 0.5 + 0.3 * np.sin(X * params.noise_scale * 10) * np.cos(Y * params.noise_scale * 10)
        
        # Apply elevation redistribution
        if params.elevation_exponent != 1.0:
            noise_map = np.power(noise_map, params.elevation_exponent)
        
        return noise_map
    
    def _fallback_noise_2d(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Fallback 2D noise function when noise library unavailable."""
        try:
            # Simple sine-based noise that should always work
            noise = np.sin(X * 12.9898 + Y * 78.233) * 43758.5453
            # Ensure we get values between -1 and 1
            noise = noise % 1.0  # Get fractional part (0 to 1)
            noise = noise * 2.0 - 1.0  # Convert to -1 to 1 range
            return noise
        except Exception as e:
            # Emergency fallback - just return random-looking but deterministic values
            self.logger.warning(f"Fallback noise failed: {e}, using emergency noise")
            return np.random.RandomState(42).uniform(-1, 1, X.shape)
    
    def _combine_elevations(self, base: np.ndarray, noise: np.ndarray, 
                           params: AdvancedTerrainParameters) -> np.ndarray:
        """Combine base elevation with noise elevation."""
        # Weight the combination based on elevation variance
        noise_weight = params.elevation_variance
        base_weight = 1.0 - noise_weight
        
        combined = base_weight * base + noise_weight * noise
        
        # Preserve water areas (don't let noise push them above sea level)
        water_mask = base <= 0.2  # Areas that should remain water
        combined[water_mask] = base[water_mask]  # Keep original water elevation
        
        return combined
    
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
    
    def _generate_fallback_terrain(self, params: AdvancedTerrainParameters) -> 'TerrainMap':
        """Generate a basic fallback terrain when everything else fails."""
        self.logger.info(f"Generating fallback terrain {params.map_size}x{params.map_size}")
        
        size = params.map_size
        
        # Create basic terrain
        terrain_types = np.full((size, size), TerrainType.GRASS, dtype=object)
        elevation = np.full((size, size), 0.5)
        
        # Set random seed for reproducible fallback
        np.random.seed(params.random_seed)
        
        # Add some basic features
        for i in range(size):
            for j in range(size):
                # Add water near edges
                if i < size * 0.1 or i > size * 0.9 or j < size * 0.1 or j > size * 0.9:
                    if np.random.random() < 0.3:
                        terrain_types[i, j] = TerrainType.SEA
                        elevation[i, j] = 0.1
                
                # Add some random forest patches
                elif np.random.random() < 0.1:
                    terrain_types[i, j] = TerrainType.GRASS
                    elevation[i, j] = 0.6
                    
                # Add some hills
                elif np.random.random() < 0.05:
                    terrain_types[i, j] = TerrainType.ROCK
                    elevation[i, j] = 0.8
        
        # Create terrain map
        from .terrain_generator import TerrainMap
        terrain_map = TerrainMap(size, size, params)
        terrain_map.elevation_map = elevation
        terrain_map.tiles = terrain_types
        
        return terrain_map
    
    def _assign_terrain_types(self, elevation: np.ndarray, water_mask: np.ndarray,
                             river_mask: np.ndarray, params: AdvancedTerrainParameters) -> np.ndarray:
        """Step 7: Assign terrain types based on elevation and features."""
        size = elevation.shape[0]
        terrain_types = np.full((size, size), TerrainType.GRASS, dtype=object)
        
        # Water areas (sea level and below)
        sea_mask = elevation <= 0.2
        terrain_types[sea_mask] = TerrainType.SEA
        
        # Additional water features (lakes, rivers)
        terrain_types[water_mask] = TerrainType.SEA
        terrain_types[river_mask] = TerrainType.SEA
        
        # Coastal areas (just above sea level)
        coastal_mask = (elevation > 0.2) & (elevation <= 0.3)
        terrain_types[coastal_mask] = TerrainType.COAST
        
        # Elevation-based land assignment
        low_land_mask = (elevation > 0.3) & (elevation <= 0.5)
        terrain_types[low_land_mask] = TerrainType.GRASS
        
        mid_land_mask = (elevation > 0.5) & (elevation <= 0.7)
        terrain_types[mid_land_mask] = TerrainType.GRASS
        
        high_land_mask = (elevation > 0.7) & (elevation <= 0.85)
        terrain_types[high_land_mask] = TerrainType.ROCK
        
        # Mountain peaks
        mountain_mask = elevation > 0.85
        terrain_types[mountain_mask] = TerrainType.ROCK
        
        # TODO: Add moisture and temperature considerations for more diverse biomes
        
        return terrain_types


 