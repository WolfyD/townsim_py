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
    coastline_angle: Optional[float] = None  # Override angle for coastal type (0-360°)
    
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
    
    # Debug options
    show_debug_markers: bool = False  # Show red segment markers for debugging


class AdvancedTerrainGenerator:
    """Advanced terrain generator using geological principles."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        if not NOISE_AVAILABLE:
            self.logger.warning("noise library not available - using fallback")
        
        # Debug marker storage
        self.debug_markers = []  # List of (x, y, marker_type) tuples
    
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
            
            # Clear debug markers
            self.debug_markers = []
            
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
            
            # Add debug markers if enabled
            if params.show_debug_markers:
                terrain_types = self._add_debug_markers(terrain_types, params)
            
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
            # Use user-specified angle if provided, otherwise random
            if params.coastline_angle is not None:
                coast_angle = params.coastline_angle
                self.logger.debug(f"Using user-specified coastline angle: {coast_angle:.1f}°")
            else:
                # Generate a random angled coastline (not just cardinal directions)
                coast_angle = random.uniform(0, 360)
                self.logger.debug(f"Using random coastline angle: {coast_angle:.1f}°")
            
            # Position the coastline to create realistic coastal coverage
            # Offset the coastline so there's less water (20-30% instead of 50%)
            coast_depth = random.uniform(-0.4, -0.2)  # Negative = less water
            
            config["coast_angle"] = coast_angle
            config["coast_depth"] = coast_depth
            
        elif config_type == CoastalType.ISLAND:
            # Island surrounded by water - ensure water is visible at all edges
            config["coastal_edges"] = ["north", "south", "east", "west"]
            config["island_center"] = (
                random.uniform(0.4, 0.6),  # x center (more centered)
                random.uniform(0.4, 0.6)   # y center (more centered)
            )
            config["island_size"] = random.uniform(0.35, 0.40)  # radius as fraction of map (smaller to ensure water around edges)
        
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
        # At angle 0°: land points north (up), coastline is horizontal
        # At angle 90°: land points east (right), coastline is vertical
        angle_rad = np.radians(coast_angle)
        
        # Normal vector points toward land
        # At 0°: land points north (up), so normal = (0, -1) in numpy coordinates
        # At 90°: land points east (right), so normal = (1, 0) in numpy coordinates
        # This gives us: normal_x = sin(angle), normal_y = -cos(angle)
        normal_x = np.sin(angle_rad)
        normal_y = -np.cos(angle_rad)
        
        # Position the line so it cuts through the map center
        center_x, center_y = 0.5, 0.5
        offset = coast_depth  # Offset from center
        
        # Distance from each point to the line
        distance_to_line = normal_x * (X - center_x) + normal_y * (Y - center_y) - offset
        
        # Add noise to make the coastline more natural
        coastline_noise = self._generate_coastline_noise(X, Y, params)
        
        # Create water mask: points on the "water side" of the line
        # Water is on the side opposite to where the land points
        water_mask = distance_to_line + coastline_noise < 0
        
        # Apply sea level to water areas
        result[water_mask] = 0.15  # Sea level
        
        # Apply additional smoothing to coastline transitions for flowing curves
        result = self._smooth_coastline_transitions(result, water_mask)
        
        return result
    
    def _generate_coastline_noise(self, X: np.ndarray, Y: np.ndarray, 
                                 params: AdvancedTerrainParameters) -> np.ndarray:
        """Generate segmented coastline with varying angles and smoothing at joints."""
        # Create segmented coastline with convex and concave features
        coastline_variation = np.zeros_like(X)
        
        # Set random seed for reproducible coastline segments
        np.random.seed(params.random_seed + 500)
        
        # Create random coastline segments (4-10 segments)
        num_segments = random.randint(4, 10)
        segment_positions = np.sort(np.random.uniform(0, 1, num_segments))
        
        # Generate intelligent angle adjustments with alternating tendency
        angle_adjustments = self._generate_alternating_angles(num_segments)
        
        # Generate random smoothing levels for each joint (0-10)
        smoothing_levels = np.random.uniform(0, 10, num_segments)
        
        # Record segment positions for debug markers (will be positioned after coastline is created)
        if params.show_debug_markers:
            # Store segment positions for later placement along actual coastline
            for i, seg_pos in enumerate(segment_positions):
                self.debug_markers.append((seg_pos, angle_adjustments[i], 'coastal_segment_data'))
        
        # Create coastline variation for each segment
        for i in range(num_segments):
            # Current segment position (0-1 along coastline)
            segment_pos = segment_positions[i]
            angle_adj = angle_adjustments[i]
            smoothing = smoothing_levels[i]
            
            # Create segment influence based on distance from segment center
            if i == 0:
                # First segment: influence from start to midpoint with next
                next_pos = segment_positions[i + 1] if i + 1 < num_segments else 1.0
                segment_center = segment_pos
                segment_width = (next_pos - segment_pos) * 0.5
            elif i == num_segments - 1:
                # Last segment: influence from midpoint with previous to end
                prev_pos = segment_positions[i - 1]
                segment_center = segment_pos
                segment_width = (segment_pos - prev_pos) * 0.5
            else:
                # Middle segments: influence between adjacent segments
                prev_pos = segment_positions[i - 1]
                next_pos = segment_positions[i + 1]
                segment_center = segment_pos
                segment_width = min((segment_pos - prev_pos) * 0.5, (next_pos - segment_pos) * 0.5)
            
            # Create segment influence mask
            # Use both X and Y coordinates to create varied influence patterns
            segment_influence = self._create_segment_influence(X, Y, segment_center, segment_width, angle_adj, smoothing)
            coastline_variation += segment_influence
        
        # Add base flowing pattern to connect segments smoothly
        base_flow = np.sin(X * math.pi * 2 * 0.6) * 0.004  # Subtle base flow
        coastline_variation += base_flow
        
        return coastline_variation
    
    def _generate_alternating_angles(self, num_segments: int) -> np.ndarray:
        """Generate intelligent angle adjustments with alternating tendency - SEVERE angles for visibility."""
        angle_adjustments = np.zeros(num_segments)
        
        for i in range(num_segments):
            if i == 0:
                # First segment: random angle within SEVERE range
                if random.random() < 0.5:  # 50% chance for moderate severe angle
                    angle_adjustments[i] = random.uniform(-60, 60)
                else:  # 50% chance for extremely severe angle
                    angle_adjustments[i] = random.uniform(-90, 90)
            else:
                # Subsequent segments: consider previous angle for alternating pattern
                prev_angle = angle_adjustments[i-1]
                
                # Determine tendency based on previous angle
                if abs(prev_angle) > 45:  # Previous was a big change
                    # Strong tendency to alternate direction (80% chance)
                    if random.random() < 0.8:
                        # Opposite direction with SEVERE angles
                        if prev_angle > 0:
                            # Previous was clockwise, go counter-clockwise
                            base_angle = random.uniform(-80, -20)
                        else:
                            # Previous was counter-clockwise, go clockwise
                            base_angle = random.uniform(20, 80)
                    else:
                        # 20% chance for randomness with severe angles
                        base_angle = random.uniform(-70, 70)
                else:
                    # Previous was moderate, more random but still some alternating bias
                    if random.random() < 0.6:  # 60% chance to alternate
                        if prev_angle > 0:
                            base_angle = random.uniform(-70, 20)
                        else:
                            base_angle = random.uniform(-20, 70)
                    else:
                        # 40% chance for full randomness with severe angles
                        base_angle = random.uniform(-60, 60)
                
                # Apply limits: up to 120° but usually less than 90°
                if random.random() < 0.25:  # 25% chance for maximum angle
                    max_angle = 120
                else:
                    max_angle = 90
                
                # Clamp to limits
                angle_adjustments[i] = np.clip(base_angle, -max_angle, max_angle)
        
        return angle_adjustments
    
    def _generate_alternating_radii(self, num_segments: int) -> np.ndarray:
        """Generate intelligent radius adjustments with alternating tendency for islands - GENTLE changes for natural coastlines."""
        radius_adjustments = np.zeros(num_segments)
        
        for i in range(num_segments):
            if i == 0:
                # First segment: random radius within GENTLE range for natural islands
                if random.random() < 0.7:  # 70% chance for moderate gentle change
                    radius_adjustments[i] = random.uniform(-0.08, 0.08)  # ±8%
                else:  # 30% chance for slightly larger change
                    radius_adjustments[i] = random.uniform(-0.12, 0.12)  # ±12%
            else:
                # Subsequent segments: consider previous radius for alternating pattern
                prev_radius = radius_adjustments[i-1]
                
                # Determine tendency based on previous radius
                if abs(prev_radius) > 0.06:  # Previous was a bigger change
                    # Strong tendency to alternate direction (80% chance)
                    if random.random() < 0.8:
                        # Opposite direction with GENTLE changes
                        if prev_radius > 0:
                            # Previous was outward, go inward
                            base_radius = random.uniform(-0.1, -0.02)
                        else:
                            # Previous was inward, go outward
                            base_radius = random.uniform(0.02, 0.1)
                    else:
                        # 20% chance for randomness with gentle changes
                        base_radius = random.uniform(-0.08, 0.08)
                else:
                    # Previous was moderate, more random but still some alternating bias
                    if random.random() < 0.6:  # 60% chance to alternate
                        if prev_radius > 0:
                            base_radius = random.uniform(-0.08, 0.02)
                        else:
                            base_radius = random.uniform(-0.02, 0.08)
                    else:
                        # 40% chance for full randomness with gentle changes
                        base_radius = random.uniform(-0.06, 0.06)
                
                # Apply limits: up to 15% but usually less than 10%
                if random.random() < 0.15:  # 15% chance for maximum radius
                    max_radius = 0.15
                else:
                    max_radius = 0.1
                
                # Clamp to limits
                radius_adjustments[i] = np.clip(base_radius, -max_radius, max_radius)
        
        return radius_adjustments
    
    def _create_segment_influence(self, X: np.ndarray, Y: np.ndarray, center: float, 
                                 width: float, angle_adj: float, smoothing: float) -> np.ndarray:
        """Create influence pattern for a single coastline segment."""
        # Convert angle adjustment to radians
        angle_rad = np.radians(angle_adj)
        
        # Create directional pattern based on angle adjustment
        # Positive angle = convex bulge outward, negative = concave indent inward
        if angle_adj > 0:
            # Convex feature - bulge outward
            pattern = np.sin((X - center) * math.pi / width) * np.cos(Y * math.pi * 2)
            amplitude = 0.08 * (angle_adj / 120.0)  # Scale by angle (up to 8% for 120° angle)
        else:
            # Concave feature - indent inward  
            pattern = -np.sin((X - center) * math.pi / width) * np.cos(Y * math.pi * 2)
            amplitude = 0.08 * (abs(angle_adj) / 120.0)  # Scale by angle (up to 8% for 120° angle)
        
        # Apply smoothing to the pattern
        if smoothing > 5:
            # Smooth rounded features (smoothing 5-10)
            smooth_factor = (smoothing - 5) / 5.0  # 0-1 scale
            pattern = pattern * (1 - smooth_factor) + np.sin(pattern * math.pi) * smooth_factor
        
        # Create distance-based falloff from segment center
        distance_from_center = np.abs(X - center)
        falloff = np.exp(-distance_from_center / (width * 2))  # Exponential falloff
        
        # Apply amplitude and falloff
        segment_influence = pattern * amplitude * falloff
        
        return segment_influence
    
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
        # 1. Start with a natural circular base
        base_distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        
        # 2. Add multiple influence points for organic variation (fewer points, less square)
        num_centers = random.randint(2, 4)  # 2-4 influence points (reduced from 3-6)
        organic_distance = base_distance.copy()
        
        for _ in range(num_centers):
            # Smaller offsets to keep circular shape
            offset_x = random.uniform(-0.08, 0.08)  # Reduced from -0.15, 0.15
            offset_y = random.uniform(-0.08, 0.08)  # Reduced from -0.15, 0.15
            weight = random.uniform(0.1, 0.3)  # Lighter influence
            
            cx = center_x + offset_x
            cy = center_y + offset_y
            
            # Add subtle variation, not dominant shape change
            dist_variation = np.sqrt((X - cx)**2 + (Y - cy)**2)
            organic_distance += weight * (dist_variation - base_distance)
        
        # Keep the result close to circular
        organic_distance = base_distance + (organic_distance - base_distance) * 0.3
        
        # 3. Add large-scale shape variation
        shape_noise = self._generate_island_shape_noise(X, Y, center_x, center_y, params)
        
        # 4. Add medium-scale edge variation  
        edge_noise = self._generate_island_edge_noise(X, Y, center_x, center_y, params)
        
        # Combine all influences for final island boundary
        final_radius = island_radius + shape_noise + edge_noise
        
        # Ensure island doesn't extend to map edges - enforce water boundary
        max_allowed_radius = min(
            center_x, 1.0 - center_x,  # Distance to left/right edges
            center_y, 1.0 - center_y   # Distance to top/bottom edges
        ) * 0.95  # Leave 15% buffer for water
        
        final_radius = np.minimum(final_radius, max_allowed_radius)
        
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
        
        # Apply extra smoothing specifically for islands to reduce jagged coastlines
        base_elevation = self._apply_island_smoothing(base_elevation, sea_mask)
        
        # Instead of square border, ensure natural water boundary using distance from edges
        x_coords = np.linspace(0, 1, size)
        y_coords = np.linspace(0, 1, size)
        X_border, Y_border = np.meshgrid(x_coords, y_coords)
        
        # Calculate minimum distance to any edge (creates natural circular boundary)
        edge_distance = np.minimum(
            np.minimum(X_border, 1 - X_border),  # Distance to left/right edges
            np.minimum(Y_border, 1 - Y_border)   # Distance to top/bottom edges
        )
        
        # Create natural water boundary - anything within 10% of edge becomes water
        water_boundary_mask = edge_distance < 0.03
        base_elevation[water_boundary_mask] = 0.15  # Sea level
        
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
        """Generate segmented island edge with varying angles and smoothing."""
        # Use segmented approach for island edges similar to coastlines
        rel_x = X - center_x
        rel_y = Y - center_y
        
        # Convert to polar coordinates for island edge variation
        angle = np.arctan2(rel_y, rel_x)
        distance = np.sqrt(rel_x**2 + rel_y**2)
        
        # Set random seed for reproducible island edge segments
        np.random.seed(params.random_seed + 2000)
        
        # Create random edge segments (4-10 segments around the island)
        num_segments = random.randint(4, 10)
        segment_angles = np.sort(np.random.uniform(-math.pi, math.pi, num_segments))
        
        # Generate intelligent radius adjustments with alternating tendency
        radius_adjustments = self._generate_alternating_radii(num_segments)
        
        # Generate random smoothing levels for each segment (0-10)
        smoothing_levels = np.random.uniform(0, 10, num_segments)
        
        # Record segment positions for debug markers (will be positioned after island is created)
        if params.show_debug_markers:
            # Store segment data for later placement along actual island boundary
            for i, seg_angle in enumerate(segment_angles):
                self.debug_markers.append((seg_angle, radius_adjustments[i], center_x, center_y, 'island_segment_data'))
        
        edge_variation = np.zeros_like(X)
        
        # Create edge variation for each segment
        for i in range(num_segments):
            segment_angle = segment_angles[i]
            radius_adj = radius_adjustments[i]
            smoothing = smoothing_levels[i]
            
            # Calculate segment width (angular span)
            if i == num_segments - 1:
                # Last segment: wrap around to first segment
                next_angle = segment_angles[0] + 2 * math.pi
            else:
                next_angle = segment_angles[i + 1]
            
            prev_angle = segment_angles[i - 1] if i > 0 else segment_angles[-1] - 2 * math.pi
            
            # Angular width for this segment
            segment_width = min((segment_angle - prev_angle) * 0.5, (next_angle - segment_angle) * 0.5)
            
            # Create segment influence based on angular distance
            angular_distance = np.minimum(
                np.abs(angle - segment_angle),
                np.minimum(np.abs(angle - segment_angle + 2 * math.pi), 
                          np.abs(angle - segment_angle - 2 * math.pi))
            )
            
            # Create segment influence pattern
            segment_influence = self._create_island_segment_influence(
                angular_distance, segment_width, radius_adj, smoothing, distance
            )
            edge_variation += segment_influence
        
        # Add subtle base variation to connect segments
        base_variation = np.sin(angle * 3) * 0.002  # Very subtle 3-lobe pattern (reduced from 0.005)
        edge_variation += base_variation
        
        return edge_variation
    
    def _create_island_segment_influence(self, angular_distance: np.ndarray, width: float, 
                                        radius_adj: float, smoothing: float, distance: np.ndarray) -> np.ndarray:
        """Create influence pattern for a single island edge segment."""
        # Create radial pattern based on radius adjustment
        if radius_adj > 0:
            # Convex feature - bulge outward
            pattern = np.sin(angular_distance * math.pi / width) 
            amplitude = 0.3 * radius_adj  # Scale by radius adjustment (gentle - up to 4.5% for 15% radius_adj)
        else:
            # Concave feature - indent inward
            pattern = -np.sin(angular_distance * math.pi / width)
            amplitude = 0.3 * abs(radius_adj)  # Scale by radius adjustment (gentle - up to 4.5% for 15% radius_adj)
        
        # Apply smoothing to the pattern
        if smoothing > 5:
            # Smooth rounded features (smoothing 5-10)
            smooth_factor = (smoothing - 5) / 5.0  # 0-1 scale
            pattern = pattern * (1 - smooth_factor) + np.sin(pattern * math.pi) * smooth_factor
        
        # Create angular falloff from segment center
        angular_falloff = np.exp(-angular_distance / (width * 2))  # Exponential falloff
        
        # Add distance-based modulation for more organic feel
        distance_factor = np.exp(-distance * 1.5)  # Fade effect from center
        
        # Apply amplitude and falloff
        segment_influence = pattern * amplitude * angular_falloff * (0.5 + distance_factor * 0.5)
        
        return segment_influence
    
    def _apply_island_smoothing(self, elevation: np.ndarray, water_mask: np.ndarray) -> np.ndarray:
        """Apply additional smoothing specifically for island coastlines to reduce jaggedness."""
        result = elevation.copy()
        
        try:
            from scipy.ndimage import gaussian_filter
            
            # Create a mask for coastline areas (transition between land and water)
            from scipy.ndimage import binary_dilation, binary_erosion
            
            # Expand and contract water mask to find coastline boundary
            water_expanded = binary_dilation(water_mask, iterations=3)
            water_contracted = binary_erosion(water_mask, iterations=1)
            coastline_mask = water_expanded & ~water_contracted
            
            # Apply gentle smoothing to the entire elevation map first
            smoothed_elevation = gaussian_filter(elevation, sigma=1.5, mode='reflect')
            
            # Blend smoothed version only in coastline areas
            blend_factor = 0.6  # 60% smoothed, 40% original
            result[coastline_mask] = (
                blend_factor * smoothed_elevation[coastline_mask] + 
                (1 - blend_factor) * elevation[coastline_mask]
            )
            
        except ImportError:
            # Fallback: simple averaging in coastline areas
            self.logger.debug("scipy not available - using simple island smoothing")
            
            size = elevation.shape[0]
            for y in range(2, size - 2):
                for x in range(2, size - 2):
                    if not water_mask[y, x]:  # Land pixel
                        # Check if near water (within 2 pixels)
                        near_water = False
                        for dy in range(-2, 3):
                            for dx in range(-2, 3):
                                if water_mask[y + dy, x + dx]:
                                    near_water = True
                                    break
                            if near_water:
                                break
                        
                        if near_water:
                            # Apply gentle smoothing to coastline pixels
                            neighbors = []
                            for dy in range(-1, 2):
                                for dx in range(-1, 2):
                                    neighbors.append(elevation[y + dy, x + dx])
                            result[y, x] = np.mean(neighbors) * 0.4 + elevation[y, x] * 0.6
        
        return result
    
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
    
    def _add_debug_markers(self, terrain_types: np.ndarray, params: AdvancedTerrainParameters) -> np.ndarray:
        """Add red debug markers to show segment positions along actual coastlines."""
        from .terrain_types import TerrainType
        
        result = terrain_types.copy()
        
        # Find coastline boundaries (where water meets land)
        water_mask = (terrain_types == TerrainType.SEA)
        coastline_pixels = self._find_coastline_pixels(water_mask)
        
        # Place markers along actual coastlines based on segment data
        for marker_data in self.debug_markers:
            if len(marker_data) == 3 and marker_data[2] == 'coastal_segment_data':
                # Coastal segment: (seg_pos, angle_adj, marker_type)
                seg_pos, angle_adj, marker_type = marker_data
                marker_coords = self._find_coastal_marker_position(seg_pos, coastline_pixels, params.map_size)
                
            elif len(marker_data) == 5 and marker_data[4] == 'island_segment_data':
                # Island segment: (seg_angle, radius_adj, center_x, center_y, marker_type)
                seg_angle, radius_adj, center_x, center_y, marker_type = marker_data
                marker_coords = self._find_island_marker_position(seg_angle, center_x, center_y, coastline_pixels, params.map_size)
            
            else:
                continue  # Skip unknown marker types
            
            # Add 5x5 marker at found position
            if marker_coords:
                mx, my = marker_coords
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        px, py = mx + dx, my + dy
                        if 0 <= px < params.map_size and 0 <= py < params.map_size:
                            result[py, px] = TerrainType.ROCK  # Will render as red
        
        return result
    
    def _find_coastline_pixels(self, water_mask: np.ndarray) -> List[Tuple[int, int]]:
        """Find all pixels that are on the coastline (land pixels adjacent to water)."""
        coastline_pixels = []
        height, width = water_mask.shape
        
        for y in range(height):
            for x in range(width):
                if not water_mask[y, x]:  # Land pixel
                    # Check if adjacent to water
                    adjacent_to_water = False
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < width and 0 <= ny < height:
                                if water_mask[ny, nx]:  # Adjacent water found
                                    adjacent_to_water = True
                                    break
                        if adjacent_to_water:
                            break
                    
                    if adjacent_to_water:
                        coastline_pixels.append((x, y))
        
        return coastline_pixels
    
    def _find_coastal_marker_position(self, seg_pos: float, coastline_pixels: List[Tuple[int, int]], map_size: int) -> Optional[Tuple[int, int]]:
        """Find the best coastline pixel position for a coastal segment marker."""
        if not coastline_pixels:
            return None
        
        # For coastal segments, position is based on X coordinate
        target_x = int(seg_pos * (map_size - 1))
        
        # Find coastline pixel closest to target X position
        best_pixel = None
        best_distance = float('inf')
        
        for x, y in coastline_pixels:
            distance = abs(x - target_x)
            if distance < best_distance:
                best_distance = distance
                best_pixel = (x, y)
        
        return best_pixel
    
    def _find_island_marker_position(self, seg_angle: float, center_x: float, center_y: float, 
                                   coastline_pixels: List[Tuple[int, int]], map_size: int) -> Optional[Tuple[int, int]]:
        """Find the best coastline pixel position for an island segment marker."""
        if not coastline_pixels:
            return None
        
        # Convert center to pixel coordinates
        center_px = int(center_x * map_size)
        center_py = int(center_y * map_size)
        
        # Find coastline pixel in the direction of seg_angle from center
        target_x = center_px + math.cos(seg_angle) * map_size * 0.5
        target_y = center_py + math.sin(seg_angle) * map_size * 0.5
        
        # Find coastline pixel closest to target direction
        best_pixel = None
        best_distance = float('inf')
        
        for x, y in coastline_pixels:
            distance = math.sqrt((x - target_x)**2 + (y - target_y)**2)
            if distance < best_distance:
                best_distance = distance
                best_pixel = (x, y)
        
        return best_pixel


 