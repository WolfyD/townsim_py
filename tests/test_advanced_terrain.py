#!/usr/bin/env python3
"""
Test script for the Advanced Terrain Generator

Demonstrates the new 7-step geological terrain generation approach
with performance improvements and better quality output.
"""

import sys
import time
import numpy as np
from pathlib import Path
import random

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from townsim.modules.terrain.advanced_terrain_generator import (
    AdvancedTerrainGenerator, 
    AdvancedTerrainParameters,
    CoastalType
)
from townsim.modules.terrain.terrain_generator import TerrainGenerator
from townsim.modules.terrain.terrain_parameters import TerrainParameters
from townsim.modules.terrain.terrain_types import TerrainType

def save_terrain_image(terrain_map, filename: str):
    """Save terrain map as a PNG image."""
    try:
        from PIL import Image
        
        # Create RGB image
        size = terrain_map.tiles.shape[0]
        img_array = np.zeros((size, size, 3), dtype=np.uint8)
        
        # Color mapping for terrain types
        colors = {
            TerrainType.GRASS: (34, 139, 34),    # Forest green
            TerrainType.MUD: (139, 69, 19),      # Saddle brown
            TerrainType.SAND: (238, 203, 173),   # Peach puff
            TerrainType.ROCK: (105, 105, 105),   # Dim gray
            TerrainType.COAST: (255, 218, 185),  # Peach
            TerrainType.SEA: (70, 130, 180),     # Steel blue
            TerrainType.SNOW: (255, 250, 250)    # Snow
        }
        
        # Apply colors
        for terrain_type, color in colors.items():
            mask = terrain_map.tiles == terrain_type
            img_array[mask] = color
        
        # Save image
        img = Image.fromarray(img_array)
        img.save(filename)
        print(f"Saved terrain image: {filename}")
        
    except ImportError:
        print("PIL not available - cannot save image")

def save_elevation_image(elevation_map, filename: str):
    """Save elevation map as a grayscale PNG."""
    try:
        from PIL import Image
        
        # Convert to 0-255 grayscale
        elevation_normalized = (elevation_map * 255).astype(np.uint8)
        
        # Save image
        img = Image.fromarray(elevation_normalized, mode='L')
        img.save(filename)
        print(f"Saved elevation image: {filename}")
        
    except ImportError:
        print("PIL not available - cannot save image")

def test_advanced_terrain_generator():
    """Test the new advanced terrain generator."""
    
    print("=== Advanced Terrain Generator Test ===\n")
    
    # Test parameters
    map_size = 512
    random_seed = 42
    
    # Create advanced parameters
    params = AdvancedTerrainParameters(
        map_size=map_size,
        random_seed=random_seed,
        coastal_type=CoastalType.COASTAL,  # Force coastal for better demo
        terrain_smoothness=0.3,
        max_lakes=2,
        noise_scale=5.0,
        elevation_variance=0.7,
        elevation_exponent=1.3
    )
    
    print(f"Generating {map_size}x{map_size} terrain with advanced generator...")
    print(f"Coastal type: {params.coastal_type}")
    print(f"Terrain smoothness: {params.terrain_smoothness}")
    print(f"Noise scale: {params.noise_scale}")
    print(f"Elevation variance: {params.elevation_variance}")
    print(f"Elevation exponent: {params.elevation_exponent}")
    
    # Generate terrain
    generator = AdvancedTerrainGenerator()
    start_time = time.time()
    terrain_map = generator.generate_terrain(params)
    elapsed_time = time.time() - start_time
    
    print(f"\nGeneration completed in {elapsed_time:.2f} seconds")
    print(f"That's {(map_size * map_size) / elapsed_time:.0f} tiles per second")
    
    # Save results
    save_terrain_image(terrain_map, "advanced_terrain_512x512.png")
    save_elevation_image(terrain_map.elevation_map, "advanced_elevation_512x512.png")
    
    # Print statistics
    print(f"\nTerrain Statistics:")
    print(f"Elevation range: {terrain_map.elevation_map.min():.3f} - {terrain_map.elevation_map.max():.3f}")
    print(f"Elevation mean: {terrain_map.elevation_map.mean():.3f}")
    print(f"Elevation std: {terrain_map.elevation_map.std():.3f}")
    
    # Check for water areas
    water_areas = terrain_map.elevation_map <= 0.2
    if np.any(water_areas):
        print(f"Water areas found: {np.sum(water_areas)} pixels at elevation <= 0.2")
    else:
        print("⚠ No water areas found at elevation <= 0.2")
    
    # Count terrain types
    print(f"\nTerrain type distribution:")
    total_tiles = map_size * map_size
    for terrain_type in TerrainType:
        count = np.sum(terrain_map.tiles == terrain_type)
        if count > 0:
            percentage = (count / total_tiles) * 100
            print(f"  {terrain_type.name}: {count} tiles ({percentage:.1f}%)")

def test_different_coastal_types():
    """Test different coastal configurations."""
    
    print("\n=== Testing Different Coastal Types ===\n")
    
    coastal_types = [CoastalType.LANDLOCKED, CoastalType.COASTAL, CoastalType.ISLAND]
    map_size = 256
    
    for coastal_type in coastal_types:
        print(f"Testing {coastal_type.name}...")
        
        params = AdvancedTerrainParameters(
            map_size=map_size,
            random_seed=42,
            coastal_type=coastal_type,
            terrain_smoothness=0.2,
            max_lakes=2,
            noise_scale=4.0
        )
        
        generator = AdvancedTerrainGenerator()
        start_time = time.time()
        terrain_map = generator.generate_terrain(params)
        elapsed_time = time.time() - start_time
        
        print(f"  Generated in {elapsed_time:.2f}s")
        
        # Analyze coastal separation
        if coastal_type == CoastalType.COASTAL:
            sea_count = np.sum(terrain_map.tiles == TerrainType.SEA)
            sea_percentage = (sea_count / (map_size * map_size)) * 100
            print(f"  Sea coverage: {sea_percentage:.1f}% (should be 20-30%)")
        
        # Save image
        filename = f"advanced_{coastal_type.name.lower()}_256x256.png"
        save_terrain_image(terrain_map, filename)

def test_coastal_boundary_separation():
    """Test that coastal areas have proper land/sea separation."""
    
    print("\n=== Testing Coastal Boundary Separation ===\n")
    
    # Test different coastal angles
    map_size = 256
    test_angles = [45, 135, 225, 315]  # Different angles to test varied coastlines
    
    for i, angle in enumerate(test_angles):
        print(f"Testing coastal boundary at {angle}° angle...")
        
        # Force specific angle for testing
        params = AdvancedTerrainParameters(
            map_size=map_size,
            random_seed=123 + i,  # Different seed for each test
            coastal_type=CoastalType.COASTAL,
            terrain_smoothness=0.1,  # Less smooth to show boundary clearly
            max_lakes=2,
            noise_scale=3.0
        )
        
        generator = AdvancedTerrainGenerator()
        
        # Temporarily override the coastal configuration for testing
        original_config = generator._determine_coastal_config(params)
        test_config = {
            "type": CoastalType.COASTAL,
            "coast_angle": angle,
            "coast_depth": 0.25  # Fixed depth for consistent testing
        }
        
        # Generate terrain with test configuration
        size = params.map_size
        random.seed(params.random_seed)
        np.random.seed(params.random_seed)
        
        # Create coordinate grids
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        X, Y = np.meshgrid(x, y)
        
        # Initialize noise amplitudes if not set
        if params.noise_amplitudes is None:
            params.noise_amplitudes = [1.0, 0.5, 0.25, 0.125]
        
        # Generate with specific angle
        base_elevation = generator._generate_base_elevation(size, test_config, params)
        noise_elevation = generator._generate_noise_elevation(size, params)
        elevation_map = generator._combine_elevations(base_elevation, noise_elevation, params)
        
        if params.terrain_smoothness > 0.1:
            elevation_map = generator._apply_smoothing(elevation_map, params.terrain_smoothness)
        
        water_mask = generator._generate_water_bodies(elevation_map, params)
        river_mask = generator._generate_rivers(elevation_map, params)
        final_elevation = generator._apply_water_features(elevation_map, water_mask, river_mask)
        terrain_types = generator._assign_terrain_types(final_elevation, water_mask, river_mask, params)
        
        # Create terrain map
        from townsim.modules.terrain.terrain_generator import TerrainMap
        terrain_map = TerrainMap(size, size, params)
        terrain_map.elevation_map = final_elevation
        terrain_map.tiles = terrain_types
        
        # Analyze the boundary
        sea_mask = terrain_map.tiles == TerrainType.SEA
        sea_count = np.sum(sea_mask)
        sea_percentage = (sea_count / (map_size * map_size)) * 100
        
        print(f"  Angle {angle}°: Sea coverage {sea_percentage:.1f}%")
        
        # Verify water rendering
        if sea_count > 0:
            print("  ✓ Water is being rendered")
        else:
            print("  ⚠ No water tiles found")
        
        # Save with descriptive filename
        filename = f"coastal_angle_{angle}_deg_256x256.png"
        save_terrain_image(terrain_map, filename)
    
    print("\n✓ Coastal angle testing complete - check the generated images!")
    print("Each image should show a different angled coastline with natural variation.")

def test_natural_island_edges():
    """Test that islands have natural, varied edges."""
    
    print("\n=== Testing Natural Island Edges ===\n")
    
    map_size = 256
    
    print("Testing natural island generation...")
    params = AdvancedTerrainParameters(
        map_size=map_size,
        random_seed=456,
        coastal_type=CoastalType.ISLAND,
        terrain_smoothness=0.2,
        max_lakes=2,
        noise_scale=4.0
    )
    
    generator = AdvancedTerrainGenerator()
    terrain_map = generator.generate_terrain(params)
    
    # Analyze the island
    sea_mask = terrain_map.tiles == TerrainType.SEA
    land_mask = ~sea_mask
    
    sea_count = np.sum(sea_mask)
    land_count = np.sum(land_mask)
    sea_percentage = (sea_count / (map_size * map_size)) * 100
    land_percentage = (land_count / (map_size * map_size)) * 100
    
    print(f"Sea coverage: {sea_percentage:.1f}%")
    print(f"Land coverage: {land_percentage:.1f}%")
    
    # Check elevation distribution
    print(f"Elevation range: {terrain_map.elevation_map.min():.3f} - {terrain_map.elevation_map.max():.3f}")
    
    # Verify water rendering
    if sea_count > 0:
        print("✓ Island water is being rendered")
    else:
        print("⚠ No water around island")
    
    # Save with descriptive filename
    save_terrain_image(terrain_map, "natural_island_256x256.png")
    save_elevation_image(terrain_map.elevation_map, "natural_island_elevation_256x256.png")
    
    print("✓ Natural island test complete - check natural_island_256x256.png!")
    print("The island should have irregular, natural-looking edges.")

def test_smoothness_levels():
    """Test different smoothness levels."""
    
    print("\n=== Testing Smoothness Levels ===\n")
    
    smoothness_levels = [0.0, 0.3, 0.7, 1.0]
    map_size = 256
    
    for smoothness in smoothness_levels:
        print(f"Testing smoothness level {smoothness}...")
        
        params = AdvancedTerrainParameters(
            map_size=map_size,
            random_seed=42,
            coastal_type=CoastalType.LANDLOCKED,
            terrain_smoothness=smoothness,
            max_lakes=2,
            noise_scale=4.0
        )
        
        generator = AdvancedTerrainGenerator()
        start_time = time.time()
        terrain_map = generator.generate_terrain(params)
        elapsed_time = time.time() - start_time
        
        print(f"  Generated in {elapsed_time:.2f}s")
        
        # Save image
        filename = f"advanced_smooth_{smoothness:.1f}_256x256.png"
        save_terrain_image(terrain_map, filename)

def compare_with_old_generator():
    """Compare new generator with old generator."""
    
    print("\n=== Comparing with Old Generator ===\n")
    
    map_size = 512
    random_seed = 42
    
    # Test old generator
    print("Testing old generator...")
    old_params = TerrainParameters(
        map_size=map_size,
        random_seed=random_seed,
        noise_scale=5.0,
        elevation_variance=0.7
    )
    
    old_generator = TerrainGenerator()
    start_time = time.time()
    old_terrain = old_generator.generate_terrain(old_params)
    old_time = time.time() - start_time
    
    print(f"Old generator: {old_time:.2f}s")
    save_terrain_image(old_terrain, "old_terrain_512x512.png")
    
    # Test new generator
    print("Testing new generator...")
    new_params = AdvancedTerrainParameters(
        map_size=map_size,
        random_seed=random_seed,
        coastal_type=CoastalType.COASTAL,  # Force coastal for better comparison
        terrain_smoothness=0.3,
        max_lakes=2,
        noise_scale=5.0,
        elevation_variance=0.7
    )
    
    new_generator = AdvancedTerrainGenerator()
    start_time = time.time()
    new_terrain = new_generator.generate_terrain(new_params)
    new_time = time.time() - start_time
    
    print(f"New generator: {new_time:.2f}s")
    save_terrain_image(new_terrain, "new_terrain_512x512.png")
    
    # Compare performance
    speedup = old_time / new_time
    print(f"\nSpeed comparison: New generator is {speedup:.1f}x faster")

if __name__ == "__main__":
    # Run all tests
    test_advanced_terrain_generator()
    test_different_coastal_types()
    test_coastal_boundary_separation()
    test_natural_island_edges()
    test_smoothness_levels()
    compare_with_old_generator()
    
    print("\n=== Test Complete ===")
    print("Check the generated PNG files to see the results!")
    print("\nKey improvements:")
    print("- Coastal areas now use single angled boundaries (any angle 0-360°)")
    print("- Coastlines have natural variation with noise for realistic edges")
    print("- Islands have irregular, natural-looking boundaries")
    print("- Water areas are properly rendered with blue coloring")
    print("- Lakes are limited to maximum of 2 (configurable)")
    print("- Generated files include multiple angle tests and natural island examples") 