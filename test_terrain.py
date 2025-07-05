#!/usr/bin/env python3
"""
Test script for terrain generation.

This script tests the terrain generation system and creates a simple visualization.
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from townsim.modules.terrain.terrain_generator import TerrainGenerator
from townsim.modules.terrain.terrain_parameters import TerrainParameters, PRESET_PARAMETERS
from townsim.modules.terrain.terrain_types import TerrainType, get_terrain_color
from townsim.modules.utils.logging_setup import setup_logging

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def test_terrain_generation():
    """Test basic terrain generation functionality."""
    print("Testing terrain generation...")
    
    # Setup logging
    setup_logging()
    
    # Create generator
    generator = TerrainGenerator()
    
    # Test with default parameters
    print("\n1. Testing with default parameters...")
    params = TerrainParameters(map_size=64, random_seed=42)
    terrain_map = generator.generate_terrain(params)
    
    print(f"   Generated terrain: {terrain_map.width}x{terrain_map.height}")
    print(f"   Terrain types found:")
    
    # Count terrain types
    terrain_counts = {}
    for y in range(terrain_map.height):
        for x in range(terrain_map.width):
            terrain_type = terrain_map.tiles[y, x]
            terrain_counts[terrain_type] = terrain_counts.get(terrain_type, 0) + 1
    
    for terrain_type, count in terrain_counts.items():
        percentage = (count / (terrain_map.width * terrain_map.height)) * 100
        print(f"   - {terrain_type.value}: {count} tiles ({percentage:.1f}%)")
    
    # Test with preset parameters
    print("\n2. Testing with preset parameters...")
    for preset_name, preset_params in PRESET_PARAMETERS.items():
        print(f"   Testing preset: {preset_name}")
        preset_params.map_size = 32  # Smaller for faster testing
        preset_params.random_seed = 42
        
        terrain_map = generator.generate_terrain(preset_params)
        
        # Count water vs land
        water_count = sum(1 for y in range(terrain_map.height) 
                         for x in range(terrain_map.width) 
                         if terrain_map.tiles[y, x] == TerrainType.SEA)
        land_count = terrain_map.width * terrain_map.height - water_count
        
        print(f"   - Water: {water_count} tiles, Land: {land_count} tiles")
    
    print("\n3. Testing parameter validation...")
    # Test parameter validation
    invalid_params = TerrainParameters(
        map_size=100,  # Will be rounded up to 128
        elevation_variance=1.5,  # Will be clamped to 1.0
        wind_direction=450.0,  # Will be wrapped to 90.0
        water_bodies=-1  # Will be clamped to 0
    )
    
    print(f"   Original map_size: 100, corrected: {invalid_params.map_size}")
    print(f"   Original elevation_variance: 1.5, corrected: {invalid_params.elevation_variance}")
    print(f"   Original wind_direction: 450.0, corrected: {invalid_params.wind_direction}")
    print(f"   Original water_bodies: -1, corrected: {invalid_params.water_bodies}")
    
    print("\nTerrain generation tests completed successfully!")
    return terrain_map


def create_terrain_image(terrain_map, filename="terrain_test.png"):
    """Create a visual representation of the terrain."""
    if not PIL_AVAILABLE:
        print("PIL not available - skipping image generation")
        return
    
    print(f"\nCreating terrain image: {filename}")
    
    width, height = terrain_map.width, terrain_map.height
    image = Image.new('RGB', (width, height))
    
    for y in range(height):
        for x in range(width):
            terrain_type = terrain_map.tiles[y, x]
            color = get_terrain_color(terrain_type)
            image.putpixel((x, y), color)
    
    # Scale up for better visibility
    image = image.resize((width * 4, height * 4), Image.NEAREST)
    image.save(filename)
    print(f"Terrain image saved as: {filename}")


def main():
    """Main test function."""
    print("TownSim Terrain Generation Test")
    print("=" * 40)
    
    try:
        terrain_map = test_terrain_generation()
        create_terrain_image(terrain_map)
        
        print("\nAll tests passed! ✓")
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 