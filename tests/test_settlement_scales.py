#!/usr/bin/env python3
"""
Test script for different settlement scales.

This script demonstrates terrain generation for different settlement types.
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from townsim.modules.terrain.terrain_generator import TerrainGenerator
from townsim.modules.terrain.terrain_parameters import TerrainParameters, PRESET_PARAMETERS
from townsim.modules.terrain.terrain_types import TerrainType
from townsim.modules.utils.logging_setup import setup_logging

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def test_settlement_scales():
    """Test terrain generation for different settlement scales."""
    print("Testing Settlement Scale Terrain Generation")
    print("=" * 50)
    
    # Setup logging
    setup_logging()
    
    # Create generator
    generator = TerrainGenerator()
    
    # Settlement presets to test
    settlement_presets = [
        "Small Village",
        "Large Village", 
        "Small Town",
        "Large Town"
        # Skip Small City for now as 2048x2048 takes a while
    ]
    
    for preset_name in settlement_presets:
        print(f"\n🏘️  Testing {preset_name}")
        print("-" * 30)
        
        params = PRESET_PARAMETERS[preset_name].copy()
        params.random_seed = 42  # Consistent results
        
        # Calculate area
        tiles = params.map_size * params.map_size
        area_km2 = tiles * (0.0000286)  # ~5.3m per tile = 28.6m² per tile
        
        print(f"Map Size: {params.map_size}×{params.map_size} tiles")
        print(f"Estimated Area: {area_km2:.1f} km²")
        print(f"Water Bodies: {params.water_bodies}")
        
        # Generate terrain
        terrain_map = generator.generate_terrain(params)
        
        # Analyze terrain composition
        terrain_counts = {}
        for y in range(terrain_map.height):
            for x in range(terrain_map.width):
                terrain_type = terrain_map.tiles[y, x]
                terrain_counts[terrain_type] = terrain_counts.get(terrain_type, 0) + 1
        
        total_tiles = terrain_map.width * terrain_map.height
        
        print("Terrain Composition:")
        for terrain_type, count in terrain_counts.items():
            percentage = (count / total_tiles) * 100
            print(f"  {terrain_type.value.title()}: {percentage:.1f}% ({count:,} tiles)")
        
        # Calculate buildable area (for settlement planning)
        buildable_tiles = sum(count for terrain_type, count in terrain_counts.items() 
                             if terrain_type in [TerrainType.GRASS, TerrainType.SAND, TerrainType.COAST])
        buildable_percentage = (buildable_tiles / total_tiles) * 100
        buildable_area = buildable_tiles * 0.0000286  # 28.6m² per tile
        
        print(f"Buildable Area: {buildable_percentage:.1f}% ({buildable_area:.2f} km²)")
        
        # Estimate theoretical population capacity
        # Assume ~100 people per hectare for mixed residential/commercial
        theoretical_population = buildable_area * 100 * 100  # km² to hectares * 100 people/hectare
        
        print(f"Theoretical Population Capacity: {theoretical_population:,.0f} people")
        
        # Check if it matches expected settlement type
        if preset_name == "Small Village" and theoretical_population > 1000:
            print("⚠️  Warning: Capacity exceeds village scale")
        elif preset_name in ["Large Village", "Small Town"] and theoretical_population > 20000:
            print("⚠️  Warning: Capacity exceeds town scale")
    
    print(f"\n✅ Settlement scale testing completed!")


def create_settlement_comparison():
    """Create a visual comparison of different settlement scales."""
    if not PIL_AVAILABLE:
        print("PIL not available - skipping image generation")
        return
    
    print("\n🖼️  Creating settlement scale comparison images...")
    
    generator = TerrainGenerator()
    
    # Generate smaller examples for comparison
    settlement_configs = [
        ("Village", 256, 1),
        ("Town", 512, 2), 
        ("City", 768, 4)
    ]
    
    for name, size, water_bodies in settlement_configs:
        params = TerrainParameters(
            map_size=size,
            water_bodies=water_bodies,
            random_seed=42
        )
        
        terrain_map = generator.generate_terrain(params)
        
        # Create image
        image = Image.new('RGB', (size, size))
        
        for y in range(size):
            for x in range(size):
                terrain_type = terrain_map.tiles[y, x]
                from townsim.modules.terrain.terrain_types import get_terrain_color
                color = get_terrain_color(terrain_type)
                image.putpixel((x, y), color)
        
        # Scale up for visibility
        scale_factor = max(1, 512 // size)
        image = image.resize((size * scale_factor, size * scale_factor), Image.NEAREST)
        
        filename = f"settlement_{name.lower()}_scale.png"
        image.save(filename)
        print(f"  Saved: {filename}")


def main():
    """Main test function."""
    try:
        test_settlement_scales()
        create_settlement_comparison()
        
        print("\n🎉 All settlement scale tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 