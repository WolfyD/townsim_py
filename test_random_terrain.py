#!/usr/bin/env python3
"""
Generate random terrain variations
"""

import sys
import random
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from townsim.modules.terrain.terrain_generator import TerrainGenerator
from townsim.modules.terrain.terrain_parameters import TerrainParameters
from townsim.modules.terrain.terrain_types import get_terrain_color
from townsim.modules.utils.logging_setup import setup_logging

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

def generate_random_terrain(index):
    """Generate a random terrain and save image."""
    generator = TerrainGenerator()
    
    # Create random parameters
    params = TerrainParameters(
        map_size=256,  # Smaller for speed
        random_seed=None,  # This will generate a new random seed each time
        elevation_variance=random.uniform(0.2, 0.8),
        moisture_level=random.uniform(0.3, 0.8),
        base_temperature=random.uniform(0.3, 0.8),
        water_bodies=random.randint(1, 4),
        wind_direction=random.uniform(0, 360)
    )
    
    print(f"Terrain {index}:")
    print(f"  Seed: {params.random_seed}")
    print(f"  Elevation: {params.elevation_variance:.2f}")
    print(f"  Moisture: {params.moisture_level:.2f}")
    print(f"  Temperature: {params.base_temperature:.2f}")
    print(f"  Water Bodies: {params.water_bodies}")
    print(f"  Wind: {params.wind_direction:.0f}°")
    
    # Generate terrain
    terrain_map = generator.generate_terrain(params)
    
    # Create image if possible
    if PIL_AVAILABLE:
        width, height = terrain_map.width, terrain_map.height
        image = Image.new('RGB', (width, height))
        
        for y in range(height):
            for x in range(width):
                terrain_type = terrain_map.tiles[y, x]
                color = get_terrain_color(terrain_type)
                image.putpixel((x, y), color)
        
        # Scale up for visibility
        image = image.resize((width * 2, height * 2), Image.NEAREST)
        filename = f"random_terrain_{index}.png"
        image.save(filename)
        print(f"  Saved: {filename}")
    
    print()

def main():
    print("🎲 Generating Random Terrain Variations")
    print("=" * 40)
    
    setup_logging()
    
    # Generate 3 different random terrains
    for i in range(1, 4):
        generate_random_terrain(i)
    
    print("✅ Random terrain generation complete!")
    print("Check the generated PNG files to see the differences!")

if __name__ == "__main__":
    main() 