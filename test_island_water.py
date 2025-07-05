#!/usr/bin/env python3
"""
Test script to verify island water coverage reduction.
"""

import sys
from pathlib import Path
import numpy as np

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from townsim.modules.terrain.advanced_terrain_generator import (
    AdvancedTerrainGenerator, 
    AdvancedTerrainParameters,
    CoastalType
)
from townsim.modules.terrain.terrain_types import TerrainType
from townsim.modules.utils.logging_setup import setup_logging

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

def save_terrain_image(terrain_map, filename: str):
    """Save terrain map as a PNG image."""
    if not PIL_AVAILABLE:
        print("PIL not available - cannot save image")
        return
        
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

def test_island_water_coverage():
    """Test island generation with reduced water coverage."""
    
    print("Island Water Coverage Test")
    print("=" * 50)
    
    setup_logging()
    
    # Test multiple island generations
    generator = AdvancedTerrainGenerator()
    
    for i in range(3):
        print(f"\nTesting island generation #{i+1}...")
        
        params = AdvancedTerrainParameters(
            map_size=256,
            random_seed=100 + i,  # Different seeds
            coastal_type=CoastalType.ISLAND,
            terrain_smoothness=0.2,
            max_lakes=2,
            noise_scale=4.0
        )
        
        terrain_map = generator.generate_terrain(params)
        
        # Calculate water coverage
        total_tiles = params.map_size * params.map_size
        water_tiles = np.sum(terrain_map.tiles == TerrainType.SEA)
        water_percentage = (water_tiles / total_tiles) * 100
        
        print(f"  Water coverage: {water_percentage:.1f}%")
        
        # Save image
        filename = f"island_test_{i+1}.png"
        save_terrain_image(terrain_map, filename)
        
        # Verify water is visible at edges (but not too much)
        edge_size = 10  # Check 10-pixel border
        
        # Check top edge
        top_edge = terrain_map.tiles[:edge_size, :]
        top_water = np.sum(top_edge == TerrainType.SEA)
        top_water_pct = (top_water / (edge_size * params.map_size)) * 100
        
        # Check bottom edge
        bottom_edge = terrain_map.tiles[-edge_size:, :]
        bottom_water = np.sum(bottom_edge == TerrainType.SEA)
        bottom_water_pct = (bottom_water / (edge_size * params.map_size)) * 100
        
        # Check left edge
        left_edge = terrain_map.tiles[:, :edge_size]
        left_water = np.sum(left_edge == TerrainType.SEA)
        left_water_pct = (left_water / (edge_size * params.map_size)) * 100
        
        # Check right edge
        right_edge = terrain_map.tiles[:, -edge_size:]
        right_water = np.sum(right_edge == TerrainType.SEA)
        right_water_pct = (right_water / (edge_size * params.map_size)) * 100
        
        print(f"  Edge water coverage:")
        print(f"    Top: {top_water_pct:.1f}%")
        print(f"    Bottom: {bottom_water_pct:.1f}%")
        print(f"    Left: {left_water_pct:.1f}%")
        print(f"    Right: {right_water_pct:.1f}%")
        
        # Check if water is visible on all edges
        all_edges_have_water = (top_water_pct > 0 and bottom_water_pct > 0 and 
                               left_water_pct > 0 and right_water_pct > 0)
        
        if all_edges_have_water:
            print("  ✓ Water visible on all edges")
        else:
            print("  ⚠ Water not visible on all edges")
        
        # Check if water coverage is reasonable (should be less than before)
        if water_percentage < 40:
            print(f"  ✓ Good water coverage: {water_percentage:.1f}%")
        elif water_percentage < 50:
            print(f"  ⚠ Moderate water coverage: {water_percentage:.1f}%")
        else:
            print(f"  ❌ Too much water coverage: {water_percentage:.1f}%")
    
    print("\n" + "=" * 50)
    print("✅ Island water coverage test complete!")
    print("Island water coverage should be reduced compared to before.")
    print("Water should still be visible around all edges but take up less total area.")

if __name__ == "__main__":
    test_island_water_coverage() 