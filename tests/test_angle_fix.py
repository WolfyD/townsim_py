#!/usr/bin/env python3
"""
Test script to verify the coastline angle fix.

This test generates terrain at 45° and verifies that:
- Sea is towards the lower left
- Land is towards the upper right
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

def test_45_degree_angle():
    """Test that 45° angle produces correct water/land placement."""
    
    print("Testing 45° coastline angle fix...")
    print("=" * 40)
    
    # Setup logging
    setup_logging()
    
    # Create parameters for 45° coastal terrain
    params = AdvancedTerrainParameters(
        map_size=256,
        random_seed=42,
        coastal_type=CoastalType.COASTAL,
        coastline_angle=45.0,  # Explicit 45° angle
        terrain_smoothness=0.1,  # Less smooth to show boundary clearly
        max_lakes=0,  # No lakes to avoid confusion
        noise_scale=3.0,
        elevation_variance=0.5
    )
    
    print(f"Generating terrain with {params.coastline_angle}° coastline angle...")
    
    # Generate terrain
    generator = AdvancedTerrainGenerator()
    terrain_map = generator.generate_terrain(params)
    
    # Save the result
    save_terrain_image(terrain_map, "coastline_angle_45_fixed.png")
    
    # Analyze the quadrants to verify correct placement
    size = terrain_map.tiles.shape[0]
    half_size = size // 2
    
    # Define quadrants
    upper_left = terrain_map.tiles[0:half_size, 0:half_size]
    upper_right = terrain_map.tiles[0:half_size, half_size:size]
    lower_left = terrain_map.tiles[half_size:size, 0:half_size]
    lower_right = terrain_map.tiles[half_size:size, half_size:size]
    
    # Count water (SEA) in each quadrant
    water_ul = np.sum(upper_left == TerrainType.SEA)
    water_ur = np.sum(upper_right == TerrainType.SEA)
    water_ll = np.sum(lower_left == TerrainType.SEA)
    water_lr = np.sum(lower_right == TerrainType.SEA)
    
    quadrant_size = half_size * half_size
    water_ul_pct = (water_ul / quadrant_size) * 100
    water_ur_pct = (water_ur / quadrant_size) * 100
    water_ll_pct = (water_ll / quadrant_size) * 100
    water_lr_pct = (water_lr / quadrant_size) * 100
    
    print(f"\nWater coverage by quadrant:")
    print(f"Upper Left:  {water_ul_pct:.1f}%")
    print(f"Upper Right: {water_ur_pct:.1f}%")
    print(f"Lower Left:  {water_ll_pct:.1f}%")
    print(f"Lower Right: {water_lr_pct:.1f}%")
    
    # Verify the fix
    print(f"\nVerifying fix:")
    if water_ll_pct > water_ur_pct:
        print("✓ CORRECT: More water in lower left than upper right")
        success = True
    else:
        print("✗ INCORRECT: More water in upper right than lower left")
        success = False
    
    if water_ll_pct > 50:
        print("✓ CORRECT: Lower left has significant water coverage")
    else:
        print("✗ INCORRECT: Lower left doesn't have enough water")
        success = False
    
    if water_ur_pct < 20:
        print("✓ CORRECT: Upper right has minimal water coverage")
    else:
        print("✗ INCORRECT: Upper right has too much water")
        success = False
    
    return success

def test_multiple_angles():
    """Test several different angles to ensure the fix works broadly."""
    
    print("\nTesting multiple angles...")
    print("=" * 40)
    
    test_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    
    for angle in test_angles:
        print(f"\nTesting {angle}° angle...")
        
        params = AdvancedTerrainParameters(
            map_size=128,  # Smaller for speed
            random_seed=42,
            coastal_type=CoastalType.COASTAL,
            coastline_angle=float(angle),
            terrain_smoothness=0.1,
            max_lakes=0,
            noise_scale=3.0,
            elevation_variance=0.5
        )
        
        generator = AdvancedTerrainGenerator()
        terrain_map = generator.generate_terrain(params)
        
        # Save image
        filename = f"angle_test_{angle}_deg.png"
        save_terrain_image(terrain_map, filename)
        
        # Quick analysis
        total_water = np.sum(terrain_map.tiles == TerrainType.SEA)
        total_tiles = terrain_map.tiles.shape[0] * terrain_map.tiles.shape[1]
        water_pct = (total_water / total_tiles) * 100
        
        print(f"  Water coverage: {water_pct:.1f}%")
        print(f"  Saved: {filename}")

def main():
    """Main test function."""
    print("Coastline Angle Fix Test")
    print("=" * 50)
    
    try:
        # Test the specific 45° case
        success = test_45_degree_angle()
        
        # Test multiple angles
        test_multiple_angles()
        
        print("\n" + "=" * 50)
        if success:
            print("✅ Coastline angle fix verification PASSED!")
            print("Check coastline_angle_45_fixed.png - sea should be in lower left")
        else:
            print("❌ Coastline angle fix verification FAILED!")
            print("The fix may need additional adjustments")
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 