"""
Scene generation information: block position, connections, .
"""

from pathlib import Path
from inchworm_sim.physics.scene_builder import build_scene

ROOT = Path(__file__).resolve().parent.parent

# Block positions: (x, y, z)
# Will be replaced by world's blocks() in Core
block_positions = {
    0: (0, 0, 0),
    1: (1, 0, 0),
    2: (0, 0, 1),
    3: (0, 1, 0),
    4: (1, 1, 0),
    5: (2, 0, 0),
}

# Connections between blocks
block_connections = [
    (0, 1, "px", "nx"),
    (0, 2, "pz", "nz"),
    (0, 3, "py", "ny"),
    (4, 1, "ny", "py"),
    (4, 3, "nx", "px"),
    (5, 1, "nx", "px"),
]

build_scene(block_positions, block_connections, scene_file=ROOT / "models" / "multi_block_scene.xml", out_path=ROOT / "models" / "generated_scene.xml")