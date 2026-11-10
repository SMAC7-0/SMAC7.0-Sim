"""Generate a reusable MuJoCo scene from grid coordinates.

Usage:
    python scripts/generate_multi_block_scene.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inchworm_sim.bridge import build_scene


ROOT = Path(__file__).resolve().parent.parent

# Add or remove entries without changing the XML model definition.
BLOCK_POSITIONS = {
    "block1": (0, 0, 0),
    "block2": (1, 0, 0),
    "block3": (0, 0, 1),
    "block4": (0, 1, 0),
    "block5": (1, 1, 0),
    "block6": (2, 0, 0),
}


if __name__ == "__main__":
    output = ROOT / "models" / "generated_scene.xml"
    build_scene(
        BLOCK_POSITIONS,
        scene_file=ROOT / "models" / "scene.xml",
        output_file=output,
        origin_offset=(0.0, 0.0, 0.0),
    )
    print(f"Generated {len(BLOCK_POSITIONS)} blocks in {output}")
