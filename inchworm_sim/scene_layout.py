"""Logical starting layouts for generated simulator scenes."""

from typing import Dict, Tuple

from .physics.coords import BLOCK_EDGE

GridCoord = Tuple[int, int, int]
ORIGIN_OFFSET = (0.0, 0.0, 0.0)
BLOCK_SIZE = BLOCK_EDGE

STARTING_BLOCKS: Dict[str, GridCoord] = {
    "block1": (1, 1, 0),
    "block2": (2, 1, 0),
    "block3": (0, 1, 0),
    "block4": (1, 2, 0),
    "block5": (1, 0, 0),
}

MULTI_BLOCK_DEMO_BLOCKS: Dict[str, GridCoord] = {
    "block1": (0, 0, 0),
    "block2": (1, 0, 0),
    "block3": (0, 0, 1),
    "block4": (0, 1, 0),
    "block5": (1, 1, 0),
    "block6": (2, 0, 0),
}
