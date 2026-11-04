"""Shared physical dimensions and coordinate/name conversions."""

from typing import Dict, Tuple

import numpy as np


BLOCK_EDGE = 0.0742
FACES = ("+x", "-x", "+y", "-y", "+z", "-z")
FACE_TO_SITE: Dict[str, str] = {
    "+x": "px",
    "-x": "nx",
    "+y": "py",
    "-y": "ny",
    "+z": "pz",
    "-z": "nz",
}


def grid_to_world(
    coord: Tuple[int, int, int],
    origin_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Return the center of a logical cell in world coordinates."""
    if len(coord) != 3:
        raise ValueError("grid coordinates must contain three values")
    return np.asarray(origin_offset, dtype=float) + (
        np.asarray(coord, dtype=float) + 0.5
    ) * BLOCK_EDGE


def world_to_grid(
    position: Tuple[float, float, float],
    origin_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Tuple[int, int, int]:
    """Convert a world position to the nearest logical cell."""
    value = np.asarray(position, dtype=float)
    if value.shape != (3,):
        raise ValueError("world position must contain three values")
    relative = (value - np.asarray(origin_offset, dtype=float)) / BLOCK_EDGE
    return tuple(int(np.floor(component)) for component in relative)
