"""
This document converts between grid coordinates and world coordinates for blocks.
"""
import numpy as np

BLOCK_SIZE = 0.0742  # {m}

def grid_to_world(i, j, k, size=BLOCK_SIZE):
    """
    Center of cell (i, j, k) in world coordinates.
    """
    return (np.array([i, j, k]) + 0.5) * size

def world_to_grid(pos, size=BLOCK_SIZE):
    """
    World coordinates of a point in grid coordinates.
    """
    return tuple(int(x) for x in np.floor(np.asarray(pos) / size))