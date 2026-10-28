"""Logical 3D grid representation for blocks and their face connections."""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Optional, Tuple, Union

import numpy as np


GridCoord = Tuple[int, int, int]

# Directions used to inspect and connect the six neighboring cells.
FACES = {
    "+x": (1, 0, 0),
    "-x": (-1, 0, 0),
    "+y": (0, 1, 0),
    "-y": (0, -1, 0),
    "+z": (0, 0, 1),
    "-z": (0, 0, -1),
}

OPPOSITE_FACE = {
    "+x": "-x",
    "-x": "+x",
    "+y": "-y",
    "-y": "+y",
    "+z": "-z",
    "-z": "+z",
}


class CellStatus(IntEnum):
    EMPTY = 0
    BLOCK = 1
    SEED_BLOCK = 2
    SUPPLY_DEPOT = 3
    INCOMING_BLOCK = 4


@dataclass
class GridBlock:
    """Logical block stored in one grid cell."""

    block_id: str
    grid_pos: Optional[GridCoord]
    status: CellStatus
    face_connections: Dict[str, Optional[str]] = field(
        default_factory=lambda: {face: None for face in FACES}
    )


class WorldGrid:
    """Bounds-checked logical grid using the simulator's z-up convention."""

    def __init__(
        self,
        size_x: int = 50,
        size_y: int = 50,
        size_z: int = 4,
        block_edge: float = 0.0762,
        origin_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        if block_edge <= 0:
            raise ValueError("block_edge must be positive")
        if any(size <= 0 for size in (size_x, size_y, size_z)):
            raise ValueError("grid dimensions must be positive")
        if len(origin_offset) != 3:
            raise ValueError("origin_offset must contain three values")

        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.block_edge = float(block_edge)
        self.origin_offset = np.asarray(origin_offset, dtype=float)
        # NumPy stores status values by x, y, then z.
        self.grid = np.full(
            (size_x, size_y, size_z), CellStatus.EMPTY, dtype=np.int8
        )
        # Keep both coordinate and ID lookups for fast logical updates.
        self.blocks: Dict[GridCoord, GridBlock] = {}
        self.blocks_by_id: Dict[str, GridBlock] = {}

    @staticmethod
    def _coord(coord: GridCoord) -> GridCoord:
        """Validate and normalize a grid coordinate."""
        if len(coord) != 3 or any(not isinstance(value, (int, np.integer)) for value in coord):
            raise ValueError("grid coordinates must be integer triples")
        return tuple(int(value) for value in coord)

    @staticmethod
    def _face(face: str) -> str:
        """Validate a face name and return it unchanged."""
        if face not in FACES:
            raise ValueError(f"invalid face {face!r}; expected one of {tuple(FACES)}")
        return face

    def grid_to_world(self, coord: GridCoord) -> np.ndarray:
        """Convert a grid coordinate to its world-space cell center."""
        x, y, z = self._coord(coord)
        return self.origin_offset + (np.array((x, y, z), dtype=float) + 0.5) * self.block_edge

    def world_to_grid(self, pos: Union[Tuple[float, float, float], np.ndarray]) -> GridCoord:
        """Convert a world-space position to the nearest grid coordinate."""
        position = np.asarray(pos, dtype=float)
        if position.shape != (3,):
            raise ValueError("world position must contain three values")
        relative = (position - self.origin_offset) / self.block_edge - 0.5
        return tuple(int(np.floor(value + 0.5)) for value in relative)

    def get_face_world_pos(self, coord: GridCoord, face: str) -> np.ndarray:
        """Return the world-space center of a cell face."""
        face = self._face(face)
        return self.grid_to_world(coord) + np.asarray(FACES[face], dtype=float) * (
            self.block_edge / 2.0
        )

    def in_bounds(self, coord: GridCoord) -> bool:
        """Return whether a coordinate lies inside the configured grid."""
        x, y, z = self._coord(coord)
        return 0 <= x < self.size_x and 0 <= y < self.size_y and 0 <= z < self.size_z

    def is_empty(self, coord: GridCoord) -> bool:
        """Return whether an in-bounds cell is empty."""
        return self.in_bounds(coord) and self.grid[tuple(self._coord(coord))] == CellStatus.EMPTY

    def get_neighbors(self, coord: GridCoord) -> Dict[str, GridBlock]:
        """Return occupied neighboring cells keyed by their shared face."""
        coord = self._coord(coord)
        neighbors: Dict[str, GridBlock] = {}
        for face, direction in FACES.items():
            adjacent = tuple(coord[index] + direction[index] for index in range(3))
            block = self.blocks.get(adjacent)
            if block is not None:
                neighbors[face] = block
        return neighbors

    def is_valid_placement(self, coord: GridCoord) -> bool:
        """Return whether a block can be supported at a coordinate."""
        coord = self._coord(coord)
        if not self.is_empty(coord):
            return False
        # Ground cells are supported by z=0; higher cells need a neighbor.
        return coord[2] == 0 or bool(self.get_neighbors(coord))

    def place_block(
        self,
        block_id: str,
        coord: GridCoord,
        status: CellStatus = CellStatus.BLOCK,
        enforce_rules: bool = True,
    ) -> GridBlock:
        """Place a block and connect it to all adjacent blocks."""
        coord = self._coord(coord)
        if not block_id:
            raise ValueError("block_id must not be empty")
        if block_id in self.blocks_by_id:
            raise ValueError(f"block_id {block_id!r} is already placed")
        if not isinstance(status, CellStatus):
            status = CellStatus(status)
        if enforce_rules and not self.is_valid_placement(coord):
            raise ValueError(f"invalid block placement at {coord}")
        if not self.in_bounds(coord):
            raise ValueError(f"coordinate {coord} is out of bounds")

        block = GridBlock(block_id, coord, status)
        # Keep the array and object indexes synchronized.
        self.grid[coord] = status
        self.blocks[coord] = block
        self.blocks_by_id[block_id] = block
        # Link both sides of every face shared with an existing block.
        for face, neighbor in self.get_neighbors(coord).items():
            block.face_connections[face] = neighbor.block_id
            neighbor.face_connections[OPPOSITE_FACE[face]] = block_id
        return block

    def remove_block(self, coord: GridCoord) -> Optional[GridBlock]:
        """Remove a block, clear its cell, and disconnect its neighbors."""
        coord = self._coord(coord)
        block = self.blocks.pop(coord, None)
        if block is None:
            return None
        self.grid[coord] = CellStatus.EMPTY
        self.blocks_by_id.pop(block.block_id, None)
        # Disconnect neighboring blocks before clearing the removed block.
        for face, neighbor in self.get_neighbors(coord).items():
            neighbor.face_connections[OPPOSITE_FACE[face]] = None
        block.grid_pos = None
        block.face_connections = {face: None for face in FACES}
        return block

    def get_walkable_surfaces(self) -> list[Tuple[int, int, int, str]]:
        """Return exposed faces that an inchworm can step onto."""
        surfaces = []
        for coord in self.blocks:
            for face, direction in FACES.items():
                if face == "-z":
                    continue
                adjacent = tuple(coord[index] + direction[index] for index in range(3))
                if not self.in_bounds(adjacent) or self.is_empty(adjacent):
                    surfaces.append((*coord, face))
        return surfaces