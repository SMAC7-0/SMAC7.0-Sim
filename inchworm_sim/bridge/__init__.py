"""Logical grid ↔ MuJoCo scene generation and runtime coordination."""

from typing import Any

from .coords import BLOCK_EDGE, grid_to_world, world_to_grid
from .faces import FACES, HORIZONTAL_FACES, OPPOSITE_FACE, FaceName
from .scene_builder import (
    DEFAULT_BLOCK_SPEC,
    BlockSpec,
    build_scene,
    create_temporary_scene,
)

__all__ = [
    "BLOCK_EDGE",
    "BlockSpec",
    "DEFAULT_BLOCK_SPEC",
    "FACES",
    "FaceName",
    "HORIZONTAL_FACES",
    "OPPOSITE_FACE",
    "SimBridge",
    "build_scene",
    "create_temporary_scene",
    "grid_to_world",
    "world_to_grid",
]


def __getattr__(name: str) -> Any:
    """Load SimBridge on demand so core can import face tables first."""
    if name == "SimBridge":
        from .sim_bridge import SimBridge

        return SimBridge
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
