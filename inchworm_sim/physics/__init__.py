"""Physics adapters and MuJoCo scene-generation helpers."""

from typing import Any

__all__ = [
    "MujocoBlock",
    "MujocoWorm",
    "Arm",
    "SimBridge",
    "build_scene",
    "create_temporary_scene",
    "BlockSpec",
    "DEFAULT_BLOCK_SPEC",
]


def __getattr__(name: str) -> Any:
    """Load runtime adapters only when they are requested."""
    if name == "MujocoBlock":
        from .mujoco_block import MujocoBlock

        return MujocoBlock
    if name == "MujocoWorm":
        from .mujoco_worm import MujocoWorm

        return MujocoWorm
    if name == "Arm":
        from .inchworm_ctrl import Arm

        return Arm
    if name == "SimBridge":
        from .sim_bridge import SimBridge

        return SimBridge
    if name == "build_scene":
        from .scene_builder import build_scene

        return build_scene
    if name == "create_temporary_scene":
        from .scene_builder import create_temporary_scene

        return create_temporary_scene
    if name in {"BlockSpec", "DEFAULT_BLOCK_SPEC"}:
        from .scene_builder import BlockSpec, DEFAULT_BLOCK_SPEC

        return {"BlockSpec": BlockSpec, "DEFAULT_BLOCK_SPEC": DEFAULT_BLOCK_SPEC}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
