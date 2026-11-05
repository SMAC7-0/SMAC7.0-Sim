"""Move, grab, and place a block using the logical/physical bridge."""

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from inchworm_sim.core.world_grid import WorldGrid
from inchworm_sim.physics.coords import BLOCK_EDGE
from inchworm_sim.physics.inchworm_ctrl import Arm
from inchworm_sim.physics.scene_builder import create_temporary_scene
from inchworm_sim.physics.sim_bridge import SimBridge
from inchworm_sim.scene_layout import (
    MULTI_BLOCK_DEMO_BLOCKS,
    ORIGIN_OFFSET,
)

ROOT = Path(__file__).resolve().parent.parent
TARGET_BLOCK = "block6"
TARGET_CELL = (2, 1, 0)


def _require_movement(result: bool, action: str) -> None:
    if not result:
        raise RuntimeError(f"arm failed to {action}")


def main() -> None:
    scene_path = create_temporary_scene(
        MULTI_BLOCK_DEMO_BLOCKS,
        ROOT / "models" / "scene.xml",
        origin_offset=ORIGIN_OFFSET,
    )
    try:
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        world_grid = WorldGrid(
            block_edge=BLOCK_EDGE,
            origin_offset=ORIGIN_OFFSET,
        )
        bridge = SimBridge(
            model,
            data,
            world_grid,
            tuple(MULTI_BLOCK_DEMO_BLOCKS),
        )

        with mujoco.viewer.launch_passive(model, data) as viewer:
            arm = Arm(
                model,
                data,
                viewer,
                physics_step=bridge.step,
            )
            _require_movement(arm.hold(1.0), "settle")

            target_site = model.site(f"{TARGET_BLOCK}_pz").id
            mujoco.mj_forward(model, data)
            target_position = data.site_xpos[target_site].copy()
            _require_movement(
                arm.move_to(target_position + np.array((0.0, 0.0, 0.10))),
                "move above the target block",
            )
            _require_movement(
                arm.move_to(target_position + np.array((0.0, 0.0, 0.003))),
                "move to the target face",
            )

            held_block = bridge.grab_nearest_block("ee2")
            if held_block != TARGET_BLOCK:
                raise RuntimeError(
                    f"expected to grab {TARGET_BLOCK}, got {held_block!r}"
                )

            mujoco.mj_forward(model, data)
            ee_position = data.site_xpos[bridge.worm.gripper_sites["ee2"]].copy()
            block_position = bridge.blocks[TARGET_BLOCK].get_world_pos()
            block_offset = block_position - ee_position
            destination = world_grid.grid_to_world(TARGET_CELL)
            _require_movement(
                arm.move_to(destination - block_offset + np.array((0.0, 0.0, 0.10))),
                "lift the held block",
            )
            _require_movement(
                arm.move_to(destination - block_offset),
                "move the held block to its destination",
            )
            placed_cell = bridge.release_held_block("ee2")
            if placed_cell is None:
                raise RuntimeError("bridge did not place the held block")
            print(f"Placed {TARGET_BLOCK} at {placed_cell}")
            _require_movement(arm.hold(0.5), "stabilize the placed block")

            while viewer.is_running():
                bridge.step()
                viewer.sync()
                time.sleep(model.opt.timestep)
    finally:
        scene_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
