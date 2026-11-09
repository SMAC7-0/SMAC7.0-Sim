"""Grab block 5 and swing it sideways using the logical/physical bridge."""

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
TARGET_BLOCK = "block5"
SHOW_DEBUG_MARKERS = False


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
                marker_site=f"{TARGET_BLOCK}_pz",
            )
            _require_movement(arm.hold(1.0), "settle")

            target_site = model.site(f"{TARGET_BLOCK}_pz").id
            mujoco.mj_forward(model, data)
            target_position = data.site_xpos[target_site].copy()
            ee_position = data.site_xpos[model.site("ee2_grab").id].copy()

            if SHOW_DEBUG_MARKERS:
                with viewer.lock():
                    viewer.user_scn.ngeom = 2
                    # Red sphere tracks ee2_grab; green tracks the target face.
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[0],
                        mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([0.012, 0, 0]),
                        ee_position,
                        np.eye(3).flatten(),
                        np.array([1, 0, 0, 1]),
                    )
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[1],
                        mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([0.008, 0, 0]),
                        target_position,
                        np.eye(3).flatten(),
                        np.array([0, 1, 0, 1]),
                    )

            arm.hold(1.0)
            print(f"Moving to {TARGET_BLOCK}...")
            _require_movement(
                arm.move_to(
                    np.array(
                        (ee_position[0], ee_position[1], target_position[2] + 0.10)
                    )
                ),
                "move up",
            )
            _require_movement(
                arm.move_to(target_position + np.array((0.0, 0.0, 0.10))),
                "move over the target block",
            )
            _require_movement(
                arm.move_to(target_position + np.array((0.0, 0.0, 0.003))),
                "move down onto the target face",
            )

            held_block = bridge.grab_nearest_block("ee2")
            if held_block != TARGET_BLOCK:
                raise RuntimeError(
                    f"expected to grab {TARGET_BLOCK}, got {held_block!r}"
                )

            _require_movement(arm.hold(0.8), "stabilize the grabbed block")
            print(f"Swinging {TARGET_BLOCK} sideways...")
            _require_movement(
                arm.move_to(target_position + np.array((0.0, 0.0, 0.12))),
                "lift the grabbed block",
            )
            mujoco.mj_forward(model, data)
            return_position = data.site_xpos[
                model.site("ee2_grab").id
            ].copy()
            _require_movement(arm.rotate("wrist1", np.pi / 2), "swing sideways")

            print("Testing end-effector rotation...")
            _require_movement(arm.rotate("hand2", np.pi / 2), "rotate hand2")
            _require_movement(arm.rotate("wrist2", -np.pi / 2), "rotate wrist2")
            _require_movement(arm.rotate("wrist2", np.pi / 2), "return wrist2")
            _require_movement(arm.rotate("hand2", -np.pi / 2), "return hand2")

            _require_movement(
                arm.move_to(
                    return_position,
                    step=0.0005,
                    rot_step=0.01,
                    tolerance=0.0015,
                    max_iterations=6000,
                ),
                "return to the lifted position",
            )
            _require_movement(
                arm.move_to(
                    np.array(
                        (
                            return_position[0],
                            return_position[1],
                            target_position[2] + 0.003,
                        )
                    ),
                    step=0.0005,
                    rot_step=0.01,
                    tolerance=0.0015,
                    max_iterations=6000,
                ),
                "lower the grabbed block",
            )
            print(f"Dropping {TARGET_BLOCK}...")
            placed_cell = bridge.release_held_block("ee2")
            if placed_cell is None:
                raise RuntimeError(f"bridge did not release {TARGET_BLOCK}")
            print(f"Placed {TARGET_BLOCK} at {placed_cell}")
            _require_movement(arm.hold(0.5), "stabilize the placed block")
            _require_movement(
                arm.move_to(
                    return_position,
                    step=0.0005,
                    rot_step=0.01,
                    tolerance=0.0015,
                    max_iterations=6000,
                ),
                "move away from the placed block",
            )

            while viewer.is_running():
                bridge.step()
                viewer.sync()
                time.sleep(model.opt.timestep)
    finally:
        scene_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
