"""Open the multi-block demo layout in the viewer without moving the arm."""

import time
from pathlib import Path

import mujoco
import mujoco.viewer

from inchworm_sim.core.world_grid import WorldGrid
from inchworm_sim.physics.coords import BLOCK_EDGE
from inchworm_sim.physics.scene_builder import create_temporary_scene
from inchworm_sim.physics.sim_bridge import SimBridge
from inchworm_sim.scene_layout import MULTI_BLOCK_DEMO_BLOCKS, ORIGIN_OFFSET

ROOT = Path(__file__).resolve().parent.parent


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
            while viewer.is_running():
                bridge.step()
                viewer.sync()
                time.sleep(model.opt.timestep)
    finally:
        scene_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
