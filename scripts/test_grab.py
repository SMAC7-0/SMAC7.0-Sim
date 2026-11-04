import time

import mujoco
import mujoco.viewer

from inchworm_sim.core.world_grid import WorldGrid
from inchworm_sim.physics.coords import BLOCK_EDGE
from inchworm_sim.physics.sim_bridge import SimBridge


model = mujoco.MjModel.from_xml_path("models/scene.xml")
data = mujoco.MjData(model)
world_grid = WorldGrid(
    block_edge=BLOCK_EDGE,
    origin_offset=(0.0, 0.0, 0.0),
)
bridge = SimBridge(model, data, world_grid)

print(
    "\nInteractive grab demo:\n"
    "  Move joints with the MuJoCo viewer Control sliders or keyboard shortcuts.\n"
    "  Move ee2 within 30 mm of a block and press SPACE to grab it.\n"
    "  Move the held block, then press SPACE again to release and snap it to the 3D grid.\n"
    "  With validity enabled, block1 at (1, 1, 0) is trapped by blocks2..5 and "
    "cannot be picked up.\n"
    "  Pick up an outer block, stack it at z=1, and block1 becomes removable "
    "after one horizontal neighbor is gone.\n"
    "  Press V to toggle validity checks; SPACE grabs/releases.\n"
)


def key_callback(keycode: int) -> None:
    """Toggle the ee2 gripper when the viewer's spacebar is pressed."""
    if keycode == ord("v") or keycode == ord("V"):
        bridge.world_grid.set_validity_enabled(
            not bridge.world_grid.rules_enabled
        )
        state = "ENABLED" if bridge.world_grid.rules_enabled else "DISABLED"
        print(f"Validity checks: {state}")
        return
    if keycode != ord(" "):
        return
    print("SPACE pressed")
    if bridge.worm.held_blocks["ee2"] is None:
        if bridge.release_cooldown == 0.0:
            bridge.worm.close_gripper("ee2", bridge)
        else:
            print(
                f"GRAB REJECTED: release cooldown "
                f"{bridge.release_cooldown:.2f} s remaining"
            )
    else:
        bridge.worm.open_gripper("ee2", bridge)


with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    while viewer.is_running():
        bridge.step()
        viewer.sync()
        time.sleep(model.opt.timestep)
