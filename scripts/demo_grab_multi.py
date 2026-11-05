import time
import mujoco
import mujoco.viewer
from pathlib import Path
import numpy as np
from inchworm_sim.physics import attach
from inchworm_sim.physics.inchworm_ctrl import Arm

ROOT = Path(__file__).resolve().parent.parent
model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "generated_scene.xml"))
data = mujoco.MjData(model)
ids = range(6)

with mujoco.viewer.launch_passive(model, data) as v:
    # Set camera to main_camera
    camera_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, "main_camera"
    )

    if camera_id == -1:
        raise ValueError("main_camera not found")

    v.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    v.cam.fixedcamid = camera_id

    #--------------------------------------------------
    # Add visual marker for site on EE2
    ee_site_id = model.site("ee2_grab").id
    block_site_id = model.site("block5_pz").id


    with v.lock():
        v.user_scn.ngeom = 2

        # Block face marker: green
        mujoco.mjv_initGeom(
            v.user_scn.geoms[1],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.008, 0, 0]),
            data.site_xpos[block_site_id].copy(),
            np.eye(3).flatten(),
            np.array([0, 1, 0, 1]),
        )

        # EE site marker: red
        mujoco.mjv_initGeom(
            v.user_scn.geoms[0],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.012, 0, 0]),
            data.site_xpos[ee_site_id].copy(),
            np.eye(3).flatten(),
            np.array([1, 0, 0, 1]),
        )

    #--------------------------------------------------
    # Demo: grab block 5 and swing it sideways

    arm = Arm(model, data, v)
    for _ in range(500):                                       # settle, arm pinned
        arm.tick()

    ee_site_id = model.site("ee2_grab").id

    ee  = data.site_xpos[model.site("ee2_grab").id].copy()
    top = data.site_xpos[model.site("block5_pz").id].copy()

    arm.hold(2.0)
    print("Moving to block 5...")

    arm.move_to([ee[0], ee[1], top[2] + 0.10])                 # up
    arm.move_to(top + [0, 0, 0.10])                            # over block 5
    arm.move_to(top + [0, 0, 0.003])                           # down onto its top face
    
    mujoco.mj_forward(model, data)                   
    # held = attach.grab(model, data, [5])
    held = attach.grab(model, data, [5], ee_site="ee2_grab")
    print(f"Block grabbed: {held}") # DEBUG
    arm.hold(0.8)

    # release and swing block sideways
    print("Swinging block 5 sideways...")
    arm.move_to(top + [0, 0, 0.12])                            # lift
    arm.rotate("wrist1", np.pi / 2)                            # swing sideways

    # test rotate wrist2 to see if it affects the block orientation
    pos = data.site_xpos[ee_site_id].copy()

    print("Testing rotation...")
    arm.rotate("hand2", np.pi / 2)
    arm.rotate("wrist2", -np.pi / 2)
    
    arm.rotate("wrist2", np.pi / 2)                           # swing back
    arm.rotate("hand2", -np.pi / 2)

    # drop block
    arm.move_to([pos[0], pos[1], top[2] + 0.003])                 # drop down
    print("Dropping block 5...")
    attach.release(model, data, 5)                                 # release block 5
    arm.hold(0.5)
    arm.move_to(pos)

    while arm.tick():
        pass

    