import time

import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("models/scene.xml")
data = mujoco.MjData(model)

# ------------------------------------------------------------
# IDs
# ------------------------------------------------------------

ee2_site_id = model.site("ee2_grab").id

block_site_id = model.site("block1_grab").id

weld_id = model.equality("ee2_block_weld").id

# ------------------------------------------------------------
# Grab parameters
# ------------------------------------------------------------

grab_distance = 0.004  # 2 mm
grabbed = False

# Time over which the weld becomes stiff
weld_ramp_time = 0.10  # 100 ms
weld_timer = 0.0

# Initial and final weld stiffness
soft_solref = np.array([0.10, 1.0])
hard_solref = np.array([0.01, 1.0])


# ------------------------------------------------------------
# Simulation
# ------------------------------------------------------------

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Update positions before checking grab distance
        mujoco.mj_forward(model, data)

        ee2_pos = data.site_xpos[ee2_site_id]
        block_pos = data.site_xpos[block_site_id]

        distance = np.linalg.norm(ee2_pos - block_pos)

        # ----------------------------------------------------
        # Detect grab
        # ----------------------------------------------------

        if not grabbed and distance < grab_distance:
            grabbed = True
            weld_timer = 0.0

            # Start with a very soft weld
            model.eq_solref[weld_id] = soft_solref

            # Activate the weld
            data.eq_active[weld_id] = 1

            print("BLOCK GRABBED")
            print("Distance:", distance)

        # ----------------------------------------------------
        # Gradually stiffen the weld
        # ----------------------------------------------------

        if grabbed and weld_timer < weld_ramp_time:
            weld_timer += model.opt.timestep

            alpha = min(weld_timer / weld_ramp_time, 1.0)

            # Smooth interpolation from soft -> stiff
            model.eq_solref[weld_id] = (1.0 - alpha) * soft_solref + alpha * hard_solref

        # ----------------------------------------------------
        # Step simulation
        # ----------------------------------------------------

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(model.opt.timestep)
