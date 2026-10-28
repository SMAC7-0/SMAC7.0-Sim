
from pathlib import Path
import mujoco
import mujoco.viewer

# Load the model
model = mujoco.MjModel.from_xml_path("models/generated_scene.xml")
data = mujoco.MjData(model)

# Initialize the simulation
mujoco.mj_forward(model, data)

# Open the interactive viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
