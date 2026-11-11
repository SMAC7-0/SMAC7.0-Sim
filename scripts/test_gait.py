"""Simple actuator-driven inchworm motion demo."""
from pathlib import Path
import math
import time

import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parent.parent
SCENE_PATH = ROOT / "models" / "moving-demo-scene.xml"


def main():
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    joint_names = ("wrist1", "hand1", "top", "hand2", "wrist2")
    actuator_names = ("wrist1_act", "hand1_act", "top_act", "hand2_act", "wrist2_act")
    qpos_addresses = [model.jnt_qposadr[model.joint(n).id] for n in joint_names]
    actuator_ids = [model.actuator(n).id for n in actuator_names]
    initial = [float(data.qpos[a]) for a in qpos_addresses]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.21, 0.09, 0.20]
        viewer.cam.distance = 1.35
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        start = time.perf_counter()

        while viewer.is_running():
            t = time.perf_counter() - start
            targets = list(initial)
            targets[2] = initial[2] + 0.45 * math.sin(2 * math.pi * 0.25 * t)
            targets[3] = initial[3] + 0.25 * math.sin(2 * math.pi * 0.25 * t + 1.0)
            data.ctrl[actuator_ids] = targets
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
