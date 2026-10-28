"""MuJoCo-backed block adapter."""

from typing import Tuple

import mujoco
import numpy as np


class MujocoBlock:
    """Expose a free-jointed MuJoCo block through a small block API."""

    def __init__(
        self, model: mujoco.MjModel, data: mujoco.MjData, block_id: str
    ) -> None:
        self.model = model
        self.data = data
        self.block_id = block_id
        self.body_id = model.body(block_id).id
        self.site_id = model.site(f"{block_id}_grab").id
        joint_id = model.body_jntadr[self.body_id]
        self.qpos_adr = model.jnt_qposadr[joint_id]
        self.qvel_adr = model.jnt_dofadr[joint_id]

    def get_world_pos(self) -> np.ndarray:
        """Return a copy of the block body's world-space position."""
        return self.data.xpos[self.body_id].copy()

    def get_grab_site_pos(self) -> np.ndarray:
        """Return a copy of the block's grab-site world position."""
        return self.data.site_xpos[self.site_id].copy()

    def set_pose(
        self,
        position: np.ndarray,
        quat: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    ) -> None:
        """Set an upright pose and stop all free-joint motion."""
        self.data.qpos[self.qpos_adr : self.qpos_adr + 3] = position
        self.data.qpos[self.qpos_adr + 3 : self.qpos_adr + 7] = quat
        self.data.qvel[self.qvel_adr : self.qvel_adr + 6] = 0.0
