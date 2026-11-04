"""MuJoCo-backed block adapter."""

from typing import Dict, Tuple

import mujoco
import numpy as np

from .coords import FACE_TO_SITE


class MujocoBlock:
    """Expose a free-jointed MuJoCo block through a small block API."""

    def __init__(
        self, model: mujoco.MjModel, data: mujoco.MjData, block_id: str
    ) -> None:
        self.model = model
        self.data = data
        self.block_id = block_id
        self.body_id = model.body(block_id).id
        self.face_site_ids: Dict[str, int] = {
            face: model.site(f"{block_id}_{site_suffix}").id
            for face, site_suffix in FACE_TO_SITE.items()
        }
        joint_id = model.body_jntadr[self.body_id]
        self.qpos_adr = model.jnt_qposadr[joint_id]
        self.qvel_adr = model.jnt_dofadr[joint_id]

    def get_world_pos(self) -> np.ndarray:
        """Return a copy of the block body's world-space position."""
        return self.data.xpos[self.body_id].copy()

    def get_face_site_pos(self, face: str) -> np.ndarray:
        """Return a copy of a named face site's world position."""
        try:
            site_id = self.face_site_ids[face]
        except KeyError as exc:
            raise ValueError(f"invalid block face {face!r}") from exc
        return self.data.site_xpos[site_id].copy()

    def get_grab_site_pos(self) -> np.ndarray:
        """Return the top face position for backward compatibility."""
        return self.get_face_site_pos("+z")

    def set_pose(
        self,
        position: np.ndarray,
        quat: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    ) -> None:
        """Set an upright pose and stop all free-joint motion."""
        self.data.qpos[self.qpos_adr : self.qpos_adr + 3] = position
        self.data.qpos[self.qpos_adr + 3 : self.qpos_adr + 7] = quat
        self.data.qvel[self.qvel_adr : self.qvel_adr + 6] = 0.0
