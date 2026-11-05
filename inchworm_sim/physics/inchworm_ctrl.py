"""MuJoCo end-effector movement helpers for the inchworm arm."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Callable, Optional

import mujoco
import numpy as np

ARM_JOINTS = ("wrist1", "hand1", "top", "hand2", "wrist2")
PhysicsStep = Callable[[], None]


class Arm:
    """Drive the five arm joints toward Cartesian end-effector targets."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        viewer: Optional[object] = None,
        site: str = "ee2_grab",
        physics_step: Optional[PhysicsStep] = None,
        marker_site: Optional[str] = None,
    ) -> None:
        self.model = model
        self.data = data
        self.viewer = viewer
        self.site_id = model.site(site).id
        self.physics_step = physics_step
        self.marker_site = marker_site

        joint_ids = [model.joint(name).id for name in ARM_JOINTS]
        self.qpos_addresses = np.array(
            [model.jnt_qposadr[joint_id] for joint_id in joint_ids]
        )
        self.dof_addresses = np.array(
            [model.jnt_dofadr[joint_id] for joint_id in joint_ids]
        )
        self.lower_limits = model.jnt_range[joint_ids, 0]
        self.upper_limits = model.jnt_range[joint_ids, 1]
        self.joint_positions = data.qpos[self.qpos_addresses].copy()

        mujoco.mj_forward(model, data)
        self.hold_axis = data.site_xmat[self.site_id].reshape(3, 3)[:, 2].copy()

    def _sync_viewer(self) -> bool:
        if self.viewer is None:
            return True

        viewer = self.viewer
        with viewer.lock():
            if getattr(viewer.user_scn, "ngeom", 0) > 0:
                viewer.user_scn.geoms[0].pos[:] = self.data.site_xpos[self.site_id]
            if (
                self.marker_site is not None
                and getattr(viewer.user_scn, "ngeom", 0) > 1
            ):
                marker_id = self.model.site(self.marker_site).id
                viewer.user_scn.geoms[1].pos[:] = self.data.site_xpos[marker_id]

        viewer.sync()
        time.sleep(self.model.opt.timestep)
        return viewer.is_running()

    def tick(self) -> bool:
        """Advance physics while pinning the commanded arm joint positions."""
        self.data.qpos[self.qpos_addresses] = self.joint_positions
        self.data.qvel[self.dof_addresses] = 0.0
        if self.physics_step is None:
            mujoco.mj_step(self.model, self.data)
        else:
            self.physics_step()
        return self._sync_viewer()

    def move_to(
        self,
        target: Sequence[float],
        step: float = 0.001,
        rot_step: float = 0.02,
        tolerance: float = 0.003,
        max_iterations: int = 3000,
        damping: float = 1e-3,
    ) -> bool:
        """Move the end-effector to a target while preserving its tilt."""
        target_position = np.asarray(target, dtype=float)
        if target_position.shape != (3,):
            raise ValueError("target must contain three coordinates")
        if step <= 0 or rot_step <= 0 or tolerance <= 0 or damping <= 0:
            raise ValueError("movement parameters must be positive")

        jacobian_position = np.zeros((3, self.model.nv))
        jacobian_rotation = np.zeros((3, self.model.nv))
        for _ in range(max_iterations):
            self.data.qpos[self.qpos_addresses] = self.joint_positions
            mujoco.mj_forward(self.model, self.data)
            error_position = target_position - self.data.site_xpos[self.site_id]
            axis = self.data.site_xmat[self.site_id].reshape(3, 3)[:, 2]
            error_rotation = np.cross(axis, self.hold_axis)
            distance = np.linalg.norm(error_position)
            angle = np.linalg.norm(error_rotation)
            if distance < tolerance and angle < 0.03:
                return True

            if distance > step:
                error_position *= step / distance
            if angle > rot_step:
                error_rotation *= rot_step / angle

            mujoco.mj_jacSite(
                self.model,
                self.data,
                jacobian_position,
                jacobian_rotation,
                self.site_id,
            )
            jacobian = np.vstack(
                (
                    jacobian_position[:, self.dof_addresses],
                    jacobian_rotation[:, self.dof_addresses],
                )
            )
            error = np.concatenate((error_position, error_rotation))
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping * np.eye(6),
                error,
            )
            self.joint_positions = np.clip(
                self.joint_positions + delta,
                self.lower_limits,
                self.upper_limits,
            )
            if not self.tick():
                return False

        return False

    def rotate(self, joint: str, delta: float, rate: float = 0.003) -> bool:
        """Rotate one named joint by ``delta`` radians within its limits."""
        if joint not in ARM_JOINTS:
            raise ValueError(f"unknown arm joint {joint!r}")
        if rate <= 0:
            raise ValueError("rate must be positive")

        index = ARM_JOINTS.index(joint)
        goal = float(
            np.clip(
                self.joint_positions[index] + delta,
                self.lower_limits[index],
                self.upper_limits[index],
            )
        )
        while abs(goal - self.joint_positions[index]) > 1e-4:
            self.joint_positions[index] += np.clip(
                goal - self.joint_positions[index], -rate, rate
            )
            if not self.tick():
                return False
        return True

    def hold(self, seconds: float = 1.0) -> bool:
        """Hold the current pose for the requested duration."""
        if seconds < 0:
            raise ValueError("seconds must not be negative")
        for _ in range(int(seconds / self.model.opt.timestep)):
            if not self.tick():
                return False
        return True
