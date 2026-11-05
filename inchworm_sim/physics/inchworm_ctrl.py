import time
import numpy as np
import mujoco

ARM = ["wrist1", "hand1", "top", "hand2", "wrist2"]


class Arm:
    """Temporary kinematic driver: arm joints are pinned to commanded angles, blocks stay dynamic."""

    def __init__(self, model, data, viewer=None, site="ee2_grab"):
        self.m, self.d, self.v = model, data, viewer
        jid = [model.joint(n).id for n in ARM]
        self.qadr = np.array([model.jnt_qposadr[j] for j in jid])
        self.dadr = np.array([model.jnt_dofadr[j] for j in jid])
        self.lo, self.hi = model.jnt_range[jid, 0], model.jnt_range[jid, 1]
        self.sid = model.site(site).id
        self.q = data.qpos[self.qadr].copy()
        mujoco.mj_forward(model, data)
        # pad orientation to hold: whatever the EE has in the home pose
        self.axis_hold = data.site_xmat[self.sid].reshape(3, 3)[:, 2].copy()

    def tick(self,):
        """One physics step with the arm pinned. Returns False if the viewer was closed."""
        self.d.qpos[self.qadr] = self.q
        self.d.qvel[self.dadr] = 0
        mujoco.mj_step(self.m, self.d)

        if self.v is not None:
            # Update the EE marker using the latest site position.
            with self.v.lock():
                self.v.user_scn.geoms[0].pos[:] = self.d.site_xpos[
                    self.m.site("ee2_grab").id
                ]

                self.v.user_scn.geoms[1].pos[:] = self.d.site_xpos[
                    self.m.site("block5_pz").id
                ]

            self.v.sync()
            time.sleep(self.m.opt.timestep)
            return self.v.is_running()
        return True

    def move_to(self, target, step=0.001, rot_step=0.02, tol=0.003,
                max_iters=3000, damping=1e-3):
        """Move the EE site to target (world coords) while keeping the pad parallel to the home pose."""
        m, d = self.m, self.d
        target = np.asarray(target, dtype=float)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(max_iters):
            d.qpos[self.qadr] = self.q
            mujoco.mj_forward(m, d)
            e_p = target - d.site_xpos[self.sid]
            axis = d.site_xmat[self.sid].reshape(3, 3)[:, 2]
            e_r = np.cross(axis, self.axis_hold)            # tilt error (2 DOF)
            dist, ang = np.linalg.norm(e_p), np.linalg.norm(e_r)
            if dist < tol and ang < 0.03:
                return True
            if dist > step:
                e_p = e_p * step / dist
            if ang > rot_step:
                e_r = e_r * rot_step / ang
            mujoco.mj_jacSite(m, d, jp, jr, self.sid)
            J = np.vstack([jp[:, self.dadr], jr[:, self.dadr]])
            err = np.concatenate([e_p, e_r])
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err)
            self.q = np.clip(self.q + dq, self.lo, self.hi)
            if not self.tick():
                return False
        print("move_to: did not converge (target may be out of reach)")
        return False

    def rotate(self, joint, delta, rate=0.003):
        """Rotate one joint by delta radians (clipped to its range) at `rate` rad per step."""
        i = ARM.index(joint)
        goal = float(np.clip(self.q[i] + delta, self.lo[i], self.hi[i]))
        while abs(goal - self.q[i]) > 1e-4:
            self.q[i] += np.clip(goal - self.q[i], -rate, rate)
            if not self.tick():
                return False
        return True

    def hold(self, seconds=1.0):
        for _ in range(int(seconds / self.m.opt.timestep)):
            if not self.tick():
                return False
        return True