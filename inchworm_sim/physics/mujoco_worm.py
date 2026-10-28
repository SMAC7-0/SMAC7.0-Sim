"""MuJoCo-backed inchworm adapter implementing the worm-facing API."""

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import mujoco

if TYPE_CHECKING:
    from .sim_bridge import SimBridge


class MujocoWorm:
    """Bind worm joints and grippers to a MuJoCo model."""

    JOINT_NAMES = ("wrist1", "hand1", "top", "hand2", "wrist2")
    ACTUATOR_NAMES = ("wrist1_act", "hand1_act", "top_act", "hand2_act", "wrist2_act")

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        worm_id: str = "inchworm",
    ) -> None:
        self.model = model
        self.data = data
        self.worm_id = worm_id
        self.gripper_sites = {"ee2": model.site("ee2_grab").id}
        if model.nsite and self._has_site(model, "ee1_grab"):
            self.gripper_sites["ee1"] = model.site("ee1_grab").id
        self.joint_ids = [model.joint(name).id for name in self.JOINT_NAMES]
        self.actuator_ids = [model.actuator(name).id for name in self.ACTUATOR_NAMES]
        self.held_blocks: Dict[str, Optional[str]] = {"ee2": None}
        self.connections: Dict[str, object] = {}
        self.weld_timers: Dict[str, float] = {"ee2": 0.0}
        self.soft_solref = (0.10, 1.0)
        self.hard_solref = (0.01, 1.0)

    @staticmethod
    def _has_site(model: mujoco.MjModel, name: str) -> bool:
        """Return whether a named site exists in the model."""
        try:
            model.site(name)
        except KeyError:
            return False
        return True

    def read_joint_positions(self) -> List[float]:
        """Read the five inchworm joint angles in model order."""
        return [
            float(self.data.qpos[self.model.jnt_qposadr[joint_id]])
            for joint_id in self.joint_ids
        ]

    def send_joint_positions(
        self,
        theta1: float,
        theta2: float,
        theta3: float,
        theta4: float,
        theta5: float,
        duration: float = 0.0,
    ) -> None:
        """Send target angles to the five position actuators."""
        del duration
        self.data.ctrl[self.actuator_ids] = [theta1, theta2, theta3, theta4, theta5]

    def close_gripper(self, gripper: str, sim_bridge: "SimBridge") -> Optional[str]:
        """Ask the simulation bridge to grab the nearest reachable block."""
        return sim_bridge.grab_nearest_block(gripper)

    def open_gripper(
        self, gripper: str, sim_bridge: "SimBridge"
    ) -> Optional[Tuple[int, int, int]]:
        """Ask the simulation bridge to release and place the held block."""
        return sim_bridge.release_held_block(gripper)

    def send_message(self, message: object, gripper: str) -> None:
        """Store a message placeholder for the full worm communication API."""
        self.connections[gripper] = message

    def recv_message(self, is_blocking: bool = False) -> Optional[object]:
        """Return no message until a communication backend is connected."""
        del is_blocking
        return None

    def add_connection(self, connection: object, gripper: str) -> None:
        """Register a communication connection for a gripper."""
        self.connections[gripper] = connection
