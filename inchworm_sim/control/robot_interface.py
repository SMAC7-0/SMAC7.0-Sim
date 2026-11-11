# Robot interface for both the simulated and physical robot

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class RobotState:
    joint_positions: np.ndarray
    joint_velocities: np.ndarray

    ee1_position: np.ndarray
    ee2_position: np.ndarray

    ee1_attached: bool
    ee2_attached: bool


class RobotInterface(ABC):

    @abstractmethod
    def get_state(self) -> RobotState:
        """Read the current robot state."""
        pass

    @abstractmethod
    def move_ee(
        self,
        ee_id: str,
        position: np.ndarray,
        orientation: Optional[np.ndarray] = None,
    ) -> bool:
        """Move an end effector to a target pose."""
        pass

    @abstractmethod
    def attach_gripper(
        self,
        ee_id: str,
        target=None,
    ) -> bool:
        """Attach a gripper to a surface."""
        pass

    @abstractmethod
    def detach_gripper(self, ee_id: str) -> bool:
        """Release a gripper."""
        pass

    @abstractmethod
    def command_joints(
        self,
        positions: np.ndarray,
    ) -> bool:
        """Send desired joint positions."""
        pass

    @abstractmethod
    def step(self, dt: float):
        """Advance the backend or wait for hardware."""
        pass
