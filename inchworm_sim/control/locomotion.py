
from inchworm_sim.control.robot_interface import RobotInterface


class InchwormLocomotion:

    def __init__(self, robot: RobotInterface):
        self.robot = robot

    def crawl_step(self, moving_ee, target_pose):

        # Determine which end remains anchored.

        fixed_ee = (
            "ee1" if moving_ee == "ee2" else "ee2"
        )

        state = self.robot.get_state()

        # Safety: fixed end must be attached.
        if not self._is_attached(state, fixed_ee):
            return False

        # 1. Release the moving end.
        if not self.robot.detach_gripper(moving_ee):
            return False

        # 2. Move the free end.
        success = self.robot.move_ee(
            moving_ee,
            target_pose.position,
            target_pose.orientation
        )

        if not success:
            return False

        # 3. Attach the moving end at its new position.
        if not self.robot.attach_gripper(moving_ee, target_pose):
            return False

        # 4. Verify that the new attachment succeeded.
        state = self.robot.get_state()

        if not self._is_attached(state, moving_ee):
            return False

        # 5. Only now release the original anchor.
        if not self.robot.detach_gripper(fixed_ee):
            return False

        # 6. Move the original anchor forward.
        fixed_target = self._get_next_anchor_target(
            fixed_ee,
            target_pose
        )

        if not self.robot.move_ee(
            fixed_ee,
            fixed_target.position,
            fixed_target.orientation
        ):
            return False

        # 7. Reattach.
        return self.robot.attach_gripper(
            fixed_ee,
            fixed_target
        )

    def _is_attached(self, state, ee_id):

        if ee_id == "ee1":
            return state.ee1_attached

        return state.ee2_attached

    def _get_next_anchor_target(self, ee_id, target_pose):
        # TODO:
        # Calculate the next position of the other end.
        pass
