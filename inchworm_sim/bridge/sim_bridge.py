"""Scene-level bridge between MuJoCo objects and the logical world grid."""

from __future__ import annotations

import re
from typing import Optional, Tuple

import mujoco
import numpy as np

from inchworm_sim.bridge.faces import FACES, OPPOSITE_FACE
from inchworm_sim.core.world_grid import WorldGrid

from inchworm_sim.physics.mujoco_block import MujocoBlock
from inchworm_sim.physics.mujoco_inchworm import MujocoWorm

GridCoord = Tuple[int, int, int]


class SimBridge:
    """Coordinate worm actions, block welds, and logical grid placement."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        world_grid: WorldGrid,
        block_names: Optional[Tuple[str, ...]] = None,
    ) -> None:
        self.model = model
        self.data = data
        self.world_grid = world_grid
        self.worm = MujocoWorm(model, data)
        if block_names is None:
            block_names = tuple(
                sorted(
                    (
                        model.body(body_id).name
                        for body_id in range(model.nbody)
                        if re.fullmatch(r"block\d+", model.body(body_id).name)
                    ),
                    key=lambda name: int(name[5:]),
                )
            )
        if not block_names:
            raise ValueError("the MuJoCo model contains no blockN bodies")
        self.blocks = {
            name: MujocoBlock(model, data, name) for name in block_names
        }
        self.weld_ids = {
            ("ee2", name, face): model.equality(f"ee2_{name}_{face}_weld").id
            for name in block_names
            for face in FACES
        }
        self.structural_weld_ids = {
            model.equality(equality_id).name: equality_id
            for equality_id in range(model.neq)
            if re.fullmatch(
                r"block\d+_block\d+_(?:[pn][xyz]_[pn][xyz])?_weld",
                model.equality(equality_id).name,
            )
        }
        self.grab_distance = 0.03
        self.weld_ramp_time = 0.10
        self.release_cooldown_duration = 0.25
        self.release_cooldown = 0.0
        self.register_initial_blocks()

    def detach_block_connections(self, block_id: str) -> None:
        """Disable every physical structural weld involving a block."""
        for weld_name, weld_id in self.structural_weld_ids.items():
            block_names = weld_name.removesuffix("_weld").split("_")
            if block_id in block_names:
                self.data.eq_active[weld_id] = 0

    def attach_block_connections(self, block_id: str) -> None:
        """Enable structural welds to the block's currently placed neighbors."""
        grid_block = self.world_grid.blocks_by_id.get(block_id)
        if grid_block is None or grid_block.grid_pos is None:
            return
        for neighbor in self.world_grid.get_neighbors(grid_block.grid_pos).values():
            face = next(
                face
                for face, candidate in self.world_grid.get_neighbors(
                    grid_block.grid_pos
                ).items()
                if candidate.block_id == neighbor.block_id
            )
            opposite = OPPOSITE_FACE[face]
            candidates = (
                f"{block_id}_{neighbor.block_id}_{face}_{opposite}_weld",
                f"{neighbor.block_id}_{block_id}_{opposite}_{face}_weld",
                f"{block_id}_{neighbor.block_id}_weld",
                f"{neighbor.block_id}_{block_id}_weld",
            )
            connected = False
            for weld_name in candidates:
                weld_id = self.structural_weld_ids.get(weld_name)
                if weld_id is not None:
                    self.data.eq_active[weld_id] = 1
                    connected = True
                    break
            if not connected:
                raise RuntimeError(
                    "No structural weld exists for "
                    f"{block_id} and {neighbor.block_id} on {face}"
                )

    def _align_grab_face(
        self, block_id: str, face: str, gripper: str
    ) -> Tuple[float, float]:
        """Align a selected block face with the gripper before enabling its weld."""
        block = self.blocks[block_id]
        face_site_id = block.face_site_ids[face]
        gripper_site_id = self.worm.gripper_face_sites[gripper][face]
        gripper_position = self.data.site_xpos[gripper_site_id].copy()
        gripper_rotation = self.data.site_xmat[gripper_site_id].reshape(3, 3)
        gripper_quat = np.zeros(4)
        mujoco.mju_mat2Quat(gripper_quat, gripper_rotation.reshape(9))
        local_face_position = self.model.site_pos[face_site_id].copy()
        local_face_quat = self.model.site_quat[face_site_id].copy()

        inverse_local_face_quat = np.zeros(4)
        aligned_quat = np.zeros(4)
        aligned_rotation = np.zeros(9)
        mujoco.mju_negQuat(inverse_local_face_quat, local_face_quat)
        mujoco.mju_mulQuat(
            aligned_quat, gripper_quat, inverse_local_face_quat
        )
        mujoco.mju_quat2Mat(aligned_rotation, aligned_quat)
        aligned_position = gripper_position - aligned_rotation.reshape(
            3, 3
        ) @ local_face_position

        block.set_pose(aligned_position, tuple(aligned_quat))
        mujoco.mj_forward(self.model, self.data)

        position_error = float(
            np.linalg.norm(
                self.data.site_xpos[gripper_site_id]
                - self.data.site_xpos[face_site_id]
            )
        )
        face_quat = np.zeros(4)
        face_rotation = self.data.site_xmat[face_site_id].reshape(3, 3)
        mujoco.mju_mat2Quat(face_quat, face_rotation.reshape(9))
        quat_dot = float(np.clip(abs(np.dot(gripper_quat, face_quat)), 0.0, 1.0))
        angle_error = float(2.0 * np.arccos(quat_dot))
        return position_error, angle_error

    def register_initial_blocks(self) -> None:
        """Register every scene block in the logical grid."""
        mujoco.mj_forward(self.model, self.data)
        for block_id, block in self.blocks.items():
            grid_coord = self.world_grid.world_to_grid(block.get_world_pos())
            was_valid = self.world_grid.is_valid_placement(grid_coord)
            placed = self.world_grid.place_block(block_id, grid_coord)
            print(
                f"Initial {block_id}: grid={grid_coord}, valid={was_valid}, "
                f"connections={placed.face_connections}"
            )

    def align_placed_blocks_to_grid(self) -> None:
        """Lock every currently placed block to its logical cell."""
        for block_id, grid_block in list(self.world_grid.blocks_by_id.items()):
            if grid_block.grid_pos is not None:
                self.blocks[block_id].set_pose(
                    self.world_grid.grid_to_world(grid_block.grid_pos)
                )

    def nearest_valid_cell(self, target_coord: GridCoord) -> Optional[GridCoord]:
        """Find the closest empty cell satisfying placement rules."""
        return self._nearest_cell(target_coord, require_support=True)

    def nearest_empty_cell(self, target_coord: GridCoord) -> Optional[GridCoord]:
        """Find the closest empty cell without support requirements."""
        return self._nearest_cell(target_coord, require_support=False)

    def nearest_available_block(
        self, gripper: str = "ee2"
    ) -> Tuple[Optional[str], Optional[str], Optional[float]]:
        """Return the closest placed block face and its distance."""
        gripper_site = self.worm.gripper_sites[gripper]
        distances = [
            (
                float(
                    np.linalg.norm(
                        self.data.site_xpos[gripper_site]
                        - block.get_face_site_pos(face)
                    )
                ),
                block_id,
                face,
            )
            for block_id, block in self.blocks.items()
            if block_id in self.world_grid.blocks_by_id
            for face in FACES
        ]
        if not distances:
            return None, None, None
        distance, block_id, face = min(distances)
        return block_id, face, distance

    def _nearest_cell(
        self, target_coord: GridCoord, require_support: bool
    ) -> Optional[GridCoord]:
        candidates = []
        for x in range(self.world_grid.min_x, self.world_grid.max_x + 1):
            for y in range(self.world_grid.min_y, self.world_grid.max_y + 1):
                for z in range(self.world_grid.size_z):
                    coord = (x, y, z)
                    if self.world_grid.is_empty(coord) and (
                        not require_support
                        or self.world_grid.is_valid_placement(coord)
                    ):
                        distance = sum(
                            (coord[index] - target_coord[index]) ** 2
                            for index in range(3)
                        )
                        candidates.append((distance, coord))
        return min(candidates, default=(None, None), key=lambda item: item[0])[1]

    def grab_nearest_block(self, gripper: str = "ee2") -> Optional[str]:
        """Grab the closest placed block within the gripper reach."""
        if (
            self.worm.held_blocks.get(gripper) is not None
            or self.release_cooldown > 0.0
        ):
            print("GRAB REJECTED: gripper is already holding a block or cooling down")
            return None
        block_id, face, distance = self.nearest_available_block(gripper)
        if block_id is None or face is None or distance is None:
            print("GRAB REJECTED: no blocks are available in the logical grid")
            return None
        if distance >= self.grab_distance:
            print(
                f"GRAB REJECTED: nearest block is {distance:.4f} m away "
                f"(limit {self.grab_distance:.4f} m)"
            )
            return None

        grid_coord = self.world_grid.blocks_by_id[block_id].grid_pos
        if (
            self.world_grid.rules_enabled
            and not self.world_grid.is_valid_removal(grid_coord)
        ):
            print(
                f"GRAB REJECTED by Validity Check: {block_id} at {grid_coord} "
                f"is surrounded on all 4 horizontal sides!"
            )
            return None
        self.detach_block_connections(block_id)
        self.world_grid.remove_block(grid_coord, enforce_rules=False)
        weld_id = self.weld_ids[(gripper, block_id, face)]
        mujoco.mj_forward(self.model, self.data)
        position_error, angle_error = self._align_grab_face(
            block_id, face, gripper
        )
        self.model.eq_solref[weld_id] = self.worm.soft_solref
        self.data.eq_active[weld_id] = 1
        self.worm.held_blocks[gripper] = (block_id, face)
        self.worm.weld_timers[gripper] = 0.0
        print(
            f"GRABBED {block_id} from {grid_coord} "
            f"via {face} (distance={distance:.4f} m, "
            f"position error={position_error:.6f} m, "
            f"angle error={np.degrees(angle_error):.3f} deg) - "
            "Free motion enabled"
        )
        return block_id

    def release_held_block(self, gripper: str = "ee2") -> Optional[GridCoord]:
        """Release a block and snap it to the nearest grid cell."""
        held = self.worm.held_blocks.get(gripper)
        if held is None:
            return None
        block_id, face = held

        weld_id = self.weld_ids[(gripper, block_id, face)]
        self.data.eq_active[weld_id] = 0
        mujoco.mj_forward(self.model, self.data)
        target_coord = self.world_grid.world_to_grid(
            self.blocks[block_id].get_world_pos()
        )
        target_coord = tuple(
            max(lower, min(upper, value))
            for value, lower, upper in zip(
                target_coord,
                (
                    self.world_grid.min_x,
                    self.world_grid.min_y,
                    self.world_grid.min_z,
                ),
                (
                    self.world_grid.max_x,
                    self.world_grid.max_y,
                    self.world_grid.max_z,
                ),
            )
        )
        if self.world_grid.rules_enabled:
            if self.world_grid.is_valid_placement(target_coord):
                placed_coord = target_coord
            else:
                reason = self.world_grid.placement_validation_error(target_coord)
                print(
                    f"PLACEMENT ADJUSTED: {block_id} at {target_coord} rejected: "
                    f"{reason}"
                )
                placed_coord = self.nearest_valid_cell(target_coord)
        else:
            placed_coord = self.nearest_empty_cell(target_coord)
        if placed_coord is None:
            raise RuntimeError("cannot release block: no valid empty grid cell")

        placed = self.world_grid.place_block(
            block_id, placed_coord, enforce_rules=False
        )
        self.blocks[block_id].set_pose(
            self.world_grid.grid_to_world(placed_coord)
        )
        self.attach_block_connections(block_id)
        self.release_cooldown = self.release_cooldown_duration
        self.worm.held_blocks[gripper] = None
        print(
            f"PLACED {block_id} at grid cell {placed_coord} | "
            f"Neighbors: {placed.face_connections}"
        )
        return placed_coord

    def align_held_blocks_to_grippers(self) -> None:
        """Lock every held block's grabbed face to its gripper site."""
        for gripper, held in self.worm.held_blocks.items():
            if held is not None:
                block_id, face = held
                self._align_grab_face(block_id, face, gripper)

    def step(self) -> None:
        """Align blocks to the grid and grippers, ramp welds, and step MuJoCo."""
        mujoco.mj_forward(self.model, self.data)
        self.align_placed_blocks_to_grid()
        mujoco.mj_forward(self.model, self.data)
        self.release_cooldown = max(
            0.0, self.release_cooldown - self.model.opt.timestep
        )
        for gripper, held in self.worm.held_blocks.items():
            if held is None:
                continue
            block_id, face = held
            timer = self.worm.weld_timers[gripper]
            if timer < self.weld_ramp_time:
                timer += self.model.opt.timestep
                self.worm.weld_timers[gripper] = timer
                alpha = min(timer / self.weld_ramp_time, 1.0)
                weld_id = self.weld_ids[(gripper, block_id, face)]
                self.model.eq_solref[weld_id] = (
                    1.0 - alpha
                ) * np.asarray(self.worm.soft_solref) + alpha * np.asarray(
                    self.worm.hard_solref
                )
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.align_held_blocks_to_grippers()
