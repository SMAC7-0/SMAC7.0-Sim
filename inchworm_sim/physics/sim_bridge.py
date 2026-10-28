"""Scene-level bridge between MuJoCo objects and the logical world grid."""

from typing import Dict, Optional, Tuple

import mujoco
import numpy as np

from inchworm_sim.core.world_grid import WorldGrid
from .mujoco_block import MujocoBlock
from .mujoco_worm import MujocoWorm


GridCoord = Tuple[int, int, int]


class SimBridge:
    """Coordinate worm actions, block welds, and logical grid placement."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        world_grid: WorldGrid,
        block_names: Tuple[str, ...] = ("block1", "block2"),
    ) -> None:
        self.model = model
        self.data = data
        self.world_grid = world_grid
        self.worm = MujocoWorm(model, data)
        self.blocks = {
            name: MujocoBlock(model, data, name) for name in block_names
        }
        self.weld_ids = {
            ("ee2", name): model.equality(f"ee2_{name}_weld").id
            for name in block_names
        }
        self.grab_distance = 0.03
        self.weld_ramp_time = 0.10
        self.release_cooldown_duration = 0.25
        self.release_cooldown = 0.0
        self.register_initial_blocks()

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
    ) -> Tuple[Optional[str], Optional[float]]:
        """Return the closest placed block and its distance from a gripper."""
        gripper_site = self.worm.gripper_sites[gripper]
        distances = [
            (
                float(
                    np.linalg.norm(
                        self.data.site_xpos[gripper_site]
                        - block.get_grab_site_pos()
                    )
                ),
                block_id,
            )
            for block_id, block in self.blocks.items()
            if block_id in self.world_grid.blocks_by_id
        ]
        if not distances:
            return None, None
        distance, block_id = min(distances)
        return block_id, distance

    def _nearest_cell(
        self, target_coord: GridCoord, require_support: bool
    ) -> Optional[GridCoord]:
        candidates = []
        for x in range(self.world_grid.size_x):
            for y in range(self.world_grid.size_y):
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
        block_id, distance = self.nearest_available_block(gripper)
        if block_id is None or distance is None:
            print("GRAB REJECTED: no blocks are available in the logical grid")
            return None
        if distance >= self.grab_distance:
            print(
                f"GRAB REJECTED: nearest block is {distance:.4f} m away "
                f"(limit {self.grab_distance:.4f} m)"
            )
            return None

        grid_coord = self.world_grid.blocks_by_id[block_id].grid_pos
        self.world_grid.remove_block(grid_coord)
        weld_id = self.weld_ids[(gripper, block_id)]
        # Site welds retain the current near-contact pose; do not rewrite eq_data.
        self.model.eq_solref[weld_id] = self.worm.soft_solref
        self.data.eq_active[weld_id] = 1
        self.worm.held_blocks[gripper] = block_id
        self.worm.weld_timers[gripper] = 0.0
        print(
            f"GRABBED {block_id} from {grid_coord} "
            f"(distance={distance:.4f} m) - Free motion enabled"
        )
        return block_id

    def release_held_block(self, gripper: str = "ee2") -> Optional[GridCoord]:
        """Release a block and snap it to the nearest grid cell."""
        block_id = self.worm.held_blocks.get(gripper)
        if block_id is None:
            return None

        weld_id = self.weld_ids[(gripper, block_id)]
        self.data.eq_active[weld_id] = 0
        mujoco.mj_forward(self.model, self.data)
        target_coord = self.world_grid.world_to_grid(
            self.blocks[block_id].get_world_pos()
        )
        target_coord = tuple(
            max(0, min(limit - 1, value))
            for value, limit in zip(
                target_coord,
                (
                    self.world_grid.size_x,
                    self.world_grid.size_y,
                    self.world_grid.size_z,
                ),
            )
        )
        placed_coord = self.nearest_valid_cell(target_coord)
        if placed_coord is None:
            placed_coord = self.nearest_empty_cell(target_coord)
            if placed_coord is None:
                raise RuntimeError("cannot release block: the logical grid is full")
            print(f"WARNING: placing {block_id} without support rules")

        placed = self.world_grid.place_block(
            block_id, placed_coord, enforce_rules=False
        )
        self.blocks[block_id].set_pose(
            self.world_grid.grid_to_world(placed_coord)
        )
        self.release_cooldown = self.release_cooldown_duration
        self.worm.held_blocks[gripper] = None
        print(
            f"PLACED {block_id} at grid cell {placed_coord} | "
            f"Neighbors: {placed.face_connections}"
        )
        return placed_coord

    def step(self) -> None:
        """Align placed blocks, ramp welds, and advance MuJoCo one step."""
        mujoco.mj_forward(self.model, self.data)
        self.align_placed_blocks_to_grid()
        mujoco.mj_forward(self.model, self.data)
        self.release_cooldown = max(
            0.0, self.release_cooldown - self.model.opt.timestep
        )
        for gripper, block_id in self.worm.held_blocks.items():
            if block_id is None:
                continue
            timer = self.worm.weld_timers[gripper]
            if timer < self.weld_ramp_time:
                timer += self.model.opt.timestep
                self.worm.weld_timers[gripper] = timer
                alpha = min(timer / self.weld_ramp_time, 1.0)
                weld_id = self.weld_ids[(gripper, block_id)]
                self.model.eq_solref[weld_id] = (
                    1.0 - alpha
                ) * np.asarray(self.worm.soft_solref) + alpha * np.asarray(
                    self.worm.hard_solref
                )
        mujoco.mj_step(self.model, self.data)
