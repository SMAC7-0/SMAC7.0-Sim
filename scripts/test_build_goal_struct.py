"""Interactive center-screen block building test."""

import time
from pathlib import Path
from typing import Optional, Tuple

import mujoco
import mujoco.viewer
import numpy as np

from inchworm_sim.core.world_grid import WorldGrid
from inchworm_sim.physics.sim_bridge import SimBridge


BLOCK_NAMES = tuple(f"block{index}" for index in range(1, 21))
BLOCK_EDGE = 0.0742
DEPOT_POSITION = np.array((0.0, 0.0, -10.0), dtype=float)


def _scene_block_names(model: mujoco.MjModel) -> Tuple[str, ...]:
    """Return configured block names that are present in the loaded scene."""
    names = []
    for name in BLOCK_NAMES:
        try:
            model.body(name)
        except KeyError:
            continue
        names.append(name)
    return tuple(names)


model = mujoco.MjModel.from_xml_path("models/scene.xml")
data = mujoco.MjData(model)
block_names = _scene_block_names(model)
if not block_names:
    raise RuntimeError("models/scene.xml does not contain any configured blocks")
if len(block_names) < len(BLOCK_NAMES):
    print(
        f"Warning: scene contains {len(block_names)} of the "
        f"{len(BLOCK_NAMES)} configured blocks"
    )
world_grid = WorldGrid(
    block_edge=BLOCK_EDGE,
    origin_offset=(0.0, 0.0, 0.0),
)
world_grid.rules_enabled = True
bridge = SimBridge(model, data, world_grid, block_names)

viewer_handle: Optional[object] = None


def _raycast() -> Optional[Tuple[int, np.ndarray]]:
    """Cast a ray straight forward from the center of the camera."""
    if viewer_handle is None:
        return None

    camera = viewer_handle.cam
    azimuth = np.deg2rad(camera.azimuth)
    elevation = np.deg2rad(camera.elevation)
    ray_direction = np.array(
        (
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ),
        dtype=float,
    )
    ray_origin = (
        np.asarray(camera.lookat, dtype=float) - camera.distance * ray_direction
    )
    ray_geom_id = np.array([-1], dtype=np.int32)
    ray_distance = mujoco.mj_ray(
        model,
        data,
        ray_origin,
        ray_direction,
        np.ones(6, dtype=np.uint8),
        1,
        -1,
        ray_geom_id,
    )
    if ray_geom_id[0] < 0:
        return None
    return (
        int(ray_geom_id[0]),
        ray_origin + float(ray_distance) * ray_direction,
    )


def _block_id_for_geom(geom_id: int) -> Optional[str]:
    """Return the logical block name for a hit geometry, if it is a block."""
    body_id = int(model.geom_bodyid[geom_id])
    body_name = model.body(body_id).name
    return body_name if body_name in bridge.blocks else None


def _hit_normal(geom_id: int, hit_point: np.ndarray) -> np.ndarray:
    """Find the outward normal of the box face containing a hit point."""
    body_id = int(model.geom_bodyid[geom_id])
    body_position = data.xpos[body_id]
    rotation = data.xmat[body_id].reshape(3, 3)
    local_point = rotation.T @ (hit_point - body_position)
    axis = int(np.argmax(np.abs(local_point)))
    normal = rotation[:, axis].copy()
    if local_point[axis] < 0.0:
        normal *= -1.0
    return normal


def _placement_target(
    hit_geom_id: int, hit_point: np.ndarray
) -> Tuple[int, int, int]:
    """Return the grid cell targeted by a placement ray hit."""
    block_id = _block_id_for_geom(hit_geom_id)
    if block_id is None:
        return world_grid.world_to_grid(hit_point)
    shifted_point = hit_point + _hit_normal(hit_geom_id, hit_point) * (
        BLOCK_EDGE * 0.51
    )
    return world_grid.world_to_grid(shifted_point)


def _place_block(hit_geom_id: int, hit_point: np.ndarray) -> None:
    """Place an available block in the grid cell selected by the camera."""
    target_coord = _placement_target(hit_geom_id, hit_point)
    if not world_grid.is_valid_placement(target_coord):
        reason = world_grid.placement_validation_error(target_coord)
        print(f"Placement rejected at {target_coord}: {reason}")
        return

    unused_block_id = next(
        (name for name in block_names if name not in world_grid.blocks_by_id),
        None,
    )
    if unused_block_id is None:
        print("Placement rejected: all blocks are already in use")
        return

    world_grid.place_block(unused_block_id, target_coord)
    bridge.blocks[unused_block_id].set_pose(world_grid.grid_to_world(target_coord))
    print(f"Placed {unused_block_id} at {target_coord}")


def _remove_block(hit_geom_id: int) -> None:
    """Remove a clicked block and return its physical body to the depot."""
    block_id = _block_id_for_geom(hit_geom_id)
    if block_id is None:
        return

    coord = world_grid.blocks_by_id[block_id].grid_pos
    if coord is None:
        return
    if not world_grid.is_valid_removal(coord):
        reason = world_grid.removal_validation_error(coord)
        print(f"Removal rejected at {coord}: {reason}")
        return

    world_grid.remove_block(coord)
    bridge.blocks[block_id].set_pose(DEPOT_POSITION)
    print(f"Removed {block_id} from {coord}")


def export_structure(filepath: str = "structure.xyz") -> None:
    """Save the coordinates of all blocks currently in the logical grid."""
    with open(filepath, "w", encoding="utf-8") as structure_file:
        count = 0
        for block in world_grid.blocks_by_id.values():
            if block.grid_pos is None:
                continue
            x, y, z = block.grid_pos
            structure_file.write(f"{x} {y} {z}\n")
            count += 1
    print(f"Saved {count} blocks to {filepath}")


def import_structure(filepath: str = "structure.xyz") -> None:
    """Replace the current logical and physical structure from a coordinate file."""
    structure_path = Path(filepath)
    if not structure_path.exists():
        print(f"Warning: structure file not found: {filepath}")
        return

    coordinates = [
        tuple(int(value) for value in line.split())
        for line in structure_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    for block in list(world_grid.blocks_by_id.values()):
        if block.grid_pos is None:
            continue
        block_id = block.block_id
        world_grid.remove_block(block.grid_pos, enforce_rules=False)
        bridge.blocks[block_id].set_pose(DEPOT_POSITION)

    count = 0
    for coord in coordinates:
        if len(coord) != 3:
            print(f"Warning: skipping invalid coordinate {coord}")
            continue
        if not world_grid.is_empty(coord):
            print(f"Warning: skipping occupied coordinate {coord}")
            continue
        unused_block_id = next(
            (name for name in block_names if name not in world_grid.blocks_by_id),
            None,
        )
        if unused_block_id is None:
            print("Warning: ran out of blocks while loading structure")
            break
        world_grid.place_block(unused_block_id, coord, enforce_rules=False)
        bridge.blocks[unused_block_id].set_pose(
            world_grid.grid_to_world(coord)
        )
        count += 1

    print(f"Loaded {count} blocks from {filepath}")


def key_callback(keycode: int) -> None:
    """Handle save/load shortcuts and toggle validity checks with V."""
    if keycode in (ord("e"), ord("E")):
        hit = _raycast()
        if hit is not None:
            _place_block(hit[0], hit[1])
        return
    if keycode in (ord("q"), ord("Q")):
        hit = _raycast()
        if hit is not None:
            _remove_block(hit[0])
        return
    if keycode in (ord("k"), ord("K")):
        export_structure()
        return
    if keycode in (ord("l"), ord("L")):
        import_structure()
        return
    if keycode not in (ord("v"), ord("V")):
        return
    world_grid.rules_enabled = not world_grid.rules_enabled
    state = "ENABLED" if world_grid.rules_enabled else "DISABLED"
    print(f"Validity checks: {state}")


def _update_target_highlight(viewer: object) -> None:
    """Highlight the placement cell and the block targeted for removal."""
    hit = _raycast()
    with viewer.lock():
        scene = viewer.user_scn
        if scene is None or scene.maxgeom < 2:
            raise RuntimeError("viewer user scene cannot draw target highlights")
        scene.ngeom = 0
        if hit is None:
            return

        hit_geom_id, hit_point = hit
        placement_coord = _placement_target(hit_geom_id, hit_point)
        placement_color = (
            (0.2, 1.0, 0.2, 0.4)
            if world_grid.is_valid_placement(placement_coord)
            else (1.0, 0.2, 0.2, 0.4)
        )
        box_size = np.full(3, BLOCK_EDGE / 2.0, dtype=np.float64)
        identity = np.eye(3, dtype=np.float64).reshape(-1)
        mujoco.mjv_initGeom(
            scene.geoms[0],
            mujoco.mjtGeom.mjGEOM_BOX,
            box_size,
            world_grid.grid_to_world(placement_coord),
            identity,
            np.asarray(placement_color, dtype=np.float32),
        )
        scene.ngeom = 1

        block_id = _block_id_for_geom(hit_geom_id)
        if block_id is None:
            return
        block_coord = world_grid.blocks_by_id[block_id].grid_pos
        if block_coord is None:
            return
        removal_color = (
            (0.2, 1.0, 0.2, 0.4)
            if world_grid.is_valid_removal(block_coord)
            else (1.0, 0.2, 0.2, 0.4)
        )
        mujoco.mjv_initGeom(
            scene.geoms[1],
            mujoco.mjtGeom.mjGEOM_BOX,
            box_size,
            world_grid.grid_to_world(block_coord),
            identity,
            np.asarray(removal_color, dtype=np.float32),
        )
        scene.ngeom = 2


print(
    "\nInteractive building demo:\n"
    "  Rotate the camera to aim at the ground or a block (center screen).\n"
    # "  Green/red boxes show placement and removal validity.\n"
    "  Press E to place a block at the highlighted cell.\n"
    "  Press Q to remove the highlighted block.\n"
    "  Press K to Save structure to structure.xyz\n"
    "  Press L to Load structure from structure.xyz\n"
    "  Press V to toggle validity rules.\n"
)

with mujoco.viewer.launch_passive(
    model,
    data,
    key_callback=key_callback,
) as viewer:
    viewer_handle = viewer
    while viewer.is_running():
        bridge.step()
        viewer.sync()
        # _update_target_highlight(viewer)
        viewer.sync()
        time.sleep(model.opt.timestep)
