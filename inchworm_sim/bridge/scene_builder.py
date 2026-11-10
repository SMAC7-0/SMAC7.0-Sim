"""Generate reusable MuJoCo scenes containing any number of blocks."""

import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping, Tuple, Union

from inchworm_sim.bridge.faces import FACES, OPPOSITE_FACE

from .coords import BLOCK_EDGE, grid_to_world

GridCoord = Tuple[int, int, int]
_BLOCK_COLORS = (
    (1.0, 0.2, 0.2, 1.0),
    (0.2, 1.0, 0.2, 1.0),
    (0.2, 0.4, 1.0, 1.0),
    (1.0, 0.9, 0.1, 1.0),
    (0.8, 0.2, 1.0, 1.0),
    (0.1, 0.9, 0.9, 1.0),
)


@dataclass(frozen=True)
class BlockSpec:
    """Physical properties shared by every generated block."""

    edge: float = BLOCK_EDGE
    mass: float = 0.01
    friction: Tuple[float, float, float] = (1.0, 0.005, 0.0001)
    site_size: float = 0.001


DEFAULT_BLOCK_SPEC = BlockSpec()


def _format_vector(values: Iterable[float]) -> str:
    return " ".join(f"{value:.9g}" for value in values)


def build_scene(
    block_positions: Mapping[str, GridCoord],
    scene_file: Union[Path, str],
    output_file: Union[Path, str],
    origin_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    block_spec: BlockSpec = DEFAULT_BLOCK_SPEC,
) -> None:
    """Create a scene with reusable block bodies and all required welds.

    ``scene_file`` supplies the floor, lighting, and inchworm. The generated
    file owns all block bodies and equality constraints, so the block count is
    not limited by a checked-in XML template.
    """
    if not block_positions:
        raise ValueError("block_positions must contain at least one block")
    if block_spec.edge <= 0:
        raise ValueError("block_spec.edge must be positive")
    if block_spec.mass <= 0:
        raise ValueError("block_spec.mass must be positive")

    tree = ET.parse(scene_file)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("scene XML must contain a worldbody")

    for include in list(root.findall("include")):
        if include.get("file") == "block.xml":
            root.remove(include)

    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    else:
        equality.clear()

    for block_index, (block_id, coord) in enumerate(block_positions.items()):
        if not block_id:
            raise ValueError("block IDs must not be empty")
        position = grid_to_world(coord, origin_offset)
        body = ET.SubElement(
            worldbody,
            "body",
            name=block_id,
            pos=_format_vector(position),
        )
        ET.SubElement(body, "freejoint", name=f"{block_id}_free")
        ET.SubElement(
            body,
            "geom",
            name=f"{block_id}_geom",
            type="box",
            size=_format_vector((block_spec.edge / 2,) * 3),
            mass=str(block_spec.mass),
            friction=_format_vector(block_spec.friction),
            contype="1",
            conaffinity="1",
            rgba=_format_vector(_BLOCK_COLORS[block_index % len(_BLOCK_COLORS)]),
        )
        for face, direction in FACES.items():
            ET.SubElement(
                body,
                "site",
                name=f"{block_id}_{face}",
                pos=_format_vector(
                    component * block_spec.edge / 2 for component in direction
                ),
                size=str(block_spec.site_size),
                rgba="0 0 0 0",
                group="3",
            )
        for face in FACES:
            ET.SubElement(
                equality,
                "weld",
                name=f"ee2_{block_id}_{face}_weld",
                site1=f"ee2_grab_{face}",
                site2=f"{block_id}_{face}",
                active="false",
                solref="0.1 1",
                solimp="0.999 0.9999 0.001",
            )

    positions = {}
    for block_id, coord in block_positions.items():
        normalized_coord = tuple(coord)
        if len(normalized_coord) != 3:
            raise ValueError(f"invalid coordinate for {block_id!r}: {coord!r}")
        if normalized_coord in positions:
            raise ValueError(f"multiple blocks occupy {normalized_coord}")
        positions[normalized_coord] = block_id
    block_items = list(block_positions.items())
    for (block_a, coord_a), (block_b, coord_b) in combinations(
        block_items, 2
    ):
        delta = tuple(coord_b[index] - coord_a[index] for index in range(3))
        for face, direction in FACES.items():
            opposite = OPPOSITE_FACE[face]
            is_adjacent = delta == direction
            active = "true" if is_adjacent else "false"
            ET.SubElement(
                equality,
                "weld",
                name=f"{block_a}_{block_b}_{face}_{opposite}_weld",
                site1=f"{block_a}_{face}",
                site2=f"{block_b}_{opposite}",
                active=active,
                solref="0.1 1",
                solimp="0.7 0.9 0.001",
            )

    ET.indent(tree, space="  ")
    tree.write(output_file, encoding="utf-8", xml_declaration=True)


def create_temporary_scene(
    block_positions: Mapping[str, GridCoord],
    scene_file: Union[Path, str],
    origin_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    block_spec: BlockSpec = DEFAULT_BLOCK_SPEC,
) -> Path:
    """Generate a temporary scene beside ``scene_file`` for MuJoCo loading."""
    scene_path = Path(scene_file)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=".runtime-scene-",
        suffix=".xml",
        dir=scene_path.parent,
    )
    os.close(file_descriptor)
    try:
        build_scene(
            block_positions,
            scene_path,
            temporary_name,
            origin_offset=origin_offset,
            block_spec=block_spec,
        )
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return Path(temporary_name)
