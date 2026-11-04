"""Generate reusable MuJoCo scenes containing any number of blocks."""

from pathlib import Path
from typing import Iterable, Mapping, Tuple, Union
import xml.etree.ElementTree as ET

from .coords import BLOCK_EDGE, FACE_TO_SITE, grid_to_world


GridCoord = Tuple[int, int, int]
_DIRECTIONS = {
    "+x": (1, 0, 0),
    "+y": (0, 1, 0),
    "+z": (0, 0, 1),
}
_OPPOSITE = {"+x": "-x", "+y": "-y", "+z": "-z"}


def _format_vector(values: Iterable[float]) -> str:
    return " ".join(f"{value:.9g}" for value in values)


def build_scene(
    block_positions: Mapping[str, GridCoord],
    scene_file: Union[Path, str],
    output_file: Union[Path, str],
    origin_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    block_mass: float = 0.01,
) -> None:
    """Create a scene with reusable block bodies and all required welds.

    ``scene_file`` supplies the floor, lighting, and inchworm. The generated
    file owns all block bodies and equality constraints, so the block count is
    not limited by a checked-in XML template.
    """
    if not block_positions:
        raise ValueError("block_positions must contain at least one block")
    if block_mass <= 0:
        raise ValueError("block_mass must be positive")

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

    for block_id, coord in block_positions.items():
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
            size=_format_vector((BLOCK_EDGE / 2,) * 3),
            mass=str(block_mass),
            friction="1 0.005 0.0001",
            contype="1",
            conaffinity="1",
            rgba="0.2 0.4 0.8 1",
        )
        for face, site_suffix in FACE_TO_SITE.items():
            direction = {
                "+x": (1, 0, 0), "-x": (-1, 0, 0),
                "+y": (0, 1, 0), "-y": (0, -1, 0),
                "+z": (0, 0, 1), "-z": (0, 0, -1),
            }[face]
            ET.SubElement(
                body,
                "site",
                name=f"{block_id}_{site_suffix}",
                pos=_format_vector(component * BLOCK_EDGE / 2 for component in direction),
                size="0.005",
                rgba="0 0 1 0",
            )
        for site_suffix in FACE_TO_SITE.values():
            ET.SubElement(
                equality,
                "weld",
                name=f"ee2_{block_id}_{site_suffix}_weld",
                site1="ee2_grab",
                site2=f"{block_id}_{site_suffix}",
                active="false",
                solref="0.1 1",
                solimp="0.7 0.9 0.001",
            )

    positions = {}
    for block_id, coord in block_positions.items():
        normalized_coord = tuple(coord)
        if len(normalized_coord) != 3:
            raise ValueError(f"invalid coordinate for {block_id!r}: {coord!r}")
        if normalized_coord in positions:
            raise ValueError(f"multiple blocks occupy {normalized_coord}")
        positions[normalized_coord] = block_id
    for coord, block_id in positions.items():
        for face, direction in _DIRECTIONS.items():
            neighbor_coord = tuple(coord[i] + direction[i] for i in range(3))
            neighbor_id = positions.get(neighbor_coord)
            if neighbor_id is None:
                continue
            ET.SubElement(
                equality,
                "weld",
                name=f"{block_id}_{neighbor_id}_weld",
                site1=f"{block_id}_{FACE_TO_SITE[face]}",
                site2=f"{neighbor_id}_{FACE_TO_SITE[_OPPOSITE[face]]}",
                active="true",
                solref="0.1 1",
                solimp="0.7 0.9 0.001",
            )

    ET.indent(tree, space="  ")
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
