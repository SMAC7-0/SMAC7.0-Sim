import xml.etree.ElementTree as ET
from pathlib import Path

from inchworm_sim.physics import coords

# will likely add a blueprint or path later
def build_scene(block_positions_i, block_connections_i, scene_file, out_path):
    B_SIZE = coords.BLOCK_SIZE
    HALF = B_SIZE / 2
    B_FACES = {
        "px": ( HALF, 0, 0),
        "nx": (-HALF, 0, 0),
        "py": (0,  HALF, 0),
        "ny": (0, -HALF, 0),
        "pz": (0, 0,  HALF),
        "nz": (0, 0, -HALF),
    }

    # --------------------------------------------------
    # Load the base scene
    # --------------------------------------------------

    tree = ET.parse(scene_file)
    root = tree.getroot()

    worldbody = root.find("worldbody")

    if worldbody is None:
        raise ValueError("scene.xml has no worldbody element")

    # Ensure equality section exists
    equality = root.find("equality")

    if equality is None:
        equality = ET.SubElement(root, "equality")


    # --------------------------------------------------
    # Generate blocks
    # --------------------------------------------------

    for id, (x, y, z) in block_positions_i.items():

        pos = coords.grid_to_world(x, y, z)
        body = ET.SubElement(
            worldbody,
            "body",
            name=f"block{id}",
            pos=f"{pos[0]} {pos[1]} {pos[2]}",
        )

        ET.SubElement(
            body,
            "freejoint",
            name=f"block{id}_free",
        )

        ET.SubElement(
            body,
            "geom",
            name=f"block{id}_geom",
            type="box",
            size=f"{B_SIZE/2} {B_SIZE/2} {B_SIZE/2}",
            mass="1",
            rgba="0.2 0.4 0.8 1",
            contype="1",
            conaffinity="1",
        )

        # Site for connections (w/ block and inchworm)
        for face, face_pos in B_FACES.items():
            ET.SubElement(
                body,
                "site",
                name=f"block{id}_{face}",
                pos=" ".join(map(str, face_pos)),
                size="0.005",
                rgba="0 0 1 0",
            )


    # --------------------------------------------------
    # Inchworm-to-block welds
    # All start inactive.
    # --------------------------------------------------

    for id in block_positions_i:
        for face in B_FACES:

            ET.SubElement(
                equality,
                "weld",
                name=f"ee2_block{id}_{face}_weld",
                site1="ee2_grab",
                site2=f"block{id}_{face}",
                active="false",
                solref="0.1 1",
                solimp="0.7 0.9 0.001",
            )


    # --------------------------------------------------
    # Block-to-block welds
    # These start active.
    # --------------------------------------------------

    for a, b, face_a, face_b in block_connections_i:

        if a not in block_positions_i or b not in block_positions_i:
            raise ValueError(
                f"Invalid connection: block {a} or {b} does not exist"
            )

        ET.SubElement(
            equality,
            "weld",
            name=f"block{a}_block{b}_weld",
            site1=f"block{a}_{face_a}",
            site2=f"block{b}_{face_b}",
            active="true",
            solref="0.1 1",
            solimp="0.7 0.9 0.001",
        )


    # --------------------------------------------------
    # Write generated XML
    # --------------------------------------------------

    ET.indent(tree, space="  ")

    tree.write(
        out_path,
        encoding="utf-8",
        xml_declaration=True,
    )

    print(f"Generated model: {out_path}")
    print(f"Blocks: {len(block_positions_i)}")
    print(f"Block connections: {len(block_connections_i)}")
