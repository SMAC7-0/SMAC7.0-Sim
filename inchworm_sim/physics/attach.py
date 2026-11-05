"""
Provides utility functions for managing block attachments in a MuJoCo
simulation. Supports detecting the nearest block face, dynamically
grabbing and releasing blocks using weld constraints, and managing
existing block-to-block connections.
Covers both block-to-block and end-effector-to-block welds, allowing for flexible
manipulation of blocks in the simulation environment.
"""
import re
import numpy as np
import mujoco

FACES = ("px", "nx", "py", "ny", "pz", "nz")
BB = re.compile(r"^block(\d+)_block(\d+)_weld$")

def set_weld(model, data, name, active: bool):
    data.eq_active[model.equality(name).id] = int(active) 

"""List of block-to-block welds for a given block."""
def block_welds(model, block_id, only_active=False, data=None):
    out = []
    for i in range(model.neq):
        name = model.equality(i).name
        if name.startswith("ee2"):
            continue
        ids = [int(x) for x in re.findall(r"block(\d+)", name)]
        if block_id in ids and (not only_active or data.eq_active[i]):
            out.append(name)
    return out

"""Finds the nearest block face to the end-effector site within a specified distance threshold."""
def nearest_face(model, data, ee_site, block_ids, max_dist=0.02):
    ee = data.site_xpos[model.site(ee_site).id]
    best, best_d = None, max_dist
    for bid in block_ids:
        for face in FACES:
            d = np.linalg.norm(data.site_xpos[model.site(f"block{bid}_{face}").id] - ee)
            if d < best_d:
                best, best_d = (bid, face), d
    return best

# def grab(model, data, block_ids, ee_site="ee2_grab"):
#     hit = nearest_face(model, data, ee_site, block_ids)

#     if hit is None:
#         return None
    
#     bid, face = hit
#     weld_name = f"ee2_block{bid}_{face}_weld"
#     idx = model.equality(weld_name).id

#     for w in block_welds(model, bid):        # pull it out of the structure
#         set_weld(model, data, w, False)

#     return hit
# def release(model, data, bid, face):
#     set_weld(model, data, f"ee2_block{bid}_{face}_weld", False)

"""Inchworm functions"""

"""Grab block - employs body-based welds and nearest face detection to identify the closest block face to the end-effector."""
def grab(model, data, block_ids, ee_site="ee2_grab"):
    hit = nearest_face(model, data, ee_site, block_ids)

    if hit is None:
        return None

    bid, face = hit

    # Disconnect block from its existing structure.
    for w in block_welds(model, bid):
        set_weld(model, data, w, False)

    # Identify the two bodies.
    b1 = model.body("ee2").id
    b2 = model.body(f"block{bid}").id

    idx = model.equality(f"ee2_block{bid}_weld").id

    # Current body poses in world coordinates.
    pos1 = data.xpos[b1].copy()
    pos2 = data.xpos[b2].copy()

    quat1 = data.xquat[b1].copy()
    quat2 = data.xquat[b2].copy()

    # Calculate body2 pose relative to body1.
    rot1 = data.xmat[b1].reshape(3, 3)

    rel_pos = rot1.T @ (pos2 - pos1)

    quat1_inv = np.zeros(4)
    rel_quat = np.zeros(4)

    mujoco.mju_negQuat(quat1_inv, quat1)
    mujoco.mju_mulQuat(rel_quat, quat1_inv, quat2)

    # Configure the body-based weld.
    model.eq_data[idx, 0:3] = 0.0
    model.eq_data[idx, 3:6] = rel_pos
    model.eq_data[idx, 6:10] = rel_quat

    print("EE site:", pos1)
    print("Block face:", pos2)
    print("Initial gap:", np.linalg.norm(pos2 - pos1))

    # Activate only after configuring the relative pose.
    data.eq_active[idx] = 1

    return hit

def release(model, data, bid, face=None):
    set_weld(model, data, f"ee2_block{bid}_weld", False)

"""Block-to-block welds - manages the connection between two blocks, allowing for dynamic attachment and detachment."""
