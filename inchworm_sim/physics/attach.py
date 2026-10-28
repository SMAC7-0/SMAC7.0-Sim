
import re
import numpy as np
import mujoco

FACES = ("px", "nx", "py", "ny", "pz", "nz")
BB = re.compile(r"^block(\d+)_block(\d+)_weld$")


def set_weld(model, data, name, active: bool):
    data.eq_active[model.equality(name).id] = int(active) 

"""List of block-to-block welds for a given block."""
def block_welds(model, block_id):
    out = []
    for i in range(model.neq):
        name = model.equality(i).name
        m = BB.match(name)
        if m and block_id in map(int, m.groups()):
            out.append(name)
    return out

def nearest_face(model, data, ee_site, block_ids, max_dist=0.01):
    ee = data.site_xpos[model.site(ee_site).id]
    best, best_d = None, max_dist
    for bid in block_ids:
        for face in FACES:
            d = np.linalg.norm(data.site_xpos[model.site(f"block{bid}_{face}").id] - ee)
            if d < best_d:
                best, best_d = (bid, face), d
    return best

def grab(model, data, block_ids, ee_site="ee2_grab"):
    hit = nearest_face(model, data, ee_site, block_ids)
    if hit is None:
        return None
    bid, face = hit
    for w in block_welds(model, bid):        # pull it out of the structure
        set_weld(model, data, w, False)
    set_weld(model, data, f"ee2_block{bid}_{face}_weld", True)
    return hit

def release(model, data, bid, face):
    set_weld(model, data, f"ee2_block{bid}_{face}_weld", False)