"""Canonical block face names shared by the logical grid and MuJoCo models.

Face names double as MJCF site suffixes (``block5_px``, ``ee2_grab_px``), so
they must stay valid identifier fragments.
"""

from typing import Dict, Literal, Tuple

FaceName = Literal["px", "nx", "py", "ny", "pz", "nz"]

# Outward unit normal of each face in the z-up grid frame.
FACES: Dict[str, Tuple[int, int, int]] = {
    "px": (1, 0, 0),
    "nx": (-1, 0, 0),
    "py": (0, 1, 0),
    "ny": (0, -1, 0),
    "pz": (0, 0, 1),
    "nz": (0, 0, -1),
}

OPPOSITE_FACE: Dict[str, str] = {
    "px": "nx",
    "nx": "px",
    "py": "ny",
    "ny": "py",
    "pz": "nz",
    "nz": "pz",
}

HORIZONTAL_FACES: Tuple[str, ...] = ("px", "nx", "py", "ny")
