"""Idealised geometry of the ship mock-up, and the mesh container.

Frame (basis project): x axial 0--1.8 m, y transverse, z 'up' with the hull
section centred on (y, z) = (0, 0).  Physically the mock-up stands inverted on
the free edges of its side plates (z = 0.39 m), which are therefore the
supported edges.  The section is a semicircular bilge of radius R joined
tangentially to vertical side plates up to z = ZTOP, with a flat deck at
z = ZDECK and three foundation frames whose top is z = ZFDN, where the shakers
act.

The registered mesh used by the manuscript is a preserved input,
``artifacts/preserved/geometry/mesh_geometry.npz``; this module reads it into a
Mesh and projects photogrammetric transducer positions onto the idealised
surfaces.
"""
from dataclasses import dataclass, field

import numpy as np

R = 0.45
ZTOP = 0.39
ZDECK = -0.25
ZFDN = -0.203
YDECK = float(np.sqrt(R**2-ZDECK**2))

SUBSTRUCTURES = ('bilge', 'side', 'deck', 'joint', 'bulkhead', 'foundation',
                 'frame', 'web', 'stiffener', 'cover')


@dataclass
class Mesh:
    nodes: np.ndarray            # (N, 3)
    quads: np.ndarray            # (E, 4) node indices
    sub: np.ndarray              # (E,) substructure index into SUBSTRUCTURES
    support_nodes: np.ndarray    # nodes on the supported free edges z = ZTOP
    support_len: np.ndarray      # tributary edge length of each support node (m)
    points: dict = field(default_factory=dict)  # name -> node index (sensors, drives)
    interface_pairs: np.ndarray = field(default_factory=lambda: np.empty((0, 2), int))
    interface_len: np.ndarray = field(default_factory=lambda: np.empty(0))
    config: dict = field(default_factory=dict)


def snap(p) -> tuple[np.ndarray, np.ndarray]:
    """Closest point on the idealised surfaces and the outward unit normal there.

    Sensor coordinates estimated from photographs (A1-A8, about +-5 cm) do not lie
    exactly on the shell; they are projected.  Normals: hull outward (away from the
    section axis), deck and foundation +z.
    """
    x, y, z = map(float, p)
    cands = []
    # side plates
    for s in (-1, 1):
        zz = min(max(z, 0.0), ZTOP)
        cands.append((np.array([x, s * R, zz]), np.array([0, s, 0.0])))
    # bilge
    if z < 0 or abs(y) < R:
        ang = np.arctan2(y, -z) if (y, z) != (0, 0) else 0.0
        ang = np.clip(ang, -np.pi / 2, np.pi / 2)
        q = np.array([x, R * np.sin(ang), -R * np.cos(ang)])
        cands.append((q, np.array([0, np.sin(ang), -np.cos(ang)])))
    # deck and foundation top
    cands.append((np.array([x, np.clip(y, -YDECK, YDECK), ZDECK]), np.array([0, 0, 1.0])))
    if abs(z - ZFDN) < 0.01:
        cands.append((np.array([x, y, ZFDN]), np.array([0, 0, 1.0])))
    d = [np.linalg.norm(c[0] - np.array([x, y, z])) for c in cands]
    q, n = cands[int(np.argmin(d))]
    return q, n
