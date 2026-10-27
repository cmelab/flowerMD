"""Keep tetrahedral stereocenters through soft-core DPD relaxation.

DPD lets atoms pass through each other, so a stereocenter can invert and a
later minimizer will not undo it. `protect_stereochemistry` records every
potential tetrahedral center of a parameterized `System` and adds a
periodic-dihedral restraint over its four substituents; `audit_stereochemistry`
checks the handedness afterwards. Requires RDKit. `find_ring_spears`
finds bonds threaded through rings, which no minimizer undoes either.
"""

import warnings

import hoomd
import numpy as np

from flowermd.internal.all_atom_dpd import to_sanitized_rdkit
from flowermd.internal.local_boltzmann import crosses

PREFIX = "stereo_"


def _volumes(xyz, neighbors, box=None):
    """Signed substituent-tetrahedron volume divided by its edge lengths."""
    edges = xyz[neighbors[:, :3]] - xyz[neighbors[:, 3:]]
    if box is not None:  # wrapped HOOMD positions
        edges -= box * np.rint(edges / box)
    return np.linalg.det(edges) / np.prod(np.linalg.norm(edges, axis=2), 1)


def _dihedrals(p):
    """Dihedral of the four substituents in HOOMD's convention (radians)."""
    b1, b2, b3 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 1], p[:, 3] - p[:, 2]
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    y = np.linalg.norm(b2, axis=1) * np.einsum("ij,ij->i", n1, b3)
    return np.arctan2(y, np.einsum("ij,ij->i", n1, n2))


def protect_stereochemistry(system, k=125520.0, planar_tolerance=0.05):
    """Record stereocenters and add a restraint that keeps their handedness.

    Call after `System.apply_forcefield`. Each center's four substituents
    get a dihedral ``U = k/2 [1 - cos(phi - phi0)]`` with phi0 from the
    current geometry (rounded to 1 degree to bound the number of types).
    The restraint is its own `hoomd.md.dihedral.Periodic` force, appended
    to ``system.hoomd_forcefield``, so its energy is logged and checked for
    stationarity separately; the other periodic dihedral forces get k = 0
    for its types. Every 1-4 pair of the restraint is already a 1-3 pair,
    so neighbor-list exclusions do not change.

    Parameters
    ----------
    system : flowermd.base.System, required
        A parameterized all-atom system.
    k : float, default 125520.0
        Restraint strength in kJ/mol (30000 kcal/mol).
    planar_tolerance : float, default 0.05
        Normalized volume below which a reference center counts as planar.

    Returns
    -------
    dict
        ``{"centers", "neighbors", "sign"}`` arrays for
        `audit_stereochemistry`.

    """
    from rdkit import Chem

    mol = to_sanitized_rdkit(system.system)
    centers = [
        info.centeredOn
        for info in Chem.FindPotentialStereo(mol, flagPossible=True)
        if info.type == Chem.StereoType.Atom_Tetrahedral
    ]
    if not centers:
        # Cross-check: four neighbors of four different symmetry classes.
        ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
        if any(
            len({ranks[n.GetIdx()] for n in atom.GetNeighbors()}) == 4
            for atom in mol.GetAtoms()
        ):
            warnings.warn(
                "No stereocenters were found, but some atoms have four "
                "distinct substituents. Nothing is protected."
            )
    reference = {"centers": np.array(centers, dtype=int)}
    neighbors = [
        [n.GetIdx() for n in mol.GetAtomWithIdx(c).GetNeighbors()]
        for c in centers
    ]
    if any(len(n) != 4 for n in neighbors):
        raise ValueError("Stereocenters need four explicit neighbors.")
    reference["neighbors"] = np.array(neighbors, dtype=int).reshape(-1, 4)
    if not centers:
        return {**reference, "sign": np.empty(0)}
    snap = system.hoomd_snapshot
    xyz = np.asarray(snap.particles.position, dtype=float)
    box = np.asarray(snap.configuration.box[:3])
    volume = _volumes(xyz, reference["neighbors"], box)
    if np.any(np.abs(volume) < planar_tolerance):
        raise ValueError("A reference stereocenter is near-planar.")
    reference["sign"] = np.sign(volume)
    points = xyz[reference["neighbors"]]
    points -= box * np.rint((points - points[:, :1]) / box)
    phi0 = np.round(np.degrees(_dihedrals(points))).astype(int) % 360
    names = [f"{PREFIX}{p}" for p in phi0]
    types = list(snap.dihedrals.types or []) + sorted(set(names))
    snap.dihedrals.typeid = np.concatenate(
        [
            np.asarray(snap.dihedrals.typeid, dtype=np.uint32).reshape(-1),
            [types.index(n) for n in names],
        ]
    ).astype(np.uint32)
    snap.dihedrals.group = np.concatenate(
        [
            np.asarray(snap.dihedrals.group, dtype=np.uint32).reshape(-1, 4),
            reference["neighbors"],
        ]
    ).astype(np.uint32)
    snap.dihedrals.N, snap.dihedrals.types = len(snap.dihedrals.group), types
    restraint = hoomd.md.dihedral.Periodic()
    for force in system.hoomd_forcefield:
        if isinstance(force, hoomd.md.dihedral.Periodic):
            for name in set(names):  # each force must cover every type
                force.params[name] = dict(k=0.0, d=1, n=1, phi0=0.0)
    for name in types:
        if name.startswith(PREFIX):
            phi = np.radians(int(name[len(PREFIX) :]))
            restraint.params[name] = dict(k=k, d=-1, n=1, phi0=phi)
        else:
            restraint.params[name] = dict(k=0.0, d=1, n=1, phi0=0.0)
    system.hoomd_forcefield.append(restraint)
    return reference


def audit_stereochemistry(reference, snapshot, planar_tolerance=0.05):
    """Count inverted and near-planar centers in a HOOMD snapshot.

    Parameters
    ----------
    reference : dict, required
        The return value of `protect_stereochemistry`.
    snapshot : gsd.hoomd.Frame or hoomd.Snapshot, required
        Coordinates to check, wrapped or not.
    planar_tolerance : float, default 0.05
        Normalized volume below which a center counts as near-planar.

    Returns
    -------
    dict
        ``{"n_centers", "inverted", "near_planar"}`` counts.

    """
    xyz = np.asarray(snapshot.particles.position, dtype=float)
    box = np.asarray(snapshot.configuration.box[:3])
    volume = _volumes(xyz, reference["neighbors"], box)
    return {
        "n_centers": len(volume),
        "inverted": int(np.sum(volume * reference["sign"] < 0)),
        "near_planar": int(np.sum(np.abs(volume) < planar_tolerance)),
    }


def find_ring_spears(snapshot, max_ring=8):
    """Find bonds that pass through the face of a ring.

    Each ring of at most `max_ring` atoms (from a cycle basis of the bond
    graph) is split into triangles fanning out from its centroid, which
    also covers puckered rings, and every bond that does not touch the
    ring and whose midpoint is within 0.35 nm of the centroid is tested
    for crossing one of them. Requires networkx.

    Parameters
    ----------
    snapshot : gsd.hoomd.Frame or hoomd.Snapshot, required
        Coordinates and bonds to check, wrapped or not.
    max_ring : int, default 8
        Largest ring checked.

    Returns
    -------
    list of tuple
        ``((i, j), ring)`` for each bond i-j threaded through `ring`, a
        tuple of atom indices.

    """
    import networkx as nx
    from scipy.spatial import cKDTree

    xyz = np.asarray(snapshot.particles.position, dtype=float)
    box = np.asarray(snapshot.configuration.box[:3], dtype=float)
    bonds = np.asarray(snapshot.bonds.group, dtype=int).reshape(-1, 2)
    rings = [
        r
        for r in nx.cycle_basis(nx.Graph(bonds.tolist()))
        if len(r) <= max_ring
    ]
    if not rings:
        return []

    def image(d):
        return d - box * np.rint(d / box)

    corners = [xyz[r[0]] + image(xyz[r] - xyz[r[0]]) for r in rings]
    centers = np.array([c.mean(axis=0) for c in corners])
    ends = xyz[bonds[:, 0]] + 0.5 * image(xyz[bonds[:, 1]] - xyz[bonds[:, 0]])
    trees = [
        cKDTree(np.where(x % box < box, x % box, 0.0), boxsize=box)
        for x in (centers, ends)
    ]
    found = trees[0].query_ball_tree(trees[1], 0.35)
    out = []
    for r, c, center, near in zip(rings, corners, centers, found):
        near = [b for b in near if not set(bonds[b]) & set(r)]
        if not near:
            continue
        p = xyz[bonds[near, 0]]
        p += image(center - p) - (center - p)  # nearest image of the bond
        q = p + image(xyz[bonds[near, 1]] - xyz[bonds[near, 0]])
        hit = crosses(p[:, None], q[:, None], center, c, np.roll(c, -1, axis=0))
        out += [
            (tuple(map(int, bonds[near[b]])), tuple(map(int, r)))
            for b in np.flatnonzero(hit.any(1))
        ]
    return out
