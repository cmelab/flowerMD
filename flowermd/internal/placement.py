"""Random conformations of all-atom chains that keep their built chemistry.

A chain is cut into rigid units and grown unit by unit from the first.
Each unit is placed by its parent's pose composed with a relative pose:
across a one-bond junction, the template pose turned about the junction
bond by a torsion that a torsion sampler chooses; across a ring (ladder)
junction, the template pose. Bond lengths, bond angles and stereocenters
of the template are kept.

A torsion sampler provides ``template(chain) -> (xyz, units)``, the
coordinates and rigid units to grow, ``chooser(chain, junctions)``, which
returns ``choose(index, placed, propose, rng) -> angle``, and
``max_restarts``. `choose` is called once per free junction in growth
order: `placed` holds the coordinates fixed so far (NaN for the rest),
and ``propose(angles)`` returns the coordinates of the junction's `moved`
atoms for an array of candidate angles. Returning None instead of an
angle restarts the chain, at most `max_restarts` times. `UniformTorsion`
is the simplest sampler; `flowermd.internal.local_boltzmann` has the
force-field-weighted one.
"""

import math
from collections import Counter, namedtuple

import numpy as np
from scipy.spatial.transform import Rotation

Junction = namedtuple(
    "Junction", ["parent", "child", "bonds", "torsion", "moved"]
)
Junction.__doc__ = """Bonds joining a parent unit to a child unit of a chain.

``bonds`` holds (parent atom, child atom) pairs; ``torsion`` is True for a
single bond outside any ring, whose torsion is free. ``moved`` holds the
atoms placed with the child: the child unit and the child-side atoms of its
own junctions, which lie on those junctions' axes.
"""


def repeat_units(chain):
    """Return the chain's rigid units as lists of particle indices.

    Indices refer to ``list(chain.particles())``. Each multi-atom child of
    the chain (a repeat unit built by mBuild's `Polymer` recipe) is one unit.
    A single-atom child (an end cap, or every atom of a molecule loaded
    straight from SMILES) is merged into the unit of an atom it is bonded
    to, so a small molecule is one rigid unit and a cap moves with its
    repeat. A chain without children is one unit.
    """
    particles = list(chain.particles())
    index = {p: i for i, p in enumerate(particles)}
    children = [c for c in chain.children if c.n_particles]
    units = [
        [index[p] for p in c.particles()] for c in children if c.n_particles > 1
    ]
    if not units:
        return [list(range(len(particles)))]
    owner = {}
    for unit_id, members in enumerate(units):
        for i in members:
            owner[i] = unit_id
    bonded = {i: [] for i in range(len(particles))}
    for a, b in chain.bonds():
        bonded[index[a]].append(index[b])
        bonded[index[b]].append(index[a])
    loose = [
        index[p] for c in children if c.n_particles == 1 for p in c.particles()
    ]
    for i in loose:
        home = next((owner[j] for j in bonded[i] if j in owner), None)
        if home is None:
            units.append([i])
            home = len(units) - 1
        else:
            units[home].append(i)
        owner[i] = home
    return units


def _in_ring(adjacency, near, far):
    """True if `near` is reached from `far` without crossing their bond."""
    seen, stack = {far}, [far]
    while stack:
        i = stack.pop()
        for j in adjacency[i]:
            if i == far and j == near:
                continue
            if j == near:
                return True
            if j not in seen:
                seen.add(j)
                stack.append(j)
    return False


def chain_junctions(units, bonds, n_atoms):
    """Junctions of a chain of rigid units, in growth order from unit 0.

    The units must form a tree (a linear or branched chain); two units may
    share several bonds (a ladder junction). Returns a list of `Junction`.
    """
    owner = np.full(n_atoms, -1, dtype=int)
    for unit_id, members in enumerate(units):
        owner[members] = unit_id
    adjacency = [[] for _ in range(n_atoms)]
    between = {}
    for a, b in bonds:
        adjacency[a].append(b)
        adjacency[b].append(a)
        if owner[a] != owner[b]:
            between.setdefault(frozenset((owner[a], owner[b])), []).append(
                (a, b)
            )
    neighbors = {u: [] for u in range(len(units))}
    for pair in between:
        x, y = tuple(pair)
        neighbors[x].append(y)
        neighbors[y].append(x)
    order, parent = [0], {0: None}
    for unit in order:
        for other in sorted(neighbors[unit]):
            if other in parent:
                if parent[unit] != other:
                    raise ValueError(
                        "The repeat units of a chain form a ring; only linear "
                        "and branched chains can be walked."
                    )
                continue
            parent[other] = unit
            order.append(other)
    if len(order) != len(units):
        raise ValueError("A chain's repeat units are not all bonded together.")
    pairs = {
        unit: [
            (a, b) if owner[a] == parent[unit] else (b, a)
            for a, b in between[frozenset((parent[unit], unit))]
        ]
        for unit in order[1:]
    }
    ahead = {unit: [] for unit in order}
    for unit in order[1:]:
        ahead[parent[unit]].extend(b for _, b in pairs[unit])
    return [
        Junction(
            parent[unit],
            unit,
            pairs[unit],
            len(pairs[unit]) == 1 and not _in_ring(adjacency, *pairs[unit][0]),
            np.array(sorted([*units[unit], *ahead[unit]]), dtype=int),
        )
        for unit in order[1:]
    ]


class UniformTorsion:
    """Torsion sampler: a uniform random turn at each repeat-unit junction.

    Only the bonds between repeat units turn, and the torsions inside a
    repeat unit stay as built, so chain dimensions depend on how the repeat
    unit is drawn; prefer
    `flowermd.internal.local_boltzmann.LocalBoltzmannSampler`. It needs no
    force field. See this module's docstring for the sampler interface.
    """

    max_restarts = 0

    def template(self, chain):
        """Built coordinates and repeat units of `chain`."""
        return np.asarray(chain.xyz, dtype=float), repeat_units(chain)

    def chooser(self, chain, junctions):
        """Return the per-junction choice function for `chain`."""
        return self.choose

    def choose(self, index, placed, propose, rng):
        """A uniform random angle in [0, 2 pi)."""
        return rng.uniform(0.0, 2 * math.pi)


def _about(xyz_a, xyz_b, angles):
    """4x4 rotations by `angles` about the line from `xyz_a` to `xyz_b`."""
    k = (xyz_b - xyz_a) / np.linalg.norm(xyz_b - xyz_a)
    angles = np.atleast_1d(np.asarray(angles, dtype=float))
    cos, sin = np.cos(angles)[:, None, None], np.sin(angles)[:, None, None]
    cross = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    matrix = np.zeros((len(angles), 4, 4))
    matrix[:, 3, 3] = 1.0
    # Rodrigues' rotation formula
    matrix[:, :3, :3] = (
        cos * np.eye(3) + sin * cross + (1 - cos) * np.outer(k, k)
    )
    matrix[:, :3, 3] = xyz_a - matrix[:, :3, :3] @ xyz_a
    return matrix


def _apply(matrix, xyz):
    return xyz @ matrix[..., :3, :3].swapaxes(-1, -2) + matrix[..., None, :3, 3]


def random_chain_conformation(chain, start, rng, torsion_sampler=None):
    """Random conformation of an all-atom chain by rigid-unit placement.

    The first unit gets a uniformly random orientation with its centroid at
    `start`; each later unit is placed from its parent as described in this
    module's docstring.

    Parameters
    ----------
    chain : mbuild.Compound, required
        Built chain whose children are its repeat units.
    start : numpy.ndarray, shape (3,), required
        Position of the first unit's centroid (nm).
    rng : numpy.random.Generator, required
    torsion_sampler : object, optional
        Chooses the rigid units and the torsion of each free junction; see
        this module's docstring. Defaults to `UniformTorsion`.

    Returns
    -------
    numpy.ndarray, shape (N, 3)
        New unwrapped coordinates in ``chain.particles()`` order.

    """
    sampler = torsion_sampler or UniformTorsion()
    xyz, units = sampler.template(chain)
    index = {p: i for i, p in enumerate(chain.particles())}
    bonds = [(index[a], index[b]) for a, b in chain.bonds()]
    junctions = chain_junctions(units, bonds, len(xyz))
    choose = sampler.chooser(chain, junctions)
    stops = Counter()
    for _ in range(sampler.max_restarts + 1):
        placed, stop = _grow(xyz, units, junctions, start, rng, choose)
        if stop is None:
            return placed
        stops[stop] += 1
    (a, b), count = stops.most_common(1)[0]
    raise RuntimeError(
        f"No conformation of a {len(xyz)}-atom chain was found in "
        f"{sum(stops.values())} attempts; {count} stopped at the bond "
        f"between atoms {a} and {b}. Relax the sampler's constraints (for "
        "LocalBoltzmannSampler, a smaller r_min)."
    )


def _grow(xyz, units, junctions, start, rng, choose):
    """One growth attempt: (coordinates, None), or (None, stopping bond)."""
    pose = {0: np.eye(4)}
    pose[0][:3, :3] = Rotation.random(random_state=rng).as_matrix()
    pose[0][:3, 3] = start - pose[0][:3, :3] @ xyz[units[0]].mean(axis=0)
    first = [
        *units[0],
        *(b for j in junctions if j.parent == 0 for _, b in j.bonds),
    ]
    placed = np.full_like(xyz, np.nan)
    placed[first] = _apply(pose[0], xyz[first])
    for i, j in enumerate(junctions):
        relative = np.eye(4)
        if j.torsion:
            a, b = j.bonds[0]
            # Rodrigues' formula on the moved atoms, then the parent's pose.
            k = (xyz[b] - xyz[a]) / np.linalg.norm(xyz[b] - xyz[a])
            y = xyz[j.moved] - xyz[a]
            along = np.outer(y @ k, k)
            parts = (y - along, np.cross(k, y), along)

            def propose(
                angles, parts=parts, origin=xyz[a], frame=pose[j.parent]
            ):
                cos = np.cos(angles)[:, None, None]
                sin = np.sin(angles)[:, None, None]
                local = cos * parts[0] + sin * parts[1] + parts[2] + origin
                return _apply(frame, local)

            angle = choose(i, placed, propose, rng)
            if angle is None:
                return None, (a, b)
            relative = _about(xyz[a], xyz[b], angle)[0]
        pose[j.child] = pose[j.parent] @ relative
        placed[j.moved] = _apply(pose[j.child], xyz[j.moved])
    return placed, None
