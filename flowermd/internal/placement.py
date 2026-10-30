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
angle is a dead end: the sampler is then asked for its least bad angle
with ``choose(..., least_bad=True)``, so each junction is visited once and
a chain grows in time linear in its length. Only if that is None too does
the chain start again, at most `max_restarts` times.
`UniformTorsion`
is the simplest sampler; `flowermd.internal.local_boltzmann` has the
force-field-weighted one.
"""

import math
from collections import Counter, namedtuple

import numpy as np
from scipy.spatial.transform import Rotation

from flowermd.internal.geometry import apply_pose, compose, turn_about_bond

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

    def choose(self, index, placed, propose, rng, least_bad=False):
        """A uniform random angle in [0, 2 pi)."""
        return rng.uniform(0.0, 2 * math.pi)


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
        f"between atoms {a} and {b}. Relax the sampler's constraints."
    )


def _grow(xyz, units, junctions, start, rng, choose):
    """One growth attempt: (coordinates, None), or (None, stopping bond)."""
    # A pose (R, t) carries a unit from the template into the box,
    # x -> R x + t. The first unit gets a random orientation with its
    # centroid at `start`: x -> R (x - centroid) + start.
    spin = Rotation.random(rng=rng).as_matrix()
    pose = {0: (spin, start - spin @ xyz[units[0]].mean(axis=0))}
    first = [
        *units[0],
        *(b for j in junctions if j.parent == 0 for _, b in j.bonds),
    ]
    placed = np.full_like(xyz, np.nan)
    placed[first] = apply_pose(pose[0], xyz[first])
    for i, j in enumerate(junctions):
        if j.torsion:
            a, b = j.bonds[0]
            # Rodrigues' formula: split each moved atom, measured from a,
            # into the part along the bond and the arm perpendicular to it;
            # the arm turned 90 degrees about the bond is k x y.
            k = (xyz[b] - xyz[a]) / np.linalg.norm(xyz[b] - xyz[a])
            y = xyz[j.moved] - xyz[a]
            along = np.outer(y @ k, k)
            parts = (y - along, np.cross(k, y), along)

            def propose(
                angles, parts=parts, origin=xyz[a], frame=pose[j.parent]
            ):
                """Box coordinates of the moved atoms, one set per angle."""
                cos = np.cos(angles)[:, None, None]
                sin = np.sin(angles)[:, None, None]
                # turned in the template, then carried into the box the
                # way the parent was carried
                local = cos * parts[0] + sin * parts[1] + parts[2] + origin
                return apply_pose(frame, local)

            angle = choose(i, placed, propose, rng)
            if angle is None:  # dead end: take the least bad angle
                angle = choose(i, placed, propose, rng, least_bad=True)
            if angle is None:
                return None, (a, b)
            relative = turn_about_bond(xyz[a], xyz[b], angle)
        else:  # ring (ladder) junction: the child keeps its built pose
            relative = (np.eye(3), np.zeros(3))
        # turn the child in the template, then carry it as its parent was
        pose[j.child] = compose(pose[j.parent], relative)
        placed[j.moved] = apply_pose(pose[j.child], xyz[j.moved])
    return placed, None
