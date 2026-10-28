"""Torsion sampler with local Boltzmann weights from an all-atom force field.

Each rotatable bond of a chain is turned, in growth order, to an angle drawn
from ``p(phi) ~ exp(-E(phi) / kT)``, where `E` holds only the terms that the
turn changes locally: the force field's proper torsions about the bond and
the Lennard-Jones energy between the atoms it moves and the atoms already
placed at most `max_sep` bonds away. This is a rotational-isomeric-state
style model of an unperturbed chain, which needs no knowledge of the other
chains; it gives realistic chain dimensions for backbones without bulky
side groups, and too compact ones for crowded chains (see
`LocalBoltzmannSampler`). The parameters are
the unscaled terms of a SMIRNOFF force field (those that
`flowermd.library.OpenFF_DPD` writes) or of a FlowerMD XML force field
such as `flowermd.library.OPLS_AA`.
"""

import functools
import os
import tempfile
from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist

from flowermd.internal.all_atom_dpd import (
    gmso_terms,
    openff_terms,
    to_sanitized_rdkit,
)
from flowermd.internal.geometry import crosses, dihedrals, signed_volumes

KB = 0.0083144626  # kJ/(mol K)
CLASH_WINDOW = 12  # bonds; see LocalBoltzmannSampler
BACKTRACK = 4  # junctions


@dataclass
class Step:
    """What `_choose` needs at one rotatable bond, worked out once by `_plan`.

    A "row" indexes `moved`, the atoms the turn moves (the second axis of
    the candidate coordinates that ``propose`` returns); an "atom" indexes
    the whole chain.
    """

    a: int  # the turned bond a-b; a is already placed
    b: int
    moved: np.ndarray  # atoms
    # proper torsions i-a-b-l about the bond, one entry per Fourier term
    torsion_i: np.ndarray  # placed atom i
    torsion_l: np.ndarray  # row of the moved atom l
    torsion_k: np.ndarray  # kJ/mol
    torsion_n: np.ndarray  # periodicity
    torsion_phase: np.ndarray  # radians
    # Lennard-Jones pairs (moved atom, placed atom) at most max_sep bonds apart
    pair_moved: np.ndarray  # rows
    pair_placed: np.ndarray  # atoms
    pair_sigma2: np.ndarray  # nm**2
    pair_eps: np.ndarray  # kJ/mol, 1-4 pairs already scaled
    # contact check: moved heavy atoms against nearby placed heavy atoms,
    # skipping the (heavy_moved index, nearby index) pairs within 3 bonds
    heavy_moved: np.ndarray  # rows
    nearby: np.ndarray  # atoms
    skip: tuple
    # ring-piercing check: the bonds and ring triangles of the moved atoms
    moved_bonds: np.ndarray  # indices into terms["bond_array"]
    moved_triangles: np.ndarray  # indices into terms["triangles"]


class LocalBoltzmannSampler:
    """Torsion sampler: local Boltzmann weights from an all-atom force field.

    For `flowermd.library.AtomisticRandomWalk`. Every rotatable bond turns:
    single bonds outside rings with a heavy atom beyond each end, except the
    C(=O)-O and C(=O)-N bonds of esters and amides and bonds to linear (sp)
    atoms. Ring bonds, and so ladder junctions, keep their geometry.

    Each torsion is drawn from a grid of `n_grid` cells by the Boltzmann
    weights of the force field's proper torsions about the bond plus the
    Lennard-Jones energy between the moved atoms and the placed atoms at
    most `max_sep` bonds away (no electrostatics), and placed uniformly
    within its cell. A cell is left out if it brings a moved heavy atom
    closer than `r_min` to a placed heavy atom 4 to `clash_window` bonds
    away, or pushes a bond through a ring of the chain. With no cell left
    the growth backs up `backtrack` rotatable bonds; after four dead ends at
    the same bond, that bond takes the least bad cell instead, and its
    closest contact is recorded in `accepted`. Overlaps
    further along the chain, and between chains, are left to the DPD
    relaxation.

    Parameters
    ----------
    temperature : float, default 413.0
        Temperature (K) of the Boltzmann weights; use the melt temperature.
    force_field : str or BaseXMLForcefield, default "openff-2.3.0.offxml"
        A SMIRNOFF force field (needs openff-toolkit and openff-interchange)
        or a FlowerMD XML force field such as
        ``flowermd.library.OPLS_AA()``; ``"oplsaa"`` is short for the latter.
    n_grid : int, default 72
        Grid cells per torsion.
    max_sep : int, default 4
        Largest bond separation of the Lennard-Jones pairs counted.
    r_min : float or None, default 0.30
        Closest allowed heavy-atom distance (nm); None for no check.
    max_restarts : int, default 500
        Restarts per chain before giving up, with ``on_dead_end="error"``.
    relax : bool, default True
        Minimize each chain under the force field's bonded terms before
        sampling (`relax_bonded`), so that the placement uses the force
        field's geometry rather than the built one.
    clash_window : int or None, default 12
        Largest bond separation of the heavy-atom pairs that `r_min` checks.
        None checks the whole chain, which makes each chain a self-avoiding
        walk that swells with length, unlike a chain in a melt.
    backtrack : int, default 4
        Rotatable bonds to back up at a dead end, times the number of dead
        ends since the growth last got past the furthest one.
    on_dead_end : str, default "least_bad"
        ``"least_bad"``, or ``"error"`` to restart the chain instead.

    Attributes
    ----------
    accepted : list of list of float
        For each chain, the closest contacts (nm) of its least-bad torsions.

    Notes
    -----
    The chain dimensions follow the local energy: close to experiment for
    polyethylene and polycarbonate, too compact for chains with bulky side
    groups such as polystyrene, and rough for crowded chains such as PMMA,
    which are placed largely with least-bad torsions. The energy needs a
    force field with realistic torsion and 1-4 terms; Sage 2.3.0, the
    default, is the validated one.

    """

    def __init__(
        self,
        temperature=413.0,
        force_field="openff-2.3.0.offxml",
        n_grid=72,
        max_sep=4,
        r_min=0.30,
        max_restarts=500,
        relax=True,
        clash_window=CLASH_WINDOW,
        backtrack=BACKTRACK,
        on_dead_end="least_bad",
    ):
        if isinstance(force_field, str) and force_field == "oplsaa":
            from flowermd.library.forcefields import OPLS_AA

            force_field = OPLS_AA()
        if isinstance(force_field, str):
            try:
                import openff.interchange  # noqa: F401
                import openff.toolkit  # noqa: F401
            except ImportError as error:
                raise ImportError(
                    "A SMIRNOFF force field needs openff-toolkit and "
                    "openff-interchange (conda-forge). Without them, use "
                    "LocalBoltzmannSampler(force_field=OPLS_AA()), with "
                    "OPLS_AA from flowermd.library."
                ) from error
        self.kT = KB * temperature
        self.force_field = force_field
        self.n_grid = n_grid
        self.max_sep = max_sep
        self.r_min = r_min or 0.0
        self.max_restarts = max_restarts
        self.relax = relax
        self.clash_window = clash_window
        self.backtrack = backtrack
        if on_dead_end not in ("least_bad", "error"):
            raise ValueError("on_dead_end must be 'least_bad' or 'error'.")
        self.on_dead_end = on_dead_end
        self.accepted = []  # per chain: closest distances accepted (nm)
        self._tables, self._templates, self._steps = {}, {}, {}

    def _topology(self, chain):
        """Force-field tables and rigid units of `chain`, once per topology."""
        index = {p: i for i, p in enumerate(chain.particles())}
        key = (
            tuple(p.element.symbol for p in chain.particles()),
            tuple(
                sorted(
                    tuple(sorted((index[a], index[b])))
                    for a, b in chain.bonds()
                )
            ),
        )
        if key not in self._tables:
            mol = to_sanitized_rdkit(chain)
            if isinstance(self.force_field, str):
                terms = openff_terms(mol, self.force_field)
            else:  # a BaseXMLForcefield or a GMSO ForceField
                ff = getattr(self.force_field, "gmso_ff", self.force_field)
                terms = gmso_terms(chain, ff)
            terms["heavy"] = np.array(
                [a.GetAtomicNum() > 1 for a in mol.GetAtoms()]
            )
            terms["units"] = _rigid_units(mol)
            terms["bond_array"] = np.array(list(terms["bonds"]), dtype=int)
            rings = [r for r in mol.GetRingInfo().AtomRings() if len(r) <= 8]
            terms["rings"] = np.full((len(rings), 8), -1)
            for i, r in enumerate(rings):
                terms["rings"][i, : len(r)] = r
            # fan triangles (ring, atom, next atom) about each ring's center
            terms["triangles"] = np.array(
                [
                    (i, a, r[k - 1])
                    for i, r in enumerate(rings)
                    for k, a in enumerate(r)
                ],
                dtype=int,
            ).reshape(-1, 3)
            self._tables[key] = terms
        return key, self._tables[key]

    def template(self, chain):
        """Relaxed coordinates and rigid units of `chain`."""
        key, terms = self._topology(chain)
        xyz = np.asarray(chain.xyz, dtype=float)
        if self.relax:
            cached = (key, xyz.round(6).tobytes())
            if cached not in self._templates:
                self._templates[cached] = relax_bonded(chain, self.force_field)
            xyz = self._templates[cached]
        return xyz, terms["units"]

    def chooser(self, chain, junctions):
        """Return the per-junction choice function for `chain`."""
        key, terms = self._topology(chain)
        if key not in self._steps:
            self._steps[key] = _plan(
                terms, junctions, self.max_sep, self.clash_window
            )
        self.accepted.append([])
        return functools.partial(self._choose, self._steps[key], terms)

    def _choose(
        self, steps, terms, index, placed, propose, rng, least_bad=False
    ):
        """Torsion angle (radians) for junction `index`, or None (dead end)."""
        step = steps[index]
        width = 2 * np.pi / self.n_grid
        cells = propose((np.arange(self.n_grid) + 0.5) * width)  # centers
        energy = _torsion_energy(step, placed, cells)
        energy += _lj_energy(step, placed, cells)
        energy = np.nan_to_num(energy, nan=1e300, posinf=1e300)
        closest = _closest(step, placed, cells)
        allowed = closest >= self.r_min  # the ring check is slower: only here
        allowed[allowed] = ~_pierces_ring(step, terms, placed, cells[allowed])
        if not allowed.any():
            if not least_bad:
                return None
            pierces = _pierces_ring(step, terms, placed, cells)
            if pierces.all():
                return None
            # least bad: the cells within 0.01 nm of the farthest contact
            allowed = ~pierces & (closest >= closest[~pierces].max() - 0.01)
            cell = self._draw(energy, allowed, rng)
            self.accepted[-1].append(closest[cell])
            return (cell + 0.5) * width
        while allowed.any():  # draw a cell, then check the angle within it
            cell = self._draw(energy, allowed, rng)
            angle = (cell + rng.uniform()) * width
            x = propose(np.array([angle]))
            if (
                _closest(step, placed, x)[0] >= self.r_min
                and not (_pierces_ring(step, terms, placed, x)[0])
            ):
                return angle
            allowed[cell] = False
        return None

    def _draw(self, energy, allowed, rng):
        """A grid cell drawn by its Boltzmann weight among the `allowed`."""
        weight = np.exp(-(energy - energy[allowed].min()) / self.kT)
        weight = np.where(allowed, weight, 0.0)
        return rng.choice(self.n_grid, p=weight / weight.sum())


def _torsion_energy(step, placed, x):
    """Torsion energy about the turned bond of each candidate `x` (kJ/mol)."""
    phi = dihedrals(
        placed[step.torsion_i],
        placed[step.a],
        placed[step.b],
        x[:, step.torsion_l],
    )  # (candidates, torsion terms)
    n, phase = step.torsion_n, step.torsion_phase
    return np.sum(step.torsion_k * (1 + np.cos(n * phi - phase)), axis=1)


def _lj_energy(step, placed, x):
    """Lennard-Jones energy of each candidate `x` with nearby placed atoms."""
    d = x[:, step.pair_moved] - placed[step.pair_placed]
    s6 = (step.pair_sigma2 / np.sum(d**2, axis=-1)) ** 3  # (sigma / r)**6
    return np.sum(4 * step.pair_eps * (s6 * s6 - s6), axis=1)


def _closest(step, placed, x):
    """Closest checked heavy-atom distance (nm) of each candidate `x`."""
    moved = x[:, step.heavy_moved].reshape(-1, 3)
    d = cdist(moved, placed[step.nearby]).reshape(
        len(x), len(step.heavy_moved), len(step.nearby)
    )
    d[:, step.skip[0], step.skip[1]] = np.inf  # pairs within three bonds
    return np.nanmin(d, axis=(1, 2), initial=np.inf)  # NaN: not placed yet


def _ring_centers(rings, xyz):
    """Centroids of `rings` (rows of atoms padded with -1) in `xyz`."""
    members = rings >= 0
    corners = np.where(members[..., None], xyz[..., rings, :], 0.0)
    return corners.sum(axis=-2) / members.sum(axis=1)[:, None]


def _pierces_ring(step, terms, placed, x, reach=0.3):
    """Whether each candidate pushes a bond through a ring of the chain.

    The bonds of the moved atoms are checked against the placed rings, and
    the placed bonds against the moved rings, where a bond's midpoint is
    within `reach` (nm) of a ring's center. Only the placed bonds and rings
    within reach of the sphere about the junction that holds the moved
    atoms are looked at.
    """
    out = np.zeros(len(x), dtype=bool)
    if not len(terms["triangles"]) or not len(x):
        return out
    rings, bonds = terms["rings"], terms["bond_array"]
    triangles = terms["triangles"]  # (ring, atom, next atom) fans
    # placed rings and bonds within reach of the junction (NaN: not placed)
    origin = placed[step.b]
    radius = np.max(np.linalg.norm(x - origin, axis=-1)) + reach
    ring_near = np.linalg.norm(_ring_centers(rings, placed) - origin, axis=1)
    bond_near = np.linalg.norm(placed[bonds].mean(axis=1) - origin, axis=1)
    checks = (  # moved bonds vs placed rings, placed bonds vs moved rings
        (step.moved_bonds, np.flatnonzero(ring_near[triangles[:, 0]] < radius)),
        (np.flatnonzero(bond_near < radius), step.moved_triangles),
    )
    # Coordinates of only the atoms these checks use, for each candidate:
    # the placed atoms, with the candidate's own moved atoms.
    used = np.unique(np.concatenate([triangles[t, 0] for _, t in checks]))
    ring_atoms = rings[used]
    atoms = np.unique(
        np.concatenate(
            [step.moved, *(bonds[b].ravel() for b, _ in checks)]
            + [ring_atoms[ring_atoms >= 0]]
        )
    )
    local = np.repeat(placed[atoms][None], len(x), axis=0)
    local[:, np.searchsorted(atoms, step.moved)] = x
    centers = _ring_centers(
        np.where(ring_atoms >= 0, np.searchsorted(atoms, ring_atoms), -1), local
    )
    for bond_ids, triangle_ids in checks:
        b, t = bonds[bond_ids], triangles[triangle_ids]
        p = local[:, np.searchsorted(atoms, b[:, 0])]
        q = local[:, np.searchsorted(atoms, b[:, 1])]
        c = centers[:, np.searchsorted(used, t[:, 0])]
        corner1 = local[:, np.searchsorted(atoms, t[:, 1])]
        corner2 = local[:, np.searchsorted(atoms, t[:, 2])]
        # (candidate, bond, triangle) with the bond's midpoint near the ring
        # center, leaving out bonds that touch the ring
        midpoint = (p + q) / 2
        near = (
            np.linalg.norm(midpoint[:, :, None] - c[:, None], axis=-1) < reach
        )
        touches = b[:, None, :, None] == rings[t[:, 0]][None, :, None]
        near &= ~np.any(touches, axis=(2, 3))
        k, i, j = np.nonzero(near)
        hit = crosses(p[k, i], q[k, i], c[k, j], corner1[k, j], corner2[k, j])
        out[k[hit]] = True
    return out


def rotatable_bonds(mol):
    """The bonds of an RDKit molecule that a torsion sampler may turn."""
    from rdkit import Chem

    def heavy_degree(atom):
        return sum(n.GetAtomicNum() > 1 for n in atom.GetNeighbors())

    def carbonyl(atom):  # C=O or C=N carbon
        return atom.GetAtomicNum() == 6 and any(
            b.GetBondType() == Chem.BondType.DOUBLE
            and b.GetOtherAtom(atom).GetAtomicNum() in (7, 8)
            for b in atom.GetBonds()
        )

    out = []
    for bond in mol.GetBonds():
        x, y = bond.GetBeginAtom(), bond.GetEndAtom()
        if (
            bond.GetBondType() != Chem.BondType.SINGLE
            or bond.IsInRing()
            or min(heavy_degree(x), heavy_degree(y)) < 2
            or Chem.HybridizationType.SP
            in (x.GetHybridization(), y.GetHybridization())
            or any(
                carbonyl(p) and q.GetAtomicNum() in (7, 8)
                for p, q in ((x, y), (y, x))
            )
        ):
            continue
        out.append((x.GetIdx(), y.GetIdx()))
    return out


def _rigid_units(mol):
    """Atoms connected without crossing a rotatable bond, as sorted lists."""
    graph = nx.Graph()
    graph.add_nodes_from(range(mol.GetNumAtoms()))
    graph.add_edges_from(
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()
    )
    graph.remove_edges_from(rotatable_bonds(mol))
    return sorted(sorted(unit) for unit in nx.connected_components(graph))


def _plan(terms, junctions, max_sep, clash_window=None):
    """The `Step` of each junction, in growth order."""
    n_atoms, heavy = len(terms["heavy"]), terms["heavy"]
    i, j = terms["bond_array"].T
    bond_graph = csr_matrix((np.ones(len(i)), (i, j)), shape=(n_atoms,) * 2)
    reach = max(max_sep, 3, clash_window or 0)
    sigma = np.array([terms["sigma"][i] for i in range(n_atoms)])
    eps = np.array([terms["eps"][i] for i in range(n_atoms)])
    # The junction that places each atom (-1: the first unit). The contact
    # check also counts the axis atoms of the first unit's junctions as
    # placed, since they are placed with it.
    placed_by = np.full(n_atoms, -1)
    for k in reversed(range(len(junctions))):
        placed_by[junctions[k].moved] = k
    with_first = np.zeros(n_atoms, dtype=bool)
    with_first[
        [b for jn in junctions if jn.parent == 0 for _, b in jn.bonds]
    ] = 1
    steps = []
    for k, junction in enumerate(junctions):
        a, b = junction.bonds[0]
        moved = junction.moved
        row = {atom: r for r, atom in enumerate(moved)}
        # bonds between each moved atom (rows) and every atom (columns)
        sep = dijkstra(
            bond_graph,
            directed=False,
            indices=moved,
            unweighted=True,
            limit=reach,
        )
        placed = placed_by < k
        placed[moved] = False
        # proper torsions i-a-b-l about the bond, l moved
        torsions = []
        for (p, q, r, s), values in terms["dihedrals"].items():
            if (r, q) == (a, b):  # written the other way round
                p, q, r, s = s, r, q, p
            if (q, r) == (a, b) and s in row:
                torsions += [(p, row[s], *t) for t in values]
        torsion_i, torsion_l, torsion_k, torsion_n, torsion_phase = (
            np.array(torsions).reshape(-1, 5).T
        )
        # Lennard-Jones pairs 3 to max_sep bonds apart (1-4 pairs scaled).
        # A pair already counted at an earlier junction has an atom on this
        # bond's axis, so its distance, and energy, is the same at every angle.
        pair_moved, pair_placed = np.nonzero(
            (sep >= 3) & (sep <= max_sep) & placed
        )
        mi, pj = moved[pair_moved], pair_placed
        if terms.get("combining", "lorentz") == "geometric":  # e.g. OPLS
            sigma_ij = np.sqrt(sigma[mi] * sigma[pj])
        else:  # Lorentz-Berthelot, e.g. SMIRNOFF
            sigma_ij = (sigma[mi] + sigma[pj]) / 2
        scale = np.where(sep[pair_moved, pair_placed] == 3, terms["scale14"], 1)
        # contact check: moved heavy atoms against placed heavy atoms within
        # clash_window bonds, skipping pairs within three bonds
        heavy_moved = np.flatnonzero(heavy[moved])
        if clash_window is None:
            nearby = np.flatnonzero(heavy)  # unplaced atoms are NaN
        else:
            known = (placed_by < k) | with_first
            known[moved] = False
            close = sep[heavy_moved].min(axis=0, initial=np.inf) <= clash_window
            nearby = np.flatnonzero(heavy & known & close)
        skip = np.nonzero(sep[np.ix_(heavy_moved, nearby)] <= 3)
        rings = [
            i
            for i, r in enumerate(terms["rings"])
            if set(r[r >= 0]) <= set(row)
        ]
        steps.append(
            Step(
                a=a,
                b=b,
                moved=moved,
                torsion_i=torsion_i.astype(int),
                torsion_l=torsion_l.astype(int),
                torsion_k=torsion_k,
                torsion_n=torsion_n,
                torsion_phase=torsion_phase,
                pair_moved=pair_moved,
                pair_placed=pair_placed,
                pair_sigma2=sigma_ij**2,
                pair_eps=scale * np.sqrt(eps[mi] * eps[pj]),
                heavy_moved=heavy_moved,
                nearby=nearby,
                skip=skip,
                moved_bonds=np.flatnonzero(
                    np.isin(terms["bond_array"], moved).any(1)
                ),
                moved_triangles=np.flatnonzero(
                    np.isin(terms["triangles"][:, 0], rings)
                ),
            )
        )
    return steps


def relax_bonded(chain, force_field="openff-2.3.0.offxml", force_tol=0.01):
    """Minimize a chain under its force field's bonded terms with FIRE.

    Only the bond, angle, dihedral and improper forces are kept, so the
    minimization brings bond lengths and angles to the force field's values
    and cannot invert a stereocenter through a clash; a stereocenter that
    changes handedness anyway raises ``RuntimeError``, as does a
    minimization that does not converge. Runs a `flowermd.base.Simulation`
    on the CPU in a temporary directory.

    Parameters
    ----------
    chain : mbuild.Compound, required
        All-atom chain with elements and explicit hydrogens.
    force_field : str or BaseXMLForcefield, default "openff-2.3.0.offxml"
        A SMIRNOFF force field, applied through
        `flowermd.library.OpenFF_DPD` at full strength, or a FlowerMD XML
        force field such as ``flowermd.library.OPLS_AA()``.
    force_tol : float, default 0.01
        FIRE's force tolerance (kJ/(mol nm), root mean square per degree
        of freedom); the largest force left is then about 1 kJ/(mol nm),
        which leaves bond angles within about 0.05 degrees.

    Returns
    -------
    numpy.ndarray, shape (N, 3)
        Relaxed coordinates (nm) in ``chain.particles()`` order.

    """
    import hoomd
    import mbuild as mb
    import unyt as u

    from flowermd.base import Simulation
    from flowermd.library import OpenFF_DPD, mbuildSystem
    from flowermd.library.simulations.phantom_walk import (
        run_FIRE_until_converged,
    )

    xyz = np.asarray(chain.xyz, dtype=float)
    compound = mb.clone(chain)
    compound.box = mb.Box(lengths=np.ptp(xyz, axis=0) + 3.0)
    units = dict(length=1 * u.nm, mass=1 * u.amu, energy=1 * u.Unit("kJ/mol"))
    system = mbuildSystem(compound, base_units=units)
    with tempfile.TemporaryDirectory() as tmp:
        if isinstance(force_field, str):
            force_field = OpenFF_DPD(
                system,
                os.path.join(tmp, "bonded.xml"),
                force_field=force_field,
                bonded_scale=1.0,
                dihedral_scale=1.0,
            )
        system.apply_forcefield(
            r_cut=0.35, force_field=force_field, kT=1.0, remove_charges=True
        )
        bonded = [
            force
            for force in system.hoomd_forcefield
            if not isinstance(force, hoomd.md.pair.Pair)
        ]
        sim = Simulation(
            system.hoomd_snapshot,
            bonded,
            dt=1e-3,
            device=hoomd.device.CPU(),
            gsd_file_name=os.path.join(tmp, "relax.gsd"),
            log_file_name=os.path.join(tmp, "relax.txt"),
        )
        result = run_FIRE_until_converged(
            sim,
            n_steps=2000,
            max_steps=1_000_000,
            force_tol=force_tol,
            energy_tol=1e-6,
            angmom_tol=1e9,
        )
        snap = sim.state.get_snapshot()
    if not result["converged"]:
        raise RuntimeError(
            f"Relaxing a {len(xyz)}-atom chain did not converge in "
            f"{result['steps']} FIRE steps. Pass "
            "LocalBoltzmannSampler(relax=False) to sample the built geometry."
        )
    box = np.asarray(snap.configuration.box[:3])
    relaxed = snap.particles.position + snap.particles.image * box
    relaxed = relaxed - relaxed.mean(axis=0) + xyz.mean(axis=0)
    index = {p: i for i, p in enumerate(chain.particles())}
    bonds = [(index[a], index[b]) for a, b in chain.bonds()]
    if np.any(_handedness(xyz, bonds) != _handedness(relaxed, bonds)):
        raise RuntimeError("Relaxing a chain inverted a stereocenter.")
    return relaxed


def _handedness(xyz, bonds):
    """Sign of the substituent volume at every atom with four neighbors."""
    graph = nx.Graph(list(bonds))
    quads = np.array(
        [list(graph[atom]) for atom in graph if graph.degree(atom) == 4],
        dtype=int,
    ).reshape(-1, 4)
    edges = xyz[quads[:, :3]] - xyz[quads[:, 3:]]
    return np.sign(signed_volumes(edges))
