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
from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.optimize import minimize
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
    # ring-threading check: the bonds and ring triangles of the moved atoms
    moved_bonds: np.ndarray  # indices into terms["bond_array"]
    moved_triangles: np.ndarray  # indices into terms["triangles"]


class LocalBoltzmannSampler:
    """Torsion sampler: local Boltzmann weights from a SMIRNOFF force field.

    For `flowermd.library.AtomisticRandomWalk`. Every rotatable bond turns,
    not only the bonds between repeat units: single bonds outside rings
    with a heavy atom beyond each end, except the C(=O)-O and C(=O)-N bonds
    of esters and amides and bonds to linear (sp) atoms. Ring bonds, and so
    ladder junctions, keep their geometry, and stereocenters are kept.

    Each torsion is drawn on a grid of `n_grid` angles, with a uniform
    offset within the chosen grid cell, from the Boltzmann weights of the
    force field's proper torsions about the bond plus the Lennard-Jones
    energy (the force field's combining rule, 1-4 pairs scaled by its
    factor) between the moved atoms and the placed atoms at most `max_sep`
    bonds away. Electrostatics are left out.

    A draw that brings a moved heavy atom closer than `r_min` to a placed
    heavy atom more than three and at most `clash_window` bonds away is
    redrawn. The check is local on purpose: it keeps the chain from
    folding back onto its neighbors, and leaves overlaps between parts of
    a chain further apart, like those between chains, to the DPD
    relaxation. A draw that threads a bond through a ring of the chain,
    which no relaxation undoes, is redrawn wherever it happens. Checked
    against the whole chain (``clash_window=None``), the contact check
    would make each chain a self-avoiding walk,
    whose size per mass grows with length without limit, unlike a chain in
    a melt, where excluded volume is screened; long chains would also
    rarely be placed at all. After `max_redraw` draws the growth is at a
    dead end and backs up `backtrack` rotatable bonds (more after repeated
    dead ends). After as many dead ends as the chain has rotatable bonds,
    each later dead end takes the least bad torsion: of the `n_grid` grid
    angles that thread no ring, those whose closest checked heavy-atom
    contact is within 0.01 nm of the farthest, drawn by their Boltzmann
    weights. Such contacts, often well under `r_min`, are left to DPD and
    recorded in `accepted`; `flowermd.library.AtomisticRandomWalk` warns
    once with their number. Chains whose growth never reaches that point,
    such as polyethylene and, up to about 5 kg/mol, atactic polystyrene,
    are placed exactly as with ``on_dead_end="error"``, which starts the
    chain again instead and raises ``RuntimeError`` after `max_restarts`
    restarts.

    Parameters
    ----------
    temperature : float, default 413.0
        Temperature (K) of the Boltzmann weights; use the melt temperature.
    force_field : str or BaseXMLForcefield, default "openff-2.3.0.offxml"
        A SMIRNOFF force field name or path, parameterized with OpenFF
        Interchange; or a FlowerMD XML force field such as
        ``flowermd.library.OPLS_AA()`` (or its GMSO ``ForceField``), typed
        with GMSO as `flowermd.base.System` types it, which needs no
        OpenFF. The string ``"oplsaa"`` is short for ``OPLS_AA()``.
    n_grid : int, default 72
        Grid points per torsion.
    max_sep : int, default 4
        Largest bond separation of the pairs whose Lennard-Jones energy is
        counted.
    r_min : float or None, default 0.30
        Closest allowed heavy-atom distance (nm), or None for no check.
    max_redraw : int, default 200
        Draws per torsion before a dead end.
    max_restarts : int, default 500
        Restarts per chain before giving up.
    relax : bool, default True
        Minimize each chain under the force field's bonded terms (bonds,
        angles, torsions and impropers) before sampling, so that the
        placement uses the force field's geometry rather than the built one.
    clash_window : int or None, default 12
        Largest separation, in bonds between heavy atoms, of the pairs that
        the `r_min` check compares; None compares every placed heavy atom.
        Twelve bonds reach the side groups of the next two repeat units of
        a vinyl polymer (the phenyl rings of neighboring styrene units are
        at most ten bonds apart) and about one Kuhn length of
        polyethylene. Any finite window lets the chain size per mass level
        off with length, but larger windows swell chains with side groups:
        for atactic polystyrene at 2 to 20 kg/mol, <R^2>/M is about 0.17
        A^2 mol/g with 8 bonds and 0.21-0.26 with 12 or 16, against 0.44
        measured.
    backtrack : int, default 4
        Rotatable bonds to back up at a dead end, times the number of dead
        ends since the growth last got past the furthest one; 0 starts the
        chain again at every dead end.
    on_dead_end : str, default "least_bad"
        ``"least_bad"`` or ``"error"``; see above.

    Attributes
    ----------
    accepted : list of list of float
        For each chain placed, the closest contacts (nm) of the torsions
        taken at dead ends.

    Notes
    -----
    With Sage 2.3.0, the default, and the local contact check, the
    single-chain <R^2>/M of 2, 5 and 20 kg/mol chains levels off with
    length and is 0.89-0.96 times the melt value of Fetters, Lohse and
    Colby (Physical Properties of Polymers Handbook, ch. 25 (2007),
    doi:10.1007/978-0-387-69002-5_25) for polyethylene (413 K) and
    0.92-1.04 times for bisphenol-A polycarbonate (473 K). Chains with side
    groups come out too compact, more so the more crowded the backbone:
    atactic polypropylene 0.75-0.84 and atactic polystyrene 0.47-0.58
    times (413 K; each rising slowly with length). The local energy sees only atoms at most `max_sep`
    bonds apart and misses the longer-range side-group repulsions that
    stiffen these chains; without any contact check, it is within 10 % of
    experiment for only 8 of 73 polymers surveyed (polyethylene,
    cis-polybutadiene, polycarbonate and a few others). The earlier
    whole-chain contact check hid part of this for short chains by
    swelling them. In alpha,alpha-disubstituted vinyl polymers (PMMA and
    the other methacrylates, polyisobutylene, poly(alpha-methylstyrene)),
    PTFE and other crowded chains, most torsions are dead ends: every grid
    angle brings a heavy-atom pair closer than `r_min`, since the backbone
    angles stay at the force field's equilibrium rather than opening up as
    in the real chain. These chains are placed with the least bad torsions
    (0.5-3 such contacts per repeat unit, the closest 0.04-0.2 nm), which the
    DPD relaxation and a minimization resolve; their dimensions follow
    neither the local energy nor experiment and are rough (single-chain
    <R^2>/M 0.26-1.7 times the melt value of Fetters et al. at 2-20
    kg/mol).
    The energy must come from a force field with realistic torsion and
    1-4 terms; UFF energies, for example, make polyethylene about 70 % too
    stiff. A SMIRNOFF force field requires openff-toolkit and
    openff-interchange.

    With OPLS-AA (``force_field=OPLS_AA()``), the single-chain placement C_n
    of C40 polyethylene at 413 K is about 9 (about 11 on Sage's geometry,
    whose C-C-C angle is 109.7 rather than 112.7 degrees), against about
    6.5 expected: OPLS-AA over-stabilizes trans states in long alkanes (the
    reason for L-OPLS; Siu, Pluhackova and Boeckmann, J. Chem. Theory
    Comput. 8, 1459 (2012), doi:10.1021/ct200908r), and the placement
    inherits it. For atactic polystyrene (20-mers, 450 K) it is about 7.3,
    close to Sage. Sage 2.3.0 remains the validated default.

    """

    def __init__(
        self,
        temperature=413.0,
        force_field="openff-2.3.0.offxml",
        n_grid=72,
        max_sep=4,
        r_min=0.30,
        max_redraw=200,
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
        self.r_min = r_min
        self.max_redraw = max_redraw
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
                self._templates[cached] = relax_bonded(xyz, terms)
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
        grid = (np.arange(self.n_grid) + 0.5) * width  # cell centers
        candidates = propose(grid)  # (n_grid, moved atoms, 3)
        energy = _torsion_energy(step, placed, candidates)
        energy += _lj_energy(step, placed, candidates)
        if least_bad:
            return self._least_bad(step, terms, placed, candidates, energy, rng)
        weight = np.exp(-(energy - energy.min()) / self.kT)
        for _ in range(0, self.max_redraw, 8):  # eight draws at a time
            cells = rng.choice(self.n_grid, size=8, p=weight / weight.sum())
            angles = (cells + rng.uniform(size=8)) * width  # anywhere in cell
            ok = self._allowed(step, terms, placed, propose(angles))
            if ok.any():
                return angles[ok.argmax()]
        return None

    def _allowed(self, step, terms, placed, candidates):
        """Candidates with no close contact and no bond through a ring."""
        ok = _closest(step, placed, candidates) >= (self.r_min or 0)
        ok[ok] = ~_threads(step, terms, placed, candidates[ok])
        return ok

    def _least_bad(self, step, terms, placed, candidates, energy, rng):
        """Grid angle whose closest contact is about the farthest, or None.

        Of the grid angles that thread no ring, those within 0.01 nm of the
        farthest closest contact are drawn by their Boltzmann weights.
        """
        closest = _closest(step, placed, candidates)
        threads = _threads(step, terms, placed, candidates)
        if threads.all():
            return None  # back up further
        closest[threads] = -np.inf
        good = closest >= closest.max() - 0.01
        energy = np.where(good, np.nan_to_num(energy, nan=1e300), np.inf)
        weight = np.exp(-(energy - energy.min()) / self.kT)
        g = rng.choice(self.n_grid, p=weight / weight.sum())
        self.accepted[-1].append(closest[g])
        return (g + 0.5) * 2 * np.pi / self.n_grid


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


def _threads(step, terms, placed, x, reach=0.3):
    """Whether each candidate threads a bond through a ring of the chain.

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
    # each candidate's chain: the placed atoms plus its own moved atoms
    chains = np.repeat(placed[None], len(x), axis=0)
    chains[:, step.moved] = x
    # placed rings and bonds within reach of the junction (NaN: not placed)
    origin = placed[step.b]
    radius = np.max(np.linalg.norm(x - origin, axis=-1)) + reach
    ring_near = np.linalg.norm(_ring_centers(rings, placed) - origin, axis=1)
    bond_near = np.linalg.norm(placed[bonds].mean(axis=1) - origin, axis=1)
    checks = (  # moved bonds vs placed rings, placed bonds vs moved rings
        (step.moved_bonds, np.flatnonzero(ring_near[triangles[:, 0]] < radius)),
        (np.flatnonzero(bond_near < radius), step.moved_triangles),
    )
    centers = _ring_centers(rings, chains)
    for bond_ids, triangle_ids in checks:
        b, t = bonds[bond_ids], triangles[triangle_ids]
        p, q, c = chains[:, b[:, 0]], chains[:, b[:, 1]], centers[:, t[:, 0]]
        # (candidate, bond, triangle) with the bond's midpoint near the ring
        # center, leaving out bonds that touch the ring
        midpoint = (p + q) / 2
        near = (
            np.linalg.norm(midpoint[:, :, None] - c[:, None], axis=-1) < reach
        )
        touches = b[:, None, :, None] == rings[t[:, 0]][None, :, None]
        near &= ~np.any(touches, axis=(2, 3))
        k, i, j = np.nonzero(near)
        hit = crosses(
            p[k, i], q[k, i], c[k, j], chains[k, t[j, 1]], chains[k, t[j, 2]]
        )
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


def _bond_graph(terms, atoms=None):
    """The chain's bonds as a graph, or only those among `atoms`."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(terms["heavy"])) if atoms is None else atoms)
    graph.add_edges_from(
        (i, j) for i, j in terms["bonds"] if i in graph and j in graph
    )
    return graph


def _plan(terms, junctions, max_sep, clash_window=None):
    """The `Step` of each junction, in growth order."""
    n_atoms = len(terms["heavy"])
    heavy = np.flatnonzero(terms["heavy"])
    bond_graph, heavy_graph = _bond_graph(terms), _bond_graph(terms, heavy)
    # bond separation of every atom pair at most max(max_sep, 3) bonds apart
    separation = [
        nx.single_source_shortest_path_length(bond_graph, i, max(max_sep, 3))
        for i in range(n_atoms)
    ]
    # torsion terms by central bond, both ways round: (i, l, k, n, phase)
    about = {}
    for (i, j, k, m), values in terms["dihedrals"].items():
        for a, b, c, d in ((i, j, k, m), (m, k, j, i)):
            about.setdefault((b, c), []).extend((a, d, *t) for t in values)
    sigma = np.array([terms["sigma"][i] for i in range(n_atoms)])
    eps = np.array([terms["eps"][i] for i in range(n_atoms)])
    placed = np.ones(n_atoms, dtype=bool)  # the first unit
    for j in junctions:
        placed[j.moved] = False
    # placed, as the contact check sees it: also the atoms on the axes of
    # the first unit's junctions, which are placed with it
    known = placed.copy()
    for j in junctions:
        if j.parent == 0:
            known[[b for _, b in j.bonds]] = True
    counted, steps = set(), []
    for j in junctions:
        row = {atom: r for r, atom in enumerate(j.moved)}
        a, b = j.bonds[0]
        torsions = [t for t in about.get((a, b), []) if t[1] in row]
        torsion_k, torsion_n, torsion_phase = (
            np.array([t[2:] for t in torsions]).reshape(-1, 3).T
        )
        # Each pair of atoms at most `max_sep` bonds apart is counted once,
        # at the first junction that fixes their distance.
        pairs = []
        for m in j.moved:
            for q, s in separation[m].items():
                if 3 <= s <= max_sep and placed[q] and q not in row:
                    if frozenset((m, q)) not in counted:
                        counted.add(frozenset((m, q)))
                        pairs.append((row[m], q, s))
        pair_moved, pair_placed, pair_sep = (
            np.array(pairs, dtype=int).reshape(-1, 3).T
        )
        mi, pj = j.moved[pair_moved], pair_placed
        if terms.get("combining", "lorentz") == "geometric":  # e.g. OPLS
            sigma_ij = np.sqrt(sigma[mi] * sigma[pj])
        else:  # Lorentz-Berthelot, e.g. SMIRNOFF
            sigma_ij = (sigma[mi] + sigma[pj]) / 2
        scale = np.where(pair_sep == 3, terms["scale14"], 1.0)
        heavy_moved = [r for r, m in enumerate(j.moved) if terms["heavy"][m]]
        if clash_window is None:
            nearby = heavy  # unplaced atoms are NaN and never clash
        else:
            reach = nx.multi_source_dijkstra_path_length(
                heavy_graph, set(j.moved[heavy_moved]), cutoff=clash_window
            )
            nearby = np.array(
                sorted(q for q in reach if known[q] and q not in row),
                dtype=int,
            )
        column = {atom: c for c, atom in enumerate(nearby)}
        skip = [
            (h, column[q])
            for h, r in enumerate(heavy_moved)
            for q, s in separation[j.moved[r]].items()
            if s <= 3 and q in column
        ]
        moved = set(j.moved.tolist())
        rings = [
            i for i, r in enumerate(terms["rings"]) if set(r[r >= 0]) <= moved
        ]
        steps.append(
            Step(
                a=a,
                b=b,
                moved=j.moved,
                torsion_i=np.array([t[0] for t in torsions], dtype=int),
                torsion_l=np.array([row[t[1]] for t in torsions], dtype=int),
                torsion_k=torsion_k,
                torsion_n=torsion_n,
                torsion_phase=torsion_phase,
                pair_moved=pair_moved,
                pair_placed=pair_placed,
                pair_sigma2=sigma_ij**2,
                pair_eps=scale * np.sqrt(eps[mi] * eps[pj]),
                heavy_moved=np.array(heavy_moved, dtype=int),
                nearby=nearby,
                skip=tuple(np.array(skip, dtype=int).reshape(-1, 2).T),
                moved_bonds=np.flatnonzero(
                    np.isin(terms["bond_array"], j.moved).any(1)
                ),
                moved_triangles=np.flatnonzero(
                    np.isin(terms["triangles"][:, 0], rings)
                ),
            )
        )
        placed[j.moved] = True
        known[j.moved] = True
    return steps


def relax_bonded(xyz, terms, tol=1.0, max_iter=None):
    """Minimize a molecule's bonded energy: bonds, angles and torsions.

    Nonbonded terms are left out, so the minimization only brings bond
    lengths and angles to the force field's values and cannot invert a
    stereocenter by a clash; a stereocenter that changes handedness anyway
    raises ``RuntimeError``. So does a minimization that stops before the
    largest force is below `tol`: a partly relaxed chain keeps strained
    bond angles in its middle, which change the chain's dimensions.

    Parameters
    ----------
    xyz : numpy.ndarray, shape (N, 3), required
        Coordinates (nm).
    terms : dict, required
        Tables as returned by `flowermd.internal.all_atom_dpd.openff_terms`
        or `flowermd.internal.all_atom_dpd.gmso_terms`.
    tol : float, default 1.0
        Largest force component (kJ/(mol nm)) at convergence. Bond lengths
        and angles are then within about 0.02 pm and 0.3 degrees of a much
        tighter minimization; only the floppy torsions, which the sampler
        sets anyway, are left less converged.
    max_iter : int, optional
        Most L-BFGS-B iterations, and energy evaluations, before giving up.
        Defaults to 100 per atom and at least 100,000; chains of a few
        thousand atoms need a few times 10,000.

    Returns
    -------
    numpy.ndarray, shape (N, 3)

    """
    bonds, angles, torsions = _bonded_tables(terms)

    def energy(flat):
        """Bonded energy (kJ/mol) and its gradient, flat like `flat`."""
        x = flat.reshape(-1, 3)
        parts = [
            _bond_terms(x, *bonds),
            _angle_terms(x, *angles),
            _torsion_terms(x, *torsions),
        ]
        total = sum(e for e, _ in parts)
        grad = _sum_onto_atoms(len(x), [g for _, rows in parts for g in rows])
        return total, grad.ravel()

    if max_iter is None:
        max_iter = max(100_000, 100 * len(xyz))
    result = minimize(
        energy,
        np.asarray(xyz, dtype=float).ravel(),
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": max_iter,
            "maxfun": max_iter,
            "gtol": tol,
            "ftol": 1e-15,
        },
    )
    force = np.max(np.abs(result.jac))
    if force > tol:
        raise RuntimeError(
            f"Relaxing a {len(xyz)}-atom chain stopped after {result.nit} "
            f"iterations with a largest force of {force:.3g} kJ/(mol nm), "
            f"above tol={tol} ({result.message}). Pass a larger max_iter, "
            "or LocalBoltzmannSampler(relax=False) to sample the built "
            "geometry."
        )
    relaxed = result.x.reshape(-1, 3)
    if np.any(_handedness(xyz, terms) != _handedness(relaxed, terms)):
        raise RuntimeError("Relaxing a chain inverted a stereocenter.")
    return relaxed


def _bonded_tables(terms):
    """Atom indices and parameters of every bond, angle and torsion term."""
    bond = np.array([(*ij, *p) for ij, p in terms["bonds"].items()])
    angle = np.array([(*ijk, *p) for ijk, p in terms["angles"].items()])
    torsion = np.array(
        [
            (*q, *t)
            for kind in ("dihedrals", "impropers")
            for q, values in terms[kind].items()
            for t in values
        ]
    )
    bond, angle, torsion = (
        t.reshape(-1, n) for t, n in ((bond, 4), (angle, 5), (torsion, 7))
    )
    return (
        (*bond[:, :2].T.astype(int), *bond[:, 2:].T),  # i, j, k, r0
        (*angle[:, :3].T.astype(int), *angle[:, 3:].T),  # i, j, k, k, theta0
        (*torsion[:, :4].T.astype(int), *torsion[:, 4:].T),  # i-l, k, n, phase
    )


def _bond_terms(x, i, j, k, r0):
    """Harmonic bonds, E = k/2 (r - r0)**2; energy and (atoms, gradient)."""
    d = x[j] - x[i]
    r = np.linalg.norm(d, axis=1)
    g = (k * (r - r0) / r)[:, None] * d  # dE/dx_j, along the bond
    return 0.5 * np.sum(k * (r - r0) ** 2), [(j, g), (i, -g)]


def _angle_terms(x, i, j, k, k_theta, theta0):
    """Harmonic angles i-j-k, E = k/2 (theta - theta0)**2."""
    u, v = x[i] - x[j], x[k] - x[j]  # arms from the vertex j
    ru, rv = np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1)
    cos = np.clip(np.sum(u * v, axis=1) / (ru * rv), -1.0, 1.0)
    theta = np.arccos(cos)
    # chain rule: dE/du = k (theta - theta0) * dtheta/dcos * dcos/du, with
    # dtheta/dcos = -1/sin(theta); dcos/du is the part of v perpendicular
    # to u over |u| |v|
    f = -k_theta * (theta - theta0) / np.sqrt(np.maximum(1 - cos**2, 1e-12))
    gu = f[:, None] * (v / (ru * rv)[:, None] - (cos / ru**2)[:, None] * u)
    gv = f[:, None] * (u / (ru * rv)[:, None] - (cos / rv**2)[:, None] * v)
    energy = 0.5 * np.sum(k_theta * (theta - theta0) ** 2)
    return energy, [(i, gu), (k, gv), (j, -gu - gv)]


def _torsion_terms(x, i, j, k, m, k_phi, n, phase):
    """Periodic torsions i-j-k-m, E = k (1 + cos(n phi - phase)).

    The gradient is that of Blondel and Karplus, J. Comput. Chem. 17,
    1132 (1996), which stays finite at phi = 0 and 180 degrees: the end
    atoms move along the normals of their planes, A and B, and the middle
    atoms take the rest so that the four gradients add to zero.
    """
    F, G, H = x[i] - x[j], x[j] - x[k], x[m] - x[k]
    A, B = np.cross(F, G), np.cross(H, G)  # normals of i-j-k and j-k-m
    rg = np.linalg.norm(G, axis=1)
    aa, bb = np.sum(A * A, axis=1), np.sum(B * B, axis=1)
    phi = dihedrals(x[i], x[j], x[k], x[m])
    de = -k_phi * n * np.sin(n * phi - phase)  # dE/dphi
    gi = -(de * rg / aa)[:, None] * A
    gm = (de * rg / bb)[:, None] * B
    shared = de[:, None] * (
        (np.sum(F * G, axis=1) / (aa * rg))[:, None] * A
        - (np.sum(H * G, axis=1) / (bb * rg))[:, None] * B
    )
    energy = np.sum(k_phi * (1 + np.cos(n * phi - phase)))
    return energy, [(i, gi), (m, gm), (j, shared - gi), (k, -gm - shared)]


def _sum_onto_atoms(n_atoms, contributions):
    """Add each term's gradient rows onto their atoms: shape (n_atoms, 3).

    An atom in several terms gets the sum of its rows. (``grad[atoms] +=
    rows`` would keep only one row per repeated atom.)
    """
    atoms = np.concatenate([a for a, _ in contributions])
    rows = np.concatenate([g for _, g in contributions])
    return np.column_stack(
        [
            np.bincount(atoms, weights=rows[:, c], minlength=n_atoms)
            for c in range(3)
        ]
    )


def _handedness(xyz, terms):
    """Sign of the substituent volume at every atom with four neighbors."""
    graph = nx.Graph(list(terms["bonds"]))
    quads = np.array(
        [list(graph[atom]) for atom in graph if graph.degree(atom) == 4],
        dtype=int,
    ).reshape(-1, 4)
    edges = xyz[quads[:, :3]] - xyz[quads[:, 3:]]
    return np.sign(signed_volumes(edges))
