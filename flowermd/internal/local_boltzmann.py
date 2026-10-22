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
from collections import namedtuple

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

from flowermd.internal.all_atom_dpd import (
    gmso_terms,
    openff_terms,
    to_sanitized_rdkit,
)

KB = 0.0083144626  # kJ/(mol K)
CLASH_WINDOW = 12  # bonds; see LocalBoltzmannSampler
BACKTRACK = 4  # junctions
# Per junction: the turned bond a-b; its torsions i-a-b-l (placed atom i,
# row of l among the moved atoms, and k, n, phase); the Lennard-Jones pairs
# (moved row, placed atom, sigma**2, epsilon); the moved heavy-atom rows and
# the heavy atoms that the contact check compares them with; and the (row,
# position in `near`) pairs within three bonds, which the check skips.
Step = namedtuple(
    "Step",
    [
        "a",
        "b",
        "ti",
        "tl",
        "terms",
        "pr",
        "pc",
        "sigma2",
        "eps",
        "hr",
        "near",
        "ex",
    ],
)


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
    redrawn. The check is local on purpose: it keeps rings from spearing
    each other and the chain from folding back onto its neighbors, and
    leaves overlaps between parts of a chain further apart, like those
    between chains, to the DPD relaxation. Checked against the whole chain
    (``clash_window=None``), it would make each chain a self-avoiding walk,
    whose size per mass grows with length without limit, unlike a chain in
    a melt, where excluded volume is screened; long chains would also
    rarely be placed at all. After `max_redraw` draws the growth is at a
    dead end and backs up `backtrack` rotatable bonds (more after repeated
    dead ends); after as many dead ends as the chain has rotatable bonds
    it starts again, and after `max_restarts` restarts a ``RuntimeError``
    is raised.

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
    swelling them. alpha,alpha-Disubstituted vinyl polymers (PMMA,
    polyisobutylene, poly(alpha-methylstyrene)) cannot be placed with the
    default `r_min`: every torsion at one of the first junctions brings a 1-5
    heavy-atom pair closer than `r_min`, since the backbone angles stay at
    the force field's equilibrium rather than opening up as in the real
    chain. With ``r_min=None`` they are placed but 55-80 % too compact.
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
        return functools.partial(self._choose, self._steps[key])

    def _choose(self, steps, index, placed, propose, rng):
        step = steps[index]
        width = 2 * np.pi / self.n_grid
        x = propose((np.arange(self.n_grid) + 0.5) * width)
        energy = np.zeros(self.n_grid)
        if len(step.ti):
            # Dihedral i-a-b-l: only l moves, so project once onto the
            # frame of the bond and the fixed atom i.
            axis = placed[step.b] - placed[step.a]
            axis /= np.sqrt(axis @ axis)
            v = placed[step.ti] - placed[step.a]
            v -= np.outer(v @ axis, axis)
            frame = np.stack([v, np.cross(axis, v)])
            w = x[:, step.tl] - placed[step.b]
            cos, sin = np.einsum("gtk,ctk->cgt", w, frame)
            phi = np.arctan2(sin, cos)
            k, n, phase = step.terms
            energy += np.sum(k * (1 + np.cos(n * phi - phase)), axis=1)
        if len(step.pr):
            d = x[:, step.pr] - placed[step.pc]
            s6 = (step.sigma2 / np.einsum("gpk,gpk->gp", d, d)) ** 3
            energy += np.sum(4 * step.eps * (s6 * s6 - s6), axis=1)
        cdf = np.cumsum(np.exp(-(energy - energy.min()) / self.kT))
        others = placed[step.near]
        for _ in range(0, self.max_redraw, 8):  # eight draws at a time
            u = rng.uniform(size=(2, 8))
            cell = np.searchsorted(cdf, u[0] * cdf[-1])
            angles = (np.minimum(cell, self.n_grid - 1) + u[1]) * width
            if self.r_min is None:
                return angles[0]
            d = propose(angles)[:, step.hr, None] - others
            d2 = np.einsum("bhok,bhok->bho", d, d)
            d2[:, step.ex[0], step.ex[1]] = np.inf  # within three bonds
            ok = ~np.any(d2 < self.r_min**2, axis=(1, 2))  # NaN: unplaced
            if ok.any():
                return angles[ok.argmax()]
        return None


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
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    cut = set(map(frozenset, rotatable_bonds(mol)))
    pairs = [
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        for b in mol.GetBonds()
        if frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) not in cut
    ]
    n = mol.GetNumAtoms()
    i, j = np.array(pairs, dtype=int).reshape(-1, 2).T
    graph = coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    labels = connected_components(graph, directed=False)[1]
    _, first = np.unique(labels, return_index=True)  # number units by atom
    rank = np.argsort(np.argsort(first))
    units = [[] for _ in first]
    for atom, label in enumerate(labels):
        units[rank[label]].append(atom)
    return units


def _separations(terms, n_atoms, depth):
    """Bond separation of every atom pair within `depth` bonds, per atom."""
    neighbors = [[] for _ in range(n_atoms)]
    for i, j in terms["bonds"]:
        neighbors[i].append(j)
        neighbors[j].append(i)
    out = []
    for start in range(n_atoms):
        seen, shell = {start: 0}, [start]
        for d in range(1, depth + 1):
            shell = [k for i in shell for k in neighbors[i] if k not in seen]
            seen.update((k, d) for k in shell)
        out.append(seen)
    return out


def _within(neighbors, sources, depth):
    """Atoms at most `depth` bonds from any of `sources`."""
    seen, shell = set(sources), list(sources)
    for _ in range(depth):
        shell = [k for i in shell for k in neighbors[i] if k not in seen]
        seen.update(shell)
    return seen


def _plan(terms, junctions, max_sep, clash_window=None):
    """Index and parameter arrays of each junction, in growth order."""
    n = len(terms["heavy"])
    sep = _separations(terms, n, max(max_sep, 3))
    heavy_neighbors = [[] for _ in range(n)]
    for i, j in terms["bonds"]:
        if terms["heavy"][i] and terms["heavy"][j]:
            heavy_neighbors[i].append(j)
            heavy_neighbors[j].append(i)
    by_bond = {}
    for (i, j, k, m), values in terms["dihedrals"].items():
        for a, b, c, d in ((i, j, k, m), (m, k, j, i)):
            by_bond.setdefault((b, c), []).extend((a, d, *t) for t in values)
    heavy = np.flatnonzero(terms["heavy"])
    sigma = np.array([terms["sigma"][i] for i in range(n)])
    eps = np.array([terms["eps"][i] for i in range(n)])
    placed = np.ones(n, dtype=bool)  # the first unit
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
        torsions = [t for t in by_bond.get((a, b), []) if t[1] in row]
        torsion_terms = np.array([t[2:] for t in torsions]).reshape(-1, 3).T
        # Each pair of atoms at most `max_sep` bonds apart is counted once,
        # at the first junction that fixes their distance.
        pairs = []
        for m in j.moved:
            for q, s in sep[m].items():
                if 3 <= s <= max_sep and placed[q] and q not in row:
                    if frozenset((m, q)) not in counted:
                        counted.add(frozenset((m, q)))
                        pairs.append((row[m], q, s))
        pr, pc, ps = np.array(pairs, dtype=int).reshape(-1, 3).T
        pm = j.moved[pr]
        scale = np.where(ps == 3, terms["scale14"], 1.0)
        if terms.get("combining", "lorentz") == "geometric":  # e.g. OPLS
            sigma_ij = np.sqrt(sigma[pm] * sigma[pc])
        else:  # Lorentz-Berthelot, e.g. SMIRNOFF
            sigma_ij = (sigma[pm] + sigma[pc]) / 2
        hr = [r for r, m in enumerate(j.moved) if terms["heavy"][m]]
        if clash_window is None:
            near = heavy  # unplaced atoms are NaN and never clash
        else:
            reach = _within(heavy_neighbors, j.moved[hr], clash_window)
            near = np.array(
                sorted(q for q in reach if known[q] and q not in row),
                dtype=int,
            )
        column = {atom: c for c, atom in enumerate(near)}
        ex = [
            (h, column[q])
            for h, r in enumerate(hr)
            for q, s in sep[j.moved[r]].items()
            if s <= 3 and q in column
        ]
        steps.append(
            Step(
                a,
                b,
                np.array([t[0] for t in torsions], dtype=int),
                np.array([row[t[1]] for t in torsions], dtype=int),
                torsion_terms,
                pr,
                pc,
                sigma_ij**2,
                scale * np.sqrt(eps[pm] * eps[pc]),
                np.array(hr, dtype=int),
                near,
                tuple(np.array(ex, dtype=int).reshape(-1, 2).T),
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
    bond = np.array([(*ij, *p) for ij, p in terms["bonds"].items()]).reshape(
        -1, 4
    )
    angle = np.array(
        [(*ijk, *p) for ijk, p in terms["angles"].items()]
    ).reshape(-1, 5)
    torsion = np.array(
        [
            (*q, *t)
            for kind in ("dihedrals", "impropers")
            for q, values in terms[kind].items()
            for t in values
        ]
    ).reshape(-1, 7)
    i_b, j_b = bond[:, :2].T.astype(int)
    i_a, j_a, k_a = angle[:, :3].T.astype(int)
    i_t, j_t, k_t, l_t = torsion[:, :4].T.astype(int)
    # Sums the per-term atom gradients, stacked in this order, onto atoms.
    atoms = np.concatenate([j_b, i_b, i_a, k_a, j_a, i_t, l_t, j_t, k_t])
    scatter = csr_matrix(
        (np.ones(len(atoms)), (atoms, np.arange(len(atoms)))),
        shape=(len(xyz), len(atoms)),
    )

    def energy(flat):
        x = flat.reshape(-1, 3)
        d = x[j_b] - x[i_b]
        r = np.linalg.norm(d, axis=1)
        dr = r - bond[:, 3]
        g = (bond[:, 2] * dr / r)[:, None] * d
        u, v = x[i_a] - x[j_a], x[k_a] - x[j_a]
        ru, rv = np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1)
        cos = np.clip(np.sum(u * v, axis=1) / (ru * rv), -1.0, 1.0)
        theta = np.arccos(cos)
        dtheta = theta - angle[:, 4]
        # dE/dcos = k (theta - theta0) dtheta/dcos, dtheta/dcos = -1/sin
        f = -angle[:, 3] * dtheta / np.sqrt(np.maximum(1 - cos**2, 1e-12))
        gu = f[:, None] * (v / (ru * rv)[:, None] - (cos / ru**2)[:, None] * u)
        gv = f[:, None] * (u / (ru * rv)[:, None] - (cos / rv**2)[:, None] * v)
        # Torsions, with the gradient of Blondel and Karplus (1996).
        F, G, H = x[i_t] - x[j_t], x[j_t] - x[k_t], x[l_t] - x[k_t]
        A, B = np.cross(F, G), np.cross(H, G)
        rg = np.linalg.norm(G, axis=1)
        aa, bb = np.sum(A * A, axis=1), np.sum(B * B, axis=1)
        phi = np.arctan2(
            np.sum(np.cross(B, A) * G, axis=1) / rg, np.sum(A * B, axis=1)
        )
        k, n, phase = torsion[:, 4:].T
        dphi = (-k * n * np.sin(n * phi - phase))[:, None]
        fg = (np.sum(F * G, axis=1) / (aa * rg))[:, None] * A
        hg = (np.sum(H * G, axis=1) / (bb * rg))[:, None] * B
        gi = -dphi * (rg / aa)[:, None] * A
        gl = dphi * (rg / bb)[:, None] * B
        gfh = dphi * (fg - hg)
        grad = scatter @ np.concatenate(
            [g, -g, gu, gv, -gu - gv, gi, gl, gfh - gi, -gl - gfh]
        )
        total = (
            0.5 * np.sum(bond[:, 2] * dr**2)
            + 0.5 * np.sum(angle[:, 3] * dtheta**2)
            + np.sum(k * (1 + np.cos(n * phi - phase)))
        )
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


def _handedness(xyz, terms):
    """Sign of the substituent volume at every atom with four neighbors."""
    neighbors = {}
    for i, j in terms["bonds"]:
        neighbors.setdefault(i, []).append(j)
        neighbors.setdefault(j, []).append(i)
    quads = np.array([n for n in neighbors.values() if len(n) == 4], dtype=int)
    if not len(quads):
        return np.empty(0)
    edges = xyz[quads[:, :3]] - xyz[quads[:, 3:]]
    return np.sign(np.linalg.det(edges))
