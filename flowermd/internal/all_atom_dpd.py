"""Write GMSO XML force fields for all-atom DPD.

Bonded terms come from UFF (RDKit) or a SMIRNOFF force field (OpenFF
Interchange); the pair term is a soft DPD force whose coefficients are
weighted by the Lennard-Jones well depths of each pair. Units are kJ/mol,
nm and amu.
"""

import itertools
import math
import re

import numpy as np

KINDS = ("bonds", "angles", "dihedrals", "impropers")


def bonds_of(mol):
    """Each bond of an RDKit molecule once, in time linear in its size.

    ``mol.GetBonds()`` fetches every bond by index, which RDKit does in time
    proportional to the number of bonds, so looping over it is quadratic.
    """
    for atom in mol.GetAtoms():
        for bond in atom.GetBonds():
            if bond.GetBeginAtomIdx() == atom.GetIdx():
                yield bond


KCAL = 4.184  # kJ per kcal


def molecules_rdkit(compound):
    """Each molecule of `compound` as a sanitized RDKit molecule.

    Yields ``(mol, atoms)``, `atoms` being the indices of the molecule's
    atoms in `compound`; identical molecules share one RDKit molecule.
    Sanitizing a whole system as one molecule takes time that grows faster
    than its size, so each distinct molecule is sanitized on its own.
    """
    from rdkit import Chem

    atoms = []
    frags = Chem.GetMolFrags(
        compound.to_rdkit(),
        asMols=True,
        sanitizeFrags=False,
        fragsMolAtomMapping=atoms,
    )
    cache = {}
    for frag, where in zip(frags, atoms):
        key = (
            tuple(a.GetAtomicNum() for a in frag.GetAtoms()),
            tuple(
                (b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondType())
                for b in bonds_of(frag)
            ),
        )
        if key not in cache:
            cache[key] = _sanitize(frag)
        yield cache[key], where


def write_all_atom_dpd_xml(system, file_path, terms, **settings):
    """Type `system`, rename its particles and write the XML force field.

    Each distinct molecule is parameterized once by ``terms(rdkit_mol)``.
    Atom types are named ``_<element><nnn>`` and refined until every bonded
    parameter is a function of its member types, and the particles (and
    GMSO sites, if already converted) are renamed to match, so the file
    applies by name.
    """
    compound = getattr(system, "system", system)
    where = list(molecules_rdkit(compound))
    frags = list({id(mol): mol for mol, _ in where}.values())
    data = [_merge_impropers(_fill_dihedrals(f, terms(f))) for f in frags]
    labels = dict(zip(map(id, frags), _refine(frags, data)))
    names = [None] * compound.n_particles
    for mol, atoms in where:
        for i, name in zip(atoms, labels[id(mol)]):
            names[i] = name
    for particle, name in zip(compound.particles(), names):
        particle.name = name
    if getattr(system, "gmso_system", None) is not None:
        for site, name in zip(system.gmso_system.sites, names):
            site.name = name
    _emit(file_path, data, [labels[id(f)] for f in frags], **settings)


def to_sanitized_rdkit(compound):
    """RDKit molecule of an mbuild compound, with bond orders and rings set.

    `mbuild.Compound.to_rdkit` leaves the bonds made by `mbuild.Polymer`
    between repeat units UNSPECIFIED and the molecule unsanitized, so
    RDKit finds no rings, hybridizations or stereocenters there. Those
    bonds are made single, which is what they are in such chains. It also
    marks aromatic bonds but not their atoms, which would leave aromatic
    rings (and thiophene S) typed as non-aromatic; their atoms are marked
    before sanitizing.
    """
    return _sanitize(compound.to_rdkit())


def _sanitize(mol):
    """Set the bond orders and aromatic atoms of `mol`, then sanitize it."""
    from rdkit import Chem

    for bond in bonds_of(mol):
        if bond.GetBondType() == Chem.BondType.UNSPECIFIED:
            bond.SetBondType(Chem.BondType.SINGLE)
        elif bond.GetBondType() == Chem.BondType.AROMATIC:
            bond.SetIsAromatic(True)
            bond.GetBeginAtom().SetIsAromatic(True)
            bond.GetEndAtom().SetIsAromatic(True)
    Chem.SanitizeMol(mol)
    return mol


def uff_terms(mol):
    """UFF terms (kJ/mol, nm, rad) as RDKit's UFF builder sets them up."""
    from rdkit.Chem import rdForceFieldHelpers as uff
    from rdkit.Chem.rdchem import HybridizationType as H

    if not uff.UFFHasAllMoleculeParams(mol):
        raise ValueError("UFF does not cover every atom in this molecule.")
    out = _empty(mol)
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        out["eps"][i] = uff.GetUFFVdWParams(mol, i, i)[1] * KCAL
    for bond in bonds_of(mol):
        pair = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        k, r0 = uff.GetUFFBondStretchParams(mol, *pair)
        out["bonds"][pair] = (k * KCAL * 100, r0 / 10)
    rings = mol.GetRingInfo()
    for atom in mol.GetAtoms():
        j = atom.GetIdx()
        for i, k in itertools.combinations(
            [n.GetIdx() for n in atom.GetNeighbors()], 2
        ):
            k_theta, theta0 = uff.GetUFFAngleBendParams(mol, i, j, k)
            theta0 = math.radians(theta0)
            # RDKit's special cases for sp2 centers in three- and four-rings.
            for size, outside, inside in ((3, 150.0, 60.0), (4, 135.0, 90.0)):
                if atom.GetHybridization() != H.SP2:
                    break
                if rings.IsAtomInRingOfSize(j, size):
                    a, c = (rings.IsAtomInRingOfSize(x, size) for x in (i, k))
                    if a or c:
                        theta0 = math.radians(outside if a != c else inside)
                        break
            # Harmonic at UFF's minimum, whose curvature is k_theta.
            out["angles"][(i, j, k)] = (k_theta * KCAL, theta0)
    for bond in bonds_of(mol):
        quads = [q for q in _quads(bond) if uff.GetUFFTorsionParams(mol, *q)]
        for q in quads:  # UFF splits one barrier among a bond's torsions
            n, sign = _uff_torsion_form(mol, q)
            barrier = uff.GetUFFTorsionParams(mol, *q) / len(quads) * KCAL
            # V/2 (1 - sign cos(n phi)) = k (1 + cos(n phi - phi0))
            out["dihedrals"][q] = ((barrier / 2, n, math.pi * (sign > 0)),)
    return out


def _uff_torsion_form(mol, quad):
    """Periodicity and cosine sign of a UFF torsion (RDKit TorsionAngle)."""
    from rdkit.Chem.rdchem import HybridizationType as H

    a, j, k, d = (mol.GetAtomWithIdx(x) for x in quad)
    hj, hk = j.GetHybridization(), k.GetHybridization()
    group6 = {8, 16, 34, 52, 84}
    bond = mol.GetBondBetweenAtoms(quad[1], quad[2])
    single = bond.GetBondTypeAsDouble() == 1.0
    if hj == H.SP3 and hk == H.SP3:
        both6 = j.GetAtomicNum() in group6 and k.GetAtomicNum() in group6
        return (2, -1) if single and both6 else (3, -1)
    if hj == H.SP2 and hk == H.SP2:
        return 2, 1
    if single:
        sp3, sp2 = (j, k) if hj == H.SP3 else (k, j)
        if sp3.GetAtomicNum() in group6 and sp2.GetAtomicNum() not in group6:
            return 2, -1
        if H.SP2 in (a.GetHybridization(), d.GetHybridization()):
            return 3, -1
    return 6, 1


def openff_terms(mol, force_field):
    """SMIRNOFF terms (kJ/mol, nm, rad) from Interchange's collections."""
    from openff.interchange import Interchange
    from openff.toolkit import ForceField, Molecule
    from openff.units import unit

    molecule = Molecule.from_rdkit(
        mol, allow_undefined_stereo=True, hydrogens_are_explicit=True
    )
    # Zero charges skip AM1-BCC; DPD has no electrostatics.
    molecule.partial_charges = np.zeros(molecule.n_atoms) * unit.e
    interchange = Interchange.from_smirnoff(
        ForceField(force_field),
        molecule.to_topology(),
        charge_from_molecules=[molecule],
    )

    def params(name):
        c = interchange.collections[name]
        for key, pot in c.key_map.items():
            yield key.atom_indices, c.potentials[pot].parameters

    out = _empty(mol)
    out["sigma"], out["scale14"] = {}, interchange.collections["vdW"].scale_14
    for (i,), p in params("vdW"):
        out["eps"][i] = p["epsilon"].m_as("kJ/mol")
        out["sigma"][i] = p["sigma"].m_as("nm")
    for ij, p in params("Bonds"):
        out["bonds"][ij] = (p["k"].m_as("kJ/mol/nm**2"), p["length"].m_as("nm"))
    for ijk, p in params("Angles"):
        out["angles"][ijk] = (
            p["k"].m_as("kJ/mol/rad**2"),
            p["angle"].m_as("rad"),
        )
    for name, kind in (
        ("ProperTorsions", "dihedrals"),
        ("ImproperTorsions", "impropers"),
    ):
        for idx, p in params(name):
            k = p["k"].m_as("kJ/mol") / float(p.get("idivf", 1))
            term = (k, int(p["periodicity"]), p["phase"].m_as("rad"))
            out[kind][idx] = out[kind].get(idx, ()) + (term,)
    return out


def gmso_terms(compound, force_field):
    """Terms (kJ/mol, nm, rad) of a GMSO force field applied to `compound`.

    The compound is typed as `flowermd.base.System` types it
    (`gmso.parameterization.apply`, impropers ignored), and the tables have
    the layout of `openff_terms`, atoms in particle order, plus
    ``combining``, the force field's combining rule. Torsions are written
    exactly as sums of ``k (1 + cos(n phi - phase))``, up to a constant.
    """
    from gmso.external import from_mbuild
    from gmso.parameterization import apply

    top = apply(from_mbuild(compound), force_field, identify_connections=True)
    index = {site: i for i, site in enumerate(top.sites)}

    def members(connection):
        return tuple(index[s] for s in connection.connection_members)

    def value(p, name, units):
        return float(p[name].to_value(units))

    out = {kind: {} for kind in KINDS}
    out["sigma"], out["eps"], out["mass"] = {}, {}, []
    for i, site in enumerate(top.sites):
        p = site.atom_type.parameters
        out["sigma"][i] = value(p, "sigma", "nm")
        out["eps"][i] = value(p, "epsilon", "kJ/mol")
        out["mass"].append(float(site.mass.to_value("amu")))
    out["scale14"] = float(force_field.scaling_factors["nonBonded14Scale"])
    out["combining"] = force_field.combining_rule
    for bond in top.bonds:
        p = bond.bond_type.parameters
        out["bonds"][members(bond)] = (
            value(p, "k", "kJ/(mol*nm**2)"),
            value(p, "r_eq", "nm"),
        )
    for angle in top.angles:
        p = angle.angle_type.parameters
        out["angles"][members(angle)] = (
            value(p, "k", "kJ/(mol*rad**2)"),
            value(p, "theta_eq", "rad"),
        )
    for kind in ("dihedrals", "impropers"):
        for c in getattr(top, kind):
            out[kind][members(c)] = _periodic_torsion(
                c.connection_type.parameters
            )
    return out


def _periodic_torsion(p):
    """GMSO torsion parameters as ``((k, n, phase), ...)``, kJ/mol and rad.

    Ryckaert-Bellemans ``sum_m c_m cos(psi)**m``, ``psi = phi - pi``, is
    ``sum_n a_n (-1)**n cos(n phi)``, with `a_n` the Chebyshev coefficients
    (``cos(n psi) = T_n(cos(psi))``); the OPLS (Fourier) form has
    ``0.5 k_n (1 - (-1)**n cos(n phi))``. Constant offsets are dropped.
    """
    if "c0" in p:  # RyckaertBellemansTorsionPotential
        c = [p[f"c{m}"].to_value("kJ/mol") for m in range(6)]
        a = np.polynomial.chebyshev.poly2cheb(c)
        cosines = {n: a[n] * (-1) ** n for n in range(1, len(a))}
    elif "k1" in p:  # OPLSTorsionPotential, FourierTorsionPotential
        cosines = {
            n: -0.5 * (-1) ** n * p[f"k{n}"].to_value("kJ/mol")
            for n in range(1, 5)
        }
    elif "phi_eq" in p:  # PeriodicTorsionPotential, PeriodicImproperPotential
        k = np.atleast_1d(p["k"].to_value("kJ/mol"))
        n = np.atleast_1d(np.asarray(p["n"], dtype=float))
        phase = np.atleast_1d(p["phi_eq"].to_value("rad"))
        return tuple(zip(k.tolist(), n.astype(int).tolist(), phase.tolist()))
    else:
        raise NotImplementedError(
            f"Torsion parameters {sorted(p)} are not supported; use "
            "Ryckaert-Bellemans, OPLS (Fourier) or periodic torsions."
        )
    # a cos(n phi) = |a| (1 + cos(n phi - phase)) - |a|
    terms = tuple(
        (abs(float(a)), n, 0.0 if a > 0 else math.pi)
        for n, a in cosines.items()
        if abs(a) > 1e-12
    )
    return terms or ((0.0, 1, 0.0),)


def _merge_impropers(out):
    """Sum SMIRNOFF's three trefoil terms per center into one improper.

    GMSO keeps one improper per center, keyed by the center and then the
    sorted other atoms; SMIRNOFF impropers list the center first.
    """
    merged = {}
    for idx, terms in out["impropers"].items():
        key = (idx[0], *sorted(idx[1:]))
        merged[key] = merged.get(key, ()) + terms
    out["impropers"] = {
        idx: ((sum(t[0] for t in terms), *terms[0][1:]),)
        for idx, terms in merged.items()
    }
    return out


def _quads(bond):
    j, k = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    return [
        (a.GetIdx(), j, k, d.GetIdx())
        for a in bond.GetBeginAtom().GetNeighbors()
        for d in bond.GetEndAtom().GetNeighbors()
        if a.GetIdx() != k and d.GetIdx() not in (j, a.GetIdx())
    ]


def _fill_dihedrals(mol, out):
    """Give every dihedral without a term (e.g. next to a nitrile) k = 0.

    GMSO types every dihedral of the topology, so each needs an entry.
    """
    have = out["dihedrals"]
    for bond in bonds_of(mol):
        for q in _quads(bond):
            if q not in have and q[::-1] not in have:
                have[q] = ((0.0, 1, 0.0),)
    return out


def _empty(mol):
    masses = [atom.GetMass() for atom in mol.GetAtoms()]
    return {"eps": {}, "mass": masses, **{kind: {} for kind in KINDS}}


def _key(kind, labels):
    if kind == "impropers":
        return (labels[0], *sorted(labels[1:]))
    return min(tuple(labels), tuple(reversed(labels)))


def _refine(frags, data):
    """Atom type names that make every bonded parameter a function of types.

    Starts from element and epsilon, then splits the atoms of conflicting
    terms by their neighbors' labels until no two terms with the same
    member types have different parameters. Names are zero-padded because
    GMSO 0.17 orders snapshot types by plain sort and force-field types by
    natural sort, which disagree for names such as ``_C2`` and ``_C10``.
    """
    labels = [
        [(a.GetSymbol(), round(d["eps"][a.GetIdx()], 8)) for a in f.GetAtoms()]
        for f, d in zip(frags, data)
    ]
    ids = {}
    while True:
        seen, clash = {}, set()
        for lab, d in zip(labels, data):
            for kind in KINDS:
                for idx, value in d[kind].items():
                    key = (kind, _key(kind, [lab[i] for i in idx]))
                    if seen.setdefault(key, value) != value:
                        clash.add(key)
        if not clash:
            break
        new = []
        for frag, lab, d in zip(frags, labels, data):
            split = {
                i
                for kind in KINDS
                for idx in d[kind]
                if (kind, _key(kind, [lab[i] for i in idx])) in clash
                for i in idx
            }
            new.append(lab.copy())
            for i in split:
                around = sorted(
                    lab[n.GetIdx()]
                    for n in frag.GetAtomWithIdx(i).GetNeighbors()
                )
                new[-1][i] = (
                    lab[i][0],
                    ids.setdefault((lab[i], *around), len(ids)),
                )
        if new == labels:
            raise RuntimeError("Atom-type refinement did not converge.")
        labels = new
    names = {}
    return [
        [names.setdefault(x, f"_{x[0]}{len(names):03d}") for x in lab]
        for lab in labels
    ]


def _emit(
    file_path,
    data,
    labels,
    A,
    gamma,
    r_cut,
    bonded_scale,
    dihedral_scale,
    epsilon_weighting,
    pair_overrides,
):
    """Write the GMSO XML with GMSO's writer, then add the DPD pair table."""
    import unyt as u
    from gmso import ForceField
    from gmso.core.angle_type import AngleType
    from gmso.core.atom_type import AtomType
    from gmso.core.bond_type import BondType
    from gmso.core.dihedral_type import DihedralType
    from gmso.core.improper_type import ImproperType
    from gmso.lib.potential_templates import PotentialTemplateLibrary
    from gmso.utils.expression import NullPotentialExpression

    templates = PotentialTemplateLibrary()
    ff = ForceField()
    ff.scaling_factors = {"nonBonded14Scale": 0.0, "electrostatics14Scale": 0.0}
    periodic = ("k", "n", "phi_eq"), ("kJ/mol", "dimensionless", "rad")
    kinds = {
        "bonds": (
            BondType,
            "HarmonicBondPotential",
            ("k", "r_eq"),
            ("kJ/mol/nm**2", "nm"),
        ),
        "angles": (
            AngleType,
            "HarmonicAnglePotential",
            ("k", "theta_eq"),
            ("kJ/mol/rad**2", "rad"),
        ),
        "dihedrals": (DihedralType, "PeriodicTorsionPotential", *periodic),
        "impropers": (ImproperType, "PeriodicTorsionPotential", *periodic),
    }
    eps = {}
    for lab, d in zip(labels, data):
        for i, name in enumerate(lab):
            eps[name] = d["eps"][i]
            ff.atom_types[name] = AtomType(
                name=name,
                atomclass=name,
                definition=name,
                mass=d["mass"][i] * u.amu,
                potential_expression=NullPotentialExpression(),
            )
        for kind, (cls, template, names, units) in kinds.items():
            potential = templates[template]
            store = getattr(ff, kind[:-1] + "_types")
            for idx, value in d[kind].items():
                members = _key(kind, [lab[i] for i in idx])
                multi = kind in ("dihedrals", "impropers")
                columns = [list(c) for c in zip(*(value if multi else [value]))]
                scale = dihedral_scale if kind == "dihedrals" else bonded_scale
                columns[0] = [scale * k for k in columns[0]]
                params = {
                    p: u.unyt_array(c, unit) if multi else c[0] * u.Unit(unit)
                    for p, c, unit in zip(names, columns, units)
                }
                store["~".join(members)] = cls(
                    name="~".join(members),
                    member_types=members,
                    expression=potential.expression,
                    independent_variables=potential.independent_variables,
                    parameters=params,
                )
    ff.to_xml(file_path, overwrite=True)
    # GMSO 0.17 workarounds. Its writer has no DPD pair potentials, and it
    # writes the null atom-type expression as expression="None" and drops
    # the element tag that name-matched types need; its reader accepts
    # neither, so both are fixed in the text below. gamma is in amu/ps, but
    # GMSO's HOOMDDPDForce template requires a mass/length/time unit; the
    # value reaches HOOMD unconverted, so only the label is wrong.
    eps_max = max(eps.values()) or 1.0
    overrides = pair_overrides or {}
    pairs = []
    for a, b in itertools.combinations_with_replacement(eps, 2):
        w = math.sqrt(eps[a] * eps[b]) / eps_max if epsilon_weighting else 1.0
        a_ij = overrides.get((a, b), overrides.get((b, a), A * w))
        pairs.append(
            f'<PairPotentialType name="HOOMDDPDForce" type1="{a}" type2="{b}">'
            f'<Parameters><Parameter name="A" value="{a_ij!r}"/>'
            f'<Parameter name="r_cut" value="{r_cut!r}"/>'
            f'<Parameter name="γ" value="{gamma * w!r}"/></Parameters>'
            "</PairPotentialType>"
        )
    with open(file_path, encoding="utf-8") as handle:
        text = handle.read().replace(
            '<AtomTypes expression="None">', "<AtomTypes>"
        )
    text = re.sub(r'definition="(_\w+)"', r'definition="\1" element="\1"', text)
    table = (
        '<PairPotentialTypes expression="A * (1-(r/r_cut)) - γ">'
        '<ParametersUnitDef parameter="A" unit="kJ/mol/nm"/>'
        '<ParametersUnitDef parameter="r_cut" unit="nm"/>'
        '<ParametersUnitDef parameter="γ" unit="amu/(nm*ps)"/>'
        + "".join(pairs)
        + "</PairPotentialTypes></ForceField>"
    )
    with open(file_path, "w", encoding="utf-8") as handle:
        handle.write(text.replace("</ForceField>", table))
