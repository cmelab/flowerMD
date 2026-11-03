"""Base forcefield classes."""

from abc import ABC, abstractmethod

import forcefield_utilities as ffutils
from gmso.core.forcefield import ForceField

from flowermd.internal.all_atom_dpd import KINDS, write_all_atom_dpd_xml


class BaseXMLForcefield:
    """Base XML forcefield class."""

    def __init__(self, forcefield_files=None, name=None, gmso_xml=None):
        self.forcefield_files = forcefield_files
        self.name = name
        self.gmso_xml = gmso_xml
        if all([name, forcefield_files]):
            raise ValueError("Give only one of `name` or `forcefield_files`.")
        if self.gmso_xml is True:
            self.gmso_ff = ForceField(forcefield_files or name)

        else:
            self.gmso_ff = (
                ffutils.FoyerFFs().load(forcefield_files or name).to_gmso_ff()
            )


class BaseAllAtomDPDForcefield(BaseXMLForcefield, ABC):
    """Base class for all-atom DPD forcefields written for a system.

    A subclass supplies the bonded terms and Lennard-Jones well depths of
    a molecule by implementing `_terms`; this class scales the bonded
    terms, weights a DPD pair force by the well depths, writes the result
    as a GMSO XML file at `file_path` and loads it. Atom types are named
    ``_<element><nnn>``, and the particles of `system` are renamed to
    match. See `flowermd.library.UFF_DPD`, `flowermd.library.OpenFF_DPD`
    and `flowermd.library.XML_DPD`.

    Notes
    -----
    This is designed for the all-atom `flowermd.library.PhantomWalk`
    workflow: soft overlaps relax under DPD while the scaled bonded terms
    keep the chemistry. Units are kJ/mol, nm and amu; GMSO passes the DPD
    parameters to HOOMD unconverted, so use ``base_units`` of 1 kJ/mol,
    1 nm and 1 amu. Use a time step of about 2e-4 ps. Requires RDKit.

    `_terms` is called once per distinct molecule with a sanitized RDKit
    molecule and returns a dict, indexed by RDKit atom index, with:

    - ``"eps"``: ``{i: epsilon}``, the Lennard-Jones well depth (kJ/mol)
      of every atom.
    - ``"mass"``: a list of every atom's mass (amu).
    - ``"bonds"``: ``{(i, j): (k, r_eq)}`` in kJ/(mol nm^2) and nm.
    - ``"angles"``: ``{(i, j, k): (k, theta_eq)}`` in kJ/(mol rad^2) and
      rad.
    - ``"dihedrals"``, ``"impropers"``: ``{(i, j, k, l): ((k, n, phase),
      ...)}``, sums of ``k (1 + cos(n phi - phase))`` in kJ/mol and rad.
      Impropers are centered on the first atom.

    Dihedrals without a term get ``k = 0``. A subclass that needs its own
    attributes in `_terms` must set them before calling
    ``super().__init__``, which writes the file.

    Parameters
    ----------
    system : flowermd.base.System or mbuild.Compound, required
        All-atom system with elements and explicit hydrogens.
    file_path : str, default "all_atom_dpd.xml"
        Where to write the XML file.
    A : float, default 52300.0
        DPD repulsion (kJ/(mol nm)) before weighting.
    gamma : float, default 4091.0
        DPD friction (amu/ps) before weighting.
    r_cut : float, default 0.35
        DPD cutoff (nm).
    bonded_scale : float, default 30.0
        Multiplier on the bond, angle and improper force constants.
    dihedral_scale : float, default 30.0
        Multiplier on the proper dihedral force constants. Stiffer torsions
        keep the placed torsions more closely (polyethylene C_n drops a few
        percent less at 300), but UFF's aryl-O and aryl-S torsions at 300
        hold neighboring rings nearly coplanar and push ortho hydrogens
        into each other, which the soft repulsion cannot undo.
    epsilon_weighting : bool, default True
        Weight ``A`` and ``gamma`` of each pair by
        ``sqrt(eps_i * eps_j) / eps_max``. Atom types with a zero well
        depth are weighted as if they had the smallest nonzero one, with
        a warning.
    pair_overrides : dict, optional
        ``{(type_i, type_j): A_ij}`` replacing the repulsion of those pairs.
    nlist_buffer : float, default 0.04
        Neighbor-list buffer (nm) that `System.apply_forcefield` uses when
        no ``nlist`` is given. GMSO's default of 0.4 nm gives the same
        trajectory but makes DPD about twice as slow.

    """

    def __init__(
        self,
        system,
        file_path="all_atom_dpd.xml",
        A=52300.0,
        gamma=4091.0,
        r_cut=0.35,
        bonded_scale=30.0,
        dihedral_scale=30.0,
        epsilon_weighting=True,
        pair_overrides=None,
        nlist_buffer=0.04,
    ):
        write_all_atom_dpd_xml(
            system,
            file_path,
            self._checked_terms,
            A=A,
            gamma=gamma,
            r_cut=r_cut,
            bonded_scale=bonded_scale,
            dihedral_scale=dihedral_scale,
            epsilon_weighting=epsilon_weighting,
            pair_overrides=pair_overrides,
        )
        super(BaseAllAtomDPDForcefield, self).__init__(
            forcefield_files=file_path, gmso_xml=True
        )
        self.nlist_buffer = nlist_buffer

    @abstractmethod
    def _terms(self, mol):
        """Abstract method returning the terms of an RDKit molecule."""
        pass

    def _checked_terms(self, mol):
        """`_terms` of `mol`, checked against the layout described above."""
        terms = self._terms(mol)
        missing = [k for k in ("eps", "mass", *KINDS) if k not in terms]
        if missing:
            raise ValueError(
                f"{type(self).__name__}._terms did not return {missing}."
            )
        n_atoms = mol.GetNumAtoms()
        if len(terms["mass"]) != n_atoms or set(terms["eps"]) != set(
            range(n_atoms)
        ):
            raise ValueError(
                f"{type(self).__name__}._terms must give a mass and an "
                f"epsilon for each of the {n_atoms} atoms."
            )
        return terms


class BaseHOOMDForcefield:
    """Base HOOMD forcefield class."""

    def __init__(self, hoomd_forces):
        self.hoomd_forces = hoomd_forces
        if hoomd_forces is None:
            raise NotImplementedError(
                "`hoomd_forces` must be defined in the subclass."
            )
        if not isinstance(hoomd_forces, list):
            raise TypeError("`hoomd_forces` must be a list.")
