"""All pre-defined forcefield classes for use in hoomd_organics."""
import itertools

import hoomd

from hoomd_organics.assets import FF_DIR
from hoomd_organics.base import BaseHOOMDForcefield, BaseXMLForcefield


class GAFF(BaseXMLForcefield):
    """GAFF forcefield class."""

    def __init__(self, forcefield_files=f"{FF_DIR}/gaff.xml"):
        super(GAFF, self).__init__(forcefield_files=forcefield_files)
        self.description = (
            "The General Amber Forcefield written in foyer XML format. "
            "The XML file was obtained from the antefoyer package: "
            "https://github.com/rsdefever/antefoyer/tree/master/antefoyer"
        )


class OPLS_AA(BaseXMLForcefield):
    """OPLS All Atom forcefield class."""

    def __init__(self, name="oplsaa"):
        super(OPLS_AA, self).__init__(name=name)
        self.description = "opls-aa forcefield found in the Foyer package."


class OPLS_AA_PPS(BaseXMLForcefield):
    """OPLS All Atom for PPS molecule forcefield class."""

    def __init__(self, forcefield_files=f"{FF_DIR}/pps_opls.xml"):
        super(OPLS_AA_PPS, self).__init__(forcefield_files=forcefield_files)
        self.description = (
            "Based on hoomd_organics.forcefields.OPLS_AA. "
            "Trimmed down to include only PPS parameters. "
            "One missing parameter was added manually: "
            "<Angle class1=CA class2=S class3=CA angle=1.805 k=627.6/> "
            "The equilibrium angle was determined from "
            "experimental PPS papers. The spring constant taken "
            "from the equivalent angle in GAFF."
        )


class OPLS_AA_BENZENE(BaseXMLForcefield):
    """OPLS All Atom for benzene molecule forcefield class."""

    def __init__(self, forcefield_files=f"{FF_DIR}/benzene_opls.xml"):
        super(OPLS_AA_BENZENE, self).__init__(forcefield_files=forcefield_files)
        self.description = (
            "Based on hoomd_organics.forcefields.OPLS_AA. "
            "Trimmed down to include only benzene parameters."
        )


class OPLS_AA_DIMETHYLETHER(BaseXMLForcefield):
    """OPLS All Atom for dimethyl ether molecule forcefield class."""

    def __init__(self, forcefield_files=f"{FF_DIR}/dimethylether_opls.xml"):
        super(OPLS_AA_DIMETHYLETHER, self).__init__(
            forcefield_files=forcefield_files
        )
        self.description = (
            "Based on hoomd_organics.forcefields.OPLS_AA. "
            "Trimmed down to include only dimethyl ether parameters."
        )


class FF_from_file(BaseXMLForcefield):
    """Forcefield class for loading a forcefield from an XML file."""

    def __init__(self, forcefield_files):
        super(FF_from_file, self).__init__(forcefield_files=forcefield_files)
        self.description = "Forcefield loaded from an XML file. "


class BeadSpring(BaseHOOMDForcefield):
    """Bead-spring forcefield class.

    Given a dictionary of bead types, this class creates a list
    `hoomd.md.force.Force` objects to capture bonded and non-bonded
    interactions between the beads.
    For non-bonded interactions, a Lennard-Jones potential is used.
    For bonds and angles, a harmonic potential is used.
    For dihedrals, a periodic potential is used.

    Parameters
    ----------
    r_cut : float, required
        The cutoff radius for the LJ potential.
    beads : dict, required
        A dictionary of bead types. Each bead type should be a dictionary with
        the keys "epsilon" and "sigma" that correspond to the LJ parameters.
    bonds : dict, default None
        A dictionary of bond types separated by a dash. Each bond type should
        be a dictionary with the keys "r0" and "k" that correspond to the
        harmonic bond parameters.
    angles : dict, default None
        A dictionary of angle types separated by a dash. Each angle type should
        be a dictionary with the keys "t0" and "k" that correspond to the
        harmonic angle parameters.
    dihedrals : dict, default None
        A dictionary of dihedral types separated by a dash. Each dihedral type
        should be a dictionary with the keys "phi0", "k", "d", and "n" that
        correspond to the periodic dihedral parameters.
    exclusions : list, default ["bond", "1-3"]
        A list of exclusions to use in the neighbor list. The default is to
        exclude bonded and 1-3 interactions.

    Examples
    --------
    For a simple bead-spring model with two bead types A and B, the following
    code can be used:

    ::

        ff = BeadSpring(r_cut=2.5,
                beads={"A": dict(epsilon=1.0, sigma=1.0),
                       "B": dict(epsilon=2.0, sigma=2.0)},
                bonds={"A-A": dict(r0=1.1, k=300), "A-B": dict(r0=1.1, k=300)},
                angles={"A-A-A": dict(t0=2.0, k=200),
                        "A-B-A": dict(t0=2.0, k=200)},
                dihedrals={"A-A-A-A": dict(phi0=0.0, k=100, d=-1, n=1)})

    """

    def __init__(
        self,
        r_cut,
        beads,
        bonds=None,
        angles=None,
        dihedrals=None,
        exclusions=["bond", "1-3"],
    ):
        self.beads = beads
        self.bonds = bonds
        self.angles = angles
        self.dihedrals = dihedrals
        self.r_cut = r_cut
        self.exclusions = exclusions
        hoomd_forces = self._create_forcefield()
        super(BeadSpring, self).__init__(hoomd_forces)

    def _create_forcefield(self):
        """Create the hoomd force objects."""
        forces = []
        # Create pair force:
        nlist = hoomd.md.nlist.Cell(buffer=0.40, exclusions=self.exclusions)
        lj = hoomd.md.pair.LJ(nlist=nlist)
        bead_types = [key for key in self.beads.keys()]
        all_pairs = list(itertools.combinations_with_replacement(bead_types, 2))
        for pair in all_pairs:
            epsilon0 = self.beads[pair[0]]["epsilon"]
            epsilon1 = self.beads[pair[1]]["epsilon"]
            pair_epsilon = (epsilon0 + epsilon1) / 2

            sigma0 = self.beads[pair[0]]["sigma"]
            sigma1 = self.beads[pair[1]]["sigma"]
            pair_sigma = (sigma0 + sigma1) / 2

            lj.params[pair] = dict(epsilon=pair_epsilon, sigma=pair_sigma)
            lj.r_cut[pair] = self.r_cut
        forces.append(lj)
        # Create bond-stretching force:
        if self.bonds:
            harmonic_bond = hoomd.md.bond.Harmonic()
            for bond_type in self.bonds:
                harmonic_bond.params[bond_type] = self.bonds[bond_type]
            forces.append(harmonic_bond)
        # Create bond-bending force:
        if self.angles:
            harmonic_angle = hoomd.md.angle.Harmonic()
            for angle_type in self.angles:
                harmonic_angle.params[angle_type] = self.angles[angle_type]
            forces.append(harmonic_angle)
        # Create torsion force:
        if self.dihedrals:
            periodic_dihedral = hoomd.md.dihedral.Periodic()
            for dih_type in self.dihedrals:
                periodic_dihedral.params[dih_type] = self.dihedrals[dih_type]
            forces.append(periodic_dihedral)
        return forces
