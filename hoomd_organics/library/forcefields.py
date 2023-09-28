"""All pre-defined forcefield classes for use in hoomd_organics."""
import itertools

import forcefield_utilities as ffutils
import foyer
import hoomd
import numpy as np

from hoomd_organics.assets import FF_DIR


class GAFF(foyer.Forcefield):
    """GAFF forcefield class."""

    def __init__(self, forcefield_files=f"{FF_DIR}/gaff.xml"):
        super(GAFF, self).__init__(forcefield_files=forcefield_files)
        self.description = (
            "The General Amber Forcefield written in foyer XML format. "
            "The XML file was obtained from the antefoyer package: "
            "https://github.com/rsdefever/antefoyer/tree/master/antefoyer"
        )
        self.gmso_ff = ffutils.FoyerFFs().load(forcefield_files).to_gmso_ff()


class OPLS_AA(foyer.Forcefield):
    """OPLS All Atom forcefield class."""

    def __init__(self, name="oplsaa"):
        super(OPLS_AA, self).__init__(name=name)
        self.description = "opls-aa forcefield found in the Foyer package."
        self.gmso_ff = ffutils.FoyerFFs().load(name).to_gmso_ff()


class OPLS_AA_PPS(foyer.Forcefield):
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
        self.gmso_ff = ffutils.FoyerFFs().load(forcefield_files).to_gmso_ff()


class OPLS_AA_BENZENE(foyer.Forcefield):
    """OPLS All Atom for benzene molecule forcefield class."""

    def __init__(self, forcefield_files=f"{FF_DIR}/benzene_opls.xml"):
        super(OPLS_AA_BENZENE, self).__init__(forcefield_files=forcefield_files)
        self.description = (
            "Based on hoomd_organics.forcefields.OPLS_AA. "
            "Trimmed down to include only benzene parameters."
        )
        self.gmso_ff = ffutils.FoyerFFs().load(forcefield_files).to_gmso_ff()


class OPLS_AA_DIMETHYLETHER(foyer.Forcefield):
    """OPLS All Atom for dimethyl ether molecule forcefield class."""

    def __init__(self, forcefield_files=f"{FF_DIR}/dimethylether_opls.xml"):
        super(OPLS_AA_DIMETHYLETHER, self).__init__(
            forcefield_files=forcefield_files
        )
        self.description = (
            "Based on hoomd_organics.forcefields.OPLS_AA. "
            "Trimmed down to include only dimethyl ether parameters."
        )
        self.gmso_ff = ffutils.FoyerFFs().load(forcefield_files).to_gmso_ff()


class FF_from_file(foyer.Forcefield):
    """Forcefield class for loading a forcefield from an XML file."""

    def __init__(self, xml_file):
        super(FF_from_file, self).__init__(forcefield_files=xml_file)
        self.gmso_ff = ffutils.FoyerFFs().load(xml_file).to_gmso_ff()


class BeadSpring:
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
        self.hoomd_forcefield = self._create_forcefield()

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


class TableForcefield:
    """Create a set of hoomd table potentials.

    This class provides an interface for creating hoomd table
    potentials either from arrays of energy and forces, or
    from files storing the tabulated energy and forces.

    In HOOMD-Blue, table potentials are available for:

        * Pairs: `hoomd.md.pair.Table`
        * Bonds: `hoomd.md.bond.Table`
        * Angles: `hoomd.md.angle.Table`
        * Dihedrals: `hoomd.md.dihedral.Table`

    Notes
    -----
    HOOMD table potentials are initialized using arrays of energy and forces.
    It may be most convenient to store tabulated data in files,
    in that case use the `from_files` method.


    Parameters
    ----------
    pairs: dict, optional, default None
    bonds: dict, optional, default None
    angles: dict, optional, default None
    dihedrals: dict, optional, default None

    Methods
    -------
    from_files: Create table potentials from a given `type: file_path` mapping.
        Use this method when the tabulated data
        is stored in text or binary numpy arrays

    """

    def __init__(
        self,
        pairs=None,
        bonds=None,
        angles=None,
        dihedrals=None,
        r_min=None,
        r_cut=None,
        exclusions=["bond", "1-3"],
        nlist_buffer=0.40,
    ):
        self.pairs = pairs
        self.bonds = bonds
        self.angles = angles
        self.dihedrals = dihedrals
        self.r_min = r_min
        self.r_cut = r_cut
        self.exclusions = exclusions
        self.nlist_buffer = nlist_buffer
        self.bond_width, self.angle_width, self.dih_width = self._check_widths()
        self.hoomd_forcefield = self._create_forcefield()

    @classmethod
    def from_files(
        cls,
        pairs=None,
        bonds=None,
        angles=None,
        dihedrals=None,
        exclusions=["bond", "1-3"],
        nlist_buffer=0.40,
    ):
        """Create table forefield using a `type: file_path` mapping.

        Parameters
        ----------
        pairs: dict, optional, default None
        bonds: dict, optional, default None
        angles: dict, optional, default None
        dihedrals: dict, optional, default None

        Notes
        -----
        The parameters must use a `{"type": "file_path"}` mapping.
        Following HOOMD conventions, pair types must be given as a tuple
        of `("type1", "type2")` while bonds, angles and dihedrals
        are givne as strings of `"type1-type2-type3"`

        Example
        -------
        ```
        table_forcefield = TableForcefield.from_files(
            pairs = {
                ("A", "A"): "A_pairs.txt
                ("B", "B"): "B_pairs.txt
                ("A", "B"): "AB_pairs.txt
            },
            bonds = {"A-A": "A_bonds.txt", "B-B": "B_bonds.txt"},
            angles = {"A-A-A": "A_angles.txt", "B-B-B": "B_angles.txt"},
        )
        ```

        """

        def _load_file(file):
            """Call the correct numpy method."""
            if file.split(".")[-1] in ["txt", "csv"]:
                return np.loadtxt(file)
            elif file.split(".")[-1] == "npy":
                return np.load(file)

        # Read pair files
        pair_dict = dict()
        pair_r_min = set()
        pair_r_max = set()
        if pairs:
            for pair_type in pairs:
                table = _load_file(pairs[pair_type])
                r = table[:, 0]
                pair_r_min.add(r[0])
                pair_r_max.add(r[-1])
                pair_dict[pair_type] = dict()
                pair_dict[pair_type]["U"] = table[:, 1]
                pair_dict[pair_type]["F"] = table[:, 2]
            if len(pair_r_min) != len(pair_r_max) != 1:
                raise ValueError(
                    "All pair files must have the same r-range values"
                )
        # Read bond files
        bond_dict = dict()
        if bonds:
            for bond_type in bonds:
                table = _load_file(bonds[bond_type])
                r = table[:, 0]
                r_min = r[0]
                r_max = r[-1]
                bond_dict[bond_type] = dict()
                bond_dict[bond_type]["r_min"] = r_min
                bond_dict[bond_type]["r_max"] = r_max
                bond_dict[bond_type]["U"] = table[:, 1]
                bond_dict[bond_type]["F"] = table[:, 2]
        # Read angle files
        angle_dict = dict()
        if angles:
            for angle_type in angles:
                table = _load_file(angles[angle_type])
                thetas = table[:, 0]
                if thetas[0] != 0 or not np.allclose(
                    thetas[-1], np.pi, atol=1e-5
                ):
                    raise ValueError(
                        "Angle values must be evenly spaced and "
                        "range from 0 to Pi."
                    )
                angle_dict[angle_type] = dict()
                angle_dict[angle_type]["U"] = table[:, 1]
                angle_dict[angle_type]["F"] = table[:, 2]
        # Read dihedral files
        dih_dict = dict()
        if dihedrals:
            for dih_type in dihedrals:
                table = _load_file(dihedrals[dih_type])
                thetas = table[:, 0]
                if not np.allclose(
                    thetas[0], -np.pi, atol=1e-5
                ) or not np.allclose(thetas[-1], np.pi, atol=1e-5):
                    raise ValueError(
                        "Dihedral angle values must be evenly spaced and "
                        "range from -Pi to Pi."
                    )
                dih_dict[dih_type] = dict()
                dih_dict[dih_type]["U"] = table[:, 1]
                dih_dict[dih_type]["F"] = table[:, 2]

        return cls(
            pairs=pair_dict,
            bonds=bond_dict,
            angles=angle_dict,
            dihedrals=dih_dict,
            r_min=list(pair_r_min)[0],
            r_cut=list(pair_r_max)[0],
            exclusions=exclusions,
        )

    def _create_forcefield(self):
        forces = []
        # Create pair forces
        if self.pairs:
            nlist = hoomd.md.nlist.Cell(
                buffer=self.nlist_buffer, exclusions=self.exclusions
            )
            pair_table = hoomd.md.pair.Table(
                nlist=nlist, default_r_cut=self.r_cut
            )
            for pair_type in self.pairs:
                U = self.pairs[pair_type]["U"]
                F = self.pairs[pair_type]["F"]
                if len(U) != len(F):
                    raise ValueError(
                        "The energy and force arrays are not the same size."
                    )
                pair_table.params[tuple(pair_type)] = dict(
                    r_min=self.r_min, U=U, F=F
                )
            forces.append(pair_table)
        # Create bond forces
        if self.bonds:
            bond_table = hoomd.md.bond.Table(width=self.bond_width)
            for bond_type in self.bonds:
                bond_table.params[tuple(bond_type)] = dict(
                    r_min=self.bonds[bond_type]["r_min"],
                    r_max=self.bonds[bond_type]["r_max"],
                    U=self.bonds[bond_type]["U"],
                    F=self.bonds[bond_type]["F"],
                )
            forces.append(bond_table)
        # Create angle forces
        if self.angles:
            angle_table = hoomd.md.angle.Table(width=self.angle_width)
            for angle_type in self.angles:
                angle_table.params[angle_type] = dict(
                    U=self.angles[angle_type]["U"],
                    tau=self.angles[angle_type]["F"],
                )
            forces.append(angle_table)
        # Create dihedral forces
        if self.dihedrals:
            dih_table = hoomd.md.dihedral.Table(width=self.dih_width)
            for dih_type in self.dihedrals:
                dih_table.params[dih_type] = dict(
                    U=self.dihedrals[dih_type]["U"],
                    tau=self.dihedrals[dih_type]["F"],
                )
            forces.append(dih_table)
        return forces

    def _check_widths(self):
        """Check number of points for bonds, pairs and angles."""
        bond_width = None
        for bond_type in self.bonds:
            new_width = len(self.bonds[bond_type]["U"])
            if bond_width is None:
                bond_width = new_width
            else:
                if new_width != bond_width:
                    raise ValueError(
                        "All bond types must have the same "
                        "number of points for table energies and forces."
                    )

        angle_width = None
        for angle_type in self.angles:
            new_width = len(self.angles[angle_type]["U"])
            if angle_width is None:
                angle_width = new_width
            else:
                if new_width != angle_width:
                    raise ValueError(
                        "All angle types must have the same "
                        "number of points for table energies and forces."
                    )

        dih_width = None
        for dih_type in self.dihedrals:
            new_width = len(self.dihedrals[dih_type]["U"])
            if dih_width is None:
                dih_width = new_width
            else:
                if new_width != dih_width:
                    raise ValueError(
                        "All dihedral types must have the same "
                        "number of points for table energies and forces."
                    )
        return bond_width, angle_width, dih_width
