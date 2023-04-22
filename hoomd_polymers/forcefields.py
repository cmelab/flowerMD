import foyer
import hoomd
import itertools


from hoomd_polymers.library import FF_DIR


class GAFF(foyer.Forcefield):
    def __init__(self, forcefield_files=f"{FF_DIR}/gaff.xml"):
        super(GAFF, self).__init__(forcefield_files=forcefield_files)
        self.description = (
                "The General Amber Forcefield written in foyer XML format. "
                "The XML file was obtained from the antefoyer package: "
                "https://github.com/rsdefever/antefoyer/tree/master/antefoyer"
        )


class OPLS_AA(foyer.Forcefield):
    def __init__(self, name="oplsaa"):
        super(OPLS_AA, self).__init__(name=name)
        self.description = (
                "opls-aa forcefield found in the Foyer package."
        )


class OPLS_AA_PPS(foyer.Forcefield):
    def __init__(self, forcefield_files=f"{FF_DIR}/pps_opls.xml"):
        super(OPLS_AA_PPS, self).__init__(forcefield_files=forcefield_files)
        self.description = (
               "Based on hoomd_polymers.forcefields.OPLS_AA. "
                "Trimmed down to include only PPS parameters. "
                "One missing parameter was added manually: "
                "<Angle class1=CA class2=S class3=CA angle=1.805 k=627.6/> "
                "The equilibrium angle was determined from "
                "experimental PPS papers. The spring constant taken "
                "from the equivalent angle in GAFF."
		)


class FF_from_file(foyer.Forcefield):
    def __init__(self, xml_file):
        super(FF_from_file, self).__init__(forcefield_files=xml_file)


class BeadSpring:
    def __init__(self, r_cut, beads, bonds=None, angles=None, dihedrals=None):
        self.beads = beads
        self.bonds = bonds
        self.angles = angles
        self.dihedrals = dihedrals
        self.r_cut = r_cut
        self.hoomd_forcefield = self._create_forcefield() 
        
    def _create_forcefield(self):
        forces = []
        # Create pair force:
        lj = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell(buffer=0.40))
        bead_types = [key for key in self.beads.keys()]
        all_pairs = list(
                itertools.combinations_with_replacement(bead_types, 2)
        )
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
                period_dihedral.params[dih_type] = self.dihedrals[dih_type]
            forces.append(periodic_dihedral)
        return forces
