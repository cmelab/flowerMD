import forcefield_utilities as ffutils
import foyer

from hoomd_organics.assets import FF_DIR


class GAFF(foyer.Forcefield):
    def __init__(self, forcefield_files=f"{FF_DIR}/gaff.xml"):
        super(GAFF, self).__init__(forcefield_files=forcefield_files)
        self.description = (
            "The General Amber Forcefield written in foyer XML format. "
            "The XML file was obtained from the antefoyer package: "
            "https://github.com/rsdefever/antefoyer/tree/master/antefoyer"
        )
        self.gmso_ff = ffutils.FoyerFFs().load(forcefield_files).to_gmso_ff()


class OPLS_AA(foyer.Forcefield):
    def __init__(self, name="oplsaa"):
        super(OPLS_AA, self).__init__(name=name)
        self.description = "opls-aa forcefield found in the Foyer package."
        self.gmso_ff = ffutils.FoyerFFs().load(name).to_gmso_ff()


class OPLS_AA_PPS(foyer.Forcefield):
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
    def __init__(self, forcefield_files=f"{FF_DIR}/benzene_opls.xml"):
        super(OPLS_AA_BENZENE, self).__init__(forcefield_files=forcefield_files)
        self.description = (
            "Based on hoomd_organics.forcefields.OPLS_AA. "
            "Trimmed down to include only benzene parameters."
        )
        self.gmso_ff = ffutils.FoyerFFs().load(forcefield_files).to_gmso_ff()


class OPLS_AA_DIMETHYLETHER(foyer.Forcefield):
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
    def __init__(self, xml_file):
        super(FF_from_file, self).__init__(forcefield_files=xml_file)
        self.gmso_ff = ffutils.FoyerFFs().load(xml_file).to_gmso_ff()
