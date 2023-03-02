import foyer
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
