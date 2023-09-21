import hoomd


class FF_Types:
    opls = "opls"
    pps_opls = "pps_opls"
    oplsaa = "oplsaa"
    gaff = "gaff"
    custom = "custom"
    Hoomd = "Hoomd"


class HOOMDThermostats:
    BERENDSEN = hoomd.md.methods.thermostats.Berendsen
    BUSSI = hoomd.md.methods.thermostats.Bussi
    MTTK = hoomd.md.methods.thermostats.MTTK
