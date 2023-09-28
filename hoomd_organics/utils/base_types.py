import hoomd


class FF_Types:
    XML = "XML"
    HOOMD = "HOOMD"


class HOOMDThermostats:
    BERENDSEN = hoomd.md.methods.thermostats.Berendsen
    BUSSI = hoomd.md.methods.thermostats.Bussi
    MTTK = hoomd.md.methods.thermostats.MTTK
