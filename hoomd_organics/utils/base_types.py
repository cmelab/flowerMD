import hoomd


class HOOMDThermostats:
    BERENDSEN = hoomd.md.methods.thermostats.Berendsen
    BUSSI = hoomd.md.methods.thermostats.Bussi
    MTTK = hoomd.md.methods.thermostats.MTTK
