import hoomd


class HOOMDThermostats:
    """Types of HOOMD thermostats used in NVT or NPT simulations."""

    BERENDSEN = hoomd.md.methods.thermostats.Berendsen
    """The Berendsen thermostat."""
    BUSSI = hoomd.md.methods.thermostats.Bussi
    """The Bussi-Donadio-Parrinello thermostat."""
    MTTK = hoomd.md.methods.thermostats.MTTK
    """The Nosé-Hoover thermostat."""
