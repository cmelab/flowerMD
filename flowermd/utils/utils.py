import numpy as np
import unyt as u

from flowermd.internal import Units

"""utils.py
   utility methods for flowerMD.
"""


def get_target_box_mass_density(
    density,
    mass,
    x_constraint=None,
    y_constraint=None,
    z_constraint=None,
):
    """Utility for calculating box lengths that match a given mass density.

    If no constraints are set, the target box is cubic.
    Setting constraints will hold those box vectors
    constant and adjust others to match the target density.

    Parameters
    ----------
    density : float or unyt.unyt_array or flowermd.internal.Units, required
        The density used to calculate volume.
    mass : float or unyt.unyt_array or flowermd.internal.Units, required
        The mass used to calculate volume.
    x_constraint : float, optional, defualt=None
        Fixes the box length (nm) along the x axis.
    y_constraint : float, optional, default=None
        Fixes the box length (nm) along the y axis.
    z_constraint : float, optional, default=None
        Fixes the box length (nm) along the z axis.
    """
    required_units = Units.kg_m3
    if density.units.dimensions != required_units.dimensions:
        raise ValueError(
            f"The density given has units of {density.units.dimensions} "
            f"but should have units of {required_units.dimensions}."
        )

    if not any([x_constraint, y_constraint, z_constraint]):
        Lx = Ly = Lz = _calculate_box_length(density=density, mass=mass)
    else:
        constraints = np.array([x_constraint, y_constraint, z_constraint])
        fixed_L = constraints[np.not_equal(constraints, None).nonzero()]
        L = _calculate_box_length(density=density, mass=mass, fixed_L=fixed_L)
        constraints[np.equal(constraints, None).nonzero()] = L
        Lx, Ly, Lz = constraints
    return np.array([Lx, Ly, Lz]) * u.Unit(Lx.units)


def get_target_box_number_density(
    density,
    n_beads,
    x_constraint=None,
    y_constraint=None,
    z_constraint=None,
):
    """Utility for calculating box lengths that match a given number
    density.

    If no constraints are set, the target box is cubic.
    Setting constraints will hold those box vectors
    constant and adjust others to match the target density.

    Parameters
    ----------
    density : float, or unyt.unyt_array or flowermd.internal.Units, required
        The density used to calculate volume.
    n_beads : int, required
        The number of beads used to calculate volume.
    x_constraint : float, optional, defualt=None
        Fixes the box length (nm) along the x axis.
    y_constraint : float, optional, default=None
        Fixes the box length (nm) along the y axis.
    z_constraint : float, optional, default=None
        Fixes the box length (nm) along the z axis.
    """
    required_units = u.Unit("m**-3")
    if density.units.dimensions != required_units.dimensions:
        raise ValueError(
            f"The density given has units of {density.units.dimensions} "
            f"but should have units of {required_units.dimensions}."
        )
    if not any([x_constraint, y_constraint, z_constraint]):
        Lx = Ly = Lz = _calculate_box_length(density=density, n_beads=n_beads)
    else:
        constraints = np.array([x_constraint, y_constraint, z_constraint])
        fixed_L = constraints[np.not_equal(constraints, None).nonzero()]
        L = _calculate_box_length(
            density=density, n_beads=n_beads, fixed_L=fixed_L
        )
        constraints[np.equal(constraints, None).nonzero()] = L
        Lx, Ly, Lz = constraints
    return np.array([Lx, Ly, Lz]) * u.Unit(Lx.units)


def _calculate_box_length(density, mass=None, n_beads=None, fixed_L=None):
    """Calculates box lengths that match a given density.

    See `flowermd.utils.get_target_box_mass_density` and
    `flowermd.utils.get_target_box_number_density`

    Parameters
    ----------
    density : unyt.unyt_quantity or flowermd.internal.Units, required
        Target density of the system
    mass : unyt.unyt_quantity or flowermd.internal.Units, optional
        Mass of the system.
        Use for mass density rather than number density.
    n_beads : int, optional
        Number of beads in the system.
        Use for number density rather than mass density.
    fixed_L : np.array, optional, defualt=None
        Array of fixed box lengths to be accounted for
        when solving for L

    Returns
    -------
    L : float
        Box edge length
    """
    # Check units of density
    mass_density = Units.kg_m3
    number_density = Units.n_m3
    if density.units.dimensions == mass_density.dimensions:
        if not mass:
            raise ValueError(
                f"The given density has units of {mass_density.dimensions} "
                "but the mass is not given."
            )
        vol = mass / density
    elif density.units.dimensions == number_density.dimensions:
        if not n_beads:
            raise ValueError(
                f"The given density has units of {number_density.dimensions} "
                "but the number of beads is not given."
            )
        vol = n_beads / density
    else:
        raise ValueError(
            f"Density units of {density.units.dimensions} were given. "
            f"Only mass density ({mass_density.units.dimensions}) and "
            f"number density ({number_density.dimensions}) are supported."
        )
    if fixed_L is None:
        L = vol ** (1 / 3)  # L is in units of volume
    else:
        L = vol / np.prod(fixed_L)
        if fixed_L.size == 1:  # L is units of area
            L = L ** (1 / 2)
    return L
