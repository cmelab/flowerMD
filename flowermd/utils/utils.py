import numpy as np
import unyt as u

from flowermd.utils.exceptions import ReferenceUnitError

"""utils.py
   utility methods for flowerMD.
"""


def check_return_iterable(obj):
    if isinstance(obj, dict):
        return [obj]
    if isinstance(obj, str):
        return [obj]
    try:
        iter(obj)
        return obj
    except:  # noqa: E722
        return [obj]


def validate_ref_value(ref_value, dimension):
    """Validates the reference value and checks the unit dimension.
    This function validates the reference value. The reference value can be
    provided in three ways:
        1. An unyt_quantity instance.
        2. A string with the value and unit , for example "1.0 g".
        3. A string with the value and unit separated by a "/", for example
        "1.0 kcal/mol".
    Parameters
    ----------
    ref_value : unyt_quantity or str; required
        The reference value.
    dimension : unyt_dimension; required
        The dimension of the reference value.

    Returns
    -------
    The validated reference value as an unyt.unyt_quantity instance.
    """

    def _is_valid_dimension(ref_unit):
        if ref_unit.dimensions != dimension:
            raise ReferenceUnitError(
                f"Invalid unit dimension. The reference "
                f"value must be in {dimension} "
                f"dimension."
            )
        return True

    def _parse_and_validate_unit(value, unit_str):
        if hasattr(u, unit_str):
            if unit_str == "amu":
                u_unit = u.Unit("amu")
            else:
                u_unit = getattr(u, unit_str)
            if _is_valid_dimension(u_unit):
                return float(value) * u_unit
        # if the unit contains "/" character, for example "g/mol", check if
        # the unit is a valid unit and has the correct dimension.
        if len(unit_str.split("/")) == 2:
            unit1, unit2 = unit_str.split("/")
            if hasattr(u, unit1) and hasattr(u, unit2):
                comb_unit = getattr(u, unit1) / getattr(u, unit2)
                if _is_valid_dimension(comb_unit):
                    return float(value) * comb_unit
        raise ReferenceUnitError(
            f"Invalid reference value. Please provide "
            f"a reference value with unit of "
            f"{dimension} dimension."
        )

    def _is_float(num):
        try:
            return float(num)
        except ValueError:
            raise ValueError("The reference value is not a number.")

    # if ref_value is an instance of unyt_quantity, check the dimension.
    if isinstance(ref_value, u.unyt_quantity) and _is_valid_dimension(
        ref_value.units
    ):
        return ref_value
    # if ref_value is a string, check if it is a number and if it is, check if
    # the unit exists in unyt and has the correct dimension.
    elif isinstance(ref_value, str) and len(ref_value.split()) == 2:
        value, unit_str = ref_value.split()
        value = _is_float(value)
        return _parse_and_validate_unit(value, unit_str)
    else:
        raise ReferenceUnitError(
            f"Invalid reference value. Please provide "
            f"a reference value with unit of "
            f"{dimension} dimension."
        )


def get_target_box_mass_density(
    density,
    mass,
    x_constraint=None,
    y_constraint=None,
    z_constraint=None,
):
    """Helper function to calculate box lengths that match a given mass density.

    If no constraints are set, the target box is cubic.
    Setting constraints will hold those box vectors
    constant and adjust others to match the target density.

    Parameters
    ----------
    density : float, or unyt.unyt_array, required
        The density used to calculate volume.
    mass : float, or unyt.unyt_array, required
        The mass used to calculate volume.
    x_constraint : float, optional, defualt=None
        Fixes the box length (nm) along the x axis.
    y_constraint : float, optional, default=None
        Fixes the box length (nm) along the y axis.
    z_constraint : float, optional, default=None
        Fixes the box length (nm) along the z axis.
    """
    required_units = u.Unit("kg") / u.Unit("m**3")
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
    """Helper function to calculate box lengths that match a given number
    density.

    If no constraints are set, the target box is cubic.
    Setting constraints will hold those box vectors
    constant and adjust others to match the target density.

    Parameters
    ----------
    density : float, or unyt.unyt_array, required
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
    """Helper function to calculate box lengths that match a given density.

    See `flowermd.utils.get_target_box_mass_density` and
    `flowermd.utils.get_target_box_number_density`

    Parameters
    ----------
    density : unyt.unyt_quantity, required
        Target density of the system
    mass : unyt.unyt_quantity, optional
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
    mass_density = u.Unit("kg") / u.Unit("m**3")
    number_density = u.Unit("m**-3")
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
