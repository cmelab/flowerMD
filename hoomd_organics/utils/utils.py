import numpy as np
import unyt as u

from hoomd_organics.utils.exceptions import ReferenceUnitError


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


def calculate_box_length(mass, density, fixed_L=None):
    """Calculates the required box length(s) given the
    mass of a sytem and the target density.

    Box edge length constraints can be set by set_target_box().
    If constraints are set, this will solve for the required
    lengths of the remaining non-constrained edges to match
    the target density.

    Parameters
    ----------
    mass : unyt.unyt_quantity, required
        Mass of the system
    density : unyt.unyt_quantity, required
        Target density of the system
    fixed_L : np.array, optional, defualt=None
        Array of fixed box lengths to be accounted for
        when solving for L

    Returns
    -------
    L : float
        Box edge length
    """

    vol = mass / density  # cm^3
    if fixed_L is None:
        L = vol ** (1 / 3)
    else:
        L = vol / np.prod(fixed_L)
        if fixed_L.size == 1:  # L is cm^2
            L = L ** (1 / 2)
    # L is cm
    return L
