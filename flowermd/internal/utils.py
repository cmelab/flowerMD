import unyt
import unyt as u

from flowermd.internal import Units
from flowermd.internal.exceptions import UnitError

"""utils.py
   Internal utility methods for flowerMD.
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


def validate_unit(value, dimension):
    """Validate the unit and checks the unit dimension.

    Parameters
    ----------
    value : `unit value * flowermd.Units`; required
        The unit value to be validated.
    dimension : unyt_dimension; required
        The dimension of the unit.
    """

    def _sample_unit_str(dimension):
        if dimension == u.dimensions.temperature:
            return "temperature = 300 * flowermd.Units.K"
        elif dimension == u.dimensions.mass:
            return "mass = 1.0 * flowermd.Units.g"
        elif dimension == u.dimensions.length:
            return "length = 1.0 * flowermd.Units.angstrom"
        elif dimension == u.dimensions.time:
            return "time = 1.0 * flowermd.Units.ps"
        elif dimension == u.dimensions.energy:
            return "energy = 1.0 * flowermd.Units.kcal_mol"

    if isinstance(value, (u.unyt_quantity, unyt.unyt_array, Units)):
        if value.units.dimensions != dimension:
            raise UnitError(
                f"Invalid unit dimension. The unit must be in "
                f"{dimension} dimension. Check `flowermd.Units` for "
                f"valid units."
            )
        return value
    else:
        raise UnitError(
            "The unit value must be provided from the "
            "`flowermd.Units` class. For example, "
            f"{_sample_unit_str(dimension)}. Check "
            "`flowermd.Units` for valid units."
        )
