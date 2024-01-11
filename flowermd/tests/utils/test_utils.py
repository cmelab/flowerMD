import pytest
import unyt as u

from flowermd.utils import (
    calculate_box_length,
    check_return_iterable,
    validate_ref_value,
)
from flowermd.utils.exceptions import ReferenceUnitError


class TestUtils:
    def test_check_return_iterable(self):
        assert check_return_iterable("test") == ["test"]
        assert check_return_iterable(["test"]) == ["test"]
        assert check_return_iterable({"test": "test"}) == [{"test": "test"}]
        assert check_return_iterable(1) == [1]
        assert check_return_iterable(1.0) == [1.0]
        assert check_return_iterable([1, 2, 3]) == [1, 2, 3]
        assert check_return_iterable({"test": 1}) == [{"test": 1}]
        assert check_return_iterable({"test": 1, "test2": 2}) == [
            {"test": 1, "test2": 2}
        ]

    def test_validate_ref_value(self):
        assert validate_ref_value(1.0 * u.g, u.dimensions.mass) == 1.0 * u.g
        assert validate_ref_value("1.0 g", u.dimensions.mass) == 1.0 * u.g
        assert validate_ref_value("1.0 kcal/mol", u.dimensions.energy) == (
            1.0 * u.kcal / u.mol
        )
        assert validate_ref_value("1.0 amu", u.dimensions.mass) == 1.0 * u.Unit(
            "amu"
        )
        with pytest.raises(ReferenceUnitError):
            validate_ref_value("1.0 g", u.dimensions.energy)

        with pytest.raises(ReferenceUnitError):
            validate_ref_value("1.0 kcal/invalid", u.dimensions.energy)

        with pytest.raises(ReferenceUnitError):
            validate_ref_value("1.0 invalid", u.dimensions.energy)

        with pytest.raises(ValueError):
            validate_ref_value("test g", u.dimensions.mass)

    def test_calculate_box_length_mass_density(self):
        mass = u.unyt_quantity(4.0, u.g)
        density = u.unyt_quantity(0.5, u.g / u.cm**3)
        box_length = calculate_box_length(density=density, mass=mass)
        assert box_length == 2.0 * u.cm

    def test_calculate_box_length_number_density(self):
        sigma = 1 * u.nm
        n_beads = 100
        density = 1 / sigma**3
        box_length = calculate_box_length(density=density, n_beads=n_beads)
        assert box_length == 100 ** (1 / 3) * u.nm

    def test_calculate_box_length_bad_args(self):
        mass_density = 1 * u.g / (u.cm**3)
        number_density = 1 / (1 * u.nm) ** 3
        invalid_density = 1 * u.J / u.kg
        with pytest.raises(ValueError):
            calculate_box_length(density=mass_density, n_beads=100)
        with pytest.raises(ValueError):
            calculate_box_length(density=number_density, mass=100)
        with pytest.raises(ValueError):
            calculate_box_length(density=invalid_density, mass=100)

    def test_calculate_box_length_fixed_l_1d(self):
        mass = u.unyt_quantity(6.0, u.g)
        density = u.unyt_quantity(0.5, u.g / u.cm**3)
        fixed_L = u.unyt_quantity(3.0, u.cm)
        box_length = calculate_box_length(
            mass=mass, density=density, fixed_L=fixed_L
        )
        assert box_length == 2.0 * u.cm

    def test_calculate_box_length_fixed_l_2d(self):
        mass = u.unyt_quantity(12.0, u.g)
        density = u.unyt_quantity(0.5, u.g / u.cm**3)
        fixed_L = u.unyt_array([3.0, 2.0], u.cm)
        box_length = calculate_box_length(
            mass=mass, density=density, fixed_L=fixed_L
        )
        assert box_length == 4.0 * u.cm
