import numpy as np
import pytest
import unyt as u

from flowermd.utils import (
    _calculate_box_length,
    check_return_iterable,
    get_target_box_mass_density,
    get_target_box_number_density,
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

    def test_target_box_mass_density(self):
        mass = u.unyt_quantity(4.0, u.g)
        density = u.unyt_quantity(0.5, u.g / u.cm**3)
        target_box = get_target_box_mass_density(density=density, mass=mass)
        assert target_box[0] == target_box[1] == target_box[2]
        assert np.array_equal(target_box, np.array([2 * u.cm] * 3))

    def test_target_box_one_constraint_mass(self):
        mass = u.unyt_quantity(4.0, u.g)
        density = u.unyt_quantity(0.5, u.g / u.cm**3)
        cubic_box = get_target_box_mass_density(density=density, mass=mass)
        tetragonal_box = get_target_box_mass_density(
            density=density, mass=mass, x_constraint=cubic_box[0] / 2
        )
        assert tetragonal_box[1] == tetragonal_box[2]
        assert np.allclose(tetragonal_box[1].value, np.sqrt(8), atol=1e-5)
        assert tetragonal_box[0] == cubic_box[0] / 2

    def test_target_box_two_constraint_mass(self):
        mass = u.unyt_quantity(4.0, u.g)
        density = u.unyt_quantity(0.5, u.g / u.cm**3)
        cubic_box = get_target_box_mass_density(density=density, mass=mass)
        ortho_box = get_target_box_mass_density(
            density=density,
            mass=mass,
            x_constraint=cubic_box[0] / 2,
            y_constraint=cubic_box[0] / 2,
        )
        assert ortho_box[0] == ortho_box[1] != ortho_box[2]
        assert np.allclose(ortho_box[2].value, 8, atol=1e-5)
        assert ortho_box[0] == cubic_box[0] / 2

    def test_target_box_number_density(self):
        sigma = 1 * u.nm
        n_beads = 100
        density = 1 / sigma**3
        target_box = get_target_box_number_density(
            density=density, n_beads=n_beads
        )
        L = target_box[0].value
        assert np.allclose(L**3, 100, atol=1e-8)

    def test_target_box_one_constraint_number_density(self):
        sigma = 1 * u.nm
        n_beads = 100
        density = 1 / sigma**3
        cubic_box = get_target_box_number_density(
            density=density, n_beads=n_beads
        )
        tetragonal_box = get_target_box_number_density(
            density=density,
            n_beads=n_beads,
            x_constraint=cubic_box[0] / 2,
        )
        assert tetragonal_box[1] == tetragonal_box[2] != tetragonal_box[0]
        assert np.allclose(tetragonal_box[1].value, 6.56419787945, atol=1e-5)

    def test_target_box_two_constraint_number_density(self):
        sigma = 1 * u.nm
        n_beads = 100
        density = 1 / sigma**3
        cubic_box = get_target_box_number_density(
            density=density, n_beads=n_beads
        )
        ortho_box = get_target_box_number_density(
            density=density,
            n_beads=n_beads,
            x_constraint=cubic_box[0] / 2,
            y_constraint=cubic_box[0] / 2,
        )
        assert cubic_box[0] / 2 == ortho_box[0] == ortho_box[1] != ortho_box[2]
        assert np.allclose(
            ortho_box[2].value, cubic_box[0].value * 4, atol=1e-5
        )

    def test_calculate_box_length_bad_args(self):
        mass_density = 1 * u.g / (u.cm**3)
        number_density = 1 / (1 * u.nm**3)
        with pytest.raises(ValueError):
            get_target_box_mass_density(density=number_density, mass=100)
        with pytest.raises(ValueError):
            get_target_box_number_density(density=mass_density, n_beads=100)

    def test_calculate_box_length_fixed_l_1d(self):
        mass = u.unyt_quantity(6.0, u.g)
        density = u.unyt_quantity(0.5, u.g / u.cm**3)
        fixed_L = u.unyt_quantity(3.0, u.cm)
        box_length = _calculate_box_length(
            mass=mass, density=density, fixed_L=fixed_L
        )
        assert box_length == 2.0 * u.cm

    def test_calculate_box_length_fixed_l_2d(self):
        mass = u.unyt_quantity(12.0, u.g)
        density = u.unyt_quantity(0.5, u.g / u.cm**3)
        fixed_L = u.unyt_array([3.0, 2.0], u.cm)
        box_length = _calculate_box_length(
            mass=mass, density=density, fixed_L=fixed_L
        )
        assert box_length == 4.0 * u.cm
