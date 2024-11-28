import numpy as np
import pytest
import unyt as u

from flowermd import Units
from flowermd.internal import check_return_iterable, validate_unit
from flowermd.internal.exceptions import UnitError
from flowermd.utils import (
    _calculate_box_length,
    get_target_box_mass_density,
    get_target_box_number_density,
)


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

    def test_validate_unit(self):
        value = 1.0 * Units.g
        dimension = u.dimensions.mass
        assert validate_unit(value, dimension) == value
        value = 1.0 * Units.kcal_mol
        dimension = u.dimensions.energy
        assert validate_unit(value, dimension) == value
        value = 1.0 * Units.angstrom
        dimension = u.dimensions.length
        assert validate_unit(value, dimension) == value
        value = 1.0 * Units.ps
        dimension = u.dimensions.time
        assert validate_unit(value, dimension) == value
        value = 300 * Units.K
        dimension = u.dimensions.temperature
        assert validate_unit(value, dimension) == value
        with pytest.raises(UnitError):
            validate_unit(1.0 * Units.nm, u.dimensions.mass)
        with pytest.raises(UnitError):
            validate_unit(1.0 * Units.g, u.dimensions.length)
        with pytest.raises(UnitError):
            validate_unit(1.0, u.dimensions.energy)

    def test_target_box_mass_density(self):
        mass = 4 * Units.g
        density = 0.5 * Units.g_cm3
        target_box = get_target_box_mass_density(density=density, mass=mass)
        assert target_box[0] == target_box[1] == target_box[2]
        assert np.array_equal(target_box, np.array([2] * 3) * Units.cm)

    def test_target_box_one_constraint_mass(self):
        mass = 4 * Units.g
        density = 0.5 * Units.g_cm3
        cubic_box = get_target_box_mass_density(density=density, mass=mass)
        tetragonal_box = get_target_box_mass_density(
            density=density, mass=mass, x_constraint=cubic_box[0] / 2
        )
        assert tetragonal_box[1] == tetragonal_box[2]
        assert np.allclose(tetragonal_box[1].value, np.sqrt(8), atol=1e-5)
        assert tetragonal_box[0] == cubic_box[0] / 2

    def test_target_box_two_constraint_mass(self):
        mass = 4 * Units.g
        density = 0.5 * Units.g_cm3
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
        sigma = 1
        n_beads = 100
        density = sigma * Units.n_nm3
        target_box = get_target_box_number_density(
            density=density, n_beads=n_beads
        )
        L = target_box[0].value
        assert np.allclose(L**3, 100, atol=1e-8)

    def test_target_box_one_constraint_number_density(self):
        sigma = 1
        n_beads = 100
        density = sigma * Units.n_nm3
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
        n_beads = 100
        density = 1 * Units.n_nm3
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
        mass_density = 1 * Units.g_cm3
        number_density = 1 * Units.n_nm3
        with pytest.raises(ValueError):
            get_target_box_mass_density(density=number_density, mass=100)
        with pytest.raises(ValueError):
            get_target_box_number_density(density=mass_density, n_beads=100)

    def test_calculate_box_length_fixed_l_1d(self):
        mass = 6.0 * Units.g
        density = 0.5 * Units.g_cm3
        fixed_L = 3.0 * Units.cm
        box_length = _calculate_box_length(
            mass=mass, density=density, fixed_L=fixed_L
        )
        assert box_length == 2.0 * Units.cm

    def test_calculate_box_length_fixed_l_2d(self):
        mass = 12.0 * Units.g
        density = 0.5 * Units.g_cm3
        fixed_L = [3.0, 2.0] * Units.cm
        box_length = _calculate_box_length(
            mass=mass, density=density, fixed_L=fixed_L
        )
        assert box_length == 4.0 * Units.cm
