import pytest
import unyt as u

from hoomd_organics.utils import check_return_iterable, validate_ref_value
from hoomd_organics.utils.exceptions import ReferenceUnitError


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
