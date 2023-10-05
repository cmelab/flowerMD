import pytest

from jankflow.base import BaseHOOMDForcefield, BaseXMLForcefield
from jankflow.tests import BaseTest


class TestBaseForcefield(BaseTest):
    def test_base_xml_forcefield(self, benzene_xml):
        base_xml_ff = BaseXMLForcefield(forcefield_files=benzene_xml)
        assert base_xml_ff.gmso_ff is not None

    def test_base_xml_forcefield_no_files(self):
        with pytest.raises(TypeError):
            BaseXMLForcefield(forcefield_files=None)

    def test_base_xml_forcefield_name(self):
        base_xml_ff = BaseXMLForcefield(name="oplsaa")
        assert base_xml_ff.gmso_ff is not None

    def test_base_xml_forcefield_invalid_name(self):
        with pytest.raises(Exception):
            BaseXMLForcefield(name="invalid_name")

    def test_base_xml_forcefield_invalid_files(self):
        with pytest.raises(Exception):
            BaseXMLForcefield(forcefield_files="invalid_files")

    def test_base_hoomd_forcefield(self):
        class TestHOOMDFF(BaseHOOMDForcefield):
            def __init__(self):
                hoomd_forces = []
                super().__init__(hoomd_forces=hoomd_forces)

        test_hoomd_ff = TestHOOMDFF()
        assert test_hoomd_ff.hoomd_forces == []

    def test_base_hoomd_forcefield_no_forces(self):
        class TestHOOMDFF(BaseHOOMDForcefield):
            def __init__(self):
                super().__init__(hoomd_forces=None)

        with pytest.raises(NotImplementedError):
            TestHOOMDFF()

    def test_base_hoomd_forcefield_invalid_forces_type(self):
        class TestHOOMDFF(BaseHOOMDForcefield):
            def __init__(self):
                hoomd_forces = "invalid_type"
                super().__init__(hoomd_forces=hoomd_forces)

        with pytest.raises(TypeError):
            TestHOOMDFF()
