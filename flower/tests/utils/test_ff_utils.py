import os

import pytest
from gmso.core.forcefield import ForceField

from flower.tests import BaseTest
from flower.utils.ff_utils import find_xml_ff, xml_to_gmso_ff


class TestFFUtils(BaseTest):
    def test_find_xml_ff(self):
        ff_xml_path = find_xml_ff("oplsaa.xml")
        assert os.path.exists(ff_xml_path)

    def test_find_xml_only_file_name(self):
        ff_xml_path = find_xml_ff("oplsaa")
        assert os.path.exists(ff_xml_path)

    def test_find_xml_ff_path(self, benzene_xml):
        ff_xml_path = find_xml_ff(benzene_xml)
        assert os.path.exists(ff_xml_path)

    def test_find_xml_invalid_extension(self):
        with pytest.raises(ValueError):
            find_xml_ff("oplsaa.txt")

    def test_find_xml_invalid_file(self, benzene_mol2):
        with pytest.raises(ValueError):
            find_xml_ff(benzene_mol2)

    def test_find_xml_not_supported_name(self):
        with pytest.raises(ValueError):
            find_xml_ff("oplsaa2")

    def test_find_xml_not_supported_path(self):
        with pytest.raises(ValueError):
            find_xml_ff("oplsaa2.xml")

    def test_xml_to_gmso_ff(self, benzene_xml):
        gmso_ff = xml_to_gmso_ff(benzene_xml)
        assert isinstance(gmso_ff, ForceField)
