from gmso.core.forcefield import ForceField

from hoomd_organics.tests import BaseTest
from hoomd_organics.utils import xml_to_gmso_ff


class TestFFUtils(BaseTest):
    def test_xml_to_gmso_ff(self, benzene_xml):
        gmso_ff = xml_to_gmso_ff(benzene_xml)
        assert isinstance(gmso_ff, ForceField)
