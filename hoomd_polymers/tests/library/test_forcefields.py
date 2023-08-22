import os

from hoomd_polymers.library import (
    GAFF,
    OPLS_AA,
    OPLS_AA_BENZENE,
    OPLS_AA_DIMETHYLETHER,
    OPLS_AA_PPS,
    FF_from_file,
)
from hoomd_polymers.tests.base_test import ASSETS_DIR


class TestForceFields:
    def test_GAFF(self):
        ff = GAFF()
        assert ff.gmso_ff is not None

    def test_OPLS_AA(self):
        ff = OPLS_AA()
        assert ff.gmso_ff is not None

    def test_OPLS_AA_PPS(self):
        ff = OPLS_AA_PPS()
        assert ff.gmso_ff is not None

    def test_OPPLS_AA_BENZENE(self):
        ff = OPLS_AA_BENZENE()
        assert ff.gmso_ff is not None

    def test_OPPLS_AA_DIMETHYLETHER(self):
        ff = OPLS_AA_DIMETHYLETHER()
        assert ff.gmso_ff is not None

    def test_FF_from_file(self):
        xml_file = os.path.join(ASSETS_DIR, "test_ff.xml")
        ff = FF_from_file(xml_file)
        assert ff.gmso_ff is not None

    def test_BeadSpring(self):
        return NotImplementedError
