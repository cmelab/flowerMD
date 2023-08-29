import os

import hoomd

from hoomd_organics.library import (
    GAFF,
    OPLS_AA,
    OPLS_AA_BENZENE,
    OPLS_AA_DIMETHYLETHER,
    OPLS_AA_PPS,
    BeadSpring,
    FF_from_file,
)
from hoomd_organics.tests.base_test import ASSETS_DIR


class TestXMLForceFields:
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


class TestCustomForceFields:
    def test_BeadSpring(self):
        ff = BeadSpring(
            r_cut=2.5,
            beads={
                "A": dict(epsilon=1.0, sigma=1.0),
                "B": dict(epsilon=2.0, sigma=2.0),
            },
            bonds={
                "A-A": dict(r0=1.1, k=300),
                "A-B": dict(r0=1.1, k=300),
            },
            angles={"A-A-A": dict(t0=2.0, k=200), "A-B-A": dict(t0=2.0, k=200)},
            dihedrals={"A-A-A-A": dict(phi0=0.0, k=100, d=-1, n=1)},
        )

        assert isinstance(ff.hoomd_forcefield[0], hoomd.md.pair.pair.LJ)
        assert isinstance(ff.hoomd_forcefield[1], hoomd.md.bond.Harmonic)
        assert isinstance(ff.hoomd_forcefield[2], hoomd.md.angle.Harmonic)
        assert isinstance(ff.hoomd_forcefield[3], hoomd.md.dihedral.Periodic)

        pair_types = [("A", "A"), ("A", "B"), ("B", "B")]
        for param in ff.hoomd_forcefield[0].params:
            assert param in pair_types
            if param == ("A", "A"):
                assert ff.hoomd_forcefield[0].params[param]["sigma"] == 1.0
            if param == ("B", "B"):
                assert ff.hoomd_forcefield[0].params[param]["epsilon"] == 2.0
            if param == ("A", "B"):
                assert ff.hoomd_forcefield[0].params[param]["epsilon"] == 1.5

        bond_types = [("A-A"), ("A-B")]
        for param in ff.hoomd_forcefield[1].params:
            assert param in bond_types
            assert ff.hoomd_forcefield[1].params[param]["r0"] == 1.1
            assert ff.hoomd_forcefield[1].params[param]["k"] == 300

        angle_types = [("A-A-A"), ("A-B-A")]
        for param in ff.hoomd_forcefield[2].params:
            assert param in angle_types
            assert ff.hoomd_forcefield[2].params[param]["t0"] == 2.0
            assert ff.hoomd_forcefield[2].params[param]["k"] == 200

        dihedral_types = [("A-A-A-A")]
        for param in ff.hoomd_forcefield[3].params:
            assert param in dihedral_types
            assert ff.hoomd_forcefield[3].params[param]["phi0"] == 0.0
            assert ff.hoomd_forcefield[3].params[param]["k"] == 100
            assert ff.hoomd_forcefield[3].params[param]["d"] == -1
            assert ff.hoomd_forcefield[3].params[param]["n"] == 1

    def test_TorchCustomForce(self):
        # TODO: Train a simple LJ and add it to BaseTest. Then test it here.
        pass
