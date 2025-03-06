import os

import hoomd
import numpy as np
import pytest

from flowermd.library import (
    GAFF,
    OPLS_AA,
    OPLS_AA_BENZENE,
    OPLS_AA_DIMETHYLETHER,
    OPLS_AA_PPS,
    BeadSpring,
    EllipsoidForcefield,
    FF_from_file,
    KremerGrestBeadSpring,
    TableForcefield,
)
from flowermd.tests.base_test import ASSETS_DIR


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

    def test_KremerGrestBeadSpring(self):
        ff = KremerGrestBeadSpring(bond_k=10, bond_max=2.0)
        assert isinstance(ff.hoomd_forces[0], hoomd.md.pair.LJ)
        assert isinstance(ff.hoomd_forces[1], hoomd.md.bond.FENEWCA)
        assert np.round(ff.hoomd_forces[0].r_cut[("A", "A")], 2) == 1.12
        assert ff.bond_type == "A-A"
        assert ff.pair == ("A", "A")
        ff2 = KremerGrestBeadSpring(bond_k=10, bond_max=2.0, sigma=2)
        assert np.round(ff2.hoomd_forces[0].r_cut[("A", "A")], 2) == 2 * 1.12

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

        assert isinstance(ff.hoomd_forces[0], hoomd.md.pair.pair.LJ)
        assert isinstance(ff.hoomd_forces[1], hoomd.md.bond.Harmonic)
        assert isinstance(ff.hoomd_forces[2], hoomd.md.angle.Harmonic)
        assert isinstance(ff.hoomd_forces[3], hoomd.md.dihedral.Periodic)

        pair_types = [("A", "A"), ("A", "B"), ("B", "B")]
        for param in ff.hoomd_forces[0].params:
            assert param in pair_types
            if param == ("A", "A"):
                assert ff.hoomd_forces[0].params[param]["sigma"] == 1.0
            if param == ("B", "B"):
                assert ff.hoomd_forces[0].params[param]["epsilon"] == 2.0
            if param == ("A", "B"):
                assert ff.hoomd_forces[0].params[param]["epsilon"] == 1.5

        bond_types = [("A-A"), ("A-B")]
        for param in ff.hoomd_forces[1].params:
            assert param in bond_types
            assert ff.hoomd_forces[1].params[param]["r0"] == 1.1
            assert ff.hoomd_forces[1].params[param]["k"] == 300

        angle_types = [("A-A-A"), ("A-B-A")]
        for param in ff.hoomd_forces[2].params:
            assert param in angle_types
            assert ff.hoomd_forces[2].params[param]["t0"] == 2.0
            assert ff.hoomd_forces[2].params[param]["k"] == 200

        dihedral_types = [("A-A-A-A")]
        for param in ff.hoomd_forces[3].params:
            assert param in dihedral_types
            assert ff.hoomd_forces[3].params[param]["phi0"] == 0.0
            assert ff.hoomd_forces[3].params[param]["k"] == 100
            assert ff.hoomd_forces[3].params[param]["d"] == -1
            assert ff.hoomd_forces[3].params[param]["n"] == 1

    def test_ellipsoid_ff(self):
        ellipsoid_ff = EllipsoidForcefield(
            epsilon=1.0,
            lperp=0.5,
            lpar=1.0,
            r_cut=3,
            angle_k=100,
            angle_theta0=1.57,
        )
        assert len(ellipsoid_ff.hoomd_forces) == 2
        assert isinstance(
            ellipsoid_ff.hoomd_forces[-1], hoomd.md.pair.aniso.GayBerne
        )
        assert ("_C", "_C") in list(
            dict(ellipsoid_ff.hoomd_forces[-1].params).keys()
        )
        assert ("_C", "_H") in list(
            dict(ellipsoid_ff.hoomd_forces[-1].params).keys()
        )
        assert (
            ellipsoid_ff.hoomd_forces[-1].params["_C", "_H"]["epsilon"] == 0.0
        )
        assert (
            ellipsoid_ff.hoomd_forces[-1].params["_H", "_H"]["epsilon"] == 0.0
        )
        assert ellipsoid_ff.hoomd_forces[-1].params["_C", "_H"]["lperp"] == 0.0
        assert ellipsoid_ff.hoomd_forces[-1].params["_H", "_H"]["lpar"] == 0.0


class TestTableForcefield:
    def test_from_txt_file(self):
        pair_file = os.path.join(ASSETS_DIR, "lj_pair_table.txt")
        bond_file = os.path.join(ASSETS_DIR, "bond_table.txt")
        angle_file = os.path.join(ASSETS_DIR, "angle_table.txt")
        dihedral_file = os.path.join(ASSETS_DIR, "dihedral_table.txt")
        ff = TableForcefield.from_files(
            pairs={("A", "A"): pair_file},
            bonds={"A-A": bond_file},
            angles={"A-A-A": angle_file},
            dihedrals={"A-A-A-A": dihedral_file},
        )
        pair_data = np.loadtxt(pair_file)
        bond_data = np.loadtxt(bond_file)
        angle_data = np.loadtxt(angle_file)
        dihedral_data = np.loadtxt(dihedral_file)
        assert ff.r_min == pair_data[:, 0][0]
        assert ff.r_cut == pair_data[:, 0][-1]
        assert ff.bond_width == len(bond_data[:, 0])
        assert ff.angle_width == len(angle_data[:, 0])
        assert ff.dih_width == len(dihedral_data[:, 0])

    def test_from_csv_file(self):
        pair_file = os.path.join(ASSETS_DIR, "lj_pair_table.csv")
        ff = TableForcefield.from_files(
            pairs={("A", "A"): pair_file}, delimiter=","
        )
        pair_data = np.genfromtxt(pair_file, delimiter=",")
        assert ff.r_min == pair_data[:, 0][0]
        assert ff.r_cut == pair_data[:, 0][-1]

    def test_from_npy_file(self):
        pair_file = os.path.join(ASSETS_DIR, "lj_pair_table.npy")
        bond_file = os.path.join(ASSETS_DIR, "bond_table.npy")
        angle_file = os.path.join(ASSETS_DIR, "angle_table.npy")
        dihedral_file = os.path.join(ASSETS_DIR, "dihedral_table.npy")
        ff = TableForcefield.from_files(
            pairs={("A", "A"): pair_file},
            bonds={"A-A": bond_file},
            angles={"A-A-A": angle_file},
            dihedrals={"A-A-A-A": dihedral_file},
        )
        pair_data = np.load(pair_file)
        bond_data = np.load(bond_file)
        angle_data = np.load(angle_file)
        dihedral_data = np.load(dihedral_file)
        assert ff.r_min == pair_data[:, 0][0]
        assert ff.r_cut == pair_data[:, 0][-1]
        assert ff.bond_width == len(bond_data[:, 0])
        assert ff.angle_width == len(angle_data[:, 0])
        assert ff.dih_width == len(dihedral_data[:, 0])

    def test_no_file(self):
        with pytest.raises(ValueError):
            TableForcefield.from_files(pairs={("A", "A"): "aa-pair.npy"})

    def test_bad_file_type(self):
        with pytest.raises(ValueError):
            TableForcefield.from_files(bonds={"A-A": "bond_table.bad"})
