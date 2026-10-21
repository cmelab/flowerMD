import os

import hoomd
import numpy as np
import pytest
import unyt as u

from flowermd.assets import FF_DIR
from flowermd.base import Lattice, Molecule
from flowermd.library import (
    DPD,
    GAFF,
    OPLS_AA,
    OPLS_AA_BENZENE,
    OPLS_AA_DIMETHYLETHER,
    OPLS_AA_PPS,
    UFF_DPD,
    Bead_Spring_DPD,
    BeadSpring,
    EllipsoidFF_DPD,
    EllipsoidForcefield,
    FF_from_file,
    KremerGrestBeadSpring,
    OpenFF_DPD,
    TableForcefield,
)
from flowermd.tests.base_test import ASSETS_DIR, BaseTest


class TestForceFields:
    def test_GAFF(self):
        ff = GAFF()
        assert ff.gmso_ff is not None

    def test_hoomd_dpd(self):
        ff_path = os.path.join(FF_DIR, "hoomd-dpd-hhp.xml")
        FF_from_file(forcefield_files=ff_path, gmso_xml=True)

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

    def test_Bead_Spring_DPD(self):
        ff = Bead_Spring_DPD()
        assert ff.gmso_ff is not None

    def test_FF_from_file(self):
        xml_file = os.path.join(ASSETS_DIR, "test_ff.xml")
        ff = FF_from_file(forcefield_files=xml_file, gmso_xml=False)
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

    def test_DPD(self):
        ff = DPD(
            A=2000, gamma=1000, kT=1.0, r_cut=1.05, bond_k=2000, bond_r0=1.0
        )

        assert isinstance(ff.hoomd_forces[0], hoomd.md.pair.pair.DPD)
        assert isinstance(ff.hoomd_forces[1], hoomd.md.bond.Harmonic)

        pair_types = [("_A", "_A")]
        for param in ff.hoomd_forces[0].params:
            assert param in pair_types
            if param == ("_A", "_A"):
                assert ff.hoomd_forces[0].params[param]["A"] == 2000

        bond_types = [("_A-_A")]
        for param in ff.hoomd_forces[1].params:
            assert param in bond_types
            assert ff.hoomd_forces[1].params[param]["r0"] == 1.0
            assert ff.hoomd_forces[1].params[param]["k"] == 2000

    def test_ellipsoid_ff(self):
        ellipsoid_ff = EllipsoidForcefield(
            epsilon=1.0,
            lperp=0.5,
            lpar=1.0,
            r_cut=3,
            angle_k=100,
            angle_theta0=1.57,
        )
        assert len(ellipsoid_ff.hoomd_forces) == 3
        assert isinstance(
            ellipsoid_ff.hoomd_forces[-1], hoomd.md.pair.aniso.GayBerne
        )
        assert ("X", "X") in list(
            dict(ellipsoid_ff.hoomd_forces[-1].params).keys()
        )
        assert ("R", "X") in list(
            dict(ellipsoid_ff.hoomd_forces[-1].params).keys()
        )
        assert ellipsoid_ff.hoomd_forces[-1].params["R", "X"]["epsilon"] == 0.0
        assert ellipsoid_ff.hoomd_forces[-1].params["A", "X"]["epsilon"] == 0.0
        assert ellipsoid_ff.hoomd_forces[-1].params["A", "T"]["epsilon"] == 0.0
        assert ellipsoid_ff.hoomd_forces[-1].params["A", "X"]["epsilon"] == 0.0
        assert ellipsoid_ff.hoomd_forces[-1].params["A", "T"]["lperp"] == 0.0
        assert ellipsoid_ff.hoomd_forces[-1].params["T", "X"]["lpar"] == 0.0
        assert ellipsoid_ff.hoomd_forces[-1].params["R", "X"]["lpar"] == 0.0
        assert ellipsoid_ff.hoomd_forces[-1].params["R", "A"]["lpar"] == 0.0
        assert ellipsoid_ff.hoomd_forces[-1].params["R", "T"]["lpar"] == 0.0

    def test_ellipsoid_ff_dpd(self):
        ellipsoid_ff = EllipsoidFF_DPD(
            epsilon=1.0,
            lperp=0.5,
            lpar=0.5,
            r_cut=3,
            A=500,
            gamma=1000,
            kT=1.0,
            bond_k=15000,
            angle_k=10,
            angle_theta0=2,
        )
        assert len(ellipsoid_ff.hoomd_forces) == 3
        assert isinstance(ellipsoid_ff.hoomd_forces[-1], hoomd.md.pair.DPD)
        assert ("X", "X") in list(
            dict(ellipsoid_ff.hoomd_forces[-1].params).keys()
        )
        assert ("R", "X") in list(
            dict(ellipsoid_ff.hoomd_forces[-1].params).keys()
        )
        assert ellipsoid_ff.hoomd_forces[-1].params["R", "X"]["A"] == 0.0
        assert ellipsoid_ff.hoomd_forces[-1].params["A", "X"]["A"] == 0.0
        assert ellipsoid_ff.hoomd_forces[-1].params["A", "T"]["A"] == 0.0
        assert ellipsoid_ff.hoomd_forces[-1].params["A", "X"]["A"] == 0.0
        assert ellipsoid_ff.hoomd_forces[-1].params["A", "T"]["gamma"] == 0.1
        assert ellipsoid_ff.hoomd_forces[-1].params["T", "X"]["gamma"] == 0.1
        assert ellipsoid_ff.hoomd_forces[-1].params["R", "X"]["gamma"] == 0.1
        assert ellipsoid_ff.hoomd_forces[-1].params["R", "A"]["gamma"] == 0.1


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


class TestAllAtomDPD(BaseTest):
    PS = "CC(c1ccccc1)CC(c1ccccc1)CC(c1ccccc1)C"

    def build(self, smiles, ff_class=UFF_DPD, **kwargs):
        mols = Molecule(num_mols=2, smiles=smiles)
        system = Lattice(molecules=mols, x=1.5, y=1.5, n=1)
        ff = ff_class(system, **kwargs)
        system.apply_forcefield(r_cut=0.35, force_field=ff, kT=4.184)
        return system

    def force(self, system, kind):
        forces = system.hoomd_forcefield
        return next(f for f in forces if isinstance(f, kind))

    def test_uff_matches_rdkit(self):
        from rdkit import Chem
        from rdkit.Chem import rdForceFieldHelpers

        system = self.build("CC", bonded_scale=1.0)
        ethane = Chem.AddHs(Chem.MolFromSmiles("CC"))
        k, r0 = rdForceFieldHelpers.GetUFFBondStretchParams(ethane, 0, 1)
        bond = self.force(system, hoomd.md.bond.Harmonic).params["_C000-_C000"]
        assert np.isclose(bond["k"], k * 418.4)
        assert np.isclose(bond["r0"], r0 / 10)
        assert system.reference_energy == 1 * u.Unit("kJ/mol")

    def test_bonded_scale_and_types(self):
        system = self.build(self.PS)
        types = system.hoomd_snapshot.particles.types
        assert len([t for t in types if t.startswith("_C")]) > 1
        assert all(len(t) == 5 for t in types)  # zero-padded: sorts agree
        one = self.build(self.PS, bonded_scale=1.0, dihedral_scale=1.0)
        k30 = self.force(system, hoomd.md.angle.Harmonic).params
        k1 = self.force(one, hoomd.md.angle.Harmonic).params
        assert all(np.isclose(k30[t]["k"], 30 * k1[t]["k"]) for t in k1)
        k300 = self.force(system, hoomd.md.dihedral.Periodic).params
        k1 = self.force(one, hoomd.md.dihedral.Periodic).params
        assert all(np.isclose(k300[t]["k"], 300 * k1[t]["k"]) for t in k1)

    def test_pair_weighting_and_overrides(self):
        system = self.build(self.PS, epsilon_weighting=False, A=100.0)
        dpd = self.force(system, hoomd.md.pair.DPD)
        assert {p["A"] for p in dpd.params.values()} == {100.0}
        assert {p["gamma"] for p in dpd.params.values()} == {4091.0}
        assert np.isclose(dpd.nlist.buffer, 0.04)
        pair = tuple(sorted(system.hoomd_snapshot.particles.types)[:2])
        system = self.build(self.PS, pair_overrides={pair: 7.0})
        assert self.force(system, hoomd.md.pair.DPD).params[pair]["A"] == 7.0

    def test_untyped_dihedrals_get_zero_k(self):
        system = self.build("C=CC#N")  # UFF has no torsion next to C#N
        params = self.force(system, hoomd.md.dihedral.Periodic).params
        k = [params[t]["k"] for t in system.hoomd_snapshot.dihedrals.types]
        assert 0.0 in k and max(k) > 0

    def test_aromatic_sulfur(self):
        from rdkit import Chem

        from flowermd.base import Polymer
        from flowermd.internal.all_atom_dpd import to_sanitized_rdkit

        bithiophene = Polymer(
            lengths=2,
            num_mols=1,
            smiles="c1ccsc1",
            bond_indices=[7, 8],
            bond_length=0.145,
            bond_orientation=[None, None],
        )
        mol = to_sanitized_rdkit(bithiophene.molecules[0])
        assert not any(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
        assert all(a.GetIsAromatic() for a in mol.GetAtoms() if a.IsInRing())
        reference = Chem.AddHs(Chem.MolFromSmiles("c1ccc(-c2cccs2)s1"))
        assert Chem.MolToSmiles(mol) == Chem.MolToSmiles(reference)
        UFF_DPD(bithiophene.molecules[0])  # UFF types the aromatic S too

    def test_openff(self):
        pytest.importorskip("openff.interchange")
        system = self.build(self.PS, ff_class=OpenFF_DPD)
        assert system.hoomd_snapshot.impropers.N > 0
        periodic = hoomd.md.dihedral.Periodic
        assert any(isinstance(f, periodic) for f in system.hoomd_forcefield)
