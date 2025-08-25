import mbuild as mb
import pytest

from flowermd.library import (
    PEEK,
    PEKK,
    PPS,
    EllipsoidChain,
    LJChain,
    PEKK_meta,
    PEKK_para,
    PolyEthylene,
)


class TestPolymers:
    def test_pps(self):
        chain = PPS(lengths=5, num_mols=1)
        monomer = mb.load(chain.smiles, smiles=True)
        assert chain.n_particles == (monomer.n_particles * 5) - 8
        for mol in chain._molecules:
            assert mol.name == "pps_5mer"

    def test_names_polydisperse(self):
        chain = PPS(lengths=[5, 10], num_mols=[1, 1])
        assert chain._molecules[0].name == "pps_5mer"
        assert chain._molecules[1].name == "pps_10mer"

    def test_polyethylene(self):
        chain = PolyEthylene(lengths=5, num_mols=1)
        monomer = mb.load(chain.smiles, smiles=True)
        assert chain.n_particles == (monomer.n_particles * 5) - 8
        for mol in chain._molecules:
            assert mol.name == "polyethylene_5mer"

    def test_pekk_meta(self):
        chain = PEKK_meta(lengths=5, num_mols=1)
        monomer = mb.load(chain.smiles, smiles=True)
        assert chain.n_particles == (monomer.n_particles * 5) - 8
        for mol in chain._molecules:
            assert mol.name == "pekk_meta_5mer"

    def test_pekk_random(self):
        chain = PEKK(lengths=5, num_mols=1, TI_ratio=0.50)
        assert chain.random_sequence is True
        assert chain.AB_ratio == 0.50

    def test_pekk_not_random(self):
        chain = PEKK(lengths=6, num_mols=1, TI_ratio=0.50, sequence="PM")
        assert chain.AB_ratio == 0.50
        for mol in chain._molecules:
            assert mol.name == "pekk_6mer_PM"

    def test_pekk_para(self):
        chain = PEKK_para(lengths=5, num_mols=1)
        monomer = mb.load(chain.smiles, smiles=True)
        assert chain.n_particles == (monomer.n_particles * 5) - 8
        for mol in chain._molecules:
            assert mol.name == "pekk_para_5mer"

    def test_peek(self):
        chain = PEEK(lengths=5, num_mols=1)
        monomer = mb.load(chain.smiles, smiles=True)
        assert chain.n_particles == (monomer.n_particles * 5) - 8
        for mol in chain._molecules:
            assert mol.name == "peek_5mer"

    def test_lj_chain(self):
        chain = LJChain(
            lengths=3,
            num_mols=1,
            bead_sequence=["_A"],
            bead_mass={"_A": 100},
            bond_lengths={"_A-_A": 1.5},
        )
        assert chain.n_particles == 3
        assert chain.molecules[0].mass == 300
        with pytest.warns():
            chain._align_backbones_z_axis(heavy_atoms_only=True)
        for mol in chain._molecules:
            assert mol.name == "lj_chain_3mer"

    def test_lj_chain_sequence(self):
        cg_chain = LJChain(
            lengths=3,
            num_mols=1,
            bead_sequence=["A", "B"],
            bead_mass={"A": 100, "B": 150},
            bond_lengths={"A-A": 1.5, "A-B": 1.0},
        )
        assert cg_chain.n_particles == 6
        assert cg_chain.molecules[0].mass == 300 + 450

    def test_lj_chain_sequence_bonds(self):
        LJChain(
            lengths=3,
            num_mols=1,
            bead_sequence=["A", "B"],
            bead_mass={"A": 100, "B": 150},
            bond_lengths={"A-A": 1.5, "A-B": 1.0},
        )

        LJChain(
            lengths=3,
            num_mols=1,
            bead_sequence=["A", "B"],
            bead_mass={"A": 100, "B": 150},
            bond_lengths={"A-A": 1.5, "B-A": 1.0},
        )

    def test_lj_chain_sequence_bad_bonds(self):
        with pytest.raises(ValueError):
            LJChain(
                lengths=3,
                num_mols=1,
                bead_sequence=["A", "B"],
                bead_mass={"A": 100, "B": 150},
                bond_lengths={"A-A": 1.5},
            )

    def test_lj_chain_sequence_bad_mass(self):
        with pytest.raises(ValueError):
            LJChain(
                lengths=3,
                num_mols=1,
                bead_sequence=["A", "B"],
                bead_mass={"A": 100},
                bond_lengths={"A-A": 1.5},
            )

    def test_copolymer(self):
        pass

    def test_ellipsoid_chain(self):
        chain = EllipsoidChain(
            lengths=4,
            num_mols=2,
            lpar=0.5,
            bead_mass=1,
        )
        assert chain.n_particles == 32
        assert chain.molecules[0].mass == 4
        assert chain.molecules[0].n_particles == 16
        assert chain.molecules[0].n_bonds == 10
        for mol in chain._molecules:
            assert mol.name == "ellipsoid_chain_4mer"
