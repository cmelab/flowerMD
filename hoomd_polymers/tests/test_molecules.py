import os
import pytest
import random

import numpy as np
import gsd.hoomd
from hoomd_polymers.molecules import *
#from polybinder.library import ASSETS_DIR
from base_test import BaseTest


class TestMolecules(BaseTest):
    def test_pps(self):
        chain = PPS(length=5)
        monomer = mb.load(chain.smiles_str, smiles=True)
        assert chain.n_particles == (monomer.n_particles*5)-8

    def test_polyethylene(self):
        chain = PolyEthylene(length=5)
        monomer = mb.load(chain.smiles_str, smiles=True)
        assert chain.n_particles == (monomer.n_particles*5)-8

    def test_pekk_meta(self):
        chain = PEKK_meta(length=5)
        monomer = mb.load(chain.smiles_str, smiles=True)
        assert chain.n_particles == (monomer.n_particles*5)-8

    def test_pekk_para(self):
        chain = PEKK_para(length=5)
        monomer = mb.load(chain.smiles_str, smiles=True)
        assert chain.n_particles == (monomer.n_particles*5)-8

    @pytest.mark.skip()
    def test_peek(self):
        chain = PEEK(length=5)
        monomer = mb.load(chain.smiles_str, smiles=True)
        assert chain.n_particles == (monomer.n_particles*5)-8

    def test_lj_chain(self):
        cg_chain = LJ_chain(
                length=3,
                bead_sequence=["A"],
                bead_mass={"A": 100},
                bond_lengths={"A-A": 1.5}
        )
        assert cg_chain.n_particles == 3
        assert cg_chain.mass == 300

    def test_lj_chain_sequence(self):
        cg_chain = LJ_chain(
                length=3,
                bead_sequence=["A", "B"],
                bead_mass={"A": 100, "B": 150},
                bond_lengths={"A-A": 1.5, "A-B": 1.0}
        )
        assert cg_chain.n_particles == 6
        assert cg_chain.mass == 300 + 450

    def test_lj_chain_sequence_bonds(self):
        cg_chain = LJ_chain(
                length=3,
                bead_sequence=["A", "B"],
                bead_mass={"A": 100, "B": 150},
                bond_lengths={"A-A": 1.5, "A-B": 1.0}
        )

        cg_chain_rev = LJ_chain(
                length=3,
                bead_sequence=["A", "B"],
                bead_mass={"A": 100, "B": 150},
                bond_lengths={"A-A": 1.5, "B-A": 1.0}
        )

    def test_lj_chain_sequence_bad_bonds(self):
        with pytest.raises(ValueError):
            cg_chain = LJ_chain(
                    length=3,
                    bead_sequence=["A", "B"],
                    bead_mass={"A": 100, "B": 150},
                    bond_lengths={"A-A": 1.5}
            )

    def test_lj_chain_sequence_bad_mass(self):
        with pytest.raises(ValueError):
            cg_chain = LJ_chain(
                    length=3,
                    bead_sequence=["A", "B"],
                    bead_mass={"A": 100},
                    bond_lengths={"A-A": 1.5}
            )

    def test_copolymer(self):
        pass
