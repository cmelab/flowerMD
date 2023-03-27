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
        pass

    def test_copolymer(self):
        pass
