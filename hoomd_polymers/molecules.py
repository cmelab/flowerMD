import os
import random

import mbuild as mb
from mbuild.coordinate_transform import x_axis_transform
from mbuild.lib.recipes import Polymer
from hoomd_polymers.library import MON_DIR


class CoPolymer(Polymer):
    def __init__(
            self,
            monomer_A,
            monomer_B,
            length,
            sequence=None,
            random_sequence=True,
            AB_ratio=0.50,
            seed=24
    ):
        super(CoPolymer, self).__init__()
        self.monomer_A = monomer_A(length=1)
        self.monomer_B = monomer_B(length=1)
        if random_sequence:
            self.sequence = random.choices(["A", "B"], [AB_ratio, 1-AB_ratio], k=length)
        else:
            self.sequence = sequence
        
        self.add_monomer(
                self.monomer_A,
                indices=self.monomer_A.bond_indices,
                orientation=self.monomer_A.bond_orientation,
                separation=self.monomer_A.bond_length
        )
        self.add_monomer(
                self.monomer_B,
                indices=self.monomer_B.bond_indices,
                orientation=self.monomer_B.bond_orientation,
                separation=self.monomer_B.bond_length
        )
        self.build(n=length, sequence=self.sequence)


class PolyEthylene(Polymer):
    def __init__(self, length):
        super(PolyEthylene, self).__init__()
        self.smiles_str = "CC"
        self.monomer = mb.load(self.smiles_str, smiles=True)
        self.bond_indices = [2,6]
        self.bond_length = 0.145
        self.bond_orientation = [None, None]
        self.add_monomer(
                self.monomer,
                indices=self.bond_indices,
                separation=self.bond_length
        )
        self.build(n=length, sequence="A")


class PPS(Polymer):
    def __init__(self, length):
        super(PPS, self).__init__()
        self.smiles_str = "c1ccc(S)cc1"
        self.monomer = mb.load(self.smiles_str, smiles=True)
        # Need to align monomer along xy plane due to orientation of S-H bond
        x_axis_transform(
                self.monomer,
                point_on_x_axis=self.monomer[7],
                point_on_xy_plane=self.monomer[4]
        )
        self.bond_indices = [7, 10]
        self.bond_length = 0.176
        self.bond_orientation = [[1, 0, 0], [-1, 0, 0]]
        self.add_monomer(
                self.monomer,
                indices=self.bond_indices,
                separation=self.bond_length,
                orientation=self.bond_orientation
        )
        self.build(n=length, sequence="A")


class PEEK(Polymer):
    def __init__(self, length):
        super(PEEK, self).__init__()


class PEKK_para(Polymer):
    def __init__(self, length):
        super(PEKK_para, self).__init__()
        self.smiles_str = "c1ccc(Oc2ccc(C(=O)c3ccc(C(=O))cc3)cc2)cc1"
        self.file = os.path.join(MON_DIR, "pekk_para.mol2")
        self.monomer = mb.load(self.file)
        self.bond_indices = [35, 36]
        self.bond_length = 0.148
        self.bond_orientation = [[0, 0, -1], [0, 0, 1]]
        self.add_monomer(
                self.monomer,
                indices=self.bond_indices,
                separation=self.bond_length,
                orientation=self.bond_orientation
        )
        self.build(n=length, sequence="A")


class PEKK_meta(Polymer):
    def __init__(self, length):
        super(PEKK_meta, self).__init__()
        self.smiles_str = "c1cc(Oc2ccc(C(=O)c3cc(C(=O))ccc3)cc2)ccc1"
        self.file = os.path.join(MON_DIR, "pekk_meta.mol2")
        self.monomer = mb.load(self.file)
        self.bond_indices = [35, 36]
        self.bond_length = 0.148
        self.bond_orientation = [[0, 0, -1], [0, 0, 1]]
        self.add_monomer(
                self.monomer,
                indices=self.bond_indices,
                separation=self.bond_length,
                orientation=self.bond_orientation
        )
        self.build(n=length, sequence="A")
