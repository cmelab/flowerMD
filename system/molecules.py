import mbuild as mb
from mbuild.coordinate_transform import x_axis_transform
from mbuild.lib.recipes import Polymer


class PolyEthylene(Polymer):
    def __init__(self, length):
        super(PolyEthylene, self).__init__()
        self.smiles_str = "CC"
        self.monomer = mb.load(self.smiles_str, smiles=True)
        self.add_monomer(self.monomer, indices=[2, 6], separation=0.145)
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
        self.add_monomer(
                self.monomer,
                indices=[7, 10],
                separation=0.176,
                orientation=[[1, 0, 0], [-1, 0, 0]]
        )
        self.build(n=length, sequence="A")
