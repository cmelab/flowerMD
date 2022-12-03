import mbuild as mb
from mbuild.lib.recipes import Polymer

class PolyEthylene(Polymer):
    def __init__(self, length):
        super(PolyEthylene, self).__init__()
        monomer = mb.load("CC", smiles=True)
        self.add_monomer(monomer, indices=[2, 6], separation=0.145)
        self.build(n=length, sequence="A")
