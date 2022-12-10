import mbuild as mb
from mbuild.lib.recipes import Polymer

class PolyEthylene(Polymer):
    def __init__(self, length):
        super(PolyEthylene, self).__init__()
        monomer = mb.load("CC", smiles=True)
        self.add_monomer(monomer, indices=[2, 6], separation=0.145)
        self.build(n=length, sequence="A")


class PolyStyrene(Polymer):
    def __init__(self, length):
        super(PolyStyrene, self).__init__()
        monomer = mb.load("CC", smiles=True)
        self.add_monomer(monomer, indices=[2, 6], separation=0.145)
        self.build(n=length, sequence="A")


class PolyPropylenesulfide(Polymer):
    def __init__(self, length):
        super(PolyStyrene, self).__init__()
        monomer = mb.load("CC", smiles=True)
        self.add_monomer(monomer, indices=[2, 6], separation=0.145)
        self.build(n=length, sequence="A")


class PEKK(Polymer):
    def __init__(self, length, para_meta_sequence):
        super(PEKK, self).__init__()
        para_monomer = mb.load("CC", smiles=True)
        meta_monomer = mb.load("CC", smiles=True)
        self.add_monomer(meta_monomer, indices=[2, 6], separation=0.145)
        self.add_monomer(para_monomer, indices=[2, 6], separation=0.145)
        self.build(n=length, sequence=para_meta_sequence)


class PEEK(Polymer):
    def __init__(self, length):
        super(PEEK, self).__init__()
        monomer = mb.load("CC", smiles=True)
        meta_monomer = mb.load("CC", smiles=True)
        self.add_monomer(meta_monomer, indices=[2, 6], separation=0.145)
        self.add_monomer(para_monomer, indices=[2, 6], separation=0.145)
        self.build(n=length, sequence=para_meta_sequence)


