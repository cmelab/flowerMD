import os
import random

import mbuild as mb
from mbuild.coordinate_transform import z_axis_transform
from mbuild.lib.recipes import Polymer as mbPolymer
import numpy as np

from hoomd_polymers.library import MON_DIR
from hoomd_polymers.utils import check_return_iterable


class Molecule:
    def __init__(self, load):
        self._mapping = None
        self.load = load
    
    @property
    def mapping(self):
        return self._mapping

    @mapping.setter:
    def mapping(self, mapping) 
        self._mapping = mapping

    def _build(self):
        pass

    def generator(self, n_mols):
        pass
        

class Polymer(Molecule):
    def __init__(self, n_mols, lengths):
        self.n_mols = n_mols
        self.lengths = lengths

    def _build(self, length):
        pass

    def _generator(self):
        for idx, length in self.lengths:
            for i in range(self.n_mols):
                yield self._buid(length=length)

    def __repr__(self):
        return [mol for mol in self.generator]


class SmallMolecule(Molecule):
    def __init__(self):
        pass

    def _build(self):
        pass


class PPS(Polymer):
    """Creates a Poly(phenylene-sulfide) (PPS) chain.

    Parameters
    ----------
    length : int; required
        The number of monomer repeat units in the chain
    """
    def __init__(self, lengths, n_mols):
        self.smiles_str = "c1ccc(S)cc1"
        self.file = None
        self.description = "Poly(phenylene-sulfide)"
        self.monomer = mb.load(self.smiles_str, smiles=True)
        # Need to align monomer along zx plane due to orientation of S-H bond
        z_axis_transform(
                self.monomer,
                point_on_z_axis=self.monomer[7],
                point_on_zx_plane=self.monomer[4]
        )
        self.bond_indices = [7, 10]
        self.bond_length = 0.176
        self.bond_orientation = [[0, 0, 1], [0, 0, -1]]
        self.lengths = lengths
        self.n_mols = n_mols

    def _build(self, length):
        chain = Polymer()
        chain.add_monomer(
                self.monomer,
                indices=self.bond_indices,
                separation=self.bond_length
                orientation=self.bond_orientation
        )
        chain.build(n=length, sequence="A")
        # Align the chain along the z-axis
        z_axis_transform(
                chain,
                point_on_z_axis=chain[-2],
                point_on_zx_plane=chain[-1]
        )
        return chain

    def _generate(self):
        molecules = []
        for idx, length in enumerate(self.lengths):
            for i in range(self.n_mols[idx]):
                mol = self._build(length=length)
                molecules.append(mol)
        return molecules


class PEEK(Polymer):
    def __init__(self, length):
        super(PEEK, self).__init__()


class PEKK_para:
    """Creates a Poly(ether-ketone-ketone) (PEKK) chain.
    The bonding positions of consecutive ketone groups
    takes place on the para site of the phenyl ring.

    Parameters
    ----------
    length : int; required
        The number of monomer repeat units in the chain
    """
    def __init__(self, lengths, n_mols):
        self.smiles_str = "c1ccc(Oc2ccc(C(=O)c3ccc(C(=O))cc3)cc2)cc1"
        self.file = os.path.join(MON_DIR, "pekk_para.mol2")
        self.description = ("Poly(ether-ketone-ketone) with para bonding "
                            "configuration between consectuvie "
                            "ketone linkage groups")
        self.monomer = mb.load(self.file)
        self.bond_indices = [35, 36]
        self.bond_length = 0.148
        self.bond_orientation = [[0, 0, -1], [0, 0, 1]]
        self.lengths = lengths
        self.n_mols = n_mols
    
    def _build(self, length):
        chain = Polymer()
        chain.add_monomer(
                self.monomer,
                indices=self.bond_indices,
                separation=self.bond_length
                orientation=self.bond_orientation
        )
        chain.build(n=length, sequence="A")
        return chain

    def _generate(self):
        molecules = []
        for idx, length in enumerate(self.lengths):
            for i in range(self.n_mols[idx]):
                mol = self._build(length=length)
                molecules.append(mol)
        return molecules


class PEKK_meta:
    """Creates a Poly(ether-ketone-ketone) (PEKK) chain.
    The bonding positions of consecutive ketone groups
    takes place on the meta site of the phenyl ring.

    Parameters
    ----------
    length : int; required
        The number of monomer repeat units in the chain
    """
    def __init__(self, lengths, n_mols):
        self.smiles_str = "c1cc(Oc2ccc(C(=O)c3cc(C(=O))ccc3)cc2)ccc1"
        self.file = os.path.join(MON_DIR, "pekk_meta.mol2")
        self.description = ("Poly(ether-ketone-ketone) with meta bonding "
                            "configuration between consectuvie "
                            "ketone linkage groups")
        self.monomer = mb.load(self.file)
        self.bond_indices = [35, 36]
        self.bond_length = 0.148
        self.bond_orientation = [[0, 0, -1], [0, 0, 1]]
        self.lengths = lengths
        self.n_mols = n_mols

    def _build(self, length):
        chain = Polymer()
        chain.add_monomer(
                self.monomer,
                indices=self.bond_indices,
                separation=self.bond_length
                orientation=self.bond_orientation
        )
        chain.build(n=length, sequence="A")
        return chain

    def _generate(self):
        molecules = []
        for idx, length in enumerate(self.lengths):
            for i in range(self.n_mols[idx]):
                mol = self._build(length=length)
                molecules.append(mol)
        return molecules


class LJChain:
    """Creates a coarse-grained bead-spring polymer chain.

    Parameters
    ----------
    length : int; required
        The number of times to repeat bead_sequence in a single chain.
    bead_sequence : list; optional; default ["A"]
        The sequence of bead types in the chain.
    bond_length : dict; optional; default {"A-A": 1.0}
        The bond length between connected beads (units: nm)
    bead_mass : dict; optional; default {"A": 1.0} 
        The mass of the bead types
    """
    def __init__(
            self,
            lengths,
            n_mols,
            bead_sequence=["A"],
            bead_mass={"A": 1.0},
            bond_lengths={"A-A": 1.0},
    ):
        super(LJChain, self).__init__()
        self.description = "Simple bead-spring polymer"
        self.lengths = check_return_iterable(lengths)
        self.n_mols = check_return_iterable(n_mols)
        self.bead_sequence = bead_sequence
        self.bead_mass = bead_mass
        self.bond_lengths = bond_lengths

    def _build(self, length):
        chain = mb.Compound()
        last_bead = None
        for i in range(length):
            for idx, bead_type in enumerate(self.bead_sequence):
                mass = self.bead_mass.get(bead_type, None)
                if not mass:
                    raise ValueError(
                            f"The bead mass for {bead_type} was not given "
                            "in the bead_mass dict."
                    )
                next_bead = mb.Compound(mass=mass, name=bead_type, charge=0)
                chain.add(next_bead)
                if last_bead:
                    bead_pair = "-".join([last_bead.name, next_bead.name])
                    bond_length = self.bond_lengths.get(bead_pair, None)
                    if not bond_length:
                        bead_pair_rev = "-".join([next_bead.name, last_bead.name])
                        bond_length = self.bond_lengths.get(bead_pair_rev, None)
                        if not bond_length:
                            raise ValueError(
                                    "The bond length for pair "
                                    f"{bead_pair} or {bead_pair_rev} "
                                    "is not found in the bond_lengths dict."
                            )
                    new_pos = last_bead.xyz[0] + (0, 0, bond_length)
                    next_bead.translate_to(new_pos)
                    chain.add_bond([next_bead, last_bead])
                last_bead = next_bead
        return chain

    def _generate(self):
        molecules = []
        for idx, length in enumerate(self.lengths):
            for i in range(self.n_mols[idx]):
                mol = self._build(length=length)
                molecules.append(mol)
        return molecules
