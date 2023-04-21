import os
import random

import mbuild as mb
from mbuild.coordinate_transform import z_axis_transform
from mbuild.lib.recipes import Polymer
import numpy as np

from hoomd_polymers.library import MON_DIR
from hoomd_polymers.utils import check_return_iterable


class CoPolymer(Polymer):
    """Builds a polymer consisting of two monomer types.
    
    Parameters
    ----------
    monomer_A : hoomd_polymers.molecules.Polymer; required
        Class of the A-type monomer
    monomer_B : hoomd_polymers.molecules.Polymer: required
        Class of the B-type monomer
    length : int; required
        The total number of monomers in the molecule
    sequence : str; optional; default None
        Manually define the sequence of 'A' and 'B' monomers.
        Leave as None if generating random sequences.
        Example: sequence = "AABAABAAB"
    random_sequence : bool; optional; default True
        Creates a random 'A' 'B' sequence as a function
        of the AB_ratio. Set to False when manually
        defining sequence
    AB_ratio : float; optional; default 0.50
        The relative weight of A to B monomer types.
        Used when generating random sequences.
    seed : int; optional; default 24
        Set the seed used when generating random sequences
    """
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
            random.seed(seed)
            self.sequence = random.choices(
                    ["A", "B"], [AB_ratio, 1-AB_ratio], k=length
            )
            length = 1
        else:
            self.sequence = sequence
        self.A_ratio = self.sequence.count("A")/len(self.sequence)
        self.B_ratio = self.sequence.count("B")/len(self.sequence)
        
        self.add_monomer(
                self.monomer_A.monomer,
                indices=self.monomer_A.bond_indices,
                orientation=self.monomer_A.bond_orientation,
                separation=self.monomer_A.bond_length
        )
        self.add_monomer(
                self.monomer_B.monomer,
                indices=self.monomer_B.bond_indices,
                orientation=self.monomer_B.bond_orientation,
                separation=self.monomer_B.bond_length
        )
        self.build(n=length, sequence=self.sequence)


class PolyEthylene(Polymer):
    """Creates a Poly(ethylene) chain.

    Parameters
    ----------
    length : int; required
        The number of monomer repeat units in the chain
    """
    def __init__(self, length):
        super(PolyEthylene, self).__init__()
        self.smiles_str = "CC"
        self.description = "Poly(ethylene)"
        self.file = None
        self.monomer = mb.load(self.smiles_str, smiles=True)
        self.bond_indices = [2, 6]
        self.bond_length = 0.145
        self.bond_orientation = [None, None]
        self.add_monomer(
                self.monomer,
                indices=self.bond_indices,
                separation=self.bond_length
        )
        self.build(n=length, sequence="A")
        # Align the chain along the z-axis
        z_axis_transform(
                self,
                point_on_z_axis=self[-2],
                point_on_zx_plane=self[-1]
        )


class PPS(Polymer):
    """Creates a Poly(phenylene-sulfide) (PPS) chain.

    Parameters
    ----------
    length : int; required
        The number of monomer repeat units in the chain
    """
    def __init__(self, length):
        super(PPS, self).__init__()
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
    """Creates a Poly(ether-ketone-ketone) (PEKK) chain.
    The bonding positions of consecutive ketone groups
    takes place on the para site of the phenyl ring.

    Parameters
    ----------
    length : int; required
        The number of monomer repeat units in the chain
    """
    def __init__(self, length):
        super(PEKK_para, self).__init__()
        self.smiles_str = "c1ccc(Oc2ccc(C(=O)c3ccc(C(=O))cc3)cc2)cc1"
        self.file = os.path.join(MON_DIR, "pekk_para.mol2")
        self.description = ("Poly(ether-ketone-ketone) with para bonding "
                            "configuration between consectuvie "
                            "ketone linkage groups")
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
    """Creates a Poly(ether-ketone-ketone) (PEKK) chain.
    The bonding positions of consecutive ketone groups
    takes place on the meta site of the phenyl ring.

    Parameters
    ----------
    length : int; required
        The number of monomer repeat units in the chain
    """
    def __init__(self, length):
        super(PEKK_meta, self).__init__()
        self.smiles_str = "c1cc(Oc2ccc(C(=O)c3cc(C(=O))ccc3)cc2)ccc1"
        self.file = os.path.join(MON_DIR, "pekk_meta.mol2")
        self.description = ("Poly(ether-ketone-ketone) with meta bonding "
                            "configuration between consectuvie "
                            "ketone linkage groups")
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


class LJChain():
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
