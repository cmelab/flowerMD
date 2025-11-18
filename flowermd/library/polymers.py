"""Polymer and CoPolymer example classes."""

import os

import mbuild as mb
import numpy as np
from mbuild.coordinate_transform import z_axis_transform

from flowermd import CoPolymer, Polymer
from flowermd.assets import MON_DIR


class PolyEthylene(Polymer):
    """Create a Poly(ethylene) chain.

    Parameters
    ----------
    lengths : int, required
        The number of monomer repeat units in the chain.
    num_mols : int, required
        The number of chains to create.
    name : str, default 'polyethylene'
        The name of the polymer. Setting the name is
        important for using the `speedup_by_moltag=True`
        parameter with polydisperse systems, or other
        mixtures. This helps improve performance
        for large systems.

    """

    def __init__(self, lengths, num_mols, name="polyethylene", **kwargs):
        smiles = "CC"
        bond_indices = [2, 6]
        bond_length = 0.145
        bond_orientation = [None, None]
        super(PolyEthylene, self).__init__(
            lengths=lengths,
            num_mols=num_mols,
            smiles=smiles,
            name=name,
            bond_indices=bond_indices,
            bond_length=bond_length,
            bond_orientation=bond_orientation,
            **kwargs,
        )


class PPS(Polymer):
    """Create a Poly(phenylene-sulfide) (PPS) chain.

    Parameters
    ----------
    lengths : int, required
        The number of monomer repeat units in the chain.
    num_mols : int, required
        The number of chains to create.
    name : str, default 'pps'
        The name of the polymer. Setting the name is
        important for using the `speedup_by_moltag=True`
        parameter with polydisperse systems, or other
        mixtures. This helps improve performance
        for large systems.

    """

    def __init__(self, lengths, num_mols, name="pps", **kwargs):
        smiles = "c1ccc(S)cc1"
        file = None
        bond_indices = [7, 10]
        bond_length = 0.176
        bond_orientation = [[0, 0, 1], [0, 0, -1]]
        super(PPS, self).__init__(
            lengths=lengths,
            num_mols=num_mols,
            smiles=smiles,
            name=name,
            file=file,
            bond_indices=bond_indices,
            bond_length=bond_length,
            bond_orientation=bond_orientation,
            **kwargs,
        )

    def _load(self):
        monomer = mb.load(self.smiles, smiles=True)
        # Need to align monomer along zx plane due to orientation of S-H bond
        z_axis_transform(
            monomer, point_on_z_axis=monomer[7], point_on_zx_plane=monomer[4]
        )
        return monomer


class PEEK(Polymer):
    """Create a Poly(ether-ether-ketone) (PEEK) chain.

    Parameters
    ----------
    lengths : int, required
        The number of monomer repeat units in the chain.
    num_mols : int, required
        The number of chains to create.
    name : str, default 'peek'
        The name of the polymer. Setting the name is
        important for using the `speedup_by_moltag=True`
        parameter with polydisperse systems, or other
        mixtures. This helps improve performance
        for large systems.

    """

    def __init__(self, lengths, num_mols, name="peek", **kwargs):
        smiles = "Oc1ccc(Oc2ccc(C(=O)c3ccccc3)cc2)cc1"
        file = os.path.join(MON_DIR, "peek.mol2")
        bond_indices = [35, 34]
        bond_length = 0.1376
        bond_orientation = [[-1, 0, 0], [1, 0, 0]]
        super(PEEK, self).__init__(
            lengths=lengths,
            num_mols=num_mols,
            smiles=smiles,
            name=name,
            file=file,
            bond_indices=bond_indices,
            bond_length=bond_length,
            bond_orientation=bond_orientation,
            **kwargs,
        )


class PEKK(CoPolymer):
    """Create a Poly(ether-ketone-ketone) (PEKK) chain.

    Creates a polymer chain with two different monomer types,
    represented by the para (T) and meta (I) isomeric forms of PEKK.

    Parameters
    ----------
    lengths : int, required
        The number of monomer repeat units in the chain.
    num_mols : int, required
        The number of chains to create.
    sequence : str, default None
        Manually define the sequence of para (T) and meta (I) monomers.
        Leave as None if generating random sequences.
        Example: sequence = "TTITTITTI"
    TI_ratio : float, required
        The ratio of meta to para isomers in the chain.

    """

    def __init__(
        self,
        lengths,
        num_mols,
        force_field=None,
        sequence=None,
        name="pekk",
        TI_ratio=0.50,
        seed=24,
    ):
        super(PEKK, self).__init__(
            monomer_A=PEKK_meta,
            monomer_B=PEKK_para,
            lengths=lengths,
            name=name,
            num_mols=num_mols,
            force_field=force_field,
            sequence=sequence,
            AB_ratio=TI_ratio,
            seed=seed,
        )


class PEKK_para(Polymer):
    """Create a Poly(ether-ketone-ketone) (PEKK) chain.

    The bonding positions of consecutive ketone groups
    takes place on the para site of the phenyl ring.

    Parameters
    ----------
    lengths : int, required
        The number of monomer repeat units in the chain.
    num_mols : int, required
        The number of chains to create.
    name : str, default 'pekk_para'
        The name of the polymer. Setting the name is
        important for using the `speedup_by_moltag=True`
        parameter with polydisperse systems, or other
        mixtures. This helps improve performance
        for large systems.

    """

    def __init__(self, lengths, num_mols, name="pekk_para"):
        smiles = "c1ccc(Oc2ccc(C(=O)c3ccc(C(=O))cc3)cc2)cc1"
        file = os.path.join(MON_DIR, "pekk_para.mol2")
        bond_indices = [35, 36]
        bond_length = 0.148
        bond_orientation = [[0, 0, -1], [0, 0, 1]]
        super(PEKK_para, self).__init__(
            lengths=lengths,
            num_mols=num_mols,
            smiles=smiles,
            name=name,
            file=file,
            bond_indices=bond_indices,
            bond_length=bond_length,
            bond_orientation=bond_orientation,
        )


class PEKK_meta(Polymer):
    """Create a Poly(ether-ketone-ketone) (PEKK) chain.

    The bonding positions of consecutive ketone groups
    takes place on the meta site of the phenyl ring.

    Parameters
    ----------
    lengths : int, required
        The number of monomer repeat units in the chain.
    num_mols : int, required
        The number of chains to create.
    name : str, default 'pekk_meta'
        The name of the polymer. Setting the name is
        important for using the `speedup_by_moltag=True`
        parameter with polydisperse systems, or other
        mixtures. This helps improve performance
        for large systems.

    """

    def __init__(self, lengths, num_mols, name="pekk_meta"):
        smiles = "c1cc(Oc2ccc(C(=O)c3cc(C(=O))ccc3)cc2)ccc1"
        file = os.path.join(MON_DIR, "pekk_meta.mol2")
        bond_indices = [35, 36]
        bond_length = 0.148
        bond_orientation = [[0, 0, -1], [0, 0, 1]]
        super(PEKK_meta, self).__init__(
            lengths=lengths,
            num_mols=num_mols,
            smiles=smiles,
            name=name,
            file=file,
            bond_indices=bond_indices,
            bond_length=bond_length,
            bond_orientation=bond_orientation,
        )


class LJChain(Polymer):
    """Create a coarse-grained bead-spring polymer chain.

    Parameters
    ----------
    lengths : int, required
        The number of times to repeat bead_sequence in a single chain.
    bead_sequence : list; default ["A"]
        The sequence of bead types in the chain.
    bond_length : dict; optional; default {"A-A": 1.0}
        The bond length between connected beads (units: nm).
    bead_mass : dict; default {"A": 1.0}
        The mass of the bead types.
    name : str, default 'lj_chain'
        The name of the polymer. Setting the name is
        important for using the `speedup_by_moltag=True`
        parameter with polydisperse systems, or other
        mixtures. This helps improve performance
        for large systems.

    """

    def __init__(
        self,
        lengths,
        num_mols,
        bead_sequence=["A"],
        bead_mass={"A": 1.0},
        bond_lengths={"A-A": 1.0},
        name="lj_chain",
    ):
        self.bead_sequence = bead_sequence
        self.bead_mass = bead_mass
        self.bond_lengths = bond_lengths
        super(LJChain, self).__init__(
            lengths=lengths, num_mols=num_mols, name=name
        )

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
                        bead_pair_rev = "-".join(
                            [next_bead.name, last_bead.name]
                        )
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
            chain.name = f"{self.name}_{length}mer"
        return chain


class EllipsoidChain(Polymer):
    """Create an ellipsoid polymer chain.

    This is a coarse-grained molecule where each monomer is modeled
    as an anisotropic bead (i.e. ellipsoid).

    Notes
    -----
    In order to form chains of connected ellipsoids, "ghost"
    particles of types "A" and "B" are used.

    This is meant to be used with
    `flowermd.library.forcefields.EllipsoidForcefield`
    and requires using `flowermd.utils.constraints.set_bond_constraints` to set up
    the fixed bonds correctly in HOOMD-Blue.

    Parameters
    ----------
    lengths : int, required
        The number of monomer repeat units in the chain.
    num_mols : int, required
        The number of chains to create.
    lpar : float, required
        The semi-axis length of the ellipsoid bead along its major axis.
    bead_mass : float, required
        The mass of the ellipsoid bead.
    name : str, default 'ellipsoid_chain'
        The name of the polymer. Setting the name is
        important for using the `speedup_by_moltag=True`
        parameter with polydisperse systems, or other
        mixtures. This helps improve performance
        for large systems.
    """

    def __init__(
        self,
        lengths,
        num_mols,
        lpar,
        bead_mass,
        bond_L=0.1,
        name="ellipsoid_chain",
    ):
        self.bead_mass = bead_mass
        self.lpar = lpar
        self.bond_L = bond_L
        # get the indices of the particles in a rigid body
        self.bead_constituents_types = ["X", "A", "T", "T"]
        super(EllipsoidChain, self).__init__(
            lengths=lengths, num_mols=num_mols, name=name
        )

    def _build(self, length):
        # Build up ellipsoid bead
        bead = mb.Compound(name="ellipsoid")
        center = mb.Compound(pos=(0, 0, 0), name="X", mass=self.bead_mass / 4)
        head = mb.Compound(
            pos=(0, 0, self.lpar + (self.bond_L / 2)),
            name="A",
            mass=self.bead_mass / 4,
        )
        tether_head = mb.Compound(
            pos=(0, 0, self.lpar), name="T", mass=self.bead_mass / 4
        )
        tether_tail = mb.Compound(
            pos=(0, 0, -self.lpar), name="T", mass=self.bead_mass / 4
        )
        bead.add([center, head, tether_head, tether_tail])
        bead.add_bond([center, head])

        chain = mb.Compound()
        last_bead = None
        for i in range(length):
            translate_by = np.array([0, 0, (i * self.lpar * 2) + self.bond_L])
            this_bead = mb.clone(bead)
            this_bead.translate(by=translate_by)
            chain.add(this_bead)
            if last_bead:
                chain.add_bond([this_bead.children[0], last_bead.children[1]])
                chain.add_bond([this_bead.children[3], last_bead.children[2]])
            last_bead = this_bead
        chain.name = f"{self.name}_{length}mer"

        return chain

class EllipsoidChainRand(Polymer):
    """Create an ellipsoid polymer chain in a random walk configuration, considering density and box size.

    This is a coarse-grained molecule where each monomer is modeled
    as an anisotropic bead (i.e. ellipsoid).

    Notes
    -----
    In order to form chains of connected ellipsoids, "ghost"
    particles of types "A" are used.

    This is meant to be used with
    `flowermd.library.forcefields.DPD_ellipse_FF`
    and requires using `flowermd.utils.constraints.set_bond_constraints` to set up
    the fixed bonds correctly in HOOMD-Blue.

    Parameters
    ----------
    lengths : int, required
        The number of monomer repeat units in the chain.
    num_mols : int, required
        The number of chains to create.
    lpar : float, required
        The semi-axis length of the ellipsoid bead along its major axis.
    bead_mass : float, required
        The mass of the ellipsoid bead.
    density : float, required
        Number density used to calculate box lengths for random walk position range.
    name : str, default 'ellipsoid_chain'
        The name of the polymer. Setting the name is
        important for using the `speedup_by_moltag=True`
        parameter with polydisperse systems, or other
        mixtures. This helps improve performance
        for large systems.
    """

   def __init__(
        self,
        lengths,
        num_mols,
        lpar,
        bead_mass,

        density,
        bond_L=0.1, #T-T bond length
        name="ellipsoid_chain",
    ):
        self.bead_mass = bead_mass
        self.lpar = lpar
        self.bond_L = bond_L
        self.density = density
        N=lengths*num_mols
        L = np.cbrt(N/ self.density)
        self.L = L
        self.bead_constituents_types = ["X", "A", "T", "T"]
        super(EllipsoidChainRand, self).__init__(
            lengths=lengths, num_mols=num_mols, name=name
        )

    def _build(self, length):
        bead = mb.Compound(name="ellipsoid")
        center = mb.Compound(pos=(0, 0, 0), name="X", mass=self.bead_mass / 4)
        head = mb.Compound(
            pos=(0, 0, self.lpar + (self.bond_L / 2)),
            name="A",
            mass=self.bead_mass / 4,
        )
        tether_head = mb.Compound(
            pos=(0, 0, self.lpar), name="T", mass=self.bead_mass / 4
        )
        tether_tail = mb.Compound(
            pos=(0, 0, -self.lpar), name="T", mass=self.bead_mass / 4
        )
        bead.add([center, head, tether_head, tether_tail])
        bead.add_bond([center, head])

        chain = mb.Compound()
        last_bead = None




        rand_range = ((self.L/2) - (self.lpar+(self.bond_L / 2))) #reducing step size for random walk
        print('range',rand_range)
        for i in range(length):
            translate_by = np.random.uniform(low=-1, high=1, size=(3,))
            translate_by /= np.linalg.norm(translate_by)*self.bond_L
            this_bead = mb.clone(bead)


            if last_bead:
                chain.add_bond([this_bead.children[0], last_bead.children[1]])
                chain.add_bond([this_bead.children[3], last_bead.children[2]])
                this_bead.translate(by=self.pbc(translate_by+last_bead.pos,pos_range=([rand_range]*3)))
            else:
                translate_by = np.random.uniform(low=-rand_range, high=rand_range, size=(3,))
                this_bead.translate(by=self.pbc(translate_by,pos_range=([rand_range]*3)))
            chain.add(this_bead)
            last_bead = this_bead
        chain.name = f"{self.name}_{length}mer"

        return chain
    
    def pbc(self,d,pos_range):
        """ Periodic boundary conditions for a reduced box considering position of A beads.
        """
        for i in range(3):
            while d[i] > pos_range[i] or d[i] < -(pos_range[i]):
                if d[i] < -pos_range[i]:
                    d[i] += pos_range[i]
                if d[i] > pos_range[i]:
                    d[i] -= pos_range[i]
        return d
