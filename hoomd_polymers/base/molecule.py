import itertools
import random
from typing import Union, Dict, List

import mbuild as mb
from gmso.external.convert_mbuild import from_mbuild
from grits import CG_Compound
from hoomd.md.force import Force as HForce
from mbuild.lib.recipes import Polymer as mbPolymer

from hoomd_polymers.base.base_types import FF_Types
from hoomd_polymers.utils import check_return_iterable
from .forcefield import find_xml_ff, apply_xml_ff, _validate_hoomd_ff


class Molecule:
    def __init__(self, n_mols, force_field: Union[Dict, List[HForce], str]=None, smiles=None, file=None, description=None,
                 remove_hydrogens=False):
        self.n_mols = check_return_iterable(n_mols)
        self.force_field = force_field
        self.smiles = smiles 
        self.file = file 
        self.description = description
        self.remove_hydrogens = remove_hydrogens
        self._mapping = None
        self._mb_molecule = self._load()
        self._molecules = []
        self._cg_molecules = []
        self._generate()
        self.gmso_molecule = self._convert_to_gmso(self._molecules[0])
        if self.force_field:
            self._validate_force_field()



    @property
    def molecules(self):
        """List of all instances of the molecule"""
        if self._cg_molecules:
            return self._cg_molecules
        return self._molecules
    
    @property
    def mapping(self):
        """Dictionary of particle index to bead mapping"""
        return self._mapping

    @mapping.setter
    def mapping(self, mapping_array):
        self._mapping = mapping_array

    @property
    def particle_types(self):
        return self.topology_information["particle_types"]

    @property
    def pair_types(self):
        return self.topology_information["pair_types"]

    @property
    def bond_types(self):
        return self.topology_information["bond_types"]

    @property
    def angle_types(self):
        return self.topology_information["angle_types"]

    @property
    def diherdal_types(self):
        return self.topology_information["dihedral_types"]

    @property
    def improper_types(self):
        return self.improper_types["improper_types"]
    
    def coarse_grain(self, beads=None, mapping=None):
        for comp in self.molecules:
            cg_comp = CG_Compound(comp, beads=beads, mapping=mapping)
            self._cg_molecules.append(cg_comp)

    def _load(self):
        if self.file and isinstance(self.file, str): # Loading from file takes precedent over SMILES 
            return mb.load(self.file)
        elif self.smiles and isinstance(self.smiles, str):
            return mb.load(self.smiles, smiles=True)
        else:
            raise ValueError(
                    "Unable to load from ",
                    f"File: {self.file}",
                    f"SMILES: {self.smiles}"
            )

    def _generate(self):
        for i in range(self.n_mols):
            self._molecules.append(self._mb_molecule)

    def _convert_to_gmso(self, mb_molecule):
        topology = from_mbuild(mb_molecule)
        topology.identify_connections()
        return topology

    def _particle_information(self, gmso_molecule):
        particle_types = set()
        hydrogen_types = set()
        for site in gmso_molecule.sites:
            particle_types.add(site.atom_type.name or site.name)
            if site.element.atomic_number == 1:
                hydrogen_types.add(site.atom_type.name or site.name)
        return particle_types, hydrogen_types

    def _identify_pairs(self, particle_types):
        pairs = list(itertools.combinations_with_replacement(particle_types, 2))
        return pairs

    def _identify_bond_types(self, gmso_molecule):
        bond_types = set()
        for bond in gmso_molecule.bonds:
            bond_connections = [bond.connection_members[0].atom_type.name or bond.connection_members[0].name,
                            bond.connection_members[1].atom_type.name or bond.connection_members[1].name]
            bond_types.add(tuple(bond_connections))
        return bond_types

    def _identify_angle_types(self, gmso_molecule):
        angle_types = set()
        for angle in gmso_molecule.angles:
            angle_connections = [angle.connection_members[0].atom_type.name or angle.connection_members[0].name,
                                 angle.connection_members[1].atom_type.name or angle.connection_members[1].name,
                                 angle.connection_members[2].atom_type.name or angle.connection_members[2].name]
            angle_types.add(tuple(angle_connections))
        return angle_types

    def _identify_dihedral_types(self, gmso_molecule):
        dihedral_types = set()
        for dihedral in gmso_molecule.dihedrals:
            dihedral_connections = [dihedral.connection_members[0].atom_type.name or dihedral.connection_members[0].name,
                                    dihedral.connection_members[1].atom_type.name or dihedral.connection_members[1].name,
                                    dihedral.connection_members[2].atom_type.name or dihedral.connection_members[2].name,
                                    dihedral.connection_members[3].atom_type.name or dihedral.connection_members[3].name]
            dihedral_types.add(tuple(dihedral_connections))
        return dihedral_types

    def _identify_improper_types(self, gmso_molecule):
        improper_types = set()
        for improper in gmso_molecule.impropers:
            improper_connections = [improper.connection_members[0].atom_type.name or improper.connection_members[0].name,
                                    improper.connection_members[1].atom_type.name or improper.connection_members[1].name,
                                    improper.connection_members[2].atom_type.name or improper.connection_members[2].name,
                                    improper.connection_members[3].atom_type.name or improper.connection_members[3].name]
            improper_types.add(tuple(improper_connections))
        return improper_types

    def _get_topology_information(self, gmso_molecule):
        topology_information = dict()
        particle_types, hydrogen_types = self._particle_information(gmso_molecule)
        topology_information["particle_types"] = particle_types
        topology_information["hydrogen_types"] = hydrogen_types
        topology_information["pair_types"] = self._identify_pairs(particle_types)
        topology_information["bond_types"] = self._identify_bond_types(gmso_molecule)
        topology_information["angle_types"] = self._identify_angle_types(gmso_molecule)
        topology_information["dihedral_types"] = self._identify_dihedral_types(gmso_molecule)
        topology_information["improper_types"] = self._identify_improper_types(gmso_molecule)
        return topology_information

    def _validate_force_field(self):
        self.ff_type = None
        if isinstance(self.force_field, str):
            ff_xml_path, ff_type = find_xml_ff(self.force_field)
            self.ff_type = ff_type
            self.gmso_molecule = apply_xml_ff(ff_xml_path, self.gmso_molecule)
            self.topology_information = self._get_topology_information(self.gmso_molecule)
        elif isinstance(self.force_field, List):
            self.topology_information = self._get_topology_information(self.gmso_molecule)
            _validate_hoomd_ff(self.force_field, self.topology_information)
            self.ff_type = FF_Types.Hoomd


class Polymer(Molecule):
    def __init__(
            self,
            lengths,
            n_mols,
            smiles,
            file,
            description,
            bond_indices,
            bond_length,
            bond_orientation,
            **kwargs
    ):
        self.lengths = check_return_iterable(lengths)
        self.bond_indices = bond_indices
        self.bond_length = bond_length
        self.bond_orientation = bond_orientation
        super(Polymer, self).__init__(
                n_mols=n_mols,
                smiles=smiles,
                file=file,
                description=description,
                **kwargs
        )

    @property
    def monomer(self):
        return self._mb_molecule

    def _build(self, length):
        chain = mbPolymer()
        chain.add_monomer(
                self.monomer,
                indices=self.bond_indices,
                separation=self.bond_length,
                orientation=self.bond_orientation
        )
        chain.build(n=length, sequence="A")
        return chain

    def _generate(self):
        for idx, length in enumerate(self.lengths):
            for i in range(self.n_mols[idx]):
                mol = self._build(length=length)
                self._molecules.append(mol)


class CoPolymer(Molecule):
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
            lengths,
            n_mols,
            sequence=None,
            random_sequence=True,
            AB_ratio=0.50,
            seed=24
    ):
        self.lengths = lengths
        self.monomer_A = monomer_A(lengths=[1], n_mols=[1])
        self.monomer_B = monomer_B(lengths=[1], n_mols=[1])
        self.n_mols = n_mols
        self.sequence = sequence
        self.random_sequence = random_sequence
        self.AB_ratio = AB_ratio
        self.seed = seed
        self._A_count = 0
        self._B_count = 0
        self.smiles = {"A": self.monomer_A.smiles, "B": self.monomer_B.smiles}
        self.description = {
            "A": self.monomer_A.description, "B": self.monomer_B.description
        }
        self.file = {"A": self.monomer_A.file, "B": self.monomer_B.file}
        random.seed(self.seed)
        super(CoPolymer, self).__init__(
                n_mols=n_mols,
                smiles=self.smiles,
                file=self.file,
                description=self.description
        )
    
    @property
    def A_ratio(self):
        return self._A_count / (self._A_count + self._B_count)

    @property
    def B_ratio(self):
        return self._B_count / (self._A_count + self._B_count)

    def _build(self, length, sequence):
        chain = mbPolymer()
        chain.add_monomer(
                self.monomer_A.monomer,
                indices=self.monomer_A.bond_indices,
                orientation=self.monomer_A.bond_orientation,
                separation=self.monomer_A.bond_length
        )
        chain.add_monomer(
                self.monomer_B.monomer,
                indices=self.monomer_B.bond_indices,
                orientation=self.monomer_B.bond_orientation,
                separation=self.monomer_B.bond_length
        )
        chain.build(n=length, sequence=sequence)
        return chain

    def _generate(self):
        for idx, length in enumerate(self.lengths):
            for i in range(self.n_mols[idx]):
                if self.random_sequence:
                    sequence = random.choices(
                            ["A", "B"],
                            [self.AB_ratio, 1-self.AB_ratio],
                            k=length
                    )
                    self._A_count += sequence.count("A")
                    self._B_count += sequence.count("B")
                    _length = 1
                else:
                    sequence = self.sequence
                    _length = length
                mol = self._build(length=_length, sequence=sequence)
                self._molecules.append(mol)
