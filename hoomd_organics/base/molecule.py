"""Base class for all hoomd-organics molecules, Polymers, and CoPolymers."""
import itertools
import os.path
import random
from typing import List

import mbuild as mb
from gmso.core.topology import Topology
from gmso.external.convert_mbuild import from_mbuild, to_mbuild
from gmso.parameterization import apply
from grits import CG_Compound
from mbuild.lib.recipes import Polymer as mbPolymer

from hoomd_organics.base import BaseHOOMDForcefield, BaseXMLForcefield
from hoomd_organics.utils import check_return_iterable
from hoomd_organics.utils.exceptions import ForceFieldError, MoleculeLoadError
from hoomd_organics.utils.ff_utils import _validate_hoomd_ff


class Molecule:
    """Base class for all `hoomd-organics` molecules.

    The Molecule class generates molecules from the provided input and can be
    used to initialize a molecular structure. This class also provides
    information about the molecule topology, such as the number of particles,
    bonds, angles, etc.

    Parameters
    ----------
    num_mols : int, required
        Number of molecules to generate.
        force_field : hoomd_organics.ForceField or a list of
                    `hoomd.md.force.Force` objects, default=None
            The force field to be applied to the molecule for parameterization.
            Note that setting `force_field` does not actually apply the
            forcefield to the molecule. The forcefield in this step is mainly
            used for validation purposes.
    smiles : str, default None
        The smiles string of the molecule to generate.
    file : str, default None
        The file path to the molecule to generate.
        Supported file types are: .mol2, .pdb, .sdf
    compound : mbuild Compound or GMSO Topology, default None
        The mbuild Compound or GMSO Topology of the molecule to generate.

    Notes
    -----
    The molecule can be generated from either a smiles string, a file path,
    or a mbuild Compound or GMSO Topology.

    """

    def __init__(
        self, num_mols, force_field=None, smiles=None, file=None, compound=None
    ):
        self.n_mols = num_mols
        self.force_field = force_field
        self.smiles = smiles
        self.file = file
        self.compound = compound
        self._mapping = None
        self._mb_molecule = self._load()
        self._molecules = []
        self._cg_molecules = []
        self._generate()
        self.gmso_molecule = self._convert_to_gmso(self._molecules[0])
        self._identify_topology_information(self.gmso_molecule)
        if self.force_field:
            self._validate_force_field()

    @property
    def molecules(self):
        """List of all instances of the molecule."""
        if self._cg_molecules:
            return self._cg_molecules
        return self._molecules

    @property
    def mapping(self):
        """Dictionary of particle index to coarse grained bead mapping."""
        return self._mapping

    @mapping.setter
    def mapping(self, mapping_array):
        """Set the bead mapping for coarse graining the molecule."""
        self._mapping = mapping_array

    @property
    def n_particles(self):
        """Total number of particles in all of the molecules."""
        n_particles = 0
        for molecule in self.molecules:
            n_particles += molecule.n_particles
        return n_particles

    @property
    def n_bonds(self):
        """Total number of bonds in all of the molecules."""
        n_bonds = 0
        for molecule in self.molecules:
            n_bonds += molecule.n_bonds
        return n_bonds

    @property
    def topology_information(self):
        """Dictionary of topology information for the molecule.

        The dictionary contains the following keys:

            - `particle_types`: list of all particle types.

            - `hydrogen_types`: list of all hydrogen types.

            - `particle_charge`: list of all particle charges.

            - `particle_typeid`: list of all particle type indices.

            - `pair_types`: list of all unique particle pairs.

            - `bond_types`: list of all unique bond types.

            - `angle_types`: list of all unique angle types.

            - `dihedral_types`: list of all unique dihedral types.

            - `improper_types`: list of all unique improper types.

        """
        topology_information = dict()
        topology_information["particle_types"] = self.particle_types
        topology_information["hydrogen_types"] = self.hydrogen_types
        topology_information["particle_charge"] = self.particle_charge
        topology_information["particle_typeid"] = self.particle_typeid
        topology_information["pair_types"] = self.pairs
        topology_information["bond_types"] = self.bond_types
        topology_information["angle_types"] = self.angle_types
        topology_information["dihedral_types"] = self.dihedral_types
        topology_information["improper_types"] = self.improper_types
        return topology_information

    def coarse_grain(self, beads=None):
        """Coarse grain the molecule.

        Parameters
        ----------
        beads : dict, default None
            A dictionary of bead names and their corresponding SMILES string.

        Examples
        --------
        Coarse grain atomistic benzene molecules into a single bead type called
        "A".

        >>> from hoomd_organics import Molecule
        >>> benzene = Molecule(num_mols=10, smiles="c1ccccc1")
        >>> benzene.coarse_grain(beads={"A": "c1ccccc1"})

        Warnings
        --------
        The changes applied by coarse grain are in-place. All molecule
        properties will be modified after coarse graining based on the bead
        mapping.

        """
        for comp in self.molecules:
            cg_comp = CG_Compound(comp, beads=beads)
            if cg_comp.mapping:
                self._cg_molecules.append(cg_comp)
            else:
                raise ValueError(
                    "Unable to coarse grain the molecule. "
                    "Please check the bead types."
                )
        if self._cg_molecules:
            self.gmso_molecule = self._convert_to_gmso(self._cg_molecules[0])
            self._identify_topology_information(self.gmso_molecule)

    def _load(self):
        """Load the molecule from the provided input."""
        if self.compound:
            if isinstance(self.compound, mb.Compound):
                return mb.clone(mb.clone(self.compound))
            if isinstance(self.compound, Topology):
                return to_mbuild(self.compound)
            else:
                raise MoleculeLoadError(
                    msg=f"Unsupported compound type {type(self.compound)}. "
                    f"Supported compound types are: {str(mb.Compound)}"
                )
        if self.file:
            if isinstance(self.file, str) and os.path.isfile(self.file):
                return mb.load(self.file)
            else:
                raise MoleculeLoadError(
                    msg=f"Unable to load the molecule from file {self.file}."
                )

        if self.smiles:
            if isinstance(self.smiles, str):
                return mb.load(self.smiles, smiles=True)
            else:
                raise MoleculeLoadError(
                    msg=f"Unable to load the molecule from smiles "
                    f"{self.smiles}."
                )

    def _generate(self):
        """Generate all the molecules by replicating the loaded molecule."""
        for i in range(self.n_mols):
            self._molecules.append(self._load())

    def _convert_to_gmso(self, mb_molecule):
        """Convert the mbuild molecule to a GMSO topology."""
        topology = from_mbuild(mb_molecule)
        topology.identify_connections()
        return topology

    def _identify_particle_information(self, gmso_molecule):
        """Identify the particle information from the GMSO topology.

        Particle information includes particle types, hydrogen types, particle
        type indices, and particle charges.

        Parameters
        ----------
        gmso_molecule : GMSO Topology, required
            The GMSO topology of the molecule.

        """
        self.particle_types = []
        self.hydrogen_types = []
        self.particle_typeid = []
        self.particle_charge = []
        for site in gmso_molecule.sites:
            p_name = getattr(site.atom_type, "name", None) or site.name
            if p_name not in self.particle_types:
                self.particle_types.append(p_name)
            if (
                site.element
                and site.element.atomic_number == 1
                and p_name not in self.hydrogen_types
            ):
                self.hydrogen_types.append(p_name)
            self.particle_typeid.append(self.particle_types.index(p_name))
            self.particle_charge.append(
                site.charge.to_value() if site.charge else 0
            )

    def _identify_pairs(self, particle_types):
        """Identify all unique particle pairs from the particle types.

        Parameters
        ----------
        particle_types : list, required
            List of all particle types.

        """
        self.pairs = set(
            itertools.combinations_with_replacement(particle_types, 2)
        )

    def _identify_bond_types(self, gmso_molecule):
        """Identify all unique bond types from the GMSO topology.

        Parameters
        ----------
        gmso_molecule : GMSO Topology, required
            The GMSO topology of the molecule.

        """
        self.bond_types = set()
        for bond in gmso_molecule.bonds:
            p1_name = (
                getattr(bond.connection_members[0].atom_type, "name", None)
                or bond.connection_members[0].name
            )
            p2_name = (
                getattr(bond.connection_members[1].atom_type, "name", None)
                or bond.connection_members[1].name
            )
            bond_connections = [p1_name, p2_name]
            if not tuple(bond_connections[::-1]) in self.bond_types:
                self.bond_types.add(tuple(bond_connections))

    def _identify_angle_types(self, gmso_molecule):
        """Identify all unique angle types from the GMSO topology.

        Parameters
        ----------
        gmso_molecule : GMSO Topology, required
            The GMSO topology of the molecule.

        """
        self.angle_types = set()
        for angle in gmso_molecule.angles:
            p1_name = (
                getattr(angle.connection_members[0].atom_type, "name", None)
                or angle.connection_members[0].name
            )
            p2_name = (
                getattr(angle.connection_members[1].atom_type, "name", None)
                or angle.connection_members[1].name
            )
            p3_name = (
                getattr(angle.connection_members[2].atom_type, "name", None)
                or angle.connection_members[2].name
            )
            angle_connections = [p1_name, p2_name, p3_name]
            if not tuple(angle_connections[::-1]) in self.angle_types:
                self.angle_types.add(tuple(angle_connections))

    def _identify_dihedral_types(self, gmso_molecule):
        """Identify all unique dihedral types from the GMSO topology.

        Parameters
        ----------
        gmso_molecule : GMSO Topology, required
            The GMSO topology of the molecule.

        """
        self.dihedral_types = set()
        for dihedral in gmso_molecule.dihedrals:
            p1_name = (
                getattr(dihedral.connection_members[0].atom_type, "name", None)
                or dihedral.connection_members[0].name
            )
            p2_name = (
                getattr(dihedral.connection_members[1].atom_type, "name", None)
                or dihedral.connection_members[1].name
            )
            p3_name = (
                getattr(dihedral.connection_members[2].atom_type, "name", None)
                or dihedral.connection_members[2].name
            )
            p4_name = (
                getattr(dihedral.connection_members[3].atom_type, "name", None)
                or dihedral.connection_members[3].name
            )
            dihedral_connections = [p1_name, p2_name, p3_name, p4_name]
            if not tuple(dihedral_connections[::-1]) in self.dihedral_types:
                self.dihedral_types.add(tuple(dihedral_connections))

    def _identify_improper_types(self, gmso_molecule):
        """Identify all unique improper types from the GMSO topology.

        Parameters
        ----------
        gmso_molecule : GMSO Topology, required
            The GMSO topology of the molecule.

        """
        self.improper_types = set()
        for improper in gmso_molecule.impropers:
            p1_name = (
                getattr(improper.connection_members[0].atom_type, "name", None)
                or improper.connection_members[0].name
            )
            p2_name = (
                getattr(improper.connection_members[1].atom_type, "name", None)
                or improper.connection_members[1].name
            )
            p3_name = (
                getattr(improper.connection_members[2].atom_type, "name", None)
                or improper.connection_members[2].name
            )
            p4_name = (
                getattr(improper.connection_members[3].atom_type, "name", None)
                or improper.connection_members[3].name
            )
            improper_connections = [p1_name, p2_name, p3_name, p4_name]
            if not tuple(improper_connections[::-1]) in self.improper_types:
                self.improper_types.add(tuple(improper_connections))

    def _identify_topology_information(self, gmso_molecule):
        """Identify all topology information from the GMSO topology.

        Parameters
        ----------
        gmso_molecule : GMSO Topology, required
            The GMSO topology of the molecule.

        """
        self._identify_particle_information(gmso_molecule)
        self._identify_pairs(self.particle_types)
        self._identify_bond_types(gmso_molecule)
        self._identify_angle_types(gmso_molecule)
        self._identify_dihedral_types(gmso_molecule)
        self._identify_improper_types(gmso_molecule)

    def _validate_force_field(self):
        """Validate the force field for the molecule."""
        if isinstance(self.force_field, BaseXMLForcefield):
            self.gmso_molecule = apply(
                self.gmso_molecule,
                self.force_field.gmso_ff,
                identify_connections=True,
                speedup_by_moltag=True,
                speedup_by_molgraph=False,
            )
            # Update topology information from typed gmso after applying ff.
            self._identify_topology_information(self.gmso_molecule)
        elif isinstance(self.force_field, BaseHOOMDForcefield):
            _validate_hoomd_ff(
                self.force_field.hoomd_forces, self.topology_information
            )
        elif isinstance(self.force_field, List):
            _validate_hoomd_ff(self.force_field, self.topology_information)
        else:
            raise ForceFieldError(
                "Unsupported forcefield type. Forcefields "
                "should be a subclass of "
                "`hoomd_organics.base.forcefield.BaseXMLForcefield` or "
                "`hoomd_organics.base.forcefield.BaseHOOMDForcefield` or a "
                "list of `hoomd.md.force.Force` objects. \n"
                "Please check `hoomd_organics.library.forcefields` for "
                "examples of supported forcefields."
            )


class Polymer(Molecule):
    """Builds a polymer from a monomer.

    Parameters
    ----------
    lengths : int, required
        The total number of monomers in each chain.
    num_mols : int, required
        Number of chains to generate.
    smiles : str, default None
        The smiles string of the monomer to generate.
    file : str, default None
        The file path to the monomer to generate.
    force_field : str, default None
        The force field to apply to the molecule.
    bond_indices: list, default None
        The indices of the atoms to bond.
    bond_length: float, default None
        The bond length between connected atoms (units: nm)
    bond_orientation: list, default None
        The orientation of the bond between connected atoms.

    """

    def __init__(
        self,
        lengths,
        num_mols,
        smiles=None,
        file=None,
        force_field=None,
        bond_indices=None,
        bond_length=None,
        bond_orientation=None,
        **kwargs,
    ):
        self.lengths = check_return_iterable(lengths)
        self.bond_indices = bond_indices
        self.bond_length = bond_length
        self.bond_orientation = bond_orientation
        num_mols = check_return_iterable(num_mols)
        if len(num_mols) != len(self.lengths):
            raise ValueError("Number of molecules and lengths must be equal.")
        super(Polymer, self).__init__(
            num_mols=num_mols,
            smiles=smiles,
            file=file,
            force_field=force_field,
            **kwargs,
        )

    @property
    def monomer(self):
        """The monomer of the polymer."""
        return self._mb_molecule

    def _build(self, length):
        chain = mbPolymer()
        chain.add_monomer(
            self.monomer,
            indices=self.bond_indices,
            separation=self.bond_length,
            orientation=self.bond_orientation,
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
    monomer_A : hoomd_organics.molecules.Polymer, required
        Class of the A-type monomer
    monomer_B : hoomd_organics.molecules.Polymer, required
        Class of the B-type monomer
    length : int, required
        The total number of monomers in the molecule
    sequence : str, default None
        Manually define the sequence of 'A' and 'B' monomers.
        Leave as None if generating random sequences.
        Example: sequence = "AABAABAAB"
    random_sequence : bool, default False
        Creates a random 'A' 'B' sequence as a function of the AB_ratio.
    AB_ratio : float, default 0.50
        The relative weight of A to B monomer types.
        Used when generating random sequences.
    seed : int, default 24
        Set the seed used when generating random sequences

    """

    def __init__(
        self,
        monomer_A,
        monomer_B,
        lengths,
        num_mols,
        force_field=None,
        sequence=None,
        random_sequence=False,
        AB_ratio=0.50,
        seed=24,
    ):
        self.lengths = check_return_iterable(lengths)
        self.monomer_A = monomer_A(lengths=[1], num_mols=[1])
        self.monomer_B = monomer_B(lengths=[1], num_mols=[1])
        num_mols = check_return_iterable(num_mols)
        if len(num_mols) != len(self.lengths):
            raise ValueError("Number of molecules and lengths must be equal.")
        self.sequence = sequence
        self.random_sequence = random_sequence
        self.AB_ratio = AB_ratio
        self.seed = seed
        self._A_count = 0
        self._B_count = 0
        self.smiles = [self.monomer_A.smiles, self.monomer_B.smiles]
        self.file = [self.monomer_A.file, self.monomer_B.file]
        random.seed(self.seed)
        super(CoPolymer, self).__init__(
            num_mols=num_mols,
            smiles=self.smiles,
            file=self.file,
            force_field=force_field,
        )

    @property
    def A_ratio(self):
        """The ratio of A monomers to B monomers in the CoPolymer."""
        return self._A_count / (self._A_count + self._B_count)

    @property
    def B_ratio(self):
        """The ratio of B monomers to A monomers in the CoPolymer."""
        return self._B_count / (self._A_count + self._B_count)

    def _build(self, length, sequence):
        chain = mbPolymer()
        chain.add_monomer(
            self.monomer_A.monomer,
            indices=self.monomer_A.bond_indices,
            orientation=self.monomer_A.bond_orientation,
            separation=self.monomer_A.bond_length,
        )
        chain.add_monomer(
            self.monomer_B.monomer,
            indices=self.monomer_B.bond_indices,
            orientation=self.monomer_B.bond_orientation,
            separation=self.monomer_B.bond_length,
        )
        chain.build(n=length, sequence=sequence)
        return chain

    def _load(self):
        return None

    def _generate(self):
        for idx, length in enumerate(self.lengths):
            for i in range(self.n_mols[idx]):
                if self.random_sequence:
                    sequence = random.choices(
                        ["A", "B"], [self.AB_ratio, 1 - self.AB_ratio], k=length
                    )
                    self._A_count += sequence.count("A")
                    self._B_count += sequence.count("B")
                    _length = 1
                else:
                    sequence = self.sequence
                    _length = length
                mol = self._build(length=_length, sequence=sequence)
                self._molecules.append(mol)
