import warnings
from abc import ABC, abstractmethod
from typing import List, Union, Optional

import mbuild as mb
import numpy as np
import unyt
from gmso.external import from_mbuild, to_parmed, from_parmed, to_gsd_snapshot, to_hoomd_forcefield
from mbuild.formats.hoomd_forcefield import create_hoomd_forcefield

#from hoomd_polymers import Molecule
from hoomd_polymers.base.molecule import Molecule
from hoomd_polymers.utils import scale_charges
from hoomd_polymers.utils.ff_utils import find_xml_ff, apply_xml_ff, _validate_hoomd_ff


class System(ABC):
    """Base class from which other systems inherit.

    Parameters
    ----------
    molecule : hoomd_polymers.molecule; required
    n_mols : int; required
        The number of times to replicate molecule in the system
    density : float; optional; default None
        The desired density of the system (g/cm^3). Used to set the
        target_box attribute. Can be useful when initializing
        systems at low denisty and running a shrink simulaton
        to acheive a target density.
    """
    def __init__(
            self,
            molecules: Union[List, Molecule],
            force_field: Optional[Union[List, str]],
            density: float,
            r_cut: float,
            auto_scale=False,
            base_units=None
    ):
        self.density = density
        self.r_cut = r_cut
        self.auto_scale = auto_scale
        self.base_units = base_units
        self.target_box = None
        self.typed_system = None
        self._hoomd_snapshot = None
        self._hoomd_forcefield = None
        self._reference_values = dict() 
        self.force_field = None
        self._mol_forcefields = set()
        self.molecules = []

        #ToDo: create an instance of the Molecule class and validate forcefield
        if isinstance(molecules, List):
            for mol in molecules:
                if isinstance(mol, Molecule):
                    self.molecules.extend(mol.molecules)
                    self._mol_forcefields.add(mol.force_field)
                else:
                    self.molecules.extend(mol)
        elif isinstance(molecules, Molecule):
            self.molecules = molecules.molecules
            self._mol_forcefields.add(mol.force_field)

        self.system = self._build_system()
        self.gmso_system = self._convert_to_gmso()

    @abstractmethod
    def _build_system(self):
        pass

    @property
    def n_molecules(self):
        return len(self.molecules)

    @property
    def n_particles(self):
        return sum([mol.n_particles for mol in self.molecules])

    @property
    def mass(self):
        return sum(mol.mass for mol in self.molecules)

    @property
    def box(self):
        return self.system.box

    @property
    def reference_length(self):
        return self._reference_values.get("length", None)

    @property
    def reference_mass(self):
        return self._reference_values.get("mass", None)

    @property
    def reference_energy(self):
        return self._reference_values.get("energy", None)

    @property
    def reference_values(self):
        return self._reference_values

    @reference_length.setter
    def reference_length(self, length):
        self._reference_values["length"] = length

    @reference_energy.setter
    def reference_length(self, energy):
        self._reference_values["energy"] = energy 

    @reference_mass.setter
    def reference_length(self, mass):
        self._reference_values["mass"] = mass 

    @reference_values.setter
    def reference_values(self, ref_value_dict):
        self._reference_values = ref_value_dict

    @property
    def hoomd_snapshot(self):
        if not self._hoomd_snapshot:
            self._hoomd_snapshot = self._create_hoomd_snapshot()
        return self._hoomd_snapshot
    
    @property
    def hoomd_forcefield(self):
        if not self._hoomd_forcefield:
            self._hoomd_forcefield = self._create_hoomd_forcefield()
        return self._hoomd_forcefield

    def remove_hydrogens(self):
        """Call this method to remove hydrogen atoms from the system.
        The masses and charges of the hydrogens are absorbed into 
        the heavy atoms they were bonded to.
        """
        parmed_struc = to_parmed(self.gmso_system)
        # Try by element first:
        hydrogens = [a for a in parmed_struc.atoms if a.element == 1]
        if len(hydrogens) == 0: # Try by mass
            hydrogens = [a for a in parmed_struc.atoms if a.mass == 1.008]
            if len(hydrogens) == 0:
                warnings.warn(
                        "Hydrogen atoms could not be found by element or mass"
                )
        for h in hydrogens:
            h.atomic_number = 1
            bonded_atom = h.bond_partners[0]
            bonded_atom.mass += h.mass
            bonded_atom.charge += h.charge
        parmed_struc.strip(
                [a.atomic_number == 1 for a in parmed_struc.atoms]
        )
        self.gmso_system = from_parmed(parmed_struc)
        if self._hoomd_snapshot:
            self._hoomd_snapshot = self._create_hoomd_snapshot()
        if self._hoomd_forcefield:
            self._hoomd_forcefield = self._create_hoomd_forcefield()

    def remove_charges(self):
        pass

    def scale_charges(self):
        pass

    def to_gsd(self, file_name):
        with gsd.hoomd.open(file_name, "wb") as traj:
            traj.append(self.hoomd_snapshot)

    def _convert_to_gmso(self):
        topology = from_mbuild(self.system)
        topology.identify_connections()
        return topology

    def _create_hoomd_forcefield(self):
        force_list = []
        ff, refs = to_hoomd_forcefield(
                top=self.gmso_system,
                r_cut=self.r_cut,
                base_units=self._reference_values
        )
        for force in ff:
            force_list.extend(ff[force])
        return force_list

    def _create_hoomd_snapshot(self):
        snap, refs = to_gsd_snapshot(
                top=self.gmso_system,
                auto_scale=self.auto_scale,
                base_units=self._reference_values
        )
        return snap
    
    #TODO: Change this to a hidden function; add conditional based on ff types 
    def apply_forcefield(self):
        ff_xml_path, ff_type = find_xml_ff(tuple(self._mol_forcefields)[0])
        self.gmso_system = apply_xml_ff(ff_xml_path, self.gmso_system)
        if self.auto_scale:
            epsilons = [s.atom_type.parameters["epsilon"] for s in self.gmso_system.sites]
            sigmas = [s.atom_type.parameters["sigma"] for s in self.gmso_system.sites]
            masses = [s.mass for s in self.gmso_system.sites]
            self._reference_values["energy"] = np.max(epsilons) * epsilons[0].unit_array
            self._reference_values["length"] = np.max(sigmas) * sigmas[0].unit_array
            self._reference_values["mass"] = np.max(masses) * masses[0].unit_array.to("amu")

    def _apply_forcefield(
            self,
            forcefield,
            remove_hydrogens=False,
            scale_parameters=True,
            remove_charges=False,
            make_charge_neutral=False,
            r_cut=2.5
    ):
        if len(self.molecules) == 1:
            use_residue_map = True
        else:
            use_residue_map = False
        self.typed_system = forcefield.apply(
                structure=self.system, use_residue_map=use_residue_map
        )
        if remove_hydrogens:
            print("Removing hydrogen atoms and adjusting heavy atoms")
            # Try by element first:
            hydrogens = [a for a in self.typed_system.atoms if a.element == 1]
            if len(hydrogens) == 0: # Try by mass
                hydrogens = [a for a in self.typed_system.atoms if a.mass == 1.008]
                if len(hydrogens) == 0:
                    warnings.warn(
                            "Hydrogen atoms could not be found by element or mass"
                    )
            for h in hydrogens:
                h.atomic_number = 1
                bonded_atom = h.bond_partners[0]
                bonded_atom.mass += h.mass
                bonded_atom.charge += h.charge
            self.typed_system.strip(
                    [a.atomic_number == 1 for a in self.typed_system.atoms]
            )
        if remove_charges:
            for atom in self.typed_system.atoms:
                atom.charge = 0
        if make_charge_neutral and not remove_charges:
            print("Adjust charges to make system charge neutral")
            new_charges = scale_charges(
                    charges=np.array([a.charge for a in self.typed_system.atoms]),
                    n_particles=len(self.typed_system.atoms)
            )
            for idx, charge in enumerate(new_charges):
                self.typed_system.atoms[idx].charge = charge

        init_snap, forcefield, refs = create_hoomd_forcefield(
                structure=self.typed_system,
                r_cut=r_cut,
                auto_scale=scale_parameters
        )
        self._hoomd_objects = [init_snap, forcefield]
        self._reference_values = refs

    def set_target_box(
            self, x_constraint=None, y_constraint=None, z_constraint=None
    ):
        """Set the target volume of the system during
        the initial shrink step.
        If no constraints are set, the target box is cubic.
        Setting constraints will hold those box vectors
        constant and adjust others to match the target density.

        Parameters
        -----------
        x_constraint : float, optional, defualt=None
            Fixes the box length (nm) along the x axis
        y_constraint : float, optional, default=None
            Fixes the box length (nm) along the y axis
        z_constraint : float, optional, default=None
            Fixes the box length (nm) along the z axis

        """
        if not any([x_constraint, y_constraint, z_constraint]):
            Lx = Ly = Lz = self._calculate_L()
        else:
            constraints = np.array([x_constraint, y_constraint, z_constraint])
            fixed_L = constraints[np.where(constraints!=None)]
            #Conv from nm to cm for _calculate_L
            fixed_L *= 1e-7
            L = self._calculate_L(fixed_L = fixed_L)
            constraints[np.where(constraints==None)] = L
            Lx, Ly, Lz = constraints

        self.target_box = np.array([Lx, Ly, Lz])

    def visualize(self):
        if self.system:
            self.system.visualize().show()
        else:
            raise ValueError(
                    "The initial configuraiton has not been created yet."
            )

    def _calculate_L(self, fixed_L=None):
        """Calculates the required box length(s) given the
        mass of a sytem and the target density.

        Box edge length constraints can be set by set_target_box().
        If constraints are set, this will solve for the required
        lengths of the remaining non-constrained edges to match
        the target density.

        Parameters
        ----------
        fixed_L : np.array, optional, defualt=None
            Array of fixed box lengths to be accounted for
            when solving for L

        """
        # Convert from amu to grams
        M = self.mass * 1.66054e-24
        vol = (M / self.density) # cm^3
        if fixed_L is None:
            L = vol**(1/3)
        else:
            L = vol / np.prod(fixed_L)
            if len(fixed_L) == 1: # L is cm^2
                L = L**(1/2)
        # Convert from cm back to nm
        L *= 1e7
        return L



class Pack(System):
    """Uses PACKMOL via mbuild.packing.fill_box.
    The box used for packing is expanded to allow PACKMOL to place all of the molecules.

    Parameters
    ----------
    packing_expand_factor : int; optional, default 5

    """
    def __init__(
            self,
            molecules: Union[List, Molecule],
            force_field: Optional[Union[List, str]],
            density: float,
            r_cut: float,
            auto_scale=False,
            base_units=None,
            packing_expand_factor=5,
            edge=0.2,
    ):
        self.packing_expand_factor = packing_expand_factor
        self.edge = edge
        super(Pack, self).__init__(
                molecules=molecules,
                density=density,
                force_field=force_field,
                r_cut=r_cut,
                auto_scale=auto_scale,
                base_units=base_units
        )

    def _build_system(self):
        self.set_target_box()
        system = mb.packing.fill_box(
                compound=self.molecules,
                n_compounds=[1 for i in self.molecules],
                box=list(self.target_box*self.packing_expand_factor),
                overlap=0.2,
                edge=self.edge
        )
        return system


class Lattice(System):
    """Places the molecules in a lattice configuration.
    Assumes two molecules per unit cell.

    Parameters
    ----------
    x : float; required
        The distance (nm) between lattice points in the x direction.
    y : float; required
        The distance (nm) between lattice points in the y direction.
    n : int; required
        The number of times to repeat the unit cell in x and y
    lattice_vector : array-like
        The vector between points in the unit cell
    """
    def __init__(
            self,
            molecules,
            density,
            x,
            y,
            n,
            basis_vector=[0.5, 0.5, 0],
            z_adjust=1.0,
    ):
        self.x = x
        self.y = y
        self.n = n
        self.basis_vector = basis_vector
        super(Lattice, self).__init__(molecules=molecules, density=density)

    def _build_system(self):
        next_idx = 0
        system = mb.Compound()
        for i in range(self.n):
            layer = mb.Compound()
            for j in range(self.n):
                try:
                    comp1 = self.molecules[next_idx]
                    comp2 = self.molecules[next_idx + 1]
                    comp2.translate(self.basis_vector)
                    unit_cell = mb.Compound(subcompounds=[comp1, comp2])
                    unit_cell.translate((0, self.y*j, 0))
                    layer.add(unit_cell)
                    next_idx += 2
                except IndexError:
                    pass
            layer.translate((self.x*i, 0, 0))
            self.system.add(layer)
        bounding_box = self.system.get_boundingbox()
        x_len = bounding_box.lengths[0]
        y_len = bounding_box.lengths[1]
        self.set_target_box(x_constraint=x_len, y_constraint=y_len)
        return system
