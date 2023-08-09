import warnings
from abc import ABC, abstractmethod
from typing import List

import gsd
import mbuild as mb
import numpy as np
import unyt as u
from gmso.external import from_mbuild, to_parmed, from_parmed, to_gsd_snapshot, \
    to_hoomd_forcefield
from gmso.parameterization import apply

from hoomd_polymers.base.molecule import Molecule
from hoomd_polymers.utils import xml_to_gmso_ff, check_return_iterable, FF_Types
from hoomd_polymers.utils.exceptions import MoleculeLoadError, \
    ReferenceUnitError, ForceFieldError


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
            molecules,
            density: float,
            r_cut: float,
            force_field=None,
            auto_scale=False,
            remove_hydrogens=False,
            remove_charges=False,
            scale_charges=False,
            base_units=None
    ):
        self._molecules = check_return_iterable(molecules)
        self._force_field = None
        if force_field:
            self._force_field = check_return_iterable(force_field)
        self.density = density
        self.r_cut = r_cut
        self.auto_scale = auto_scale
        self.remove_hydrogens = remove_hydrogens
        self.remove_charges = remove_charges
        self.scale_charges = scale_charges
        self.target_box = None
        self.all_molecules = []
        self._hoomd_snapshot = None
        self._hoomd_forcefield = []
        self.base_units = base_units
        self._reference_values = dict()
        self._gmso_forcefields_dict = dict()
        self.gmso_system = None

        # Collecting all molecules
        self.n_mol_types = 0
        for mol_item in self._molecules:
            if isinstance(mol_item, Molecule):
                mol_item.assign_mol_name(str(self.n_mol_types))
                self.all_molecules.extend(mol_item.molecules)
                # if ff is provided in Molecule class
                if mol_item.force_field:
                    if mol_item.ff_type == FF_Types.Hoomd:
                        self._hoomd_forcefield.extend(mol_item.force_field)
                    else:
                        self._gmso_forcefields_dict[
                            str(self.n_mol_types)] = xml_to_gmso_ff(
                            mol_item.force_field)
                self.n_mol_types += 1
            elif isinstance(mol_item, mb.Compound):
                mol_item.name = str(self.n_mol_types)
                self.all_molecules.append(mol_item)
            elif isinstance(mol_item, List):
                for sub_mol in mol_item:
                    if isinstance(sub_mol, mb.Compound):
                        sub_mol.name = str(self.n_mol_types)
                        self.all_molecules.append(sub_mol)
                    else:
                        raise MoleculeLoadError(
                            msg=f"Unsupported compound type {type(sub_mol)}. "
                                f"Supported compound types are: {str(mb.Compound)}")
                self.n_mol_types += 1

        # Collecting all force-fields only if xml force-field is provided
        if self._force_field:
            for i in range(self.n_mol_types):
                if not self._gmso_forcefields_dict.get(str(i)):
                    if i < len(self._force_field):
                        # if there is a ff for each molecule type
                        ff_index = i
                    else:
                        # if there is only one ff for all molecule types
                        ff_index = 0
                    if getattr(self._force_field[ff_index], "gmso_ff"):
                        self._gmso_forcefields_dict[str(i)] = self._force_field[
                            ff_index].gmso_ff
                    else:
                        raise ForceFieldError(
                            msg=f"GMSO Force field in {self._force_field[ff_index]} is not provided.")
        self.system = self._build_system()
        self.gmso_system = self._convert_to_gmso()
        self._apply_forcefield()

    @abstractmethod
    def _build_system(self):
        pass

    @property
    def n_molecules(self):
        return len(self.all_molecules)

    @property
    def n_particles(self):
        if self.gmso_system:
            return self.gmso_system.n_sites
        return sum([mol.n_particles for mol in self.all_molecules])

    @property
    def mass(self):
        if self.gmso_system:
            return sum(float(site.mass.to("amu").value) for site in
                       self.gmso_system.sites)
        return sum(mol.mass for mol in self.all_molecules)

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
    def reference_length(self, length, unit=None):
        if isinstance(length, u.array.unyt_quantity):
            self._reference_values["length"] = length
        elif isinstance(unit, str) and (
                isinstance(length, float) or isinstance(length, int)):
            self._reference_values["length"] = length * getattr(u, unit)
        else:
            raise ReferenceUnitError(
                f"Invalid reference length input.Please provide reference length (number) and "
                f"unit (string) or pass length value as an {str(u.array.unyt_quantity)}.")

    @reference_energy.setter
    def reference_energy(self, energy, unit=None):
        if isinstance(energy, u.array.unyt_quantity):
            self._reference_values["energy"] = energy
        elif isinstance(unit, str) and (
                isinstance(energy, float) or isinstance(energy, int)):
            self._reference_values["energy"] = energy * getattr(u, unit)
        else:
            raise ReferenceUnitError(
                f"Invalid reference energy input.Please provide reference energy (number) and "
                f"unit (string) or pass energy value as an {str(u.array.unyt_quantity)}.")

    @reference_mass.setter
    def reference_mass(self, mass, unit=None):
        if isinstance(mass, u.array.unyt_quantity):
            self._reference_values["mass"] = mass
        elif isinstance(unit, str) and (
                isinstance(mass, float) or isinstance(mass, int)):
            self._reference_values["mass"] = mass * getattr(u, unit)
        else:
            raise ReferenceUnitError(
                f"Invalid reference mass input.Please provide reference mass (number) and "
                f"unit (string) or pass mass value as an {str(u.array.unyt_quantity)}.")

    @reference_values.setter
    def reference_values(self, ref_value_dict):
        ref_keys = ["length", "mass", "energy"]
        for k in ref_keys:
            if k not in ref_value_dict.keys():
                raise ValueError(f"Missing reference for {k}.")
            if not isinstance(ref_value_dict[k], u.array.unyt_quantity):
                raise ReferenceUnitError(
                    f"{k} reference value must be of type {str(u.array.unyt_quantity)}")
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

    def _remove_hydrogens(self):
        """Call this method to remove hydrogen atoms from the system.
        The masses and charges of the hydrogens are absorbed into
        the heavy atoms they were bonded to.
        """
        parmed_struc = to_parmed(self.gmso_system)
        # Try by element first:
        hydrogens = [a for a in parmed_struc.atoms if a.element == 1]
        if len(hydrogens) == 0:  # Try by mass
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
        if len(hydrogens) > 0:
            self.gmso_system = from_parmed(parmed_struc)

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
            auto_scale=self.auto_scale,
            base_units=self._reference_values if self.auto_scale else self.base_units
        )
        for force in ff:
            force_list.extend(ff[force])
        return force_list

    def _create_hoomd_snapshot(self):
        snap, refs = to_gsd_snapshot(
            top=self.gmso_system,
            auto_scale=self.auto_scale,
            base_units=self._reference_values if self.auto_scale else self.base_units
        )
        return snap

    def _apply_forcefield(self):
        if self._force_field:
            self.gmso_system = apply(
                self.gmso_system,
                self._gmso_forcefields_dict,
                identify_connections=True,
                use_molecule_info=True
            )
        if self.remove_charges:
            for site in self.gmso_system.sites:
                site.charge = 0
        if self.scale_charges and not self.remove_charges:
            pass
            # TODO: Scale charges from self.gmso_system
        if self.auto_scale:
            epsilons = [s.atom_type.parameters["epsilon"] for s in
                        self.gmso_system.sites]
            sigmas = [s.atom_type.parameters["sigma"] for s in
                      self.gmso_system.sites]
            masses = [s.mass for s in self.gmso_system.sites]
            self._reference_values["energy"] = np.max(epsilons) * epsilons[
                0].unit_array
            self._reference_values["length"] = np.max(sigmas) * sigmas[
                0].unit_array
            self._reference_values["mass"] = np.max(masses) * masses[
                0].unit_array.to("amu")

        if self.remove_hydrogens:
            self._remove_hydrogens()

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
            fixed_L = constraints[np.where(constraints != None)]
            # Conv from nm to cm for _calculate_L
            fixed_L *= 1e-7
            L = self._calculate_L(fixed_L=fixed_L)
            constraints[np.where(constraints == None)] = L
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
        vol = (M / self.density)  # cm^3
        if fixed_L is None:
            L = vol ** (1 / 3)
        else:
            L = vol / np.prod(fixed_L)
            if len(fixed_L) == 1:  # L is cm^2
                L = L ** (1 / 2)
        # Convert from cm back to nm
        L *= 1e7
        return L


class Pack(System):
    """Uses PACKMOL via mbuild.packing.fill_box.
    The box used for packing is expanded to allow PACKMOL
    to more easily place all the molecules.

    Parameters
    ----------
    packing_expand_factor : int; optional, default 5

    """

    def __init__(
            self,
            molecules,
            density: float,
            r_cut: float,
            force_field=None,
            auto_scale=False,
            remove_hydrogens=False,
            remove_charges=False,
            scale_charges=False,
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
            remove_hydrogens=remove_hydrogens,
            remove_charges=remove_charges,
            scale_charges=scale_charges,
            base_units=base_units
        )

    def _build_system(self):
        self.set_target_box()
        system = mb.packing.fill_box(
            compound=self.all_molecules,
            n_compounds=[1 for i in self.all_molecules],
            box=list(self.target_box * self.packing_expand_factor),
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
            density: float,
            r_cut: float,
            x: float,
            y: float,
            n: int,
            basis_vector=[0.5, 0.5, 0],
            force_field=None,
            auto_scale=False,
            remove_hydrogens=False,
            remove_charges=False,
            scale_charges=False,
            base_units=None,
    ):
        self.x = x
        self.y = y
        self.n = n
        self.basis_vector = basis_vector
        super(Lattice, self).__init__(
            molecules=molecules,
            density=density,
            force_field=force_field,
            r_cut=r_cut,
            auto_scale=auto_scale,
            remove_hydrogens=remove_hydrogens,
            remove_charges=remove_charges,
            scale_charges=scale_charges,
            base_units=base_units
        )

    def _build_system(self):
        next_idx = 0
        system = mb.Compound()
        for i in range(self.n):
            layer = mb.Compound()
            for j in range(self.n):
                try:
                    comp1 = self.all_molecules[next_idx]
                    comp2 = self.all_molecules[next_idx + 1]
                    comp2.translate(self.basis_vector)
                    unit_cell = mb.Compound(subcompounds=[comp1, comp2])
                    unit_cell.translate((0, self.y * j, 0))
                    layer.add(unit_cell)
                    next_idx += 2
                except IndexError:
                    pass
            layer.translate((self.x * i, 0, 0))
            system.add(layer)
        bounding_box = system.get_boundingbox()
        x_len = bounding_box.lengths[0]
        y_len = bounding_box.lengths[1]
        self.set_target_box(x_constraint=x_len, y_constraint=y_len)
        return system
