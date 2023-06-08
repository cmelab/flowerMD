from abc import ABC, abstractmethod
from mbuild.formats.hoomd_forcefield import create_hoomd_forcefield
import numpy as np
import unyt
from gmso.external import from_mbuild, to_gsd_snapshot
import warnings

from hoomd_polymers.utils import scale_charges
from typing import List, Union, Optional
from hoomd_polymers import Molecule
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
    def __init__(self, molecule: Union[List, Molecule], force_field: Optional[Union[List, str]], density: float):
        self.density = density
        self.target_box = None
        self.typed_system = None
        self._hoomd_objects = None
        self._reference_values = None
        self.force_field = None
        self.molecules = []
        if isinstance(molecule, List):

            for mol_list in molecule:
                self.molecules.extend(mol_list)
        elif isinstance(molecule, Molecule):
            self.molecules = molecule.molecules

        if force_field:
            self.force_field = force_field
        elif isinstance(molecule, Molecule) and molecule.force_field:
            self.force_field = molecule.force_field

        self.system = self.build_system()

    @abstractmethod
    def build_system(self):
        pass


    @property
    def n_molecules(self):
        return len(self.molecules)

    @property
    def n_particles(self):
        return sum([mol.n_particles for mol in self.molecules])

    @property
    def mass(self):
        if not self.system:
            return sum(i.mass for i in self.molecules)
        else:
            return self.system.mass

    @property
    def box(self):
        return self.system.box

    @property
    def hoomd_snapshot(self):
        if not self._hoomd_objects:
            raise ValueError(
                    "The hoomd snapshot has not yet been created. "
                    "Create a Hoomd snapshot and forcefield by applying "
                    "a forcefield using System.apply_forcefield()."
            )
        else:
            return self._hoomd_objects[0]

    @property
    def hoomd_forcefield(self):
        if not self._hoomd_objects:
            raise ValueError(
                    "The hoomd forcefield has not yet been created. "
                    "Create a Hoomd snapshot and forcefield by applying "
                    "a forcefield using System.apply_forcefield()."
            )
        else:
            return self._hoomd_objects[1]

    @property
    def reference_distance(self):
        return self._reference_values.distance * unyt.angstrom

    @property
    def reference_mass(self):
        return self._reference_values.mass * unyt.amu

    @property
    def reference_energy(self):
        return self._reference_values.energy * unyt.kcal / unyt.mol
    
    def to_hoomd_snapshot(self, auto_scale=False, base_units=None):
        topology = from_mbuild(self.system)
        topology.identify_connections()
        snap, refs = to_gsd_snapshot(
                top=topology,
                auto_scale=auto_scale,
                base_units=base_units
        )
        return snap

    def to_gsd(self):
        pass

    def apply_forcefield(
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
