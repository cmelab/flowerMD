import mbuild as mb
from mbuild.formats.hoomd_forcefield import create_hoomd_forcefield
import numpy as np
from numba import jit
import unyt


from hoomd_polymers.utils import scale_charges, check_return_iterable


class System:
    def __init__(self, molecule, density, n_mols, mol_kwargs={}):
        self.density = density
        self.n_mols = check_return_iterable(n_mols)
        self._molecules = check_return_iterable(molecule)
        self.mol_kwargs = check_return_iterable(mol_kwargs)
        selftarget_box = None
        self.system = None
        self.typed_system = None
        self._hoomd_objects = None
        self._reference_values = None
        self.molecules = []

        for mol, n, kw_args, in zip(
                self._molecules,
                self.n_mols,
                self.mol_kwargs
        ):
            for i in range(n):
                self.molecules.append(mol(**kw_args))

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

    def apply_forcefield(
            self,
            forcefield,
            remove_hydrogens=False,
            scale_parameters=True,
            remove_charges=False,
            make_charge_neutral=False
    ):
        if len(self._molecules) == 1:
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
                r_cut=2.5,
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
    def __init__(
            self,
            molecule,
            density,
            n_mols,
            mol_kwargs={},
            packing_expand_factor=5
    ):
        super(Pack, self).__init__(molecule, density, n_mols, mol_kwargs)
        self.packing_expand_factor = packing_expand_factor
        self._build()

    def _build(self):
        self.set_target_box()
        self.system = mb.packing.fill_box(
                compound=self.molecules,
                n_compounds=[1 for i in self.molecules],
                box=list(self.target_box*self.packing_expand_factor),
                overlap=0.2,
                edge=0.2
        )


class Lattice(System):
    def __init__(
            self,
            molecule,
            density,
            n_mols,
            x,
            y,
            n,
            mol_kwargs={},
            basis_vector=[0.5, 0.5, 0],
            z_adjust=1.0,
    ):
        super(Lattice, self).__init__(molecule, density, n_mols, mol_kwargs)
        self.x = x
        self.y = y
        self.n = n
        self.basis_vector = basis_vector
        self._build()

    def _build(self):
        next_idx = 0
        self.system = mb.Compound()
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
