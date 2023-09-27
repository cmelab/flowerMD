"""System class for arranging Molecules into a box."""
import warnings
from abc import ABC, abstractmethod
from typing import List

import gsd
import mbuild as mb
import numpy as np
import unyt as u
from gmso.external import from_mbuild, to_gsd_snapshot, to_hoomd_forcefield
from gmso.parameterization import apply

from hoomd_organics.base.molecule import Molecule
from hoomd_organics.utils import (
    FF_Types,
    calculate_box_length,
    check_return_iterable,
    validate_ref_value,
    xml_to_gmso_ff,
)
from hoomd_organics.utils.exceptions import ForceFieldError, MoleculeLoadError


class System(ABC):
    """
    Base class from which other systems inherit.

    System class is used to create a system of molecules and arrange them into
    a box. If a force field is provided, the system will be parameterized.
    Two important properties of the system are `hoomd_snapshot`, which is the
    snapshot of the system in HOOMD format, and `hoomd_forcefield`, which is
    the list of HOOMD forces. These properties will be used to initialize the
    simulation object later.

    System class is used to create a system of molecules and arrange them into
    a box. If a force field is provided, the system will be parameterized.
    Two important properties of the system are `hoomd_snapshot`, which is the
    snapshot of the system in HOOMD format, and `hoomd_forcefield`, which is
    the list of HOOMD forces. These properties will be used to initialize the
    simulation object later.

    Parameters
    ----------
    molecules : hoomd_organics.Molecule or a list of Molecule objects, required
        The molecules to be placed in the system.
    density : float, required
        The desired density of the system (g/cm^3). Used to set the
        target_box attribute. Can be useful when initializing
        systems at low density and running a shrink simulation
        to achieve a target density.
    r_cut : float, required
        The cutoff radius for the Lennard-Jones potential.
    force_field : hoomd_organics.ForceField or a list of ForceField objects,
                default=None
        The force field to be applied to the system for parameterization.
    auto_scale : bool, default=False
        Set to true to use reduced simulation units.
        distance, mass, and energy are scaled by the largest value
        present in the system for each.
    remove_hydrogens : bool, default False
        Set to true to remove hydrogen atoms from the system.
        The masses and charges of the hydrogens are absorbed into
        the heavy atoms they were bonded to.
    remove_charges : bool, default False
        Set to true to remove charges from the system.
    scale_charges : bool, default False
        Set to true to scale charges to net zero.
    base_units : dict, default {}
        Dictionary of base units to use for scaling.
        Dictionary keys are "length", "mass", and "energy". Values should be an
        unyt array of the desired base unit.

    Warnings
    --------
    The ``force_field`` parameter in ``System`` class must be initialized
    from the pre-defined force field classes in
    ``hoomd_organics.library.forcefields`` module that are based on xml-based
    force fields.

    Forcefields defined as a list of `Hoomd.md.force.Force
    <https://hoomd-blue.readthedocs.io/en/stable/module-md-force.html>`_ objects
    must be directly passed to the ``hoomd_organics.Simulation`` class.
    Please refer to the ``Simulation`` class documentation for more details.


    """

    def __init__(
        self,
        molecules,
        density: float,
        force_field=None,
        auto_scale=False,
        base_units=dict(),
    ):
        self._molecules = check_return_iterable(molecules)
        self._force_field = None
        if force_field:
            self._force_field = check_return_iterable(force_field)
        self.density = density
        self.auto_scale = auto_scale
        self.all_molecules = []
        self.gmso_system = None
        self._reference_values = base_units
        self._hoomd_snapshot = None
        self._hoomd_forcefield = []
        self._gmso_forcefields_dict = dict()
        self._target_box = None
        # Reference values used when last writing snapshot and forcefields
        self._ff_refs = dict()
        self._snap_refs = dict()
        self._ff_kwargs = dict()

        # Collecting all molecules
        self.n_mol_types = 0
        for mol_item in self._molecules:
            if isinstance(mol_item, Molecule):
                if self._force_field:
                    mol_item._assign_mol_name(str(self.n_mol_types))
                self.all_molecules.extend(mol_item.molecules)
                # if ff is provided in Molecule class
                if mol_item.force_field:
                    if mol_item.ff_type == FF_Types.Hoomd:
                        self._hoomd_forcefield.extend(mol_item.force_field)
                    else:
                        self._gmso_forcefields_dict[
                            str(self.n_mol_types)
                        ] = xml_to_gmso_ff(mol_item.force_field)
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
                            f"Supported compound types are: "
                            f"{str(mb.Compound)}"
                        )
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
                            ff_index
                        ].gmso_ff
                    else:
                        raise ForceFieldError(
                            msg=f"GMSO Force field in "
                            f"{self._force_field[ff_index]} is not "
                            f"provided."
                        )
        # Create mBuild system
        self.system = self._build_system()
        # Create GMSO topology
        self.gmso_system = self._convert_to_gmso()

    @abstractmethod
    def _build_system(self):
        """Abstract method to arrange molecules into a box."""
        pass

    @property
    def n_molecules(self):
        """Total number of molecules in the system."""
        return len(self.all_molecules)

    @property
    def n_particles(self):
        """Total number of particles in the system."""
        return self.gmso_system.n_sites

    @property
    def mass(self):
        """Total mass of the system in amu."""
        if self.gmso_system:
            return sum(
                float(site.mass.to("amu").value)
                for site in self.gmso_system.sites
            )
        return sum(mol.mass for mol in self.all_molecules)

    @property
    def net_charge(self):
        """Net charge of the system."""
        return sum(
            site.charge if site.charge else 0 for site in self.gmso_system.sites
        )

    @property
    def box(self):
        """The box of the system."""
        # TODO: Use gmso system here?
        return self.system.box

    @property
    def reference_length(self):
        """The reference length and unit of the system.

        If `auto_scale` is set to True, this is the length factor that is used
        to scale the system to reduced units. If `auto_scale` is set to False,
        the default value is 1.0 with the unit of nm.

        """
        return self._reference_values.get("length", None)

    @property
    def reference_mass(self):
        """The reference mass and unit of the system.

        If `auto_scale` is set to True, this is the mass factor that is used
        to scale the system to reduced units. If `auto_scale` is set to False,
        the default value is 1.0 with the unit of amu.

        """
        return self._reference_values.get("mass", None)

    @property
    def reference_energy(self):
        """The reference energy and unit of the system.

        If `auto_scale` is set to True, this is the energy factor that is used
        to scale the system to reduced units. If `auto_scale` is set to False,
        the default value is 1.0 with the unit of kJ/mol.
        """
        return self._reference_values.get("energy", None)

    @property
    def reference_values(self):
        """The reference values of the system in form of a dictionary."""
        return self._reference_values

    @reference_length.setter
    def reference_length(self, length):
        """Set the reference length of the system along with a unit of length.

        Parameters
        ----------
        length : string or unyt.unyt_quantity, required
            The reference length of the system.
            It can be provided in the following forms:
            1) A string with the format of "value unit", for example "1 nm".
            2) A unyt.unyt_quantity object with the correct dimension. For
            example, unyt.unyt_quantity(1, "nm").

        """
        validated_length = validate_ref_value(length, u.dimensions.length)
        self._reference_values["length"] = validated_length

    @reference_energy.setter
    def reference_energy(self, energy):
        """Set the reference energy of the system along with a unit of energy.

        Parameters
        ----------
        energy : string or unyt.unyt_quantity, required
            The reference energy of the system.
            It can be provided in the following forms:
            1) A string with the format of "value unit", for example "1 kJ/mol".
            2) A unyt.unyt_quantity object with the correct dimension. For
            example, unyt.unyt_quantity(1, "kJ/mol").

        """
        validated_energy = validate_ref_value(energy, u.dimensions.energy)
        self._reference_values["energy"] = validated_energy

    @reference_mass.setter
    def reference_mass(self, mass):
        """Set the reference mass of the system along with a unit of mass.

        Parameters
        ----------
        mass : string or unyt.unyt_quantity, required
            The reference mass of the system.
            It can be provided in the following forms:
            1) A string with the format of "value unit", for example "1 amu".
            2) A unyt.unyt_quantity object with the correct dimension. For
            example, unyt.unyt_quantity(1, "amu").

        """
        validated_mass = validate_ref_value(mass, u.dimensions.mass)
        self._reference_values["mass"] = validated_mass

    @reference_values.setter
    def reference_values(self, ref_value_dict):
        """Set all the reference values of the system at once as a dictionary.

        Parameters
        ----------
        ref_value_dict : dict, required
            A dictionary of reference values. The keys of the dictionary must
            be "length", "mass", and "energy". The values of the dictionary
            should follow the same format as the values of the reference
            length, mass, and energy.

        """
        ref_keys = ["length", "mass", "energy"]
        for k in ref_keys:
            if k not in ref_value_dict.keys():
                raise ValueError(f"Missing reference for {k}.")
            self.__setattr__(f"reference_{k}", ref_value_dict[k])

    @property
    def hoomd_snapshot(self):
        """The snapshot of the system in form of a HOOMD snapshot."""
        if self._snap_refs != self.reference_values:
            self._hoomd_snapshot = self._create_hoomd_snapshot()
        if self._hoomd_snapshot is None:  # Hasn't been created yet
            self._hoomd_snapshot = self._create_hoomd_snapshot()
        return self._hoomd_snapshot

    @property
    def hoomd_forcefield(self):
        """List of HOOMD forces."""
        if self._ff_refs != self.reference_values and self._force_field:
            self._hoomd_forcefield = self._create_hoomd_forcefield(
                **self._ff_kwargs
            )
        return self._hoomd_forcefield

    @property
    def target_box(self):
        """The target box size of the system in form of a numpy array.

        If reference length is set, the target box is in reduced units.

        Notes
        -----
        The `target_box` property can be passed to
        `hoomd_orgaics.base.Simulation.run_update_volume` method to reach the
        target density.

        """
        if self.reference_length:
            return self._target_box / self.reference_length.value
        else:
            return self._target_box

    def remove_hydrogens(self):
        """Call this method to remove hydrogen atoms from the system.

        The masses and charges of the hydrogens are absorbed into
        the heavy atoms they were bonded to.

        """
        # Try by element first:
        hydrogens = [
            site
            for site in self.gmso_system.sites
            if site.element.atomic_number == 1
        ]
        # If none found by element; try by mass
        if len(hydrogens) == 0:
            hydrogens = [
                site
                for site in self.gmso_system.sites
                if site.mass.to("amu").value == 1.008
            ]
            if len(hydrogens) == 0:
                warnings.warn(
                    "Hydrogen atoms could not be found by element or mass"
                )
        for h in hydrogens:
            # Find bond and other site in bond, add mass and charge
            for bond in self.gmso_system.iter_connections_by_site(
                site=h, connections=["bonds"]
            ):
                for site in bond.connection_members:
                    if site is not h:
                        site.mass += h.mass
                        site.charge += h.charge
            self.gmso_system.remove_site(site=h)
        # If a snap shot was already made, need to re-create it w/o hydrogens
        if self._hoomd_snapshot:
            self._create_hoomd_snapshot()

    def _scale_charges(self):
        """Scale charges to net zero.

        If the net charge does not sum to zero after applying the forcefield,
        this method equally shifts negative charges and positive charges
        across all particles to reach a net charge of zero.

        """
        charges = np.array(
            [
                site.charge if site.charge else 0
                for site in self.gmso_system.sites
            ]
        )
        net_charge = sum(charges)
        abs_charge = sum(abs(charges))
        if abs_charge != 0:
            for site in self.gmso_system.sites:
                site.charge -= abs(site.charge if site.charge else 0) * (
                    net_charge / abs_charge
                )

    def to_gsd(self, file_name):
        """Write the system's `hoomd_snapshot` to a GSD file."""
        with gsd.hoomd.open(file_name, "wb") as traj:
            traj.append(self.hoomd_snapshot)

    def _convert_to_gmso(self):
        """Convert the mbuild system to a gmso system."""
        topology = from_mbuild(self.system)
        topology.identify_connections()
        return topology

    def _create_hoomd_forcefield(self, r_cut, nlist_buffer, pppm_kwargs):
        """Create a list of HOOMD forces."""
        force_list = []
        ff, refs = to_hoomd_forcefield(
            top=self.gmso_system,
            r_cut=r_cut,
            nlist_buffer=nlist_buffer,
            pppm_kwargs=pppm_kwargs,
            auto_scale=self.auto_scale,
            base_units=self._reference_values
            if self._reference_values
            else None,
        )
        for force in ff:
            force_list.extend(ff[force])
        self._ff_refs = self._reference_values.copy()
        return force_list

    def _create_hoomd_snapshot(self):
        """Create a HOOMD snapshot."""
        snap, refs = to_gsd_snapshot(
            top=self.gmso_system,
            auto_scale=self.auto_scale,
            base_units=self._reference_values
            if self._reference_values
            else None,
        )
        self._snap_refs = self._reference_values.copy()
        return snap

    def apply_forcefield(
        self,
        r_cut,
        auto_scale=False,
        scale_charges=False,
        remove_charges=False,
        remove_hydrogens=False,
        pppm_resolution=(8, 8, 8),
        pppm_order=4,
        nlist_buffer=0.4,
    ):
        """Apply the forcefield to the system.

        Parameters
        ----------
        r_cut : float
            The cutoff radius for the Lennard-Jones interactions.

        auto_scale : bool, default=False
            Set to true to use reduced simulation units.
            distance, mass, and energy are scaled by the largest value
            present in the system for each.
        scale_charges : bool, default False
            Set to true to scale charges to net zero.
        remove_charges : bool, default False
            Set to true to remove charges from the system.
        remove_hydrogens : bool, default False
            Set to true to remove hydrogen atoms from the system.
            The masses and charges of the hydrogens are absorbed into
            the heavy atoms they were bonded to.
        pppm_resolution : tuple, default=(8, 8, 8)
            The resolution used in
            `hoomd.md.long_range.pppm.make_pppm_coulomb_force` representing
            number of grid points in the x, y, and z directions.
        ppmp_order : int, default=4
            The order used in
            `hoomd.md.long_range.pppm.make_pppm_coulomb_force` representing
            number of grid points in each direction to assign charges to.
        nlist_buffer : float, default=0.4
            Neighborlist buffer for simulation cell.

        """
        if not self._force_field:
            # TODO: Better erorr message
            raise ValueError(
                "This method can only be used when the System is "
                "initialized with an XML type forcefield."
            )
        self.gmso_system = apply(
            self.gmso_system,
            self._gmso_forcefields_dict,
            identify_connections=True,
            speedup_by_moltag=True,
            speedup_by_molgraph=False,
        )

        if remove_charges:
            for site in self.gmso_system.sites:
                site.charge = 0

        if scale_charges and not remove_charges:
            self._scale_charges()

        epsilons = [
            s.atom_type.parameters["epsilon"] for s in self.gmso_system.sites
        ]
        sigmas = [
            s.atom_type.parameters["sigma"] for s in self.gmso_system.sites
        ]
        masses = [s.mass for s in self.gmso_system.sites]

        energy_scale = np.max(epsilons) if self.auto_scale else 1.0
        length_scale = np.max(sigmas) if self.auto_scale else 1.0
        mass_scale = np.max(masses) if self.auto_scale else 1.0

        self._reference_values["energy"] = energy_scale * epsilons[0].unit_array
        self._reference_values["length"] = length_scale * sigmas[0].unit_array
        self._reference_values["mass"] = mass_scale * masses[0].unit_array

        if remove_hydrogens:
            self.remove_hydrogens()

        pppm_kwargs = {"resolution": pppm_resolution, "order": pppm_order}
        self._ff_kwargs = {
            "r_cut": r_cut,
            "nlist_buffer": nlist_buffer,
            "pppm_kwargs": pppm_kwargs,
        }
        self._hoomd_forcefield = self._create_hoomd_forcefield(
            r_cut=r_cut, nlist_buffer=nlist_buffer, pppm_kwargs=pppm_kwargs
        )
        self._hoomd_snapshot = self._create_hoomd_snapshot()

    def set_target_box(
        self, x_constraint=None, y_constraint=None, z_constraint=None
    ):
        """Set the target box size of the system.

        If no constraints are set, the target box is cubic.
        Setting constraints will hold those box vectors
        constant and adjust others to match the target density.

        Parameters
        ----------
        x_constraint : float, optional, defualt=None
            Fixes the box length (nm) along the x axis.
        y_constraint : float, optional, default=None
            Fixes the box length (nm) along the y axis.
        z_constraint : float, optional, default=None
            Fixes the box length (nm) along the z axis.

        """
        if not any([x_constraint, y_constraint, z_constraint]):
            Lx = Ly = Lz = self._calculate_L()
        else:
            constraints = np.array([x_constraint, y_constraint, z_constraint])
            fixed_L = constraints[np.not_equal(constraints, None).nonzero()]
            # Conv from nm to cm for _calculate_L
            fixed_L *= 1e-7
            L = self._calculate_L(fixed_L=fixed_L)
            constraints[np.equal(constraints, None).nonzero()] = L
            Lx, Ly, Lz = constraints

        self._target_box = np.array([Lx, Ly, Lz])

    def visualize(self):
        """Visualize the system."""
        if self.system:
            self.system.visualize().show()
        else:
            raise ValueError(
                "The initial configuraiton has not been created yet."
            )

    def _calculate_L(self, fixed_L=None):
        """Calculate the box length.

        Calculate the required box length(s) given the mass of a system and
        the target density.
        Box edge length constraints can be set by set_target_box().
        If constraints are set, this will solve for the required
        lengths of the remaining non-constrained edges to match
        the target density.

        Parameters
        ----------
        fixed_L : np.array, optional, defualt=None
            Array of fixed box lengths to be accounted for when solving for L.

        """
        mass_quantity = u.unyt_quantity(self.mass, u.g / u.mol).to("g")
        density_quantity = u.unyt_quantity(self.density, u.g / u.cm**3)
        if fixed_L is not None:
            fixed_L = u.unyt_array(fixed_L, u.cm)
        L = calculate_box_length(
            mass_quantity, density_quantity, fixed_L=fixed_L
        )
        return L.to("nm").value


class Pack(System):
    """Uses PACKMOL via mbuild.packing.fill_box.

    The box used for packing is expanded to allow PACKMOL
    to more easily place all the molecules.

    Parameters
    ----------
    packing_expand_factor : int, default 5
        The factor by which to expand the box for packing.
    edge : float, default 0.2
        The space (nm) between the edge of the box and the molecules.


    .. warning::

        Note that the default `packing_expand_factor` for pack is 5, which means
        that the box density will not be the same as the specified density.
        This is because in some cases PACKMOL will not be able to fit all the
        molecules into the box if the target box is too small, therefore, we
        need to expand the box by a factor (default:5) to allow PACKMOL to fit
        all the molecules.

        In order to get the specified density there are two options:

        1. set the `packing_expand_factor` to 1, which will not expand the box.
        However, this may result in PACKMOL errors if the box is too small.

        2. Update the box volume after creating the simulation object to the
        target box length. This property is called `target_box`.

    """

    def __init__(
        self,
        molecules,
        density: float,
        force_field=None,
        auto_scale=False,
        base_units=dict(),
        packing_expand_factor=5,
        edge=0.2,
    ):
        self.packing_expand_factor = packing_expand_factor
        self.edge = edge
        super(Pack, self).__init__(
            molecules=molecules,
            density=density,
            force_field=force_field,
            auto_scale=auto_scale,
            base_units=base_units,
        )

    def _build_system(self):
        self.set_target_box()
        system = mb.packing.fill_box(
            compound=self.all_molecules,
            n_compounds=[1 for i in self.all_molecules],
            box=list(self._target_box * self.packing_expand_factor),
            overlap=0.2,
            edge=self.edge,
        )
        return system


class Lattice(System):
    """Places the molecules in a lattice configuration.

    Assumes two molecules per unit cell.

    Parameters
    ----------
    x : float, required
        The distance (nm) between lattice points in the x direction.
    y : float, required
        The distance (nm) between lattice points in the y direction.
    n : int, required
        The number of times to repeat the unit cell in x and y.
    basis_vector : array-like, default [0.5, 0.5, 0]
        The vector between points in the unit cell.

    """

    def __init__(
        self,
        molecules,
        density: float,
        x: float,
        y: float,
        n: int,
        basis_vector=[0.5, 0.5, 0],
        force_field=None,
        auto_scale=False,
        base_units=dict(),
    ):
        self.x = x
        self.y = y
        self.n = n
        self.basis_vector = basis_vector
        super(Lattice, self).__init__(
            molecules=molecules,
            density=density,
            force_field=force_field,
            auto_scale=auto_scale,
            base_units=base_units,
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
                    # TODO: what if comp1 and comp2 have different names?
                    unit_cell = mb.Compound(
                        subcompounds=[comp1, comp2], name=comp1.name
                    )
                    unit_cell.translate((0, self.y * j, 0))
                    layer.add(unit_cell)
                    next_idx += 2
                except IndexError:
                    pass
            layer.translate((self.x * i, 0, 0))
            system.add(layer)
        bounding_box = system.get_boundingbox()
        # Add lattice constants to box lengths to account for boundaries
        x_len = bounding_box.lengths[0] + self.x
        y_len = bounding_box.lengths[1] + self.y
        z_len = bounding_box.lengths[2] + 0.2
        self.set_target_box(x_constraint=x_len, y_constraint=y_len)
        # Center the lattice in its box
        system.box = mb.box.Box(np.array([x_len, y_len, z_len]))
        system.translate_to(
            (system.box.Lx / 2, system.box.Ly / 2, system.box.Lz / 2)
        )
        return system
