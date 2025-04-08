"""System class for arranging Molecules into a box."""

import pickle
import warnings
from abc import ABC, abstractmethod
from typing import List, Union

import gsd
import mbuild as mb
import numpy as np
import unyt as u
from gmso.external import from_mbuild, to_gsd_snapshot, to_hoomd_forcefield
from gmso.parameterization import apply

from flowermd.base.forcefield import BaseHOOMDForcefield, BaseXMLForcefield
from flowermd.base.molecule import Molecule
from flowermd.internal import Units, check_return_iterable, validate_unit
from flowermd.internal.exceptions import ForceFieldError, MoleculeLoadError
from flowermd.utils import (
    get_target_box_mass_density,
    get_target_box_number_density,
)


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
    molecules : flowermd.Molecule or a list of Molecule objects, required
        The molecules to be placed in the system.
    base_units : dict, default {}
        Dictionary of base units to use for scaling.
        Dictionary keys are "length", "mass", and "energy". Values should be an
        unyt array of the desired base unit.
    kwargs
        See classes that inherit from System for kwargs

    Warnings
    --------
    The ``force_field`` parameter in ``System`` class must be initialized
    from the pre-defined force field classes in
    ``flowermd.library.forcefields`` module that are based on xml-based
    force fields.

    Forcefields defined as a list of `Hoomd.md.force.Force
    <https://hoomd-blue.readthedocs.io/en/stable/module-md-force.html>`_ objects
    must be directly passed to the ``flowermd.Simulation`` class.
    Please refer to the ``Simulation`` class documentation for more details.


    """

    def __init__(
        self,
        molecules,
        base_units=dict(),
        **kwargs,
    ):
        self._molecules = check_return_iterable(molecules)
        self.all_molecules = []
        self.gmso_system = None
        self._reference_values = base_units
        self._hoomd_snapshot = None
        self._hoomd_forcefield = []
        self._gmso_forcefields_dict = dict()
        # Reference values used when last writing snapshot and forcefields
        self._ff_refs = dict()
        self._snap_refs = dict()
        self._ff_kwargs = dict()
        self.auto_scale = False

        # Collecting all molecules
        self.n_mol_types = 0
        self._mol_type_idx = []
        for mol_item in self._molecules:
            if isinstance(mol_item, Molecule):
                # keep track of molecule types indices to assign to sites
                # before applying forcefield
                self._mol_type_idx.extend(
                    [self.n_mol_types] * mol_item.n_particles
                )
                self.all_molecules.extend(mol_item.molecules)
                # if ff is provided in the Molecule class use that as the ff
                if mol_item.force_field:
                    if isinstance(mol_item.force_field, BaseHOOMDForcefield):
                        self._hoomd_forcefield.extend(
                            mol_item.force_field.hoomd_forces
                        )
                    elif isinstance(mol_item.force_field, BaseXMLForcefield):
                        self._gmso_forcefields_dict[str(self.n_mol_types)] = (
                            mol_item.force_field.gmso_ff
                        )
                    elif isinstance(mol_item.force_field, list):
                        self._hoomd_forcefield.extend(mol_item.force_field)
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

        # Create mBuild system
        self.system = self._build_system(**kwargs)
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
        if self.gmso_system:
            return self.gmso_system.n_sites
        else:
            return sum(mol.n_particles for mol in self.all_molecules)

    @property
    def mass(self):
        """Total mass of the system in amu."""
        if self.gmso_system:
            return sum(site.mass.to("amu") for site in self.gmso_system.sites)
        return sum(mol.mass * u.Unit("amu") for mol in self.all_molecules)

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
        length : reference length * `flowermd.Units`, required
            The reference length of the system.
            It can be provided in the following form of:
            value * `flowermd.Units`, for example 1 * `flowermd.Units.angstrom`.

        """
        if self.auto_scale:
            warnings.warn(
                "`auto_scale` was set to True for this system. "
                "Setting reference length manually disables auto "
                "scaling."
            )
        validated_length = validate_unit(length, u.dimensions.length)
        self._reference_values["length"] = validated_length

    @reference_energy.setter
    def reference_energy(self, energy):
        """Set the reference energy of the system along with a unit of energy.

        Parameters
        ----------
        energy : reference energy * `flowermd.Units`, required
            The reference energy of the system.
            It can be provided in the following form of:
            value * `flowermd.Units`, for example 1 * `flowermd.Units.kcal/mol`.

        """
        if self.auto_scale:
            warnings.warn(
                "`auto_scale` was set to True for this system. "
                "Setting reference energy manually disables auto "
                "scaling."
            )
        validated_energy = validate_unit(energy, u.dimensions.energy)
        self._reference_values["energy"] = validated_energy

    @reference_mass.setter
    def reference_mass(self, mass):
        """Set the reference mass of the system along with a unit of mass.

        Parameters
        ----------
        mass : reference mass * `flowermd.Units`, required
            The reference mass of the system.
            It can be provided in the following form of:
            value * `flowermd.Units`, for example 1 * `flowermd.Units.amu`.
        """
        if self.auto_scale:
            warnings.warn(
                "`auto_scale` was set to True for this system. "
                "Setting reference mass manually disables auto "
                "scaling."
            )
        validated_mass = validate_unit(mass, u.dimensions.mass)
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
        if self.auto_scale:
            warnings.warn(
                "`auto_scale` was set to True for this system. "
                "Setting reference values manually disables auto "
                "scaling."
            )

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
        if (
            self._ff_refs != self.reference_values
            and self._gmso_forcefields_dict
        ):
            self._hoomd_forcefield = self._create_hoomd_forcefield(
                **self._ff_kwargs
            )
        return self._hoomd_forcefield

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
        # If a snapshot was already made, need to re-create it w/o hydrogens
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
                site.charge -= abs(
                    site.charge if site.charge else 0 * u.Unit("C")
                ) * (net_charge / abs_charge)

    def pickle_forcefield(self, file_path="forcefield.pickle"):
        """Pickle the list of HOOMD forces.

        This method useful for saving the Hoomd force objects
        generated by System.apply_forcefield() which can then
        be used to initialize a Simulation at a later time.

        Parameters
        ----------
        file_path : str, default "forcefield.pickle"
            The path to save the pickle file to.

        """
        if not self.hoomd_forcefield:
            raise ValueError(
                "A forcefield has not yet been applied. "
                "See System.apply_forcefield()"
            )
        f = open(file_path, "wb")
        pickle.dump(self.hoomd_forcefield, f)

    def to_gsd(self, file_name):
        """Write the system's `hoomd_snapshot` to a GSD file."""
        with gsd.hoomd.open(file_name, "w") as traj:
            traj.append(self.hoomd_snapshot)

    def save_reference_values(self, file_path="reference_values.pickle"):
        """Save the reference values of the system to a pickle file.

        Parameters
        ----------
        file_path : str, default "reference_values.pickle"
            The path to save the pickle file to.

        """
        if not self.reference_values:
            raise ValueError(
                "Reference values have not been set. "
                "See System.reference_values"
            )
        f = open(file_path, "wb")
        pickle.dump(self.reference_values, f)

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
            auto_scale=False,
            base_units=(
                self._reference_values if self._reference_values else None
            ),
        )
        for force in ff:
            force_list.extend(ff[force])
        self._ff_refs = self._reference_values.copy()
        return force_list

    def _create_hoomd_snapshot(self):
        """Create a HOOMD snapshot."""
        snap, refs = to_gsd_snapshot(
            top=self.gmso_system,
            auto_scale=False,
            base_units=(
                self._reference_values if self._reference_values else None
            ),
        )
        self._snap_refs = self._reference_values.copy()
        return snap

    def _validate_forcefield(self, input_forcefield):
        if input_forcefield is None and not self._gmso_forcefields_dict:
            raise ForceFieldError(
                "Forcefield is not provided. Valid forcefield "
                "must be provided either during Molecule "
                "initialization or when calling the "
                "`apply_forcefield` method of the System "
                "class."
            )

        if input_forcefield and self._gmso_forcefields_dict:
            raise ForceFieldError(
                "Forcefield is provided both during Molecule "
                "initialization and when calling the "
                "`apply_forcefield` method of the System "
                "class. Please provide the forcefield only "
                "once."
            )

        if input_forcefield:
            _force_field = check_return_iterable(input_forcefield)
            if not all(
                isinstance(ff, (BaseHOOMDForcefield, BaseXMLForcefield))
                for ff in _force_field
            ):
                raise ForceFieldError(
                    "Forcefield must be an instance of either "
                    " `BaseHOOMDForcefield` or "
                    "`BaseXMLForcefield`. \n"
                    "Please check "
                    "`flowermd.library.forcefields` for "
                    "examples of supported forcefields."
                )
            # Collecting all force-fields into a dict with mol_type index as key
            for i in range(self.n_mol_types):
                if not self._gmso_forcefields_dict.get(str(i)):
                    if i < len(_force_field):
                        # if there is a ff for each molecule type
                        ff_index = i
                    else:
                        # if there is only one ff for all molecule types
                        ff_index = 0
                    self._gmso_forcefields_dict[str(i)] = _force_field[
                        ff_index
                    ].gmso_ff

    def _assign_site_mol_type_idx(self):
        """Assign molecule type index to the gmso sites."""
        for i, site in enumerate(self.gmso_system.sites):
            site.group = str(self._mol_type_idx[i])

    def apply_forcefield(
        self,
        r_cut,
        force_field=None,
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
        force_field : flowermd.ForceField or a list of ForceField objects,
                default=None
            The force field to be applied to the system for parameterization.
            If a list of force fields is provided, the length of the list must
            be equal to the number of molecule types in the system.
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
        self.auto_scale = auto_scale
        self._validate_forcefield(force_field)

        if self._gmso_forcefields_dict:
            # assign names to all the gmso sites based on mol_type to
            # match the keys in self._gmso_forcefields_dict before applying ff
            self._assign_site_mol_type_idx()
        self.gmso_system = apply(
            self.gmso_system,
            self._gmso_forcefields_dict,
            match_ff_by="group",
            identify_connections=True,
            speedup_by_moltag=True,
            speedup_by_molgraph=False,
        )

        if remove_charges:
            for site in self.gmso_system.sites:
                site.charge = 0

        if scale_charges and not remove_charges:
            self._scale_charges()

        if not self._reference_values:
            epsilons = [
                s.atom_type.parameters["epsilon"]
                for s in self.gmso_system.sites
            ]
            sigmas = [
                s.atom_type.parameters["sigma"] for s in self.gmso_system.sites
            ]
            masses = [s.mass for s in self.gmso_system.sites]

            energy_scale = np.max(epsilons) if self.auto_scale else 1.0
            length_scale = np.max(sigmas) if self.auto_scale else 1.0
            mass_scale = np.max(masses) if self.auto_scale else 1.0

            self._reference_values["energy"] = (
                energy_scale * epsilons[0].unit_array
            )
            self._reference_values["length"] = (
                length_scale * sigmas[0].unit_array
            )
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

    def visualize(self):
        """Visualize the system."""
        if self.system:
            self.system.visualize().show()
        else:
            raise ValueError(
                "The initial configuraiton has not been created yet."
            )


class Pack(System):
    """Uses PACKMOL via mbuild.packing.fill_box.

    The box used for packing is expanded to allow PACKMOL
    to more easily place all the molecules.

    Parameters
    ----------
    density : float or unyt_quantity or flowermd.internal.Units, required
        The desired density of the system. Used to set the
        target_box attribute. Can be useful when initializing
        systems at low density and running a shrink simulation
        to achieve a target density. If no unit is provided, assuming the
        density is in g/cm**3.
    packing_expand_factor : int, default 5
        The factor by which to expand the box for packing.
    edge : float, default 0.2
        The space (nm) between the edge of the box and the molecules.
    overlap : float, default 0.2
        Minimum separation (nm) between particles of different molecules.
    seed : int, default 12345
        Change seed to be passed to PACKMOL for different starting positions
    kwargs
        Arguments to be passed into mbuild.packing.fill_box


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
        density: Union[int, float, u.unyt_quantity, u.unyt_array, Units],
        base_units=dict(),
        packing_expand_factor=5,
        edge=0.2,
        overlap=0.2,
        seed=12345,
        fix_orientation=False,
        **kwargs,
    ):
        if isinstance(density, (int, float)):
            warnings.warn(
                "Units for density were not given, assuming units of g/cm**3."
            )
            self.density = density * Units.g_cm3
        else:
            self.density = density
        self.packing_expand_factor = packing_expand_factor
        self.edge = edge
        self.overlap = overlap
        self.seed = seed
        self.fix_orientation = fix_orientation
        super(Pack, self).__init__(
            molecules=molecules, base_units=base_units, **kwargs
        )

    def _build_system(self, **kwargs):
        mass_density = Units.kg_m3
        number_density = Units.n_m3
        if self.density.units.dimensions == mass_density.dimensions:
            target_box = get_target_box_mass_density(
                density=self.density, mass=self.mass
            ).to("nm")
        elif self.density.units.dimensions == number_density.dimensions:
            target_box = get_target_box_number_density(
                density=self.density, n_beads=self.n_particles
            ).to("nm")
        else:
            raise ValueError(
                f"Density dimensions of {self.density.units.dimensions} "
                "were given, but only mass density "
                f"({mass_density.dimensions}) and "
                f"number density ({number_density.dimensions}) are supported."
            )

        system = mb.packing.fill_box(
            compound=self.all_molecules,
            n_compounds=[1 for i in self.all_molecules],
            box=list(target_box * self.packing_expand_factor),
            overlap=self.overlap,
            seed=self.seed,
            edge=self.edge,
            fix_orientation=self.fix_orientation,
            **kwargs,
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

    Notes
    -----
    The system is built in a way that the long axis of the
    molecules is aligned with the z direction, and the
    lattice is made by repeating and translating in the
    x and y directions.

    See the `periodic_bond_axis` paramter in `flowermd.base.Polymer`
    if you wish to form head-tail bonds across the periodic boundary
    in the lattice.

    """

    def __init__(
        self,
        molecules,
        x: float,
        y: float,
        n: int,
        basis_vector=[0.5, 0.5, 0],
        base_units=dict(),
    ):
        self.x = x
        self.y = y
        self.n = n
        self.basis_vector = basis_vector
        super(Lattice, self).__init__(
            molecules=molecules,
            base_units=base_units,
        )

    def _build_system(self):
        for mol in self._molecules:
            mol._align_backbones_z_axis(heavy_atoms_only=True)
        next_idx = 0
        system = mb.Compound()
        for i in range(self.n):
            layer = mb.Compound()
            for j in range(self.n):
                try:
                    comp1 = self.all_molecules[next_idx]
                    comp2 = self.all_molecules[next_idx + 1]
                    comp2.translate(
                        self.basis_vector * np.array([self.x, self.y, 0])
                    )
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
        # Center the lattice in its box
        system.box = mb.box.Box(np.array([x_len, y_len, z_len]))
        system.translate_to(
            (system.box.Lx / 2, system.box.Ly / 2, system.box.Lz / 2)
        )
        return system
