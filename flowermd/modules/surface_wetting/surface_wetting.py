"""Module for simulating surface wetting."""

import warnings

import gsd.hoomd
import hoomd
import numpy as np
import unyt as u

from flowermd.base import Simulation
from flowermd.internal import Units
from flowermd.modules.surface_wetting.utils import combine_forces
from flowermd.utils import HOOMDThermostats, get_target_box_mass_density


class DropletSimulation(Simulation):
    """Simulation which creates a droplet."""

    def __init__(
        self,
        initial_state,
        forcefield,
        reference_values=dict(),
        dt=0.0001,
        device=hoomd.device.auto_select(),
        seed=42,
        gsd_write_freq=1e4,
        gsd_file_name="trajectory.gsd",
        log_write_freq=1e3,
        log_file_name="log.txt",
        thermostat=HOOMDThermostats.MTTK,
    ):
        super(DropletSimulation, self).__init__(
            initial_state=initial_state,
            forcefield=forcefield,
            reference_values=reference_values,
            dt=dt,
            device=device,
            seed=seed,
            gsd_write_freq=gsd_write_freq,
            gsd_file_name=gsd_file_name,
            log_write_freq=log_write_freq,
            log_file_name=log_file_name,
            thermostat=thermostat,
        )

    def run_droplet(
        self,
        shrink_temperature,
        shrink_duration,
        shrink_period,
        shrink_density,
        expand_temperature,
        expand_duration,
        expand_period,
        hold_temperature,
        hold_duration,
        final_density,
        tau_kt,
    ):
        """Run droplet simulation.

        The steps for creating a droplet are:
        1. Shrink the box to a high density (i.e. `shrink_density`) at a high
        temperature (i.e. `shrink_temperature`) to get the droplet to form.
        2. Expand the box back to a low density (i.e. 'final_density') at a low
        temperature (i.e. `expand_temperature`). Keeping the temperature low will keep
        the droplet from falling apart.
        3. Run the simulation at the `final_density` and low temperature
        (i.e. `hold_temperature`) to equilibrate the droplet.


        Parameters
        ----------
        shrink_temperature : float or flowermd.internal.Units or hoomd.variant.Ramp, required
            The temperature to run the simulation at while shrinking.
        shrink_duration : int or flowermd.internal.Units, required
            The number of steps (unitless) or time length (with units) to run the simulation while shrinking.
        shrink_period : int or flowermd.internal.Units, required
            The number of steps (unitless) or time length (with units) between updates to the box size while shrinking.
        shrink_density : float or flowermd.internal.Units, required
            The high density to shrink the box to.
            Note: If unitless, the units of the density are in g/cm^3.
        expand_temperature : float or flowermd.internal.Units or hoomd.variant.Ramp, required
            The temperature to run the simulation at while expanding.
        expand_duration : int or flowermd.internal.Units, required
            The number of steps (unitless) or time length (with units) to run the simulation while expanding.
        expand_period : int or flowermd.internal.Units, required
            The number of steps (unitless) or time length (with units) between updates to the box size while expanding.
        hold_temperature : float or flowermd.internal.Units or hoomd.variant.Ramp, required
            The temperature to run the simulation at while equilibrating.
        hold_duration : int or flowermd.internal.Units, required
            The number of steps (unitless) or time length (with units) to run the simulation while equilibrating.
        final_density : float or flowermd.internal.Units, required
            The low density to equilibrate the box to.
            Note: If unitless, the units of the density are in g/cm^3.
        tau_kt : float, required
            The time constant for the thermostat.

        """
        # Shrink down to high density
        if not isinstance(
            shrink_density, (u.array.unyt_quantity, u.unyt_quantity, Units)
        ) and not isinstance(
            final_density, (u.array.unyt_quantity, u.unyt_quantity, Units)
        ):
            warnings.warn(
                "Units for density were not given, assuming units of g/cm**3."
            )
            target_box_shrink = get_target_box_mass_density(
                density=shrink_density * Units.g_cm3,
                mass=self.mass.to("g"),
            )
            target_box_final = get_target_box_mass_density(
                density=final_density * Units.g_cm3,
                mass=self.mass.to("g"),
            )
        else:
            mass_density = Units.kg_m3
            number_density = Units.n_m3
            if shrink_density.units.dimensions == mass_density.units.dimensions:
                target_box_shrink = get_target_box_mass_density(
                    density=shrink_density, mass=self.mass.to("g")
                )
                target_box_final = get_target_box_mass_density(
                    density=final_density, mass=self.mass.to("g")
                )
            elif (
                shrink_density.units.dimensions
                == number_density.units.dimensions
            ):
                raise ValueError(
                    "For now, only mass density is supported "
                    "in the surface wetting module."
                )
        # Shrink down to higher density
        self.run_update_volume(
            duration=shrink_duration,
            period=shrink_period,
            temperature=shrink_temperature,
            tau_kt=tau_kt,
            final_box_lengths=target_box_shrink,
            write_at_start=True,
        )
        # Expand back up to low density
        self.run_update_volume(
            duration=expand_duration,
            period=expand_period,
            temperature=expand_temperature,
            tau_kt=tau_kt,
            final_box_lengths=target_box_final,
        )
        # Run at low density
        self.run_NVT(
            duration=hold_duration, temperature=hold_temperature, tau_kt=tau_kt
        )


class InterfaceBuilder:
    """Builds an interface with droplet on top of a surface.

    Create the snapshot and forces for the surface droplet simulation.

    This class creates a new snapshot that combines the surface and droplet
    snapshots by putting the droplet particles on top of the surface particles
    with the correct spacing (i.e. gap) between the two.
    Also combines the hoomd forces from the surface and droplet simulations,
    and adds the forces for the new pair interactions between the droplet
    and surface particles.

    Droplet reference values are used as the reference to scale the mass,
    energy and length values in the new snapshot.

    Parameters
    ----------
    surface_snapshot : hoomd.snapshot.Snapshot or str, required
        A snapshot of the surface simulation, or a path to a GSD file
        of the surface simulation.
    surface_ff : List of hoomd.md.force.Force, required
        List of HOOMD force objects used in the surface simulation.
    drop_snapshot : hoomd.snapshot.Snapshot or str, required
        A snapshot of the droplet simulation, or a path
        to a GSD file of the droplet simulation.
    drop_ff : List of hoomd.md.force.Force, required
        List of HOOMD force objects used in the droplet simulation.
    drop_ref_values : dict, required
        Dictionary of reference values for the droplet simulation.
    box_height : unyt.unyt_quantity or float, required
        The height of the simulation box.
    gap : unyt.unyt_quantity or float, required
        The gap between the droplet and the surface.

    """

    def __init__(
        self,
        surface_snapshot,
        surface_ff,
        drop_snapshot,
        drop_ff,
        drop_ref_values,
        box_height,
        gap,
    ):
        if isinstance(drop_snapshot, str):
            drop_snapshot = gsd.hoomd.open(drop_snapshot)[-1]
        if isinstance(surface_snapshot, str):
            surface_snapshot = gsd.hoomd.open(surface_snapshot)[-1]
        self.surface_ff = surface_ff
        self.surface_snapshot = surface_snapshot
        self.drop_snapshot = drop_snapshot
        self.drop_ff = drop_ff
        self.reference_values = drop_ref_values
        self.box_height = box_height / drop_ref_values["length"]
        self.gap = gap / drop_ref_values["length"]

        self._surface_n = self.surface_snapshot.particles.N
        self._drop_n = self.drop_snapshot.particles.N

        # get snapshot of the combined system
        if set(self.surface_snapshot.particles.types).intersection(
            set(self.drop_snapshot.particles.types)
        ):
            raise NotImplementedError(
                "handle cases where there are common "
                "particle types between the surface and "
                "droplet."
            )
        self._wetting_snapshot = self._build_snapshot()

        # get forces of the combined system
        self._wetting_forces = combine_forces(
            self.drop_ff,
            self.surface_ff,
            self.drop_ptypes,
            self.surface_ptypes,
        )

    @property
    def hoomd_snapshot(self):
        """The snapshot containing the surface and droplet particles."""
        return self._wetting_snapshot

    @property
    def hoomd_forces(self):
        """The forces for the surface and droplet particles."""
        return self._wetting_forces

    def _build_snapshot(self):
        """Build a snapshot by combining the surface and droplet snapshots."""
        wetting_snapshot = gsd.hoomd.Frame()
        wetting_snapshot.particles.N = self._surface_n + self._drop_n

        # set up snapshot particles
        self.surface_ptypes = [
            f"_{ptype}" for ptype in self.surface_snapshot.particles.types
        ]
        self.drop_ptypes = self.drop_snapshot.particles.types

        wetting_snapshot.particles.types = (
            self.surface_ptypes + self.drop_ptypes
        )

        wetting_snapshot.particles.typeid = np.concatenate(
            (
                self.surface_snapshot.particles.typeid,
                self.drop_snapshot.particles.typeid + len(self.surface_ptypes),
            ),
            axis=None,
        )
        wetting_snapshot.particles.mass = np.concatenate(
            (
                self.surface_snapshot.particles.mass,
                self.drop_snapshot.particles.mass,
            ),
            axis=None,
        )
        wetting_snapshot.particles.charge = np.concatenate(
            (
                self.surface_snapshot.particles.charge,
                self.drop_snapshot.particles.charge,
            ),
            axis=None,
        )

        # create the surface wetting box
        wetting_box = self._create_box()
        wetting_snapshot.configuration.box = wetting_box
        # put the surface particles in the box and add droplet particles on top
        wetting_snapshot.particles.position = self._adjust_particle_positions(
            wetting_box
        )

        # set up bonds
        wetting_snapshot.bonds.N = (
            self.surface_snapshot.bonds.N + self.drop_snapshot.bonds.N
        )
        wetting_snapshot.bonds.types = (
            self.surface_snapshot.bonds.types + self.drop_snapshot.bonds.types
        )
        wetting_snapshot.bonds.typeid = np.concatenate(
            (
                self.surface_snapshot.bonds.typeid,
                self.drop_snapshot.bonds.typeid
                + len(self.surface_snapshot.bonds.types),
            ),
            axis=None,
        )
        wetting_snapshot.bonds.group = np.concatenate(
            (
                self.surface_snapshot.bonds.group,
                self.drop_snapshot.bonds.group + self._surface_n,
            ),
            axis=None,
        )

        # set up angles
        wetting_snapshot.angles.N = (
            self.surface_snapshot.angles.N + self.drop_snapshot.angles.N
        )
        wetting_snapshot.angles.types = (
            self.surface_snapshot.angles.types + self.drop_snapshot.angles.types
        )
        wetting_snapshot.angles.typeid = np.concatenate(
            (
                self.surface_snapshot.angles.typeid,
                self.drop_snapshot.angles.typeid
                + len(self.surface_snapshot.angles.types),
            ),
            axis=None,
        )
        wetting_snapshot.angles.group = np.concatenate(
            (
                self.surface_snapshot.angles.group,
                self.drop_snapshot.angles.group + self._surface_n,
            ),
            axis=None,
        )

        # set up dihedrals
        wetting_snapshot.dihedrals.N = (
            self.surface_snapshot.dihedrals.N + self.drop_snapshot.dihedrals.N
        )
        wetting_snapshot.dihedrals.types = (
            self.surface_snapshot.dihedrals.types
            + self.drop_snapshot.dihedrals.types
        )
        wetting_snapshot.dihedrals.typeid = np.concatenate(
            (
                self.surface_snapshot.dihedrals.typeid,
                self.drop_snapshot.dihedrals.typeid
                + len(self.surface_snapshot.dihedrals.types),
            ),
            axis=None,
        )
        wetting_snapshot.dihedrals.group = np.concatenate(
            (
                self.surface_snapshot.dihedrals.group,
                self.drop_snapshot.dihedrals.group + self._surface_n,
            ),
            axis=None,
        )
        # set up pairs
        wetting_snapshot.pairs.N = (
            self.surface_snapshot.pairs.N + self.drop_snapshot.pairs.N
        )
        # rename surface pair types (add '_' to the beginning)
        surface_pair_types = []
        for pair in self.surface_snapshot.pairs.types:
            p1, p2 = pair.split("-")
            surface_pair_types.append(f"_{p1}-_{p2}")
        wetting_snapshot.pairs.types = (
            surface_pair_types + self.drop_snapshot.pairs.types
        )
        wetting_snapshot.pairs.typeid = np.concatenate(
            (
                self.surface_snapshot.pairs.typeid,
                self.drop_snapshot.pairs.typeid
                + len(self.surface_snapshot.pairs.types),
            ),
            axis=None,
        )
        wetting_snapshot.pairs.group = np.concatenate(
            (
                self.surface_snapshot.pairs.group,
                self.drop_snapshot.pairs.group + self._surface_n,
            ),
            axis=None,
        )

        return wetting_snapshot

    def _create_box(self):
        """Create the wetting simulation box."""
        wetting_sim_box = np.copy(self.surface_snapshot.configuration.box)
        # use box height for z
        wetting_sim_box[2] = self.box_height
        return wetting_sim_box

    def _adjust_particle_positions(self, wetting_box):
        """Place the surface and droplet particles in the wetting box."""
        # shift surface particles to the bottom of the box
        surface_z_shift = (
            np.abs(
                min(self.surface_snapshot.particles.position[:, 2])
                - (-wetting_box[2] / 2)
            )
            - 0.1
        )
        surface_pos = self.surface_snapshot.particles.position - np.array(
            [0, 0, surface_z_shift]
        )
        # find center of the droplet and shift the droplet particles to origin
        drop_pos = self.drop_snapshot.particles.position - np.mean(
            self.drop_snapshot.particles.position, axis=0
        )
        # shift drop particles z position to be at the top of surface
        z_shift = (
            np.abs(min(drop_pos[:, 2]) - max(surface_pos[:, 2])) - self.gap
        )
        drop_pos[:, 2] -= z_shift
        wetting_pos = np.concatenate((surface_pos, drop_pos), axis=0)
        return wetting_pos


class WettingSimulation(Simulation):
    """Simulation of surface wetting.

    Parameters
    ----------
    initial_state : hoomd.snapshot.Snapshot or str, required
        A snapshot to initialize a simulation from, or a path
        to a GSD file to initialize a simulation from.
        This snapshot contains the surface and droplet particles.
    forcefield : List of hoomd.md.force.Force, required
        List of HOOMD force objects used in the simulation.
        This forcefield contains the surface and droplet forces.
    fix_surface : bool, optional, default=True
        If `True`, the surface particles are not integrated over.
    """

    def __init__(
        self,
        initial_state,
        forcefield,
        fix_surface=True,
        reference_values=dict(),
        dt=0.0001,
        device=hoomd.device.auto_select(),
        seed=42,
        gsd_write_freq=1e4,
        gsd_file_name="wetting.gsd",
        log_write_freq=1e3,
        log_file_name="wetting_log.txt",
        thermostat=HOOMDThermostats.MTTK,
    ):
        self._fix_surface = fix_surface
        super(WettingSimulation, self).__init__(
            initial_state=initial_state,
            forcefield=forcefield,
            reference_values=reference_values,
            dt=dt,
            device=device,
            seed=seed,
            gsd_write_freq=gsd_write_freq,
            gsd_file_name=gsd_file_name,
            log_write_freq=log_write_freq,
            log_file_name=log_file_name,
            thermostat=thermostat,
        )
        self.fix_surface = fix_surface

    @property
    def fix_surface(self):
        """If `True`, the surface particles are not integrated over."""
        return self._fix_surface

    @fix_surface.setter
    def fix_surface(self, value):
        if not isinstance(value, bool):
            raise ValueError(
                "Set to `True` to not integrate over surface particles, "
                "or set to `False` to integrate over surface particles."
            )
        self._fix_surface = value
        if self.fix_surface:
            snapshot = self.state.get_snapshot()
            droplet_types = [
                i for i in snapshot.particles.types if not i.startswith("_")
            ]
            self.integrate_group = hoomd.filter.Type(droplet_types)
        else:
            self.integrate_group = hoomd.filter.All()
