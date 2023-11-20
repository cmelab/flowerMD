"""Module for simulating surface wetting."""
import gsd.hoomd
import hoomd
import numpy as np
import unyt as u
from gmso.external import to_gsd_snapshot, to_hoomd_forcefield
from utils import combine_forces

from flowermd.base import Simulation
from flowermd.utils import HOOMDThermostats


class SurfaceDropletCreator:
    """Create a droplet on a surface."""

    def __init__(
        self, surface, drop_snapshot, drop_ff, drop_ref_values, box_height, gap
    ):
        """Create the snapshot and forces for the surface droplet simulation.

        Creates a new snapshot that combines the surface and droplet snapshots
        by adding the droplet particles on top of the surface particles with
        the correct spacing (i.e. gap) between the two.
        Also combines the hoomd forces from the surface and droplet simulations,
        and adds the forces for the pair interactions between the droplet and
        surface particles.

        Droplet reference values are used as the reference to scale the mass,
        energy and length values in the new snapshot.

        Parameters
        ----------
        surface : flowermd.library.surfaces.Graphene, required
            The surface to place the droplet on.
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
            The gap between the droplet and the surface.'

        """
        self.surface = surface
        self.drop_snapshot = drop_snapshot
        self.drop_ff = drop_ff
        self.drop_ref_values = drop_ref_values
        self.box_height = box_height
        self.gap = gap

        # get surface snapshot and forces
        (
            self.surface_snapshot,
            self.surface_ref_values,
        ) = self._create_surface_snapshot()
        self._surface_n = self.surface_snapshot.particles.N
        self._drop_n = self.drop_snapshot.particles.N
        # save surface snapshot
        with gsd.hoomd.open("surface.gsd", "w") as f:
            f.append(self.surface_snapshot)
        self.surface_ff = self._create_surface_forces()
        # get snapshot of the combined system
        if set(self.surface_snapshot.particles.types).intersection(
            set(self.drop_snapshot.particles.types)
        ):
            raise NotImplementedError(
                "handle cases where there are common "
                "particle types between the surface and "
                "droplet."
            )
        self._combined_snapshot = self._build_snapshot()

        # get forces of the combined system
        self._combined_forces = combine_forces(
            self.drop_ff,
            self.surface_ff,
            self.drop_ptypes,
            self.surface_ptypes,
        )

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
        wetting_snapshot.configuration.box = self._create_box()
        # put the surface particles in the box and add droplet particles on top
        wetting_snapshot.particles.position = self._adjust_particle_positions()

        # set up bonds
        wetting_snapshot.bonds.N = (
            self.surface_snapshot.bonds.N + self.drop_snapshot.bonds.N
        )
        wetting_snapshot.bonds.types = (
            self.surface_snapshot.bonds.types + self.drop_snapshot.bonds.types
        )
        wetting_snapshot.bonds.typeid = np.concatenate(
            self.surface_snapshot.bonds.typeid,
            self.drop_snapshot.bonds.typeid
            + len(self.surface_snapshot.bonds.types),
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
            self.surface_snapshot.angles.typeid,
            self.drop_snapshot.angles.typeid
            + len(self.surface_snapshot.angles.types),
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
            self.surface_snapshot.dihedrals.typeid,
            self.drop_snapshot.dihedrals.typeid
            + len(self.surface_snapshot.dihedrals.types),
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
            self.surface_snapshot.pairs.typeid,
            self.drop_snapshot.pairs.typeid
            + len(self.surface_snapshot.pairs.types),
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

    def _adjust_particle_positions(self):
        """Place the surface and droplet particles in the wetting box."""
        # place surface particles in the box
        surface_pos = np.copy(self.surface_snapshot.particles.position)
        # find center of the droplet and shift the droplet particles to origin
        drop_pos = self.drop_snapshot.particles.position - np.mean(
            self.drop_snapshot.particles.position, axis=0
        )
        # shift drop particles z position to be at the top of surface
        z_shift = (
            np.abs(max(drop_pos[:, 2]) - max(surface_pos[:, 2])) - self.gap
        )
        drop_pos[:, 2] -= z_shift
        wetting_pos = np.concatenate((surface_pos, drop_pos), axis=0)
        return wetting_pos

    def _create_surface_snapshot(self):
        """Get the surface snapshot."""
        snap, refs = to_gsd_snapshot(
            top=self.surface.gmso_molecule,
            auto_scale=False,
            base_units=self.drop_ref_values,
        )
        return snap, refs

    def _create_surface_forces(self, surface_r_cut=2.5):
        """Get the surface forces."""
        force_list = []
        ff, refs = to_hoomd_forcefield(
            top=self.surface.gmso_molecule,
            r_cut=surface_r_cut,
            auto_scale=False,
            base_units=self.drop_ref_values,
        )
        for force in ff:
            force_list.extend(ff[force])
        return force_list


class DropletSimulation(Simulation):
    """Simulation which creates a droplet."""

    def __init__(
        self,
        initial_state,
        forcefield,
        r_cut=2.5,
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
            r_cut=r_cut,
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
        shrink_kT,
        shrink_steps,
        shrink_period,
        expand_kT,
        expand_steps,
        expand_period,
        hold_kT,
        hold_steps,
        final_density,
        tau_kt,
    ):
        """Run droplet simulation."""
        # Shrink down to high density
        self.run_update_volume(
            n_steps=shrink_steps,
            period=shrink_period,
            kT=shrink_kT,
            tau_kt=tau_kt,
            final_density=1.4 * (u.g / (u.cm**3)),
            write_at_start=True,
        )
        # Expand back up to low density
        self.run_update_volume(
            n_steps=expand_steps,
            period=expand_period,
            kT=expand_kT,
            tau_kt=tau_kt,
            final_density=final_density * (u.g / (u.cm**3)),
        )
        # Run at low density
        self.run_NVT(n_steps=hold_steps, kT=hold_kT, tau_kt=tau_kt)


class WettingSimulation(Simulation):
    """For simulating welding of an interface joint."""

    def __init__(
        self,
        initial_state,
        forcefield,
        r_cut=2.5,
        reference_values=dict(),
        dt=0.0001,
        device=hoomd.device.auto_select(),
        seed=42,
        gsd_write_freq=1e4,
        gsd_file_name="weld.gsd",
        log_write_freq=1e3,
        log_file_name="sim_data.txt",
        thermostat=HOOMDThermostats.MTTK,
    ):
        super(WettingSimulation, self).__init__(
            initial_state=initial_state,
            forcefield=forcefield,
            r_cut=r_cut,
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
