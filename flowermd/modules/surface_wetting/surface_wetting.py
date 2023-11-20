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

        # get forces of the combined system
        self._combined_forces = combine_forces(
            self.drop_ff,
            self.surface_ff,
            self.drop_snapshot.particles.types,
            self.surface_snapshot.particles.types,
        )
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

    def _build_snapshot(self):
        """Build a snapshot by combining the surface and droplet snapshots."""
        wetting_snapshot = gsd.hoomd.Frame()
        wetting_snapshot.particles.N = self._surface_n + self._drop_n

        self.surface_ptypes = self.surface_ptypes = [
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
        wetting_sim_box = self._create_box()
        wetting_snapshot.configuration.box = wetting_sim_box
        # put the surface particles in the box and add droplet particles on top
        wetting_snapshot.particles.position = (
            self._place_surface_droplet_particles(wetting_sim_box)
        )

        wetting_snapshot.bonds.N = (
            self.surface_snapshot.bonds.N + self.drop_snapshot.bonds.N
        )

        wetting_snapshot.angles.N = (
            self.surface_snapshot.angles.N + self.drop_snapshot.angles.N
        )

        wetting_snapshot.dihedrals.N = (
            self.surface_snapshot.dihedrals.N + self.drop_snapshot.dihedrals.N
        )
        wetting_snapshot.pairs.N = (
            self.surface_snapshot.pairs.N + self.drop_snapshot.pairs.N
        )

    def _create_box(self):
        """Create the wetting simulation box."""
        wetting_sim_box = [0, 0, 0, 0, 0, 0]
        # for x, y use the max of the surface and droplet box dimensions
        wetting_sim_box[0] = max(
            self.surface_snapshot.configuration.box[0],
            self.drop_snapshot.configuration.box[0],
        )
        wetting_sim_box[1] = max(
            self.surface_snapshot.configuration.box[1],
            self.drop_snapshot.configuration.box[1],
        )
        # use box height for z
        wetting_sim_box[2] = self.box_height
        return wetting_sim_box

    def _place_surface_droplet_particles(self, wetting_sim_box):
        """Place the surface and droplet particles in the wetting box."""
        # place surface particles in the box
        surface_pos = np.copy(self.surface_snapshot.particles.position)
        # shift drop particles to be centered in the box at (0, 0, 0)

        # shift drop particles z position to be at the top of surface

        return surface_pos

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
