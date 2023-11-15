"""Module for simulating surface wetting."""
import hoomd
import numpy as np
import unyt as u
from gmso.external import to_gsd_snapshot, to_hoomd_forcefield

from flowermd.base import Simulation
from flowermd.utils import HOOMDThermostats


class SurfaceDropletCreator:
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

        self._drop_r_cut = self._get_drop_r_cut()
        # get surface snapshot and forces
        (
            self._surface_snapshot,
            self._surface_ref_values,
        ) = self._create_surface_snapshot()
        self._surface_forces = self._create_surface_forces()

        # get droplet and surface particle types
        self._all_particle_types = list(
            set(
                self._surface_snapshot.particles.types
                + self.drop_snapshot.particles.types
            )
        )

        self._combined_forces = self._combine_forces()

    def _build(self):
        """Duplicates the slab and builds the interface."""
        pass

    def _get_drop_r_cut(self):
        """Find the r_cut value from the droplet LJ forces."""
        for force in self.drop_ff:
            if isinstance(force, hoomd.md.pair.LJ):
                lj_r_cut = force.r_cut.values()[0]
                return lj_r_cut

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

    def _retrieve_forces(self, hoomd_force_list):
        """Retrieve individual force objects from the hoomd force list."""
        force_dict = {}
        for force in hoomd_force_list:
            if isinstance(force, hoomd.md.pair.LJ):
                force_dict["lj"] = force
            elif isinstance(force, hoomd.md.long_range.pppm.Coulomb):
                force_dict["coulomb"] = force
            elif isinstance(force, hoomd.md.bond.Bond):
                force_dict["bond"] = force
            elif isinstance(force, hoomd.md.angle.Angle):
                force_dict["angle"] = force
            elif isinstance(force, hoomd.md.dihedral.Dihedral):
                force_dict["dihedral"] = force
        return force_dict

    def _combine_forces(self):
        """Combine the surface and droplet forces."""
        combined_force_list = []

        # retrieve individual force objects from the droplet and surface
        # force lists
        drop_forces_dict = self._retrieve_forces(self.drop_ff)
        surface_forces_dict = self._retrieve_forces(self._surface_forces)

        # combine LJ forces
        combined_force_list.append(
            self._combine_lj(drop_forces_dict["lj"], surface_forces_dict["lj"])
        )

        # combine Coulomb forces

        # combine bond forces

        # combine angle forces

        # combine dihedral forces

        return combined_force_list

    def _combine_lj(self, drop_lj, surface_lj, combining_rule="geometric"):
        """Combine the droplet and surface LJ forces."""
        if combining_rule not in ["geometric", "lorentz"]:
            raise ValueError("combining_rule must be 'geometric' or 'lorentz'")

        lj = drop_lj.copy()

        # add the surface LJ parameters to the droplet LJ parameters
        for k, v in surface_lj.params.items():
            if k not in lj.params.keys():
                lj.params[k] = v
        for k, v in surface_lj.r_cut.items():
            if k not in lj.r_cut.keys():
                lj.r_cut[k] = v

        # find new pairs and add them to the droplet LJ pairs
        for drop_ptype in self.drop_snapshot.particles.types:
            for surface_ptype in self._surface_snapshot.particles.types:
                if (drop_ptype, surface_ptype) not in lj.params.keys() and (
                    surface_ptype,
                    drop_ptype,
                ) not in lj.params.keys():
                    epsilon = np.sqrt(
                        drop_lj.params[(drop_ptype, drop_ptype)]["epsilon"]
                        * surface_lj.params[(surface_ptype, surface_ptype)][
                            "epsilon"
                        ]
                    )
                    if combining_rule == "geometric":
                        sigma = np.sqrt(
                            drop_lj.params[(drop_ptype, drop_ptype)]["sigma"]
                            * surface_lj.params[(surface_ptype, surface_ptype)][
                                "sigma"
                            ]
                        )
                    else:
                        # combining_rule == 'lorentz'
                        sigma = 0.5 * (
                            drop_lj.params[(drop_ptype, drop_ptype)]["sigma"]
                            + surface_lj.params[(surface_ptype, surface_ptype)][
                                "sigma"
                            ]
                        )

                    lj.params[(drop_ptype, surface_ptype)] = {
                        "sigma": sigma,
                        "epsilon": epsilon,
                    }
                    lj.r_cut[(drop_ptype, surface_ptype)] = self._drop_r_cut
        return lj


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
