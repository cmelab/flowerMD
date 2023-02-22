from itertools import combinations_with_replacement as combo
import operator
import os

import gsd.hoomd
import hoomd
import hoomd.md
from mbuild.formats.hoomd_forcefield import create_hoomd_forcefield
import numpy as np
import parmed as pmd


class Simulation:
    """The simulation context management class.

    This class takes the output of the Initialization class
    and sets up a hoomd-blue simulation.

    Parameters
    ----------
    initial_state : gsd.hoomd.Snapshot or str
        A snapshot to initialize a simulation from, or a path
        to a GSD file to initialize a simulation from.
    forcefield : list
        List of hoomd force objects to add to the integrator.
    r_cut : float, default 2.5
        Cutoff radius for potentials (in simulation distance units)
    dt : float, default 0.0001
        Initial value for dt, the ize of simulation timestep
    auto_scale : bool, default True
        Set to true to use reduced simulation units.
        distance, mass, and energy are scaled by the largest value
        present in the system for each.
    gsd_write : int, default 1e4
        Period to write simulation snapshots to gsd file.
    gsd_file_name : str, default "trajectory.gsd"
        The file name to use for the GSD file
    log_write : int, default 1e3
        Period to write simulation data to the log file.
    log_file_name : str, default "sim_data.txt"
        The file name to use for the .txt log file
    seed : int, default 42
        Seed passed to integrator when randomizing velocities.
    restart : str, default None
        Path to gsd file from which to restart the simulation

    Methods
    -------

    """
    def __init__(
        self,
        initial_state,
        forcefield=None,
        r_cut=2.5,
        dt=0.0001,
        seed=42,
        restart=None,  #TODO: Restart logic
        gsd_write_freq=1e4,
        gsd_file_name="trajectory.gsd",
        log_write_freq=1e3,
        log_file_name="sim_data.txt"
    ):
        self.initial_state = initial_state
        self.forcefield = forcefield
        self.r_cut = r_cut
        self._dt = dt
        self.gsd_write_freq = gsd_write_freq
        self.log_write_freq = log_write_freq
        self.gsd_file_name = gsd_file_name
        self.log_file_name = log_file_name
        self.log_quantities = [
            "kinetic_temperature",
            "potential_energy",
            "kinetic_energy",
            "volume",
            "pressure",
            "pressure_tensor",
        ]
        self.device = hoomd.device.auto_select()
        self.sim = hoomd.Simulation(device=self.device, seed=seed)
        self._integrate_group = hoomd.filter.All()
        self.integrator = None
        self._wall_forces = dict()
        if isinstance(self.initial_state, str): # Load from a GSD file
            print("Initializing simulation state from a GSD file.")
            self.sim.create_state_from_gsd(self.initial_state)
        elif isinstance(self.initial_state, hoomd.snapshot.Snapshot):
            print("Initializing simulation state from a snapshot.")
            self.sim.create_state_from_snapshot(self.initial_state)
        elif isinstance(self.initial_state, gsd.hoomd.Snapshot):
            print("Initializing simulation state from a snapshot.")
            self.sim.create_state_from_snapshot(self.initial_state)
        # Add a gsd and thermo props logger to sim operations
        self._add_hoomd_writers()

    @property
    def timestep(self):
        """"""
        return self.sim.timestep

    @property
    def atom_types(self):
        """"""
        snap = self.sim.state.get_snapshot()
        return snap.particles.types

    @property
    def box_lengths(self):
        box = self.sim.state.box
        return np.array([box.Lx, box.Ly, box.Lz])

    #TODO: Fix nlist functions
    @property
    def nlist(self):
        """"""
        return self.forcefield[0].nlist

    @nlist.setter
    def nlist(self, hoomd_nlist, buffer=0.4):
        """"""
        self.forcefield[0].nlist = hoomd_nlist(buffer)

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        self._dt = value
        if self.integrator:
            self.sim.operations.integrator.dt = self.dt


    @property
    def integrate_group(self):
        """"""
        return self._integrate_group

    @integrate_group.setter
    def integrate_group(self, group):
        """"""
        self._integrate_group = group

    @property
    def method(self):
        if self.integrator:
            return self.sim.operations.integrator.methods[0]
        else:
            raise RuntimeError(
                    "No integrator, or method has been set yet. "
                    "These will be set once one of the run functions "
                    "have been called for the first time."
            )

    def add_force(self, hoomd_force):
        """"""
        self.forcefield.append(hoomd_force)
        if self.integrator:
            self.integrator.forces.append(hoomd_force)

    def remove_force(self, hoomd_force):
        """"""
        self.forcefield.remove(hoomd_force)
        if self.integrator:
            self.integrator.forces.remove(hoomd_force)

    def scale_epsilon(self, scale_factor):
        """"""
        lj_forces = self._lj_pair_force()
        for k in lj_forces.params.keys():
            epsilon = lj_forces.params[k]['epsilon']
            lj_forces.params[k]['epsilon'] = epsilon * scale_factor

    def set_integrator_method(self, integrator_method, method_kwargs):
        """Creates an initial (or updates the existing) method used by
        Hoomd's integrator. This doesn't need to be called directly;
        instead the various run functions use this method to update
        the integrator method as needed.

        Parameters:
        -----------
        integrrator_method : hoomd.md.method; required
            Instance of one of the hoomd.md.method options
        method_kwargs : dict; required
            A diction of parameter:value for the integrator method used

        """
        if not self.integrator: # Integrator and method not yet created
            self.integrator = hoomd.md.Integrator(dt=self.dt)
            self.integrator.forces = self.forcefield
            self.sim.operations.add(self.integrator)
            new_method = integrator_method(**method_kwargs)
            self.sim.operations.integrator.methods = [new_method]
        else: # Replace the existing integrator method
            self.integrator.methods.remove(self.method)
            new_method = integrator_method(**method_kwargs)
            self.integrator.methods.append(new_method)

    def add_walls(self, wall_axis, sigma, epsilon, r_cut, r_extrap=0):
        """"""
        wall_axis = np.asarray(wall_axis)
        box = self.sim.state.box
        wall_origin = wall_axis * np.array([box.Lx/2, box.Ly/2, box.Lz/2])
        wall_normal = -wall_origin
        wall_origin2 = -wall_origin
        wall_normal2 = -wall_normal
        wall1 = hoomd.wall.Plane(origin=wall_origin, normal=wall_normal)
        wall2 = hoomd.wall.Plane(origin=wall_origin2, normal=wall_normal2)
        lj_walls = hoomd.md.external.wall.LJ(walls=[wall1, wall2])
        lj_walls.params[self.atom_types] = {
                "epsilon": epsilon,
                "sigma": sigma,
                "r_cut": r_cut,
                "r_extrap": r_extrap
        }
        self.add_force(lj_walls)
        self._wall_forces[tuple(wall_axis)] = (
                lj_walls,
                {"sigma": sigma,
                 "epsilon": epsilon,
                 "r_cut": r_cut,
                 "r_extrap": r_extrap}
        )

    def remove_walls(self, wall_axis):
        """"""
        wall_force = self._wall_forces[wall_axis][0]
        self.remove_force(wall_force)


    def run_update_volume(
            self,
            n_steps,
            period,
            kT,
            tau_kt,
            final_box_lengths,
            thermalize_particles=True
    ):
        """Runs an NVT simulation while shrinking or expanding
        the simulation volume to the given final volume.

        Parameters:
        -----------
        n_steps : int, required
            Number of steps to run during shrinking
        period : int, required
            The number of steps ran between box updates
        kT : int or hoomd.variant.Ramp; required
            The temperature to use during shrinking.
        tau_kt : float; required
            Thermostat coupling period (in simulation time units)
        final_box_lengths : np.ndarray, shape=(3,), dtype=float; required
            The final box edge lengths in (x, y, z) order

        """
        resize_trigger = hoomd.trigger.Periodic(period)
        box_ramp = hoomd.variant.Ramp(
                A=0, B=1, t_start=self.sim.timestep, t_ramp=int(n_steps)
        )
        initial_box = self.sim.state.box
        final_box = hoomd.Box(
                Lx=final_box_lengths[0],
                Ly=final_box_lengths[1],
                Lz=final_box_lengths[2]
        )
        box_resizer = hoomd.update.BoxResize(
                box1=initial_box,
                box2=final_box,
                variant=box_ramp,
                trigger=resize_trigger
        )
        self.sim.operations.updaters.append(box_resizer)
        self.set_integrator_method(
                integrator_method=hoomd.md.methods.NVT,
                method_kwargs={
                    "tau": tau_kt, "filter": self.integrate_group, "kT": kT
                },
        )
        if thermalize_particles:
            self._thermalize_system(kT)
        if not self._wall_forces:
            self.sim.run(n_steps)
        else:
            start_timestep = self.sim.timestep
            while self.sim.timestep < box_ramp.t_start + box_ramp.t_ramp:
                self.sim.run(period)
                self._update_walls()
            self.sim.run(1)
            self._update_walls()

    def run_langevin(
            self,
            n_steps,
            kT,
            alpha,
            tally_reservoir_energy=False,
            default_gamma=1.0,
            default_gamma_r=(1.0, 1.0, 1.0),
            thermalize_particles=True
    ):
        """"""
        self.set_integrator_method(
                integrator_method=hoomd.md.methods.Langevin,
                method_kwargs={
                        "filter": self.integrate_group,
                        "kT": kT,
                        "alpha": alpha,
                        "tally_reservoir_energy": tally_resivoir_energy,
                        "default_gamma": default_gamma,
                        "default_gamma_r": default_gamma_r,
                    }
        )
        if thermalize_particles:
            self._thermalize_system(kT)
        self.run(n_steps)

    def run_NPT(
            self,
            n_steps,
            kT,
            pressure,
            tau_kt,
            tau_pressure,
            couple="xyz",
            box_dof=[True, True, True, False, False, False],
            rescale_all=False,
            gamma=0.0,
            thermalize_particles=True
    ):
        """"""
        self.set_integrator_method(
                integrator_method=hoomd.md.methods.NPT,
                method_kwargs={
                    "kT": kT,
                    "S": pressure,
                    "tau": tau_kt,
                    "tauS": tau_pressure,
                    "couple": couple,
                    "box_dof": box_dof,
                    "rescale_all": rescale_all,
                    "gamma": gamma,
                    "filter": self.integrate_group,
                    "kT": kT
                }
        )
        if thermalize_particles:
            self._thermalize_system(kT)
        self.sim.run(n_steps)

    def run_NVT(self, n_steps, kT, tau_kt, thermalize_particles=True):
        """"""
        self.set_integrator_method(
                integrator_method=hoomd.md.methods.NVT,
                method_kwargs={
                    "tau": tau_kt, "filter": self.integrate_group, "kT": kT
                }
        )
        if thermalize_particles:
            self._thermalize_system(kT)
        self.sim.run(n_steps)

    def run_NVE(self, n_steps):
        """"""
        self.set_integrator_method(
                integrator_method=hoomd.md.methods.NVE,
                method_kwargs={"filter": self.integrate_group}
        )
        self.sim.run(n_steps)

    def temperature_ramp(self, n_steps, kT_start, kT_final):
        return hoomd.variant.Ramp(
                A=kT_start,
                B=kT_final,
                t_start=self.sim.timestep,
                t_ramp=int(n_steps)
        )

    def _update_walls(self):
        for wall_axis in self._wall_forces:
            wall_force = self._wall_forces[wall_axis][0]
            wall_kwargs = self._wall_forces[wall_axis][1]
            self.remove_force(wall_force)
            #self.sim.operations.integrator.forces.remove(wall_force)
            self.add_walls(wall_axis, **wall_kwargs)

    def _thermalize_system(self, kT):
        if isinstance(kT, hoomd.variant.Ramp):
            self.sim.state.thermalize_particle_momenta(
                    filter=self.integrate_group, kT=kT.range[0]
            )
        else:
            self.sim.state.thermalize_particle_momenta(
                    filter=self.integrate_group, kT=kT
            )

    #TODO: Better way to access this
    def _lj_pair_force(self):
        lj_force = [
                f for f in self.forcefield if
                isinstance(f, hoomd.md.pair.pair.LJ)][0]
        if lj_force is None:
            raise ValueError(
                    "The current hoomd forcefield does not contain "
                    "LJ pair forces"
            )
        return lj_force

    def _add_hoomd_writers(self):
        """Creates gsd and log writers"""
        gsd_writer = hoomd.write.GSD(
                filename=self.gsd_file_name,
                trigger=hoomd.trigger.Periodic(int(self.gsd_write_freq)),
                mode="wb",
                dynamic=["momentum"]
        )

        logger = hoomd.logging.Logger(categories=["scalar", "string"])
        logger.add(self.sim, quantities=["timestep", "tps"])
        thermo_props = hoomd.md.compute.ThermodynamicQuantities(
                filter=self.integrate_group
        )
        self.sim.operations.computes.append(thermo_props)
        logger.add(thermo_props, quantities=self.log_quantities)

        for f in self.forcefield:
            logger.add(f, quantities=["energy"])

        table_file = hoomd.write.Table(
            output=open(self.log_file_name, mode="w", newline="\n"),
            trigger=hoomd.trigger.Periodic(period=int(self.log_write_freq)),
            logger=logger,
            max_header_len=None,
        )
        self.sim.operations.writers.append(gsd_writer)
        self.sim.operations.writers.append(table_file)
