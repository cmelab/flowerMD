import pickle
import warnings

import gsd.hoomd
import hoomd
import hoomd.md
import numpy as np
import unyt as u

from hoomd_organics.utils import (
    HOOMDThermostats,
    StdOutLogger,
    UpdateWalls,
    calculate_box_length,
    validate_ref_value,
)
from hoomd_organics.utils.exceptions import ReferenceUnitError


class Simulation(hoomd.simulation.Simulation):
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

    Methods
    -------

    """

    def __init__(
        self,
        initial_state,
        forcefield=None,
        reference_values=dict(),
        r_cut=2.5,
        dt=0.0001,
        device=hoomd.device.auto_select(),
        seed=42,
        gsd_write_freq=1e4,
        gsd_file_name="trajectory.gsd",
        log_write_freq=1e3,
        log_file_name="sim_data.txt",
    ):
        super(Simulation, self).__init__(device, seed)
        self.initial_state = initial_state
        self._forcefield = forcefield
        self.r_cut = r_cut
        self.gsd_write_freq = int(gsd_write_freq)
        self.log_write_freq = int(log_write_freq)
        self._std_out_freq = int(
            (self.gsd_write_freq + self.log_write_freq) / 2
        )
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
        self.integrator = None
        self._dt = dt
        self._reference_values = dict()
        self._reference_values = reference_values
        self._integrate_group = hoomd.filter.All()
        self._wall_forces = dict()
        self._create_state(self.initial_state)
        # Add a gsd and thermo props logger to sim operations
        self._add_hoomd_writers()
        self._thermostat = HOOMDThermostats.BUSSI

    @classmethod
    def from_system(cls, system, **kwargs):
        """Initialize a simulation from a `hoomd_organics.base.System`
        object."""

        if system.hoomd_forcefield:
            return cls(
                initial_state=system.hoomd_snapshot,
                forcefield=system.hoomd_forcefield,
                reference_values=system.reference_values,
                **kwargs,
            )
        elif kwargs.get("forcefield", None):
            return cls(
                initial_state=system.hoomd_snapshot,
                reference_values=system.reference_values,
                **kwargs,
            )
        else:
            raise ValueError(
                "No forcefield provided. Please provide a forcefield "
                "or a system with a forcefield."
            )

    @classmethod
    def from_snapshot_forces(cls, initial_state, forcefield, **kwargs):
        """Initialize a simulation from an initial state object and a
        list of HOOMD forces."""
        return cls(initial_state=initial_state, forcefield=forcefield, **kwargs)

    @property
    def forces(self):
        if self.integrator:
            return self.operations.integrator.forces
        else:
            return self._forcefield

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
    def reference_length(self, length):
        validated_length = validate_ref_value(length, u.dimensions.length)
        self._reference_values["length"] = validated_length

    @reference_energy.setter
    def reference_energy(self, energy):
        validated_energy = validate_ref_value(energy, u.dimensions.energy)
        self._reference_values["energy"] = validated_energy

    @reference_mass.setter
    def reference_mass(self, mass):
        validated_mass = validate_ref_value(mass, u.dimensions.mass)
        self._reference_values["mass"] = validated_mass

    @reference_values.setter
    def reference_values(self, ref_value_dict):
        ref_keys = ["length", "mass", "energy"]
        for k in ref_keys:
            if k not in ref_value_dict.keys():
                raise ValueError(f"Missing reference for {k}.")
            self.__setattr__(f"reference_{k}", ref_value_dict[k])

    @property
    def box_lengths_reduced(self):
        box = self.state.box
        return np.array([box.Lx, box.Ly, box.Lz])

    @property
    def box_lengths(self):
        if self.reference_length:
            return self.box_lengths_reduced * self.reference_length
        else:
            warnings.warn(
                "Reference length is not specified. Using HOOMD's unit-less "
                "length instead. You can set reference length value and unit "
                "with `reference_length()` method. "
            )
            return self.box_lengths_reduced

    @property
    def volume_reduced(self):
        return np.prod(self.box_lengths_reduced)

    @property
    def volume(self):
        return np.prod(self.box_lengths)

    @property
    def mass_reduced(self):
        with self.state.cpu_local_snapshot as snap:
            return sum(snap.particles.mass)

    @property
    def mass(self):
        if self.reference_mass:
            return self.mass_reduced * self.reference_mass
        else:
            warnings.warn(
                "Reference mass is not specified. Using HOOMD's unit-less mass "
                "instead. You can set reference mass value and unit with "
                "`reference_mass()` method. "
            )
            return self.mass_reduced

    @property
    def density_reduced(self):
        return self.mass_reduced / self.volume_reduced

    @property
    def density(self):
        return self.mass / self.volume

    @property
    def nlist(self):
        """"""
        return self._lj_force().nlist

    @nlist.setter
    def nlist(self, hoomd_nlist, buffer=0.4):
        """"""
        self._lj_force().nlist = hoomd_nlist(buffer)

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        self._dt = value
        if self.integrator:
            self.operations.integrator.dt = self.dt

    @property
    def real_timestep(self):
        if self._reference_values.get("mass"):
            mass = self._reference_values["mass"].to("kg")
        else:
            mass = 1 * u.kg
        if self._reference_values.get("length"):
            dist = self.reference_length.to("m")
        else:
            dist = 1 * u.m
        if self._reference_values.get("energy"):
            energy = self.reference_energy.to("J")
        else:
            energy = 1 * u.J
        tau = (mass * (dist**2)) / energy
        timestep = self.dt * (tau**0.5)
        return timestep

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
            return self.operations.integrator.methods[0]
        else:
            raise RuntimeError(
                "No integrator, or method has been set yet. "
                "These will be set once one of the run functions "
                "have been called for the first time."
            )

    @property
    def thermostat(self):
        return self._thermostat

    @thermostat.setter
    def thermostat(self, thermostat):
        if not issubclass(
            self._thermostat, hoomd.md.methods.thermostats.Thermostat
        ):
            raise ValueError(
                f"Invalid thermostat. Please choose from: {HOOMDThermostats}"
            )
        self._thermostat = thermostat

    def add_force(self, hoomd_force):
        """"""
        self._forcefield.append(hoomd_force)
        if self.integrator:
            self.integrator.forces.append(hoomd_force)

    def remove_force(self, hoomd_force):
        """"""
        self._forcefield.remove(hoomd_force)
        if self.integrator:
            self.integrator.forces.remove(hoomd_force)

    def adjust_epsilon(self, scale_by=None, shift_by=None, type_filter=None):
        """"""
        lj_forces = self._lj_force()
        for k in lj_forces.params.keys():
            if type_filter and k not in type_filter:
                continue
            epsilon = lj_forces.params[k]["epsilon"]
            if scale_by:
                lj_forces.params[k]["epsilon"] = epsilon * scale_by
            elif shift_by:
                lj_forces.params[k]["epsilon"] = epsilon + shift_by

    def adjust_sigma(self, scale_by=None, shift_by=None, type_filter=None):
        """"""
        lj_forces = self._lj_force()
        for k in lj_forces.params.keys():
            if type_filter and k not in type_filter:
                continue
            sigma = lj_forces.params[k]["sigma"]
            if scale_by:
                lj_forces.params[k]["sigma"] = sigma * scale_by
            elif shift_by:
                lj_forces.params[k]["sigma"] = sigma + shift_by

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
        if not self.integrator:  # Integrator and method not yet created
            self.integrator = hoomd.md.Integrator(dt=self.dt)
            self.integrator.forces = self._forcefield
            self.operations.add(self.integrator)
            new_method = integrator_method(**method_kwargs)
            self.operations.integrator.methods = [new_method]
        else:  # Replace the existing integrator method
            self.integrator.methods.remove(self.method)
            new_method = integrator_method(**method_kwargs)
            self.integrator.methods.append(new_method)

    def add_walls(self, wall_axis, sigma, epsilon, r_cut, r_extrap=0):
        """"""
        wall_axis = np.asarray(wall_axis)
        wall_origin = wall_axis * self.box_lengths_reduced / 2
        wall_normal = -wall_axis
        wall_origin2 = -wall_origin
        wall_normal2 = -wall_normal
        wall1 = hoomd.wall.Plane(origin=wall_origin, normal=wall_normal)
        wall2 = hoomd.wall.Plane(origin=wall_origin2, normal=wall_normal2)
        lj_walls = hoomd.md.external.wall.LJ(walls=[wall1, wall2])
        lj_walls.params[self.state.particle_types] = {
            "epsilon": epsilon,
            "sigma": sigma,
            "r_cut": r_cut,
            "r_extrap": r_extrap,
        }
        self.add_force(lj_walls)
        self._wall_forces[tuple(wall_axis)] = (
            lj_walls,
            {
                "sigma": sigma,
                "epsilon": epsilon,
                "r_cut": r_cut,
                "r_extrap": r_extrap,
            },
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
        final_box_lengths=None,
        final_density=None,
        thermalize_particles=True,
        write_at_start=True,
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
        final_box_lengths : np.ndarray, shape=(3,), dtype=float; optional
            The final box edge lengths in (x, y, z) order
        write_at_start : bool; optional default True
            When set to True, triggers writers that evaluate to True
            for the initial step to execute before the next simulation
            time step.
        final_density : float; optional
            The final density of the simulation

        """
        if final_box_lengths is None and final_density is None:
            raise ValueError(
                "Must provide either `final_box_lengths` or `final_density`"
            )
        if final_box_lengths is not None and final_density is not None:
            raise ValueError(
                "Cannot provide both `final_box_lengths` and `final_density`."
            )
        if final_box_lengths is not None:
            final_box = hoomd.Box(
                Lx=final_box_lengths[0],
                Ly=final_box_lengths[1],
                Lz=final_box_lengths[2],
            )
        else:
            if not self.reference_values:
                raise ReferenceUnitError(
                    "Missing simulation units. Please "
                    "provide units for mass, length, and"
                    " energy."
                )

            if isinstance(final_density, u.unyt_quantity):
                density_quantity = final_density.to(u.g / u.cm**3)
            else:
                density_quantity = u.unyt_quantity(
                    final_density, u.g / u.cm**3
                )
            mass_g = self.mass.to("g")
            L = calculate_box_length(mass_g, density_quantity)
            # convert L from cm to reference units
            L = (
                L.to(self.reference_length.units) / self.reference_length.value
            ).value
            final_box = hoomd.Box(Lx=L, Ly=L, Lz=L)

        resize_trigger = hoomd.trigger.Periodic(period)
        box_ramp = hoomd.variant.Ramp(
            A=0, B=1, t_start=self.timestep, t_ramp=int(n_steps)
        )
        initial_box = self.state.box

        box_resizer = hoomd.update.BoxResize(
            box1=initial_box,
            box2=final_box,
            variant=box_ramp,
            trigger=resize_trigger,
        )
        self.operations.updaters.append(box_resizer)
        self.set_integrator_method(
            integrator_method=hoomd.md.methods.NVT,
            method_kwargs={
                "tau": tau_kt,
                "filter": self.integrate_group,
                "kT": kT,
            },
        )
        if thermalize_particles:
            self._thermalize_system(kT)

        if self._wall_forces:
            wall_update = UpdateWalls(sim=self)
            wall_updater = hoomd.update.CustomUpdater(
                trigger=resize_trigger, action=wall_update
            )
            self.operations.updaters.append(wall_updater)
        std_out_logger = StdOutLogger(n_steps=n_steps, sim=self)
        std_out_logger_printer = hoomd.update.CustomUpdater(
            trigger=hoomd.trigger.Periodic(self._std_out_freq),
            action=std_out_logger,
        )
        self.operations.updaters.append(std_out_logger_printer)
        self.run(steps=n_steps + 1, write_at_start=write_at_start)
        self.operations.updaters.remove(std_out_logger_printer)

    def run_langevin(
        self,
        n_steps,
        kT,
        alpha,
        tally_reservoir_energy=False,
        default_gamma=1.0,
        default_gamma_r=(1.0, 1.0, 1.0),
        thermalize_particles=True,
        write_at_start=True,
    ):
        """"""
        self.set_integrator_method(
            integrator_method=hoomd.md.methods.Langevin,
            method_kwargs={
                "filter": self.integrate_group,
                "kT": kT,
                "alpha": alpha,
                "tally_reservoir_energy": tally_reservoir_energy,
                "default_gamma": default_gamma,
                "default_gamma_r": default_gamma_r,
            },
        )
        if thermalize_particles:
            self._thermalize_system(kT)
        std_out_logger = StdOutLogger(n_steps=n_steps, sim=self)
        std_out_logger_printer = hoomd.update.CustomUpdater(
            trigger=hoomd.trigger.Periodic(self._std_out_freq),
            action=std_out_logger,
        )
        self.operations.updaters.append(std_out_logger_printer)
        self.run(steps=n_steps, write_at_start=write_at_start)
        self.operations.updaters.remove(std_out_logger_printer)

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
        thermalize_particles=True,
        write_at_start=True,
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
                "kT": kT,
            },
        )
        if thermalize_particles:
            self._thermalize_system(kT)
        std_out_logger = StdOutLogger(n_steps=n_steps, sim=self)
        std_out_logger_printer = hoomd.update.CustomUpdater(
            trigger=hoomd.trigger.Periodic(self._std_out_freq),
            action=std_out_logger,
        )
        self.operations.updaters.append(std_out_logger_printer)
        self.run(steps=n_steps, write_at_start=write_at_start)
        self.operations.updaters.remove(std_out_logger_printer)

    def run_NVT(
        self,
        n_steps,
        kT,
        tau_kt,
        thermalize_particles=True,
        write_at_start=True,
    ):
        """"""
        self.set_integrator_method(
            integrator_method=hoomd.md.methods.NVT,
            method_kwargs={
                "tau": tau_kt,
                "filter": self.integrate_group,
                "kT": kT,
            },
        )
        if thermalize_particles:
            self._thermalize_system(kT)
        std_out_logger = StdOutLogger(n_steps=n_steps, sim=self)
        std_out_logger_printer = hoomd.update.CustomUpdater(
            trigger=hoomd.trigger.Periodic(self._std_out_freq),
            action=std_out_logger,
        )
        self.operations.updaters.append(std_out_logger_printer)
        self.run(steps=n_steps, write_at_start=write_at_start)
        self.operations.updaters.remove(std_out_logger_printer)

    def run_NVE(self, n_steps, write_at_start=True):
        """"""
        self.set_integrator_method(
            integrator_method=hoomd.md.methods.NVE,
            method_kwargs={"filter": self.integrate_group},
        )
        std_out_logger = StdOutLogger(n_steps=n_steps, sim=self)
        std_out_logger_printer = hoomd.update.CustomUpdater(
            trigger=hoomd.trigger.Periodic(self._std_out_freq),
            action=std_out_logger,
        )
        self.operations.updaters.append(std_out_logger_printer)
        self.run(steps=n_steps, write_at_start=write_at_start)
        self.operations.updaters.remove(std_out_logger_printer)

    def run_displacement_cap(
        self,
        n_steps,
        maximum_displacement=1e-3,
        write_at_start=True,
    ):
        """NVE based integrator that Puts a cap on the maximum displacement
        per time step.

        DisplacementCapped method is mostly useful for initially relaxing a
        system with overlapping particles. Putting a cap on the max particle
        displacement prevents Hoomd Particle Out of Box execption.
        Once the system is relaxed, other run methods (NVE, NVT, etc) can be
        used.

        Parameters:
        -----------
        n_steps : int, required
            Number of steps to run during shrinking
        maximum_displacement : maximum displacement per step (length)

        """
        self.set_integrator_method(
            integrator_method=hoomd.md.methods.DisplacementCapped,
            method_kwargs={
                "filter": self.integrate_group,
                "maximum_displacement": maximum_displacement,
            },
        )
        std_out_logger = StdOutLogger(n_steps=n_steps, sim=self)
        std_out_logger_printer = hoomd.update.CustomUpdater(
            trigger=hoomd.trigger.Periodic(self._std_out_freq),
            action=std_out_logger,
        )
        self.operations.updaters.append(std_out_logger_printer)
        self.run(steps=n_steps, write_at_start=write_at_start)
        self.operations.updaters.remove(std_out_logger_printer)

    def temperature_ramp(self, n_steps, kT_start, kT_final):
        return hoomd.variant.Ramp(
            A=kT_start, B=kT_final, t_start=self.timestep, t_ramp=int(n_steps)
        )

    def pickle_forcefield(self, file_path="forcefield.pickle"):
        f = open(file_path, "wb")
        pickle.dump(self._forcefield, f)

    def save_restart_gsd(self, file_path="restart.gsd"):
        hoomd.write.GSD.write(self.state, filename=file_path)

    def _thermalize_system(self, kT):
        if isinstance(kT, hoomd.variant.Ramp):
            self.state.thermalize_particle_momenta(
                filter=self.integrate_group, kT=kT.range[0]
            )
        else:
            self.state.thermalize_particle_momenta(
                filter=self.integrate_group, kT=kT
            )

    def _lj_force(self):
        if not self.integrator:
            lj_force = [
                f
                for f in self._forcefield
                if isinstance(f, hoomd.md.pair.pair.LJ)
            ][0]
        else:
            lj_force = [
                f
                for f in self.integrator.forces
                if isinstance(f, hoomd.md.pair.pair.LJ)
            ][0]
        return lj_force

    def _create_state(self, initial_state):
        if isinstance(initial_state, str):  # Load from a GSD file
            print("Initializing simulation state from a GSD file.")
            self.create_state_from_gsd(initial_state)
        elif isinstance(initial_state, hoomd.snapshot.Snapshot):
            print("Initializing simulation state from a snapshot.")
            self.create_state_from_snapshot(initial_state)
        elif isinstance(initial_state, gsd.hoomd.Snapshot):
            print("Initializing simulation state from a snapshot.")
            self.create_state_from_snapshot(initial_state)

    def _add_hoomd_writers(self):
        """Creates gsd and log writers"""

        gsd_logger = hoomd.logging.Logger(
            categories=["scalar", "string", "sequence"]
        )
        logger = hoomd.logging.Logger(categories=["scalar", "string"])
        gsd_logger.add(self, quantities=["timestep", "tps"])
        logger.add(self, quantities=["timestep", "tps"])
        thermo_props = hoomd.md.compute.ThermodynamicQuantities(
            filter=self.integrate_group
        )
        self.operations.computes.append(thermo_props)
        gsd_logger.add(thermo_props, quantities=self.log_quantities)
        logger.add(thermo_props, quantities=self.log_quantities)

        for f in self._forcefield:
            logger.add(f, quantities=["energy"])
            gsd_logger.add(f, quantities=["energy"])

        gsd_writer = hoomd.write.GSD(
            filename=self.gsd_file_name,
            trigger=hoomd.trigger.Periodic(int(self.gsd_write_freq)),
            mode="wb",
            dynamic=["momentum"],
            logger=gsd_logger,
        )

        table_file = hoomd.write.Table(
            output=open(self.log_file_name, mode="w", newline="\n"),
            trigger=hoomd.trigger.Periodic(period=int(self.log_write_freq)),
            logger=logger,
            max_header_len=None,
        )
        self.operations.writers.append(gsd_writer)
        self.operations.writers.append(table_file)
