"""Base simulation class for flowerMD."""

import inspect
import os
import pickle
import tempfile
import warnings
from collections.abc import Iterable

import gsd.hoomd
import hoomd
import hoomd.md
import numpy as np
import unyt as u

from flowermd.internal import Units, validate_unit
from flowermd.utils.actions import StdOutLogger, UpdateWalls
from flowermd.utils.base_types import HOOMDThermostats


class Simulation(hoomd.simulation.Simulation):
    """The simulation context management class.

    This class takes the output of the Initialization class
    and sets up a hoomd-blue simulation.

    Parameters
    ----------
    initial_state : hoomd.snapshot.Snapshot or str, required
        A snapshot to initialize a simulation from, or a path
        to a GSD file to initialize a simulation from.
    forcefield : List of hoomd.md.force.Force, required
        List of HOOMD force objects to add to the integrator.
    reference_values : dict, default {}
        A dictionary of reference values for mass, length, and energy.
    dt : float, default 0.0001
        Initial value for dt, the size of simulation timestep.
    device : hoomd.device, default hoomd.device.auto_select()
        The CPU or GPU device to use for the simulation.
    seed : int, default 42
        Seed passed to integrator when randomizing velocities.
    gsd_write : int, default 1e4
        Period to write simulation snapshots to gsd file.
    gsd_file_name : str, default "trajectory.gsd"
        The file name to use for the GSD file
    gsd_max_buffer_size : int, default 64 * 1024 * 1024
        Size (in bytes) to buffer in memory before writing GSD to file.
    log_write : int, default 1e3
        Period to write simulation data to the log file.
    log_file_name : str, default "sim_data.txt"
        The file name to use for the .txt log file
    thermostat : flowermd.utils.HOOMDThermostats, default
        HOOMDThermostats.MTTK
        The thermostat to use for the simulation.

    """

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
        gsd_max_buffer_size=64 * 1024 * 1024,
        log_write_freq=1e3,
        log_file_name="sim_data.txt",
        thermostat=HOOMDThermostats.MTTK,
        rigid_constraint=None,
        integrate_rotational_dof=False,
    ):
        if not isinstance(forcefield, Iterable) or isinstance(forcefield, str):
            raise ValueError(
                "forcefield must be a sequence of hoomd.md.force.Force objects."
            )
        else:
            for obj in forcefield:
                if not isinstance(obj, hoomd.md.force.Force):
                    raise ValueError(
                        "forcefield must be a sequence of "
                        "hoomd.md.force.Force objects."
                    )
        super(Simulation, self).__init__(device, seed)
        self.initial_state = initial_state
        self._forcefield = forcefield
        self.gsd_write_freq = int(gsd_write_freq)
        self.maximum_write_buffer_size = gsd_max_buffer_size
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
        self._kT = None
        self._reference_values = dict()
        self._reference_values = reference_values
        if rigid_constraint and not isinstance(
            rigid_constraint, hoomd.md.constrain.Rigid
        ):
            raise ValueError(
                "Invalid rigid constraint. Please provide a "
                "hoomd.md.constrain.Rigid object."
            )
        self._rigid_constraint = rigid_constraint
        self._integrate_group = self._create_integrate_group(
            rigid=True if rigid_constraint else False
        )
        self._wall_forces = dict()
        self._create_state(self.initial_state)
        # Add a gsd and thermo props logger to sim operations
        self._add_hoomd_writers()
        self._thermostat = thermostat
        self.integrate_rotational_dof = integrate_rotational_dof

    @classmethod
    def from_system(cls, system, **kwargs):
        """Initialize a simulation from a `flowermd.base.System` object.

        Parameters
        ----------
        system : flowermd.base.System, required
            A `flowermd.base.System` object.

        """
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
    def from_simulation_pickle(cls, file_path):
        with open(file_path, "rb") as f:
            string = f.read(len(b"FLOWERMD"))
            if string != b"FLOWERMD":
                raise ValueError(
                    "It appears this pickle file "
                    "was not created by flowermd.base.Simulation. "
                    "See flowermd.base.Simulation.save_simulation()."
                )
            data = pickle.load(f)

        state = data["state"]
        forces = data["forcefield"]
        for force in forces:
            if isinstance(force, hoomd.md.external.wall.LJ):
                new_walls = []
                for _wall in force.walls:
                    new_walls.append(
                        hoomd.wall.Plane(
                            origin=_wall.origin, normal=_wall.normal
                        )
                    )
                new_wall = hoomd.md.external.wall.LJ(walls=new_walls)
                for param in force.params:
                    new_wall.params[param] = force.params[param]
                forces.remove(force)
                forces.append(new_wall)
        ref_values = data["reference_values"]
        sim_kwargs = data["sim_kwargs"]
        return cls(
            initial_state=state,
            forcefield=list(forces),
            reference_values=ref_values,
            **sim_kwargs,
        )

    @classmethod
    def from_snapshot_forces(cls, initial_state, forcefield, **kwargs):
        """Initialize a simulation from an initial state and HOOMD forces.

        Parameters
        ----------
        initial_state : gsd.hoomd.Snapshot or str
            A snapshot to initialize a simulation from, or a path to a GSD file.
        forcefield : List of HOOMD force objects, required
            List of HOOMD force objects to add to the integrator.

        """
        return cls(initial_state=initial_state, forcefield=forcefield, **kwargs)

    @property
    def forces(self):
        """The list of forces in the simulation."""
        if self.integrator:
            return self.operations.integrator.forces
        else:
            return self._forcefield

    @property
    def reference_length(self):
        """The reference length for the simulation."""
        return self._reference_values.get("length", None)

    @property
    def reference_mass(self):
        """The reference mass for the simulation."""
        return self._reference_values.get("mass", None)

    @property
    def reference_energy(self):
        """The reference energy for the simulation."""
        return self._reference_values.get("energy", None)

    @property
    def reference_values(self):
        """The reference values for the simulation in form of a dictionary."""
        return self._reference_values

    @reference_length.setter
    def reference_length(self, length):
        """Set the reference length of the system along with a unit of length.

        Parameters
        ----------
        length : reference length * `flowermd.internal.Units`, required
            The reference length of the system.
            It can be provided in the following form of:
            value * `flowermd.internal.Units`, for example 1 * `flowermd.internal.Units.angstrom`.

        """
        validated_length = validate_unit(length, u.dimensions.length)
        self._reference_values["length"] = validated_length

    @reference_energy.setter
    def reference_energy(self, energy):
        """Set the reference energy of the system along with a unit of energy.

        Parameters
        ----------
        energy : reference energy * `flowermd.internal.Units`, required
            The reference energy of the system.
            It can be provided in the following form of:
            value * `flowermd.internal.Units`, for example 1 * `flowermd.internal.Units.kcal/mol`.

        """
        validated_energy = validate_unit(energy, u.dimensions.energy)
        self._reference_values["energy"] = validated_energy

    @reference_mass.setter
    def reference_mass(self, mass):
        """Set the reference mass of the system along with a unit of mass.

        Parameters
        ----------
        mass : reference mass * `flowermd.internal.Units`, required
            The reference mass of the system.
            It can be provided in the following form of:
            value * `flowermd.internal.Units`, for example 1 * `flowermd.internal.Units.amu`.
        """
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
        ref_keys = ["length", "mass", "energy"]
        for k in ref_keys:
            if k not in ref_value_dict.keys():
                raise ValueError(f"Missing reference for {k}.")
            self.__setattr__(f"reference_{k}", ref_value_dict[k])

    @property
    def box_lengths_reduced(self):
        """The simulation box lengths in reduced units."""
        box = self.state.box
        return np.array([box.Lx, box.Ly, box.Lz])

    @property
    def box_lengths(self):
        """The simulation box lengths.

        If reference length is set, the box lengths are scaled back to the
        non-reduced values.

        """
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
        """The simulation volume in reduced units."""
        return np.prod(self.box_lengths_reduced)

    @property
    def volume(self):
        """The simulation volume."""
        return np.prod(self.box_lengths)

    @property
    def mass_reduced(self):
        """The total mass of the system in reduced units."""
        with self.state.cpu_local_snapshot as snap:
            if self._rigid_constraint:
                last_body_tag = -1
                for body_tag in snap.particles.body:
                    if body_tag > last_body_tag:
                        last_body_tag += 1
                    else:
                        break
                return sum(snap.particles.mass[last_body_tag + 1 :])
            else:
                return sum(snap.particles.mass)

    @property
    def mass(self):
        """The total mass of the system.

        If reference mass is set, the mass is scaled back to the non-reduced
        value.

        """
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
        """The density of the system in reduced units."""
        return self.mass_reduced / self.volume_reduced

    @property
    def density(self):
        """The density of the system."""
        return self.mass / self.volume

    @property
    def nlist(self):
        """The neighbor list used by the Lennard-Jones pair force."""
        return self._lj_force().nlist

    @nlist.setter
    def nlist(self, hoomd_nlist, buffer=0.4):
        """Set the neighbor list used by the Lennard-Jones pair force.

        Parameters
        ----------
        hoomd_nlist : hoomd.md.nlist.NeighborList, required
            The neighbor list to use.
        buffer : float,  default 0.4
            The buffer width to use for the neighbor list.

        """
        self._lj_force().nlist = hoomd_nlist(buffer)

    @property
    def dt(self):
        """The simulation timestep."""
        return self._dt

    @dt.setter
    def dt(self, value):
        """Set the simulation timestep."""
        self._dt = value
        if self.integrator:
            self.operations.integrator.dt = self.dt

    @property
    def real_time_length(self):
        """The simulation time length in nanoseconds."""
        return (self.timestep * self.real_timestep).to("ns")

    @property
    def real_timestep(self):
        """The simulation timestep in femtoseconds."""
        if self._reference_values.get("mass"):
            mass = self._reference_values["mass"].to("kg")
        else:
            mass = 1 * Units.kg
        if self._reference_values.get("length"):
            dist = self.reference_length.to("m")
        else:
            dist = 1 * Units.m
        if self._reference_values.get("energy"):
            energy = self.reference_energy.to("J")
        else:
            energy = 1 * Units.J
        tau = (mass * (dist**2)) / energy
        timestep = self.dt * (tau**0.5)
        return timestep.to("fs")

    @property
    def reduced_temperature(self):
        """The temperature of the simulation in reduced units."""
        if not self._kT:
            raise ValueError(
                "Temperature is not set. Please specify the temperature when "
                "running the simulation, using one of the following run"
                " methods: `run_nvt`, `run_npt`, `run_update_volume`."
            )
        return self._kT

    @property
    def real_temperature(self):
        """The temperature of the simulation in Kelvin."""
        if self._reference_values.get("energy"):
            energy = self.reference_energy.to("J")
        else:
            energy = 1 * Units.J
        temperature = (
            self.reduced_temperature * energy
        ) / u.boltzmann_constant_mks
        return temperature

    @property
    def reduced_pressure(self):
        """The pressure of the simulation in reduced units."""
        if not self._reduced_pressure:
            raise ValueError(
                "Pressure is not set. Please specify the pressure when "
                "running the simulation, using the `run_npt` method."
            )
        return self._reduced_pressure

    @property
    def real_pressure(self):
        """The pressure of the simulation in Pascals."""
        if self._reference_values.get("energy"):
            energy = self.reference_energy.to("J/mol")
        else:
            energy = 1 * Units.J / Units.mol
        if self._reference_values.get("length"):
            length = self.reference_length.to("m")
        else:
            length = 1 * Units.m
        pressure = (self.reduced_pressure * energy) / (length**3)
        return pressure.to("Pa")

    def _temperature_to_kT(self, temperature):
        """Convert temperature to kT."""
        if self._reference_values.get("energy"):
            energy = self.reference_energy.to("J")
        else:
            energy = 1 * Units.J
        temperature = temperature.to("K")
        kT = (temperature * u.boltzmann_constant_mks) / energy
        return float(kT)

    def _pressure_to_reduced_pressure(self, pressure):
        """Convert pressure to reduced units."""
        if self._reference_values.get("energy"):
            energy = self.reference_energy.to("J/mol")
        else:
            energy = (1 * Units.J).to("J/mol")
        if self._reference_values.get("length"):
            length = self.reference_length.to("m")
        else:
            length = 1 * Units.m
        pressure = pressure.to("Pa")
        reduced_pressure = (pressure * (length**3)) / energy
        return float(reduced_pressure)

    def _time_length_to_n_steps(self, time_length):
        """Convert time length to number of steps."""
        time_length = time_length.to("s")
        real_timestep = self.real_timestep.to("s")
        return int(time_length / real_timestep)

    def _setup_temperature(self, temperature):
        """Set the temperature of the simulation."""
        if isinstance(temperature, (float, int, hoomd.variant.scalar.Ramp)):
            # assuming temperature is kT
            return temperature
        else:
            validated_temperature = validate_unit(
                temperature, u.dimensions.temperature
            )
            return self._temperature_to_kT(validated_temperature)

    def _setup_pressure(self, pressure):
        """Set the pressure of the simulation."""
        if isinstance(pressure, (float, int, hoomd.variant.scalar.Ramp)):
            # assuming pressure is in reduced units.
            return pressure
        else:
            validated_pressure = validate_unit(pressure, u.dimensions.pressure)
            return self._pressure_to_reduced_pressure(validated_pressure)

    def _setup_n_steps(self, duration):
        """Set the number of steps to run the simulation."""
        if isinstance(duration, (int, float)):
            # assuming duration is num steps
            return duration
        else:
            validated_duration = validate_unit(duration, u.dimensions.time)
            return self._time_length_to_n_steps(validated_duration)

    def _setup_period(self, period):
        """Set the period for the simulation."""
        if isinstance(period, int):
            # assuming period is num steps
            return period
        else:
            validated_period = validate_unit(period, u.dimensions.time)
            return self._time_length_to_n_steps(validated_period)

    @property
    def integrate_group(self):
        """The group of particles to apply the integrator to.

        Default is all particles.
        """
        return self._integrate_group

    @integrate_group.setter
    def integrate_group(self, group):
        """Set the group of particles to apply the integrator to.

        Checkout [HOOMD's documentation]
        (https://hoomd-blue.readthedocs.io/en/stable/module-hoomd-filter.html)
        for more information on Particle Filters.
        """
        self._integrate_group = group

    @property
    def method(self):
        """The integrator method used by the simulation."""
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
        """The thermostat used for the simulation."""
        return self._thermostat

    @thermostat.setter
    def thermostat(self, thermostat):
        """Set the thermostat used for the simulation.

        The thermostat must be a selected from
        `flowermd.utils.HOOMDThermostats`.

        Parameters
        ----------
        thermostat : flowermd.utils.HOOMDThermostats, required
            The type of thermostat to use.
        """
        if not issubclass(
            self._thermostat, hoomd.md.methods.thermostats.Thermostat
        ):
            raise ValueError(
                f"Invalid thermostat. Please choose from: {HOOMDThermostats}"
            )
        self._thermostat = thermostat

    def add_force(self, hoomd_force):
        """Add a force to the simulation.

        Parameters
        ----------
        hoomd_force : hoomd.md.force.Force, required
            The force to add to the simulation.

        """
        self._forcefield.append(hoomd_force)
        if self.integrator:
            self.integrator.forces.append(hoomd_force)

    def remove_force(self, hoomd_force):
        """Remove a force from the simulation.

        Parameters
        ----------
        hoomd_force : hoomd.md.force.Force, required
            The force to remove from the simulation.

        """
        self._forcefield.remove(hoomd_force)
        if self.integrator:
            self.integrator.forces.remove(hoomd_force)

    def adjust_epsilon(self, scale_by=None, shift_by=None, type_filter=None):
        """Adjust the epsilon parameter of the Lennard-Jones pair force.

        Parameters
        ----------
        scale_by : float, default None
            The factor to scale epsilon by.
        shift_by : float, default None
            The amount to shift epsilon by.
        type_filter : list of str, default None
            A list of particle pair types to apply the adjustment to.

        """
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
        """Adjust the sigma parameter of the Lennard-Jones pair force.

        Parameters
        ----------
        scale_by : float, default None
            The factor to scale sigma by.
        shift_by : float, default None
            The amount to shift sigma by.
        type_filter : list of str, default None
            A list of particle pair types to apply the adjustment to.

        """
        lj_forces = self._lj_force()
        for k in lj_forces.params.keys():
            if type_filter and k not in type_filter:
                continue
            sigma = lj_forces.params[k]["sigma"]
            if scale_by:
                lj_forces.params[k]["sigma"] = sigma * scale_by
            elif shift_by:
                lj_forces.params[k]["sigma"] = sigma + shift_by

    def _initialize_thermostat(self, thermostat_kwargs):
        """Initialize the thermostat used by the simulation.

        Parameters
        ----------
        thermostat_kwargs : dict, required
            A dictionary of parameter:value for the thermostat.
        """
        required_thermostat_kwargs = {}
        for k in inspect.signature(self.thermostat).parameters:
            if k not in thermostat_kwargs.keys():
                raise ValueError(
                    f"Missing required parameter {k} for thermostat."
                )
            required_thermostat_kwargs[k] = thermostat_kwargs[k]
        return self.thermostat(**required_thermostat_kwargs)

    def set_integrator_method(self, integrator_method, method_kwargs):
        """Create an initial (or updates the existing) integrator method.

        This doesn't need to be called directly;
        instead the various run functions use this method to update
        the integrator method as needed.

        Parameters
        ----------
        integrrator_method : hoomd.md.method, required
            Instance of one of the `hoomd.md.method` options.
        method_kwargs : dict, required
            A diction of parameter:value for the integrator method used.

        """
        if not self.integrator:  # Integrator and method not yet created
            self.integrator = hoomd.md.Integrator(
                dt=self.dt,
                integrate_rotational_dof=(
                    True
                    if (self._rigid_constraint or self.integrate_rotational_dof)
                    else False
                ),
            )
            if self._rigid_constraint:
                self.integrator.rigid = self._rigid_constraint
            self.integrator.forces = self._forcefield
            self.operations.add(self.integrator)
            new_method = integrator_method(**method_kwargs)
            self.operations.integrator.methods = [new_method]
        else:  # Replace the existing integrator method
            self.integrator.methods.remove(self.method)
            new_method = integrator_method(**method_kwargs)
            self.integrator.methods.append(new_method)

    def add_walls(self, wall_axis, sigma, epsilon, r_cut, r_extrap=0):
        """Add `hoomd.md.external.wall.LJ` forces to the simulation.

        Parameters
        ----------
        wall_axis : np.ndarray, shape=(3,), dtype=float, required
            The axis of the wall in (x, y, z) order.
        sigma : float, required
            The sigma parameter of the LJ wall.
        epsilon : float, required
            The epsilon parameter of the LJ wall.
        r_cut : float, required
            The cutoff radius of the LJ wall.
        r_extrap : float, default 0
            The extrapolation radius of the LJ wall.

        """
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
        """Remove LJ walls from the simulation.

        Parameters
        ----------
        wall_axis : np.ndarray, shape=(3,), dtype=float, required
            The axis of the wall in (x, y, z) order.

        """
        wall_force = self._wall_forces[wall_axis][0]
        self.remove_force(wall_force)

    def run_update_volume(
        self,
        final_box_lengths,
        temperature,
        tau_kt,
        duration,
        period,
        thermalize_particles=True,
        write_at_start=True,
    ):
        """Run an NVT simulation while shrinking or expanding simulation box.

        The simulation box is updated using `hoomd.update.BoxResize` and the
        final box lengths are set to `final_box_lengths`.

        See `flowermd.utils.get_target_volume_mass_density` and
        `flowermd.utils.get_target_volume_number_density` which are
        helper functions that can be used to get `final_box_lengths`.

        Parameters
        ----------
        final_box_lengths : np.ndarray or unyt.array.unyt_array, shape=(3,), required # noqa: E501
            The final box edge lengths in (x, y, z) order.
        temperature : flowermd.internal.Units or float or int, required
            The temperature to use during volume update. If no unit is provided,
            the temperature is assumed to be kT (temperature times Boltzmann
            constant).
        tau_kt : float, required
            Thermostat coupling period (in simulation time units).
        duration : int or flowermd.internal.Units, required
            The number of steps or time length to run the simulation. If no unit
            is provided, the time is assumed to be the number of steps.
        period : int or flowermd.internal.Units, required
            The number of steps or time length between box updates. If no unit
            is provided, the period is assumed to be the number of steps.
        write_at_start : bool, default True
            When set to True, triggers writers that evaluate to True
            for the initial step to execute before the next simulation
            time step.


        Examples
        --------
        In this example, a low density system is initialized with `Pack`
        and a box matching a density of 1.1 g/cm^3 is passed into
        `final_box_lengths`.

        ::

            import unyt
            from flowermd.base import Pack, Simulation
            from flowermd.library import PPS, OPLS_AA_PPS

            pps_mols = PPS(num_mols=20, lengths=15)
            pps_system = Pack(
                molecules=[pps_mols],
                density=0.5,
            )
            pps_system.apply_forcefield(
                r_cut=2.5,
                force_field=OPLS_AA_PPS(),
                auto_scale=True,
                scale_charges=True
            )
            sim = Simulation.from_system(pps_system)
            target_box = flowermd.utils.get_target_box_mass_density(
                density=1.1 * unyt.g/unyt.cm**3, mass=sim.mass.to("g")
            )
            sim.run_update_volume(
                final_box_lengths=target_box,
                temperature=1.0,
                tau_kt=1.0,
                duration=1e4,
                period=100
            )

        """
        self._kT = self._setup_temperature(temperature)
        _n_steps = self._setup_n_steps(duration)
        _period = self._setup_period(period)
        if self.reference_length and hasattr(final_box_lengths, "to"):
            ref_unit = self.reference_length.units
            final_box_lengths = final_box_lengths.to(ref_unit)
            final_box_lengths /= self.reference_length

        final_box = hoomd.Box(
            Lx=final_box_lengths[0],
            Ly=final_box_lengths[1],
            Lz=final_box_lengths[2],
        )
        resize_trigger = hoomd.trigger.Periodic(_period)
        box_ramp = hoomd.variant.Ramp(
            A=0, B=1, t_start=self.timestep, t_ramp=int(_n_steps)
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
            integrator_method=hoomd.md.methods.ConstantVolume,
            method_kwargs={
                "thermostat": self._initialize_thermostat(
                    {"kT": self._kT, "tau": tau_kt}
                ),
                "filter": self.integrate_group,
            },
        )
        if thermalize_particles:
            self._thermalize_system(self._kT)

        if self._wall_forces:
            wall_update = UpdateWalls(sim=self)
            wall_updater = hoomd.update.CustomUpdater(
                trigger=resize_trigger, action=wall_update
            )
            self.operations.updaters.append(wall_updater)
        std_out_logger = StdOutLogger(n_steps=_n_steps, sim=self)
        std_out_logger_printer = hoomd.update.CustomUpdater(
            trigger=hoomd.trigger.Periodic(self._std_out_freq),
            action=std_out_logger,
        )
        self.operations.updaters.append(std_out_logger_printer)
        self.run(steps=_n_steps + 1, write_at_start=write_at_start)
        self.operations.updaters.remove(std_out_logger_printer)
        self.operations.updaters.remove(box_resizer)

    def run_langevin(
        self,
        duration,
        temperature,
        tally_reservoir_energy=False,
        default_gamma=1.0,
        default_gamma_r=(1.0, 1.0, 1.0),
        thermalize_particles=True,
        write_at_start=True,
    ):
        """Run the simulation using the Langevin dynamics integrator.

        Parameters
        ----------
        duration : int or flowermd.internal.Units, required
            The number of steps or time length to run the simulation. If no unit
            is provided, the time is assumed to be the number of steps.
        temperature : flowermd.internal.Units or float or int, required
            The temperature to use during the simulation. If no unit is
            provided, the temperature is assumed to be kT (temperature times
            Boltzmann constant).
        tally_reservoir_energy : bool, default False
            When set to True, energy exchange between the thermal reservoir
             and the particles is tracked.
        default_gamma : float, default 1.0
            The default drag coefficient to use for all particles.
        default_gamma_r : tuple of floats, default (1.0, 1.0, 1.0)
            The default rotational drag coefficient to use for all particles.
        thermalize_particles : bool, default True
            When set to True, assigns random velocities to all particles.
        write_at_start : bool, default True
            When set to True, triggers writers that evaluate to True
            for the initial step to execute before the next simulation
            time step.

        """
        self._kT = self._setup_temperature(temperature)
        _n_steps = self._setup_n_steps(duration)
        self.set_integrator_method(
            integrator_method=hoomd.md.methods.Langevin,
            method_kwargs={
                "filter": self.integrate_group,
                "kT": self._kT,
                "tally_reservoir_energy": tally_reservoir_energy,
                "default_gamma": default_gamma,
                "default_gamma_r": default_gamma_r,
            },
        )
        if thermalize_particles:
            self._thermalize_system(self._kT)
        std_out_logger = StdOutLogger(n_steps=_n_steps, sim=self)
        std_out_logger_printer = hoomd.update.CustomUpdater(
            trigger=hoomd.trigger.Periodic(self._std_out_freq),
            action=std_out_logger,
        )
        self.operations.updaters.append(std_out_logger_printer)
        self.run(steps=_n_steps, write_at_start=write_at_start)
        self.operations.updaters.remove(std_out_logger_printer)

    def run_NPT(
        self,
        pressure,
        tau_pressure,
        temperature,
        tau_kt,
        duration,
        couple="xyz",
        box_dof=[True, True, True, False, False, False],
        rescale_all=False,
        gamma=0.0,
        thermalize_particles=True,
        write_at_start=True,
    ):
        """Run the simulation in the NPT ensemble.

        Parameters
        ----------
        pressure: flowermd.internal.Units or int or hoomd.variant.Ramp, required
            The pressure to use during the simulation. If no unit is provided,
            the pressure is assumed to in reduced units
            pressure * (Avogadro constant × reduced length^3 / reduced energy).
        tau_pressure: float, required
            Barostat coupling period.
        temperature: flowermd.internal.Units or float or int, required
            The temperature to use during the simulation. If no unit is
            provided, the temperature is assumed to be kT (temperature times
            Boltzmann constant).
        tau_kt: float, required
            Thermostat coupling period (in simulation time units).
        duration: int or flowermd.internal.Units, required
            The number of steps or time length to run the simulation. If no unit
            is provided, the time is assumed to be the number of steps.
        couple: str, default "xyz"
            Couplings of diagonal elements of the stress tensor/
        box_dof: list of bool;
                optional default [True, True, True, False, False, False]
            Degrees of freedom of the box.
        rescale_all: bool, default False
            Rescale all particles, not just those in the group.
        gamma: float, default 0.0
            Friction constant for the box degrees of freedom,
        thermalize_particles: bool, default True
            When set to True, assigns random velocities to all particles.
        write_at_start : bool, default True
            When set to True, triggers writers that evaluate to True
            for the initial step to execute before the next simulation
            time step.

        """
        self._kT = self._setup_temperature(temperature)
        self._reduced_pressure = self._setup_pressure(pressure)
        _n_steps = self._setup_n_steps(duration)
        self.set_integrator_method(
            integrator_method=hoomd.md.methods.ConstantPressure,
            method_kwargs={
                "S": pressure,
                "tauS": tau_pressure,
                "couple": couple,
                "box_dof": box_dof,
                "rescale_all": rescale_all,
                "gamma": gamma,
                "filter": self.integrate_group,
                "thermostat": self._initialize_thermostat(
                    {"kT": self._kT, "tau": tau_kt}
                ),
            },
        )
        if thermalize_particles:
            self._thermalize_system(self._kT)
        std_out_logger = StdOutLogger(n_steps=_n_steps, sim=self)
        std_out_logger_printer = hoomd.update.CustomUpdater(
            trigger=hoomd.trigger.Periodic(self._std_out_freq),
            action=std_out_logger,
        )
        self.operations.updaters.append(std_out_logger_printer)
        self.run(steps=_n_steps, write_at_start=write_at_start)
        self.operations.updaters.remove(std_out_logger_printer)

    def run_NVT(
        self,
        temperature,
        tau_kt,
        duration,
        thermalize_particles=True,
        write_at_start=True,
    ):
        """Run the simulation in the NVT ensemble.

        Parameters
        ----------
        temperature: flowermd.internal.Units or float or int, required
            The temperature to use during the simulation. If no unit is
            provided, the temperature is assumed to be kT (temperature times
            Boltzmann constant).
        tau_kt: float, required
            Thermostat coupling period (in simulation time units).
        duration: int or flowermd.internal.Units, required
            The number of steps or time length to run the simulation. If no unit
            is provided, the time is assumed to be the number of steps.
        thermalize_particles: bool, default True
            When set to True, assigns random velocities to all particles.
        write_at_start : bool, default True
            When set to True, triggers writers that evaluate to True
            for the initial step to execute before the next simulation
            time step.

        Examples
        --------
        In this example, a system is initialized with `Pack` and a simulation
        is run in the NVT ensemble at 300 K for 1 ns.

        ::

            import unyt
            from flowermd.base import Pack, Simulation
            from flowermd.library import PPS, OPLS_AA_PPS

            pps_mols = PPS(num_mols=20, lengths=15)
            pps_system = Pack(
                molecules=[pps_mols],
                density=0.5,
            )
            pps_system.apply_forcefield(
                r_cut=2.5,
                force_field=OPLS_AA_PPS(),
                auto_scale=True,
                scale_charges=True
            )
            sim = Simulation.from_system(pps_system)
            sim.run_NVT(
                temperature=300 * flowermd.internal.Units.K,
                tau_kt=1.0,
                duration= 1 * flowermd.internal.Units.ns,
            )

        """
        self._kT = self._setup_temperature(temperature)
        _n_steps = self._setup_n_steps(duration)
        self.set_integrator_method(
            integrator_method=hoomd.md.methods.ConstantVolume,
            method_kwargs={
                "thermostat": self._initialize_thermostat(
                    {"kT": self._kT, "tau": tau_kt}
                ),
                "filter": self.integrate_group,
            },
        )
        if thermalize_particles:
            self._thermalize_system(self._kT)
        std_out_logger = StdOutLogger(n_steps=_n_steps, sim=self)
        std_out_logger_printer = hoomd.update.CustomUpdater(
            trigger=hoomd.trigger.Periodic(self._std_out_freq),
            action=std_out_logger,
        )
        self.operations.updaters.append(std_out_logger_printer)
        self.run(steps=_n_steps, write_at_start=write_at_start)
        self.operations.updaters.remove(std_out_logger_printer)

    def run_NVE(self, duration, write_at_start=True):
        """Run the simulation in the NVE ensemble.

        Parameters
        ----------
        duration : int or flowermd.internal.Units, required
            The number of steps or time length to run the simulation. If no unit
            is provided, the time is assumed to be the number of steps.
        write_at_start : bool, default True
            When set to True, triggers writers that evaluate to True
            for the initial step to execute before the next simulation
            time step.

        """
        _n_steps = self._setup_n_steps(duration)
        self.set_integrator_method(
            integrator_method=hoomd.md.methods.ConstantVolume,
            method_kwargs={"filter": self.integrate_group},
        )
        std_out_logger = StdOutLogger(n_steps=_n_steps, sim=self)
        std_out_logger_printer = hoomd.update.CustomUpdater(
            trigger=hoomd.trigger.Periodic(self._std_out_freq),
            action=std_out_logger,
        )
        self.operations.updaters.append(std_out_logger_printer)
        self.run(steps=_n_steps, write_at_start=write_at_start)
        self.operations.updaters.remove(std_out_logger_printer)

    def run_displacement_cap(
        self,
        duration,
        maximum_displacement=1e-3,
        write_at_start=True,
    ):
        """NVE integrator with a cap on the maximum displacement per time step.

        DisplacementCapped method is mostly useful for initially relaxing a
        system with overlapping particles. Putting a cap on the max particle
        displacement prevents Hoomd Particle Out of Box execption.
        Once the system is relaxed, other run methods (NVE, NVT, etc) can be
        used.

        Parameters
        ----------
        duration : int or flowermd.internal.Units, required
            The number of steps or time length to run the simulation. If no unit
            is provided, the time is assumed to be the number of steps.
        maximum_displacement : float, default 1e-3
            Maximum displacement per step (length)
        write_at_start : bool, default True
            When set to True, triggers writers that evaluate to True
            for the initial step to execute before the next simulation
            time step.

        """
        _n_steps = self._setup_n_steps(duration)
        self.set_integrator_method(
            integrator_method=hoomd.md.methods.DisplacementCapped,
            method_kwargs={
                "filter": self.integrate_group,
                "maximum_displacement": maximum_displacement,
            },
        )
        std_out_logger = StdOutLogger(n_steps=_n_steps, sim=self)
        std_out_logger_printer = hoomd.update.CustomUpdater(
            trigger=hoomd.trigger.Periodic(self._std_out_freq),
            action=std_out_logger,
        )
        self.operations.updaters.append(std_out_logger_printer)
        self.run(steps=_n_steps, write_at_start=write_at_start)
        self.operations.updaters.remove(std_out_logger_printer)

    def temperature_ramp(
        self,
        duration,
        temperature_start,
        temperature_final,
    ):
        """Create a temperature ramp.

        Parameters
        ----------
        duration : int or flowermd.internal.Units, required
            The number of steps or time length to run the simulation. If no unit
            is provided, the time is assumed to be the number of steps.
        temperature_start : unyt.unyt_quantity or float, required
            The initial temperature. If unitless, the temperature is
            assumed to be kT (temperature times Boltzmann constant).
        temperature_final : unyt.unyt_quantity or float, required
            The final temperature. If unitless, the temperature is
            assumed to be kT (temperature times Boltzmann constant).

        """
        _kT_start = self._setup_temperature(temperature_start)
        _kT_final = self._setup_temperature(temperature_final)
        _n_steps = self._setup_n_steps(duration)
        return hoomd.variant.Ramp(
            A=_kT_start,
            B=_kT_final,
            t_start=self.timestep,
            t_ramp=int(_n_steps),
        )

    def pickle_forcefield(
        self, file_path="forcefield.pickle", save_walls=False
    ):
        """Pickle the list of HOOMD forces.

        This method useful for saving the forcefield of a simulation to a file
        and reusing it for restarting a simulation or running a different
        simulation.

        Parameters
        ----------
        file_path : str, default "forcefield.pickle"
            The path to save the pickle file to.
        save_walls : bool, default False
            Determines if any wall forces are saved.

        Notes
        -----
        Wall forces are not able to be reused when starting
        a simulation. If your simulation has wall forces,
        set `save_walls` to `False` and manually re-add them
        in the new simulation if needed.

        Examples
        --------
        In this example, a simulation is initialized and run for 1000 steps. The
        forcefield is then pickled and saved to a file. The forcefield is then
        loaded from the pickle file and used to run a tensile simulation.

        ::

            from flowermd.base import Pack, Simulation
            from flowermd.library import PPS, OPLS_AA_PPS, Tensile
            import pickle

            pps_mols = PPS(num_mols=10, lengths=5)
            pps_system = Pack(molecules=[pps_mols], force_field=OPLS_AA_PPS(),
                              r_cut=2.5, density=0.5, auto_scale=True,
                              scale_charges=True)
            sim = Simulation(initial_state=pps_system.hoomd_snapshot,
                             forcefield=pps_system.hoomd_forcefield)
            sim.run_NVT(duration=1e3, temperature=1.0, tau_kt=1.0)
            sim.pickle_forcefield("pps_forcefield.pickle")
            with open("pps_forcefield.pickle", "rb") as f:
                pps_forcefield = pickle.load(f)

            tensile_sim = Tensile(initial_state=pps_system.hoomd_snapshot,
                                  forcefield=pps_forcefield,
                                   tensile_axis=(1, 0, 0))
            tensile_sim.run_tensile(strain=0.05, temperature=2.0, duration=1e3, period=10)

        """
        if self._wall_forces and save_walls is False:
            forces = []
            for force in self._forcefield:
                if not isinstance(force, hoomd.md.external.wall.LJ):
                    forces.append(force)
        else:
            forces = self._forcefield
        f = open(file_path, "wb")
        pickle.dump(forces, f)

    def save_restart_gsd(self, file_path="restart.gsd"):
        """Save a GSD file of the current simulation state.

        This method is useful for saving the state of a simulation to a file
        and reusing it for restarting a simulation or running a different
        simulation.

        Parameters
        ----------
        file_path : str, default "restart.gsd"
            The path to save the GSD file to.

        Examples
        --------
        This example is similar to the example in `pickle_forcefield`. The only
        difference is that the simulation state is also saved to a GSD file.

        ::

            from flowermd.base import Pack, Simulation
            from flowermd.library import PPS, OPLS_AA_PPS, Tensile
            import pickle

            pps_mols = PPS(num_mols=10, lengths=5)
            pps_system = Pack(molecules=[pps_mols], force_field=OPLS_AA_PPS(),
                              r_cut=2.5, density=0.5, auto_scale=True,
                              scale_charges=True)
            sim = Simulation(initial_state=pps_system.hoomd_snapshot,
                             forcefield=pps_system.hoomd_forcefield)
            sim.run_NVT(duration=1e3, temperature=1.0, tau_kt=1.0)
            sim.pickle_forcefield("pps_forcefield.pickle")
            sim.save_restart_gsd("pps_restart.gsd")
            with open("pps_forcefield.pickle", "rb") as f:
                pps_forcefield = pickle.load(f)

            tensile_sim = Tensile(initial_state="pps_restart.gsd",
                                  forcefield=pps_forcefield,
                                  tensile_axis=(1, 0, 0))
            tensile_sim.run_tensile(strain=0.05, temperature=2.0, duration=1e3, period=10)


        """
        hoomd.write.GSD.write(self.state, filename=file_path)

    def save_simulation(self, file_path="simulation.pickle"):
        """Save a pickle file with everything needed to retart a simulation.

        This method is useful for saving the state of a simulation to a file
        and reusing it for restarting a simulation or running a different
        simulation.

        Parameters
        ----------
        file_path : str, default "simulation.pickle"
            The path to save the pickle file to.

        Notes
        -----
        This method creates a dictionary that contains the
        simulation's forcefield, references values, and snapshot.
        The key:value pairs are:

            'reference_values': dict of str:float
            'forcefield': list of hoomd forces
            'state': gsd.hoomd.Frame
            'sim_kwargs': dict of flowermd.base.Simulation kwargs
        """
        # Make a temp restart gsd file.
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_file_path = os.path.join(tmp_dir, "temp.gsd")
            self.save_restart_gsd(temp_file_path)
            with gsd.hoomd.open(temp_file_path, "r") as traj:
                snap = traj[0]
                os.remove(temp_file_path)
        # Save dict of kwargs needed to restart simulation.
        sim_kwargs = {
            "dt": self.dt,
            "gsd_write_freq": self.gsd_write_freq,
            "log_write_freq": self.log_write_freq,
            "gsd_max_buffer_size": self.maximum_write_buffer_size,
            "seed": self.seed,
        }
        # Create the final dict that holds everything.
        sim_dict = {
            "reference_values": self.reference_values,
            "forcefield": self._forcefield,
            "state": snap,
            "sim_kwargs": sim_kwargs,
        }
        # Add a header to the pickle file.
        # This will be checked in Simulation.from_simulation_pickle.
        flower_string = b"FLOWERMD"
        with open(file_path, "wb") as f:
            f.write(flower_string)
            pickle.dump(sim_dict, f)

    def flush_writers(self):
        """Flush all write buffers to file."""
        for writer in self.operations.writers:
            if hasattr(writer, "flush"):
                writer.flush()

    def _thermalize_system(self, kT):
        """Assign random velocities to all particles.

        Parameters
        ----------
        kT : float or hoomd.variant.Ramp, required
            The temperature to use during the thermalization.

        """
        if isinstance(kT, hoomd.variant.Ramp):
            self.state.thermalize_particle_momenta(
                filter=self.integrate_group, kT=kT.range[0]
            )
        else:
            self.state.thermalize_particle_momenta(
                filter=self.integrate_group, kT=kT
            )

    def _lj_force(self):
        """Return the Lennard-Jones pair force."""
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

    def _create_integrate_group(self, rigid):
        if rigid:
            return hoomd.filter.Rigid(("center", "free"))
        return hoomd.filter.All()

    def _create_state(self, initial_state):
        """Create the simulation state.

        If initial_state is a snapshot, the state is created from the snapshot.
        If initial_state is a GSD file, the state is created from the GSD file.

        """
        if isinstance(initial_state, str):  # Load from a GSD file
            print("Initializing simulation state from a GSD file.")
            self.create_state_from_gsd(initial_state)
        elif isinstance(initial_state, hoomd.snapshot.Snapshot):
            print(
                "Initializing simulation state from a hoomd.snapshot.Snapshot."
            )
            self.create_state_from_snapshot(initial_state)
        elif isinstance(initial_state, gsd.hoomd.Frame):
            print("Initializing simulation state from a gsd.hoomd.Frame.")
            self.create_state_from_snapshot(initial_state)

    def _add_hoomd_writers(self):
        """Create gsd and log writers."""
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
            dynamic=["momentum", "property"],
            filter=hoomd.filter.All(),
            logger=gsd_logger,
        )
        gsd_writer.maximum_write_buffer_size = self.maximum_write_buffer_size

        table_file = hoomd.write.Table(
            output=open(self.log_file_name, mode="w", newline="\n"),
            trigger=hoomd.trigger.Periodic(period=int(self.log_write_freq)),
            logger=logger,
            max_header_len=None,
        )
        self.operations.writers.append(gsd_writer)
        self.operations.writers.append(table_file)
