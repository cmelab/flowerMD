"""Base simulation class for flowerMD."""

import inspect
import pickle
import warnings
from collections.abc import Iterable

import gsd.hoomd
import hoomd
import hoomd.md
import numpy as np
import unyt as u

from flowermd.internal import validate_ref_value
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
    ):
        if not isinstance(forcefield, Iterable) or isinstance(forcefield, str):
            raise ValueError(
                "forcefield must be a sequence of "
                "hoomd.md.force.Force objects."
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
        self._integrate_group = hoomd.filter.All()
        self._wall_forces = dict()
        self._create_state(self.initial_state)
        # Add a gsd and thermo props logger to sim operations
        self._add_hoomd_writers()
        self._thermostat = thermostat

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
    def real_timestep(self):
        """The simulation timestep in real units."""
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
    def real_temperature(self):
        """The temperature of the simulation in Kelvin."""
        if not self._kT:
            raise ValueError(
                "Temperature is not set. Please specify the temperature when "
                "running the simulation, using one of the following run"
                " methods: `run_nvt`, `run_npt`, `run_update_volume`."
            )
        if self._reference_values.get("energy"):
            energy = self.reference_energy.to("J")
        else:
            energy = 1 * u.J
        temperature = (self._kT * energy) / u.boltzmann_constant_mks
        return temperature

    def _temperature_to_kT(self, temperature):
        """Convert temperature to kT."""
        if self._reference_values.get("energy"):
            energy = self.reference_energy.to("J")
        else:
            energy = 1 * u.J
        if isinstance(temperature, u.unyt_array) or isinstance(
            temperature, u.unyt_quantity
        ):
            temperature = temperature.to("K")
        else:
            temperature = temperature * u.K
        kT = (temperature * u.boltzmann_constant_mks) / energy
        return float(kT)

    def _setup_temperature(self, kT=None, temperature=None):
        if kT and temperature:
            raise ValueError(
                "Both kT and temperature are provided. Please provide only one."
            )
        if not kT and not temperature:
            raise ValueError(
                "Either kT or temperature  must be "
                "provided for the simulation."
            )
        if kT:
            return kT
        else:
            return self._temperature_to_kT(temperature)

    def _time_length_to_n_steps(self, time_length):
        """Convert time length to number of steps."""
        if isinstance(time_length, u.unyt_array) or isinstance(
            time_length, u.unyt_quantity
        ):
            time_length = time_length.to("s")
        else:
            time_length = time_length * u.s
        real_timestep = self.real_timestep.to("s")
        return int(time_length / real_timestep)

    def _setup_n_steps(self, n_steps=None, time_length=None):
        if n_steps and time_length:
            raise ValueError(
                "Both n_steps and time_length are provided. Please provide only"
                " one."
            )
        if not n_steps and not time_length:
            raise ValueError(
                "Either n_steps or time_length must be provided for the "
                "simulation."
            )
        if n_steps:
            return n_steps
        else:
            return self._time_length_to_n_steps(time_length)

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
        period,
        tau_kt,
        n_steps=None,
        time_length=None,
        kT=None,
        temperature=None,
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
        n_steps : int, required
            Number of steps to run during volume update.
        period : int, required
            The number of steps ran between each box update iteration.
        kT : float or hoomd.variant.Ramp, required
            The temperature to use during volume update.
        tau_kt : float, required
            Thermostat coupling period (in simulation time units).
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
                force_field=OPLS_AA_PPS(),
                r_cut=2.5,
                density=0.5,
                auto_scale=True,
                scale_charges=True
            )
            sim = Simulation(
                initial_state=pps_system.hoomd_snapshot,
                forcefield=pps_system.hoomd_forcefield
            )
            target_box = flowermd.utils.get_target_box_mass_density(
                density=1.1 * unyt.g/unyt.cm**3, mass=sim.mass.to("g")
            )
            sim.run_update_volume(
                n_steps=1e4, kT=1.0, tau_kt=1.0, final_box_lengths=target_box
            )

        """
        self._kT = self._setup_temperature(kT, temperature)
        _n_steps = self._setup_n_steps(n_steps, time_length)
        if self.reference_length and hasattr(final_box_lengths, "to"):
            ref_unit = self.reference_length.units
            final_box_lengths = final_box_lengths.to(ref_unit)
            final_box_lengths /= self.reference_length

        final_box = hoomd.Box(
            Lx=final_box_lengths[0],
            Ly=final_box_lengths[1],
            Lz=final_box_lengths[2],
        )
        resize_trigger = hoomd.trigger.Periodic(period)
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
        n_steps=None,
        time_length=None,
        kT=None,
        temperature=None,
        tally_reservoir_energy=False,
        default_gamma=1.0,
        default_gamma_r=(1.0, 1.0, 1.0),
        thermalize_particles=True,
        write_at_start=True,
    ):
        """Run the simulation using the Langevin dynamics integrator.

        Parameters
        ----------
        n_steps : int, required
            Number of steps to run the simulation.
        kT : int or hoomd.variant.Ramp, required
            The temperature to use during the simulation.
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
        self._kT = self._setup_temperature(kT, temperature)
        _n_steps = self._setup_n_steps(n_steps, time_length)
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
        tau_kt,
        kT=None,
        temperature=None,
        n_steps=None,
        time_length=None,
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
        n_steps: int, required
            Number of steps to run the simulation.
        kT: int or hoomd.variant.Ramp, required
            The temperature to use during the simulation.
        pressure: int or hoomd.variant.Ramp, required
            The pressure to use during the simulation.
        tau_kt: float, required
            Thermostat coupling period (in simulation time units).
        tau_pressure: float, required
            Barostat coupling period.
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
        self._kT = self._setup_temperature(kT, temperature)
        _n_steps = self._setup_n_steps(n_steps, time_length)
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
        tau_kt,
        kT=None,
        temperature=None,
        n_steps=None,
        time_length=None,
        thermalize_particles=True,
        write_at_start=True,
    ):
        """Run the simulation in the NVT ensemble.

        Parameters
        ----------
        n_steps: int, required
            Number of steps to run the simulation.
        kT: int or hoomd.variant.Ramp, required
            The temperature to use during the simulation.
        tau_kt: float, required
            Thermostat coupling period (in simulation time units).
        thermalize_particles: bool, default True
            When set to True, assigns random velocities to all particles.
        write_at_start : bool, default True
            When set to True, triggers writers that evaluate to True
            for the initial step to execute before the next simulation
            time step.

        """
        self._kT = self._setup_temperature(kT, temperature)
        _n_steps = self._setup_n_steps(n_steps, time_length)
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

    def run_NVE(self, n_steps=None, time_length=None, write_at_start=True):
        """Run the simulation in the NVE ensemble.

        Parameters
        ----------
        n_steps: int, required
            Number of steps to run the simulation.
        write_at_start : bool, default True
            When set to True, triggers writers that evaluate to True
            for the initial step to execute before the next simulation
            time step.

        """
        _n_steps = self._setup_n_steps(n_steps, time_length)
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
        n_steps=None,
        time_length=None,
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
        n_steps : int, required
            Number of steps to run the simulation.
        maximum_displacement : float, default 1e-3
            Maximum displacement per step (length)

        write_at_start : bool, default True
            When set to True, triggers writers that evaluate to True
            for the initial step to execute before the next simulation
            time step.

        """
        _n_steps = self._setup_n_steps(n_steps, time_length)
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
        n_steps=None,
        time_length=None,
        kT_start=None,
        temperature_start=None,
        kT_final=None,
        temperature_final=None,
    ):
        """Create a temperature ramp.

        Parameters
        ----------
        n_steps : int, required
            The number of steps to ramp the temperature over.
        kT_start : float, required
            The starting temperature.
        kT_final : float, required
            The final temperature.

        """
        _kT_start = self._setup_temperature(kT_start, temperature_start)
        _kT_final = self._setup_temperature(kT_final, temperature_final)
        _n_steps = self._setup_n_steps(n_steps, time_length)
        return hoomd.variant.Ramp(
            A=_kT_start,
            B=_kT_final,
            t_start=self.timestep,
            t_ramp=int(_n_steps),
        )

    def pickle_forcefield(self, file_path="forcefield.pickle"):
        """Pickle the list of HOOMD forces.

        This method useful for saving the forcefield of a simulation to a file
        and reusing it for restarting a simulation or running a different
        simulation.

        Parameters
        ----------
        file_path : str, default "forcefield.pickle"
            The path to save the pickle file to.

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
            sim.run_NVT(n_steps=1e3, kT=1.0, tau_kt=1.0)
            sim.pickle_forcefield("pps_forcefield.pickle")
            with open("pps_forcefield.pickle", "rb") as f:
                pps_forcefield = pickle.load(f)

            tensile_sim = Tensile(initial_state=pps_system.hoomd_snapshot,
                                  forcefield=pps_forcefield,
                                   tensile_axis=(1, 0, 0))
            tensile_sim.run_tensile(strain=0.05, kT=2.0, n_steps=1e3, period=10)

        """
        f = open(file_path, "wb")
        pickle.dump(self._forcefield, f)

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
            sim.run_NVT(n_steps=1e3, kT=1.0, tau_kt=1.0)
            sim.pickle_forcefield("pps_forcefield.pickle")
            sim.save_restart_gsd("pps_restart.gsd")
            with open("pps_forcefield.pickle", "rb") as f:
                pps_forcefield = pickle.load(f)

            tensile_sim = Tensile(initial_state="pps_restart.gsd",
                                  forcefield=pps_forcefield,
                                  tensile_axis=(1, 0, 0))
            tensile_sim.run_tensile(strain=0.05, kT=2.0, n_steps=1e3, period=10)


        """
        hoomd.write.GSD.write(self.state, filename=file_path)

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
                "Initializing simulation state from a hoomd.snapshot.Snapshot"
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
