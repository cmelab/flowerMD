"""Tensile simulation class."""

import hoomd
import numpy as np

from flowermd.base.simulation import Simulation
from flowermd.utils import HOOMDThermostats, PullParticles


class Tensile(Simulation):
    """Tensile simulation class.

    Parameters
    ----------
    tensile_axis : tuple of int, required
        The axis along which to apply the tensile strain.
    fix_ratio : float, default=0.20
        The ratio of the box length to fix particles at each end
        of the tensile axis.

    """

    def __init__(
        self,
        initial_state,
        forcefield,
        tensile_axis,
        fix_ratio=0.20,
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
        super(Tensile, self).__init__(
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
        self.tensile_axis = np.asarray(tensile_axis)
        self.fix_ratio = fix_ratio
        self._axis_index = np.where(self.tensile_axis != 0)[0]
        self.initial_box = self.box_lengths_reduced
        self.initial_length = self.initial_box[self._axis_index]
        self.fix_length = self.initial_length * fix_ratio
        # Set up walls of fixed particles:
        snapshot = self.state.get_snapshot()
        positions = snapshot.particles.position[:, self._axis_index]
        box_max = self.initial_length / 2
        box_min = -box_max
        left_tags = np.where(positions < (box_min + self.fix_length))[0]
        right_tags = np.where(positions > (box_max - self.fix_length))[0]
        self.fix_left = hoomd.filter.Tags(left_tags.astype(np.uint32))
        self.fix_right = hoomd.filter.Tags(right_tags.astype(np.uint32))
        all_fixed = hoomd.filter.Union(self.fix_left, self.fix_right)
        # Set the group of particles to be integrated over
        self.integrate_group = hoomd.filter.SetDifference(
            hoomd.filter.All(), all_fixed
        )

    @property
    def strain(self):
        """The current strain of the simulation."""
        delta_L = (
            self.box_lengths_reduced[self._axis_index] - self.initial_length
        )
        return delta_L / self.initial_length

    def run_tensile(self, strain, duration, temperature, tau_kt, period):
        """Run a tensile test simulation.

        Parameters
        ----------
        strain : float, required
            The strain to apply to the simulation.
        duration : int or flowermd.internal.Units, required
            The number of steps or time length to run the simulation. If unitless,
             the time is assumed to be the number of steps.
        temperature : flowermd.internal.Units or float or int, required
            The temperature to use during volume update. If no unit is provided,
            the temperature is assumed to be kT (temperature times Boltzmann
            constant).
        tau_kt : float, required
            Thermostat coupling period (in simulation time units).
        period : int or flowermd.internal.Units, required
            The number of steps or time length between box updates. If no unit
            is provided, the period is assumed to be the number of steps.

        """
        current_length = self.box_lengths_reduced[self._axis_index]
        final_length = current_length * (1 + strain)
        final_box = np.copy(self.box_lengths_reduced)
        final_box[self._axis_index] = final_length
        n_steps = self._setup_n_steps(duration)
        shift_by = (final_length - current_length) / (n_steps // period)
        resize_trigger = hoomd.trigger.Periodic(period)
        box_ramp = hoomd.variant.Ramp(
            A=0, B=1, t_start=self.timestep, t_ramp=int(n_steps)
        )
        box_resizer = hoomd.update.BoxResize(
            box1=self.box_lengths_reduced,
            box2=final_box,
            variant=box_ramp,
            trigger=resize_trigger,
            filter=hoomd.filter.Null(),
        )
        particle_puller = PullParticles(
            shift_by=shift_by / 2,
            axis=self.tensile_axis,
            neg_filter=self.fix_left,
            pos_filter=self.fix_right,
        )
        particle_updater = hoomd.update.CustomUpdater(
            trigger=resize_trigger, action=particle_puller
        )
        self.operations.updaters.append(box_resizer)
        self.operations.updaters.append(particle_updater)
        self.run_NVT(
            duration=n_steps + 1, temperature=temperature, tau_kt=tau_kt
        )
