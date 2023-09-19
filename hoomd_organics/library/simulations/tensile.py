import hoomd
import numpy as np

from hoomd_organics.base.simulation import Simulation
from hoomd_organics.utils.actions import PullParticles


class Tensile(Simulation):
    def __init__(
        self,
        initial_state,
        forcefield,
        tensile_axis,
        fix_ratio=0.20,
        r_cut=2.5,
        dt=0.0001,
        device=hoomd.device.auto_select(),
        seed=42,
        restart=None,
        gsd_write_freq=1e4,
        gsd_file_name="trajectory.gsd",
        log_write_freq=1e3,
        log_file_name="log.txt",
    ):
        super(Tensile, self).__init__(
            initial_state=initial_state,
            forcefield=forcefield,
            r_cut=r_cut,
            dt=dt,
            device=device,
            seed=seed,
            gsd_write_freq=gsd_write_freq,
            gsd_file_name=gsd_file_name,
            log_write_freq=log_write_freq,
            log_file_name=log_file_name,
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
        delta_L = (
            self.box_lengths_reduced[self._axis_index] - self.initial_length
        )
        return delta_L / self.initial_length

    def run_tensile(self, strain, n_steps, kT, tau_kT, period):
        # self._thermalize_system(kT=kT)
        current_length = self.box_lengths_reduced[self._axis_index]
        final_length = current_length * (1 + strain)
        final_box = np.copy(self.box_lengths_reduced)
        final_box[self._axis_index] = final_length
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
        # self.set_integrator_method(
        #    integrator_method=hoomd.md.methods.NVE,
        #    method_kwargs={"filter": self.integrate_group},
        # )
        # self.set_integrator_method(
        #    integrator_method=hoomd.md.methods.NVE,
        #    method_kwargs={"filter": self.integrate_group},
        # )
        # self.run(n_steps + 1)
        self.run_NVT(n_steps=n_steps, kT=kT, tau_kt=tau_kT)
