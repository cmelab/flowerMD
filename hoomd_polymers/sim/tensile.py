import hoomd
import numpy as np

from hoomd_polymers.sim.simulation import Simulation


class Tensile(Simulation):
    def __init__(
            self,
            initial_state,
            forcefield,
            tensile_axis,
            fix_ratio=0.20,
            r_cut=2.5,
            dt=0.0001,
            seed=42,
            restart=None,
            gsd_write_freq=1e4,
            gsd_file_name="trajectory.gsd",
            log_write_freq=1e3,
            log_file_name="sim_data.txt"
    ):
        super(Tensile, self).__init__(
                initial_state=initial_state,
                forcefield=forcefield,
                r_cut=r_cut,
                dt=dt,
                seed=seed,
                restart=restart,
                gsd_write_freq=gsd_write_freq,
                gsd_file_name=gsd_file_name,
                log_write_freq=log_write_freq,
        )
        self.tensile_axis=tensile_axis.lower()
        self.fix_ratio = fix_ratio
        axis_array_dict = {
                "x": np.array([1,0,0]),
                "y": np.array([0,1,0]),
                "z": np.array([0,0,1])
        }
        axis_dict = {"x": 0, "y": 1, "z": 2}
        self._axis_index = axis_dict[self.tensile_axis]
        self._axis_array = axis_array_dict[self.tensile_axis]
        # Set up final box length after tensile test
        self.initial_box = self.box_lengths
        self.initial_length = self.initial_box[self._axis_index]
        #self.final_length = self.initial_length * (1+strain)
        #self.final_box = np.copy(self.initial_box)
        #self.final_box[self._axis_index] = self.final_length
        # Set up walls of fixed particles:
        snapshot = self.sim.state.get_snapshot()
        self.fix_length = self.initial_length * fix_ratio
        positions = snapshot.particles.position[:,self._axis_index]
        box_max = self.initial_length / 2
        box_min = -box_max
        left_tags = np.where(positions < (box_min + self.fix_length))[0] 
        right_tags = np.where(positions > (box_max - self.fix_length))[0] 
        self.fix_left = hoomd.filter.Tags(left_tags.astype(np.uint32))
        self.fix_right = hoomd.filter.Tags(right_tags.astype(np.uint32))
        self.all_fixed = hoomd.filter.Union(self.fix_left, self.fix_right)
        # Set the group of particles to be integrated over
        self.integrate_group = hoomd.filter.SetDifference(
                hoomd.filter.All(), self.all_fixed
        )

        @property
        def strain(self):
            delta_L = self.box_lengths[self._axis_index] - self.initial_length
            return delta_L / self.initial_length 

        def _shift_particles(self, shift_by):
            snap = self.sim.state.get_snapshot()
            snap.particles.position[
                    self.fix_left.tags]-=(self._axis_array*(shift_by/2))
            snap.particles.position[
                    self.fix_right.tags]+=(self._axis_array*(shift_by/2))
            self.sim.state.set_snapshot(snap)

        def run_tesile(self, strain, kT, n_steps, period):
            current_length = self.box_lengths[self._axis_index]
            final_length = current_length * (1 + strain)
            final_box = np.copy(self.box_lengths)
            final_box[self._axis_index] = final_length
            # Set up box resizer
            resize_trigger = hoomd.trigger.Periodic(period)
            box_ramp = hoomd.variant.Ramp(
                    A=0, B=1, t_start=self.sim.timestep, t_ramp=int(n_steps)
            )
            box_resizer = hoomd.update.BoxResize(
                    box1=self.box_lengths,
                    box2=final_box,
                    variant=box_ramp,
                    trigger=resize_trigger
            )
            self.sim.operations.updaters.append(box_resizer)
            self.set_integrator_method(
                integrator_method=hoomd.md.methods.NVE,
                method_kwargs={"filter": self.integrate_group}
            )

            last_length = initial_length
            while self.sim.timestep < box_ramp.t_start + box_ramp.t_ramp:
                self.sim.run(period)
                shift_by = self.box_lengths[self._axis_index] - last_length
                self._shift_particles(shift_by)
                last_length = self.box_lengths
