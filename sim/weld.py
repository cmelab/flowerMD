from hoomd_polymers import Simulation

def Weld(Simulation):
    def __init__(
            self,
            initial_state,
            forcefield,
            weld_axis,
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
        super(Weld, self).__init__(
                initial_state=initial_state,
                forcefield=forcefield,
                r_cut=rcut,
                dt=dt,
                seed=seed,
                restart=resart,
                gsd_write_freq=gsd_write_freq,
                gsd_file_name=gsd_file_name,
                log_write_freq=log_write_freq,
        )
        self.weld_axis=weld_axis.lower()
        self.fix_ratio = fix_ratio
        force_dict = {
                "x": np.array([1,0,0]),
                "y": np.array([0,1,0]),
                "z": np.array([0,0,1])
        }
        position_dict = {"x": 0, "y": 1, "z": 2}
        # TODO: do we need the box length here? Probably not
        axis_box_length = getattr(self.sim.state.box, f"L{self.weld_axis}")
        snap = sim.state.get_snapshot()
        positions = snap.particles.position[:,position_dict[self.weld_axis]]
        left_tags = np.where(positions < 0)[0]
        right_tags = np.where(positions > 0)[0]
        left_group = hoomd.filter.Tags(left_tags)
        right_group = hoomd.filter.Tags(right_tags)
        left_force = hoomd.md.force.Constant(filter=left_group)
        right_force = hoomd.md.force.Constant(filter=right_group)
        for _type in snap.particles.types:
            left_force.constant_force[_type] = force_dict[self.weld_axis]
            right_force.constant_force[_type] = -1*force_dict[self.weld_axis]
