"""DPD energy relaxation simulation class."""

import hoomd

from flowermd.base.simulation import Simulation
from flowermd.utils.dpd_utils import simulation_energy_end


class DPDInit(Simulation):
    """ """

    def __init__(
        self,
        initial_state,
        forcefield,
        A,
        r,
        r_cut,
        num_pol,
        num_mon,
        sim_steps_incr,
        box,
        reference_values=dict(),
        dt=0.0001,
        device=hoomd.device.auto_select(),
        seed=42,
        gsd_write_freq=1e4,
        gsd_file_name="trajectory.gsd",
        log_write_freq=1e3,
        log_file_name="log.txt",
    ):
        self.A = A
        self.r = r
        self.r_cut = r_cut
        self.num_pol = num_pol
        self.num_mon = num_mon
        self.sim_steps_incr = sim_steps_incr
        self.L = box
        super(DPDInit, self).__init__(
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
        )
        print(self.A, self.r, self.r_cut, self.density)
        self.run_NVE(n_steps=1)
        while not simulation_energy_end(
            A=self.A,
            r=self.r,
            r_cut=self.r_cut,
            num_pol=self.num_pol,
            num_mon=self.num_mon,
            density=(self.L**3),
            log_file_name=self.log_file_name,
        ):
            self.run_NVE(n_steps=self.sim_steps_incr)
            for writer in self.operations.writers:
                if hasattr(writer, "flush"):
                    writer.flush()
