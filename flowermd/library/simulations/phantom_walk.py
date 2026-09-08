"""DPD energy relaxation simulation class."""

import hoomd

from flowermd.base.simulation import Simulation
from flowermd.utils.dpd_utils import simulation_energy_end


class PhantomWalk(Simulation):
    """ Run an initial energy relaxation of overlapping particles with DPD."""

    def __init__(
        self,
        initial_state,
        forcefield,
        n_steps_dpd,
        n_steps_fire,
        reference_values=dict(),
        dt=0.001,
        device=hoomd.device.auto_select(),
        seed=42,
        gsd_write_freq=1e4,
        gsd_file_name="trajectory.gsd",
        log_write_freq=1e3,
        log_file_name="log.txt",
    ):
        self.n_steps_dpd = n_steps_dpd
        self.n_steps_fire = n_steps_fire
        super(PhantomWalk, self).__init__(
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
        self.run_NVE(n_steps=self.n_steps_dpd,write_at_start=False)
        self.run_FIRE(n_steps=self.n_steps_fire,dt=self.dt)
        for writer in self.operations.writers:
            if hasattr(writer, "flush"):
                writer.flush()
        
