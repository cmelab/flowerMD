"""Module for simulating surface wetting."""
import hoomd
import unyt as u

from jankflow.base.simulation import Simulation
from jankflow.utils import HOOMDThermostats


class Surface:
    """For creating an interface between two slabs.

    Parameters
    ----------
    gsd_file : str
        Path to gsd file of the slab.
    interface_axis : tuple
        Axis along which the interface is to be created.
        The slab file is duplicated and translated along this axis.
    gap : float, required
        Distance (in simulation units) between the two slabs at the interface.
    wall_sigma : float
        Sigma parameter used for the wall potential when creating the slabs.

    """

    def __init__(self, drop_gsd_file, surface_gsd_file, gap):
        self.drop_gsd_file = drop_gsd_file
        self.surface_gsd_file = surface_gsd_file
        self.gap = gap

    def _build(self):
        """Duplicates the slab and builds the interface."""
        pass


class DropletSimulation(Simulation):
    """Simulation which creates a droplet."""

    def __init__(
        self,
        initial_state,
        forcefield,
        r_cut=2.5,
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
        super(DropletSimulation, self).__init__(
            initial_state=initial_state,
            forcefield=forcefield,
            r_cut=r_cut,
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

    def run_droplet(
        self,
        kT,
        tau_kt,
        shrink_steps,
        expand_steps,
        hold_steps,
        shrink_period,
        expand_period,
        final_density,
    ):
        """Run droplet simulation."""
        # Shrink down to high density
        self.run_update_volume(
            n_steps=shrink_steps,
            period=shrink_period,
            kT=kT,
            tau_kt=tau_kt,
            final_density=1.4 * (u.g / (u.cm**3)),
            write_at_start=True,
        )
        # Expand back up to low density
        self.run_update_volume(
            n_steps=expand_steps,
            period=expand_period,
            kT=kT,
            tau_kt=tau_kt,
            final_density=final_density * (u.g / (u.cm**3)),
            resize_filter=hoomd.filter.Null()
        )
        # Run at low density
        self.run_NVT(n_steps=hold_steps, kT=kT, tau_kt=tau_kt)


class WettingSimulation(Simulation):
    """For simulating welding of an interface joint."""

    def __init__(
        self,
        initial_state,
        forcefield,
        r_cut=2.5,
        reference_values=dict(),
        dt=0.0001,
        device=hoomd.device.auto_select(),
        seed=42,
        gsd_write_freq=1e4,
        gsd_file_name="weld.gsd",
        log_write_freq=1e3,
        log_file_name="sim_data.txt",
        thermostat=HOOMDThermostats.MTTK,
    ):
        super(WettingSimulation, self).__init__(
            initial_state=initial_state,
            forcefield=forcefield,
            r_cut=r_cut,
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
