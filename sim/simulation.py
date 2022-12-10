from itertools import combinations_with_replacement as combo
import operator
import os

import gsd.hoomd
import hoomd
import hoomd.md
from mbuild.formats.hoomd_forcefield import create_hoomd_forcefield
import numpy as np
import parmed as pmd


class Simulation:
    """The simulation context management class.

    This class takes the output of the Initialization class
    and sets up a hoomd-blue simulation.

    Parameters
    ----------
    system : Parmed Structure 
        The system created in system.py 
    r_cut : float, default 2.5
        Cutoff radius for potentials (in simulation distance units)
    tau_kt : float, default 0.1
        Thermostat coupling period (in simulation time units)
    tau_p : float, default None
        Barostat coupling period (in simulation time units)
    nlist : str, default `Cell`
        Type of neighborlist to use. Options are "Cell", "Tree", and "Stencil".
        See https://hoomd-blue.readthedocs.io/en/latest/module-md-nlist.html
    dt : float, default 0.0001
        Size of simulation timestep (in simulation time units)
    auto_scale : bool, default True
        Set to true to use reduced simulation units.
        distance, mass, and energy are scaled by the largest value
        present in the system for each.
    gsd_write : int, default 1e4
        Period to write simulation snapshots to gsd file.
    log_write : int, default 1e3
        Period to write simulation data to the log file.
    seed : int, default 42
        Seed passed to integrator when randomizing velocities.
    restart : str, default None
        Path to gsd file from which to restart the simulation
    
    Methods
    -------
    shrink: Runs a hoomd simulation
        Shrinks the simulation volume to the target box set in
        polybinder.system.System()
    quench: Runs a hoomd simulation
        Run a simulation at a single temperature in NVT or a single
        temperature and pressure in NPT
    anneal: Runs a hoomd simulation
        Define a schedule of temperature and steps to follow over the
        course of the simulation. Can be used in NVT or NPT at a single
        pressure.
    tensile: Runs a hoomd simulation
        Use this simulation method to perform a tensile test on the 
        simulation volume. 

    """
    def __init__(
        self,
        system,
        r_cut=2.5,
        nlist="hoomd.md.nlist.Cell",
        dt=0.0003,
        auto_scale=True,
        gsd_write=1e4,
        log_write=1e3,
        seed=42,
        restart=None,
        pppm_kwargs={"Nx": 16, "Ny": 16, "Nz": 16}
    ):
        self.r_cut = r_cut
        self.ref_mass = max([atom.mass for atom in self.system.atoms])
        pair_coeffs = list(set(
            (atom.type, atom.epsilon, atom.sigma)for atom in self.system.atoms
                        )
        )
        self.ref_energy = max(pair_coeffs, key=operator.itemgetter(1))[1]
        self.ref_distance = max(pair_coeffs, key=operator.itemgetter(2))[2]
        self.target_box = system.target_box * 10 / self.ref_distance
        self.restart = restart
        self.pppm_kwargs = pppm_kwargs
        self.log_quantities = [
            "kinetic_temperature",
            "potential_energy",
            "kinetic_energy",
            "volume",
            "pressure",
            "pressure_tensor",
        ]
        self.device = hoomd.device.auto_select()
        self.sim = hoomd.Simulation(device=self.device, seed=seed)
        self.integrator = None
        if self.restart:
            self.sim.create_state_from_gsd(self.restart)
        else:
            self.init_snap, self.forcefield, refs = create_hoomd_forcefield(
                    structure=self.system,
                    r_cut=self.r_cut,
                    auto_scale=self.auto_scale,
                    pppm_kwargs=self.pppm_kwargs 
            )
            self.sim.create_state_from_snapshot(self.init_snap)

        # Default nlist is Cell, change to Tree if needed
        if isinstance(self.nlist, hoomd.md.nlist.Tree):
            exclusions = self.forcefield[0].nlist.exclusions
            self.forcefield[0].nlist = self.nlist(buffer=0.4)
            self.forcefield[0].nlist.exclusions = exclusions

        # Set up remaining hoomd objects 
        self._all = hoomd.filter.All()
        gsd_writer, table_file, = hoomd_writers(
                group=self._all, sim=self.sim, forcefield=self.forcefield
        )
        self.sim.operations.writers.append(gsd_writer)
        self.sim.operations.writers.append(table_file)

    def set_integrator(
            self,
            integrator_method,
            method_kwargs,
            integrator_kwargs={"dt": self.dt}
    ):
        # No integrator and method has been created yet
        if not self.integrator:
            self.integrator = hoomd.md.Integrator(**integrator_kwargs)
            self.integrator.forces = self.forcefield
            self.sim.operations.add(self.integrator)
            self.method = integrator_method(**method_kwargs) 
            self.sim.operations.integrator.methods = [self.method]
        # Update the existing integrator and method
        else:
            self._update_integrator_method(
                    self, integrator_method, method_kwargs
            )

    def _update_integrator_method(self, integrator_method, method_kwargs):
        self.integrator.methods.remove(self.method)
        self.method = integrator_method(**kwargs)
        self.integrator.methods.append(self.method)
        #self.sim.operations.integrator.methods.append(self.method)

    def add_box_resizer(self, n_steps, resize_period):
        resize_trigger = hoomd.trigger.Periodic(resize_period)
        box_ramp = hoomd.variant.Ramp(
                A=0, B=1, t_start=self.sim.timestep, t_ramp=int(n_steps)
        )
        initial_box = self.sim.state.box
        final_box = hoomd.Box(
                Lx=self.target_box[0],
                Ly=self.target_box[1],
                Lz=self.target_box[2]
        )
        box_resize = hoomd.update.BoxResize(
                box1=initial_box,
                box2=final_box
                variant=box_ramp,
                trigger=resize_trigger
        )
        self.sim.operations.updaters.append(box_resize)

    def shrink_box(
            self,
            n_steps,
            period,
            kT_start,
            kT_finish,
            dt,
            tau_kt
    ):
        if kT_start != kT_finish:
            kT = hoomd.variant.Ramp(
                    A=kT_init,
                    B=kT_final,
                    t_start=self.sim.timestep,
                    t_ramp=int(n_steps)
            )
        else:
            kT = kT_start
        self.set_integrator(
                integrator_method="hoomd.md.methods.NVT",
                integrator_kwargs={"dt": dt},
                method_kwargs={"tau": tau_kt, "filter": self._all, "kT": kT},
        )
        self.sim.state.thermalize_particle_momenta(
                filter=self._all, kT=kT_init
        )
        self.add_box_resizer(n_steps, period)
        self.sim.run(n_steps)
    
    def run_langevin_dynamics(
            self,
            n_steps,
            kT,
            tally_reservoir_energy=False,
            default_gamma=1.0,
            default_gamma_r=(1.0, 1.0, 1.0)
    ):
        pass

    def run_nvt(self, n_steps, kT, tau_kt):
        self.set_integrator(
                integrator_method="hoomd.md.methods.NVT",
                integrator_kwargs={"dt": dt},
                method_kwargs={"tau": tau_kt, "filter": self._all, "kT": kT},
        )
        sim.run(n_steps)

    def run_npt(
            self,
            n_steps,
            kT,
            pressure,
            tau_kt,
            tau_pressure,
            couple="xyz",
            box_dof=[True, True, True, False, False, False],
            rescale_all=False,
            gamma=0.0
    ):
        self.set_integrator(
                integrator_method="hoomd.md.methods.NPT",
                integrator_kwargs={"dt": dt},
                method_kwargs={
                    "kT": kT,
                    "S": pressure,
                    "tau": tau_kt,
                    "tauS": tau_pressure,
                    "couple": couple,
                    "box_dof": box_dof,
                    "rescale_all": rescale_all,
                    "gamma": gamma,
                    "filter": self._all, "kT": kT
                }
        )
        sim.run(n_steps)

    def temperature_ramp(
            self,
            n_steps,
            kT_start,
            kT_final,
            period,
    ):
        return hoomd.variant.Ramp(
                A=kT_init,
                B=kT_final,
                t_start=self.sim.timestep,
                t_ramp=int(n_steps)
        )










        



