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
        tau_kt=0.1,
        tau_p=None,
        nlist="Cell",
        dt=0.0003,
        auto_scale=True,
        gsd_write=1e4,
        log_write=1e3,
        seed=42,
        restart=None,
    ):
        self.ref_mass = max([atom.mass for atom in self.system.atoms])
        pair_coeffs = list(set(
            (atom.type, atom.epsilon, atom.sigma)for atom in self.system.atoms
                        )
        )
        self.ref_energy = max(pair_coeffs, key=operator.itemgetter(1))[1]
        self.ref_distance = max(pair_coeffs, key=operator.itemgetter(2))[2]
        self.target_box = system.target_box * 10 / self.ref_distance
        self.log_quantities = [
            "kinetic_temperature",
            "potential_energy",
            "kinetic_energy",
            "volume",
            "pressure",
            "pressure_tensor",
        ]
        self.device = hoomd.device.auto_select()
        self.sim = hoomd.Simulation(device=self.device, seed=self.seed)
        if self.restart:
            self.sim.create_state_from_gsd(self.restart)
        else:
            self.init_snap, self.forcefield, refs = create_hoomd_forcefield(
                    structure=self.system,
                    r_cut=self.r_cut,
                    auto_scale=self.auto_scale,
                    pppm_kwargs = {"Nx": 16, "Ny": 16, "Nz": 16}
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
        self.integrator = hoomd.md.Integrator(**integrator_kwargs)
        self.method = integrator_method(**integrator_kwargs) 

    def update_integrator(self, integrator_method, integrator_kwargs):
        pass





