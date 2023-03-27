from base_test import BaseTest
import pytest

import gsd.hoomd
import hoomd
import numpy as np

from hoomd_polymers.sim import Simulation


class TestSimulate(BaseTest):
    def test_initialize_from_snap(self, polyethylene_system):
        sim = Simulation(
                initial_state=polyethylene_system.hoomd_snapshot,
                forcefield=polyethylene_system.hoomd_forcefield
        )

    def test_no_reference_values(self, polyethylene_system):
        sim = Simulation(
                initial_state=polyethylene_system.hoomd_snapshot,
                forcefield=polyethylene_system.hoomd_forcefield
        )
        assert np.array_equal(sim.box_lengths_reduced, sim.box_lengths)
        assert sim.density_reduced == sim.density
        assert sim.volume_reduced == sim.volume
        assert sim.mass_reduced == sim.mass

    def test_reference_values(self, polyethylene_system):
        sim = Simulation(
                initial_state=polyethylene_system.hoomd_snapshot,
                forcefield=polyethylene_system.hoomd_forcefield
        )
        sim.reference_mass = 2
        sim.reference_distance = 2
        assert np.array_equal(2*sim.box_lengths_reduced, sim.box_lengths)
        assert 2*sim.mass_reduced == sim.mass
        assert sim.volume_reduced*8 == sim.volume
        assert sim.density_reduced == sim.density * 4

    def test_NVT(self, polyethylene_system):
        sim = Simulation(
                initial_state=polyethylene_system.hoomd_snapshot,
                forcefield=polyethylene_system.hoomd_forcefield
        )
        sim.run_NVT(kT=1.0, tau_kt=0.01, n_steps=500)

    def test_NPT(self, polyethylene_system):
        sim = Simulation(
                initial_state=polyethylene_system.hoomd_snapshot,
                forcefield=polyethylene_system.hoomd_forcefield
        )
        sim.run_NPT(
                kT=1.0,
                n_steps=500,
                pressure=0.0001,
                tau_kt=0.001,
                tau_pressure=0.01
        )

    def test_langevin(self, polyethylene_system):
        sim = Simulation(
                initial_state=polyethylene_system.hoomd_snapshot,
                forcefield=polyethylene_system.hoomd_forcefield
        )
        sim.run_langevin(n_steps=500, kT=1.0, alpha=0.5)

    def test_NVE(self, polyethylene_system):
        sim = Simulation(
                initial_state=polyethylene_system.hoomd_snapshot,
                forcefield=polyethylene_system.hoomd_forcefield
        ) 
        sim.run_NVE(n_steps=500)


