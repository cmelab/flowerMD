import os
import pickle

import hoomd
import numpy as np

from hoomd_organics import Simulation
from hoomd_organics.tests import BaseTest


class TestSimulate(BaseTest):
    def test_initialize_from_snap(self, benzene_system):
        Simulation(
            initial_state=benzene_system.hoomd_snapshot,
            forcefield=benzene_system.hoomd_forcefield,
        )

    def test_no_reference_values(self, benzene_system):
        sim = Simulation(
            initial_state=benzene_system.hoomd_snapshot,
            forcefield=benzene_system.hoomd_forcefield,
        )
        assert np.array_equal(sim.box_lengths_reduced, sim.box_lengths)
        assert sim.density_reduced == sim.density
        assert sim.volume_reduced == sim.volume
        assert sim.mass_reduced == sim.mass

    def test_reference_values(self, benzene_system):
        sim = Simulation(
            initial_state=benzene_system.hoomd_snapshot,
            forcefield=benzene_system.hoomd_forcefield,
        )
        sim.reference_values = benzene_system.reference_values
        assert np.isclose(float(sim.mass.value), benzene_system.mass, atol=1e-4)
        assert np.allclose(benzene_system.box.lengths, sim.box_lengths.value)

    def test_NVT(self, benzene_system):
        sim = Simulation(
            initial_state=benzene_system.hoomd_snapshot,
            forcefield=benzene_system.hoomd_forcefield,
        )
        sim.run_NVT(kT=1.0, tau_kt=0.01, n_steps=500)
        assert isinstance(sim.method, hoomd.md.methods.NVT)

    def test_NPT(self, benzene_system):
        sim = Simulation(
            initial_state=benzene_system.hoomd_snapshot,
            forcefield=benzene_system.hoomd_forcefield,
        )
        sim.run_NPT(
            kT=1.0,
            n_steps=500,
            pressure=0.0001,
            tau_kt=0.001,
            tau_pressure=0.01,
        )
        assert isinstance(sim.method, hoomd.md.methods.NPT)

    def test_langevin(self, benzene_system):
        sim = Simulation(
            initial_state=benzene_system.hoomd_snapshot,
            forcefield=benzene_system.hoomd_forcefield,
        )
        sim.run_langevin(n_steps=500, kT=1.0, alpha=0.5)
        assert isinstance(sim.method, hoomd.md.methods.Langevin)

    def test_NVE(self, benzene_system):
        sim = Simulation(
            initial_state=benzene_system.hoomd_snapshot,
            forcefield=benzene_system.hoomd_forcefield,
        )
        sim.run_NVE(n_steps=500)
        assert isinstance(sim.method, hoomd.md.methods.NVE)

    def test_displacement_cap(self, benzene_system):
        sim = Simulation(
            initial_state=benzene_system.hoomd_snapshot,
            forcefield=benzene_system.hoomd_forcefield,
        )
        sim.run_displacement_cap(n_steps=500, maximum_displacement=1e-4)
        assert isinstance(sim.method, hoomd.md.methods.DisplacementCapped)

    def test_update_volume(self, benzene_system):
        sim = Simulation(
            initial_state=benzene_system.hoomd_snapshot,
            forcefield=benzene_system.hoomd_forcefield,
        )
        sim.run_update_volume(
            kT=1.0,
            tau_kt=0.01,
            n_steps=500,
            period=1,
            final_box_lengths=sim.box_lengths * 0.5,
        )

    def test_update_volume_walls(self, benzene_system):
        sim = Simulation(
            initial_state=benzene_system.hoomd_snapshot,
            forcefield=benzene_system.hoomd_forcefield,
        )
        sim.add_walls(wall_axis=(1, 0, 0), sigma=1.0, epsilon=1.0, r_cut=1.12)
        sim.run_update_volume(
            kT=1.0,
            tau_kt=0.01,
            n_steps=500,
            period=5,
            final_box_lengths=sim.box_lengths * 0.5,
        )

    def test_change_methods(self, benzene_system):
        sim = Simulation(
            initial_state=benzene_system.hoomd_snapshot,
            forcefield=benzene_system.hoomd_forcefield,
        )
        sim.run_NVT(kT=1.0, tau_kt=0.01, n_steps=0)
        assert isinstance(sim.method, hoomd.md.methods.NVT)
        sim.run_NPT(
            kT=1.0, tau_kt=0.01, tau_pressure=0.1, pressure=0.001, n_steps=0
        )
        assert isinstance(sim.method, hoomd.md.methods.NPT)

    def test_change_dt(self, benzene_system):
        sim = Simulation(
            initial_state=benzene_system.hoomd_snapshot,
            forcefield=benzene_system.hoomd_forcefield,
        )
        sim.run_NVT(kT=1.0, tau_kt=0.01, n_steps=0)
        sim.dt = 0.003
        sim.run_NVT(kT=1.0, tau_kt=0.01, n_steps=0)
        assert sim.dt == 0.003

    def test_scale_epsilon(self, benzene_system):
        sim = Simulation(
            initial_state=benzene_system.hoomd_snapshot,
            forcefield=benzene_system.hoomd_forcefield,
        )
        epsilons = []
        for param in sim._lj_force().params:
            epsilons.append(sim._lj_force().params[param]["epsilon"])
        sim.adjust_epsilon(scale_by=0.5)
        epsilons_scaled = []
        for param in sim._lj_force().params:
            epsilons_scaled.append(sim._lj_force().params[param]["epsilon"])
        for i, j in zip(epsilons, epsilons_scaled):
            assert np.allclose(i * 0.5, j, atol=1e-3)

    def test_shift_epsilon(self, benzene_system):
        sim = Simulation(
            initial_state=benzene_system.hoomd_snapshot,
            forcefield=benzene_system.hoomd_forcefield,
        )
        epsilons = []
        for param in sim._lj_force().params:
            epsilons.append(sim._lj_force().params[param]["epsilon"])
        sim.adjust_epsilon(shift_by=1.0)
        epsilons_scaled = []
        for param in sim._lj_force().params:
            epsilons_scaled.append(sim._lj_force().params[param]["epsilon"])
        for i, j in zip(epsilons, epsilons_scaled):
            assert np.allclose(i + 1, j, atol=1e-3)

    def test_scale_sigma(self, benzene_system):
        sim = Simulation(
            initial_state=benzene_system.hoomd_snapshot,
            forcefield=benzene_system.hoomd_forcefield,
        )
        sigmas = []
        for param in sim._lj_force().params:
            sigmas.append(sim._lj_force().params[param]["sigma"])
        sim.adjust_sigma(scale_by=0.5)
        sigmas_scaled = []
        for param in sim._lj_force().params:
            sigmas_scaled.append(sim._lj_force().params[param]["sigma"])
        for i, j in zip(sigmas, sigmas_scaled):
            assert np.allclose(i * 0.5, j, atol=1e-3)

    def test_shift_sigma(self, benzene_system):
        sim = Simulation(
            initial_state=benzene_system.hoomd_snapshot,
            forcefield=benzene_system.hoomd_forcefield,
        )
        sigmas = []
        for param in sim._lj_force().params:
            sigmas.append(sim._lj_force().params[param]["sigma"])
        sim.adjust_sigma(shift_by=1.0)
        sigmas_scaled = []
        for param in sim._lj_force().params:
            sigmas_scaled.append(sim._lj_force().params[param]["sigma"])
        for i, j in zip(sigmas, sigmas_scaled):
            assert np.allclose(i + 1, j, atol=1e-3)

    def test_remove_force(self, benzene_system):
        sim = Simulation(
            initial_state=benzene_system.hoomd_snapshot,
            forcefield=benzene_system.hoomd_forcefield,
        )
        sim.remove_force(sim._lj_force())
        for i in sim.forces:
            assert not isinstance(i, hoomd.md.pair.LJ)

    def test_set_integrate_group(self, benzene_system):
        sim = Simulation(
            initial_state=benzene_system.hoomd_snapshot,
            forcefield=benzene_system.hoomd_forcefield,
        )
        assert isinstance(sim.integrate_group, hoomd.filter.All)
        tag_filter = hoomd.filter.Tags([0, 1, 2, 3])
        sim.integrate_group = tag_filter
        assert not isinstance(sim.integrate_group, hoomd.filter.All)
        sim.run_NVT(n_steps=200, kT=1.0, tau_kt=0.01)

    def test_pickle_ff(self, benzene_system):
        sim = Simulation(
            initial_state=benzene_system.hoomd_snapshot,
            forcefield=benzene_system.hoomd_forcefield,
        )
        sim.pickle_forcefield("forcefield.pickle")
        assert os.path.isfile("forcefield.pickle")
        f = open("forcefield.pickle", "rb")
        hoomd_ff = pickle.load(f)

        for i, j in zip(sim.forces, hoomd_ff):
            assert type(i) is type(j)
        os.remove("forcefield.pickle")

    def test_save_restart_gsd(self, benzene_system):
        sim = Simulation(
            initial_state=benzene_system.hoomd_snapshot,
            forcefield=benzene_system.hoomd_forcefield,
        )
        sim.save_restart_gsd("restart.gsd")
        assert os.path.isfile("restart.gsd")
        sim.pickle_forcefield("forcefield.pickle")
        f = open("forcefield.pickle", "rb")
        hoomd_ff = pickle.load(f)
        Simulation(initial_state="restart.gsd", forcefield=hoomd_ff)
        os.remove("forcefield.pickle")
        os.remove("restart.gsd")
