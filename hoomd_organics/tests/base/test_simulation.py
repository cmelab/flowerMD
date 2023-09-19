import copy
import os
import pickle

import gsd.hoomd
import hoomd
import numpy as np
import pytest
import unyt as u

from hoomd_organics import Simulation
from hoomd_organics.tests import BaseTest


class TestSimulate(BaseTest):
    def test_initialize_from_system(self, benzene_system):
        sim = Simulation.from_system(benzene_system)
        sim.run_NVT(kT=1.0, tau_kt=0.01, n_steps=500)
        assert len(sim.forces) == len(benzene_system.hoomd_forcefield)
        assert sim.reference_values == benzene_system.reference_values

    def test_initialize_from_system_separate_ff(
        self, benzene_cg_system, cg_single_bead_ff
    ):
        sim = Simulation.from_system(
            benzene_cg_system, forcefield=cg_single_bead_ff
        )
        sim.run_NVT(kT=0.1, tau_kt=10, n_steps=500)

    def test_initialize_from_system_missing_ff(self, benzene_cg_system):
        with pytest.raises(ValueError):
            Simulation.from_system(benzene_cg_system)

    def test_initialize_from_state(self, benzene_system):
        Simulation.from_snapshot_forces(
            initial_state=benzene_system.hoomd_snapshot,
            forcefield=benzene_system.hoomd_forcefield,
            reference_values=benzene_system.reference_values,
        )

    def test_no_reference_values(self, benzene_system):
        sim = Simulation.from_snapshot_forces(
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
            reference_values=benzene_system.reference_values,
        )
        assert np.isclose(float(sim.mass.value), benzene_system.mass, atol=1e-4)
        assert np.allclose(benzene_system.box.lengths, sim.box_lengths.value)

    def test_set_ref_values(self, benzene_system):
        sim = Simulation(
            initial_state=benzene_system.hoomd_snapshot,
            forcefield=benzene_system.hoomd_forcefield,
        )
        ref_value_dict = {
            "length": 1 * u.angstrom,
            "energy": 3.0 * u.kcal / u.mol,
            "mass": 1.25 * u.Unit("amu"),
        }
        sim.reference_values = ref_value_dict
        assert sim.reference_length == ref_value_dict["length"]
        assert sim.reference_energy == ref_value_dict["energy"]
        assert sim.reference_mass == ref_value_dict["mass"]

    def test_set_ref_length(self, benzene_system):
        sim = Simulation(
            initial_state=benzene_system.hoomd_snapshot,
            forcefield=benzene_system.hoomd_forcefield,
        )
        sim.reference_length = 1 * u.angstrom
        assert sim.reference_length == 1 * u.angstrom

    def test_set_ref_energy(self, benzene_system):
        sim = Simulation(
            initial_state=benzene_system.hoomd_snapshot,
            forcefield=benzene_system.hoomd_forcefield,
        )
        sim.reference_energy = 3.0 * u.kcal / u.mol
        assert sim.reference_energy == 3.0 * u.kcal / u.mol

    def test_set_ref_mass(self, benzene_system):
        sim = Simulation(
            initial_state=benzene_system.hoomd_snapshot,
            forcefield=benzene_system.hoomd_forcefield,
        )
        sim.reference_mass = 1.25 * u.amu
        assert sim.reference_mass == 1.25 * u.amu

    def test_NVT(self, benzene_system):
        sim = Simulation.from_system(benzene_system)
        sim.run_NVT(kT=1.0, tau_kt=0.01, n_steps=500)
        assert isinstance(sim.method, hoomd.md.methods.NVT)

    def test_NPT(self, benzene_system):
        sim = Simulation.from_system(benzene_system)
        sim.run_NPT(
            kT=1.0,
            n_steps=500,
            pressure=0.0001,
            tau_kt=0.001,
            tau_pressure=0.01,
        )
        assert isinstance(sim.method, hoomd.md.methods.NPT)

    def test_langevin(self, benzene_system):
        sim = Simulation.from_system(benzene_system)
        sim.run_langevin(n_steps=500, kT=1.0, alpha=0.5)
        assert isinstance(sim.method, hoomd.md.methods.Langevin)

    def test_NVE(self, benzene_system):
        sim = Simulation.from_system(benzene_system)
        sim.run_NVE(n_steps=500)
        assert isinstance(sim.method, hoomd.md.methods.NVE)

    def test_displacement_cap(self, benzene_system):
        sim = Simulation.from_system(benzene_system)
        sim.run_displacement_cap(n_steps=500, maximum_displacement=1e-4)
        assert isinstance(sim.method, hoomd.md.methods.DisplacementCapped)

    def test_update_volume_target_box(self, benzene_system):
        sim = Simulation.from_system(benzene_system)
        sim.run_update_volume(
            kT=1.0,
            tau_kt=0.01,
            n_steps=500,
            period=1,
            final_box_lengths=sim.box_lengths_reduced * 0.5,
        )

    def test_update_volume_walls(self, benzene_system):
        sim = Simulation.from_system(benzene_system)
        sim.add_walls(wall_axis=(1, 0, 0), sigma=1.0, epsilon=1.0, r_cut=1.12)
        sim.run_update_volume(
            kT=1.0,
            tau_kt=0.01,
            n_steps=500,
            period=5,
            final_box_lengths=sim.box_lengths_reduced * 0.5,
        )

    def test_update_volume_density(self, benzene_system):
        sim = Simulation.from_system(benzene_system)
        sim.run_update_volume(
            kT=1.0, tau_kt=0.01, n_steps=500, period=1, final_density=0.1
        )
        assert np.isclose(
            sim.density.to(u.g / u.cm**3).value,
            (0.1 * (u.g / u.cm**3)).value,
            atol=1e-4,
        )

    def test_update_volume_by_density_factor(self, benzene_system):
        sim = Simulation.from_system(benzene_system)
        init_density = copy.deepcopy(sim.density)
        sim.run_update_volume(
            kT=1.0,
            tau_kt=0.01,
            n_steps=500,
            period=1,
            final_density=sim.density * 5,
        )
        assert np.isclose(
            sim.density.value, (init_density * 5).value, atol=1e-4
        )

    def test_update_volume_missing_values(self, benzene_system):
        sim = Simulation.from_system(benzene_system)
        with pytest.raises(ValueError):
            sim.run_update_volume(kT=1.0, tau_kt=0.01, n_steps=500, period=1)

    def test_update_volume_two_values(self, benzene_system):
        sim = Simulation.from_system(benzene_system)
        with pytest.raises(ValueError):
            sim.run_update_volume(
                kT=1.0,
                tau_kt=0.01,
                n_steps=500,
                period=1,
                final_box_lengths=sim.box_lengths_reduced * 0.5,
                final_density=0.1,
            )

    # def test_update_volume_with_density_no_ref_values(self, benzene_system):
    #     sim_no_ref = Simulation(
    #         initial_state=benzene_system.hoomd_snapshot,
    #         forcefield=benzene_system.hoomd_forcefield,
    #     )
    #     with pytest.raises(ReferenceUnitError):
    #         sim_no_ref.run_update_volume(
    #             kT=1.0,
    #             tau_kt=0.01,
    #             n_steps=500,
    #             period=1,
    #             final_density=0.1,
    #         )

    def test_change_methods(self, benzene_system):
        sim = Simulation.from_system(benzene_system)
        sim.run_NVT(kT=1.0, tau_kt=0.01, n_steps=0)
        assert isinstance(sim.method, hoomd.md.methods.NVT)
        sim.run_NPT(
            kT=1.0, tau_kt=0.01, tau_pressure=0.1, pressure=0.001, n_steps=0
        )
        assert isinstance(sim.method, hoomd.md.methods.NPT)

    def test_change_dt(self, benzene_system):
        sim = Simulation.from_system(benzene_system)
        sim.run_NVT(kT=1.0, tau_kt=0.01, n_steps=0)
        sim.dt = 0.003
        sim.run_NVT(kT=1.0, tau_kt=0.01, n_steps=0)
        assert sim.dt == 0.003

    def test_scale_epsilon(self, benzene_system):
        sim = Simulation.from_system(benzene_system)
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
        sim = Simulation.from_system(benzene_system)
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
        sim = Simulation.from_system(benzene_system)
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
        sim = Simulation.from_system(benzene_system)
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
        sim = Simulation.from_system(benzene_system)
        sim.remove_force(sim._lj_force())
        for i in sim.forces:
            assert not isinstance(i, hoomd.md.pair.LJ)

    def test_set_integrate_group(self, benzene_system):
        sim = Simulation.from_system(benzene_system)
        assert isinstance(sim.integrate_group, hoomd.filter.All)
        tag_filter = hoomd.filter.Tags([0, 1, 2, 3])
        sim.integrate_group = tag_filter
        assert not isinstance(sim.integrate_group, hoomd.filter.All)
        sim.run_NVT(n_steps=200, kT=1.0, tau_kt=0.01)

    def test_pickle_ff(self, benzene_system):
        sim = Simulation.from_system(benzene_system)
        sim.pickle_forcefield("forcefield.pickle")
        assert os.path.isfile("forcefield.pickle")
        f = open("forcefield.pickle", "rb")
        hoomd_ff = pickle.load(f)

        for i, j in zip(sim.forces, hoomd_ff):
            assert type(i) is type(j)
        os.remove("forcefield.pickle")

    def test_save_restart_gsd(self, benzene_system):
        sim = Simulation.from_system(benzene_system)
        sim.save_restart_gsd("restart.gsd")
        assert os.path.isfile("restart.gsd")
        sim.pickle_forcefield("forcefield.pickle")
        f = open("forcefield.pickle", "rb")
        hoomd_ff = pickle.load(f)
        Simulation.from_snapshot_forces(
            initial_state="restart.gsd", forcefield=hoomd_ff
        )
        os.remove("forcefield.pickle")
        os.remove("restart.gsd")

    def test_gsd_logger(self, benzene_system):
        sim = Simulation.from_system(benzene_system, gsd_write_freq=1)
        sim.run_NVT(n_steps=5, kT=1.0, tau_kt=0.001)
        expected_gsd_quantities = [
            "hoomd_organics/base/simulation/Simulation/timestep",
            "hoomd_organics/base/simulation/Simulation/tps",
            "md/compute/ThermodynamicQuantities/kinetic_temperature",
            "md/compute/ThermodynamicQuantities/potential_energy",
            "md/compute/ThermodynamicQuantities/kinetic_energy",
            "md/compute/ThermodynamicQuantities/volume",
            "md/compute/ThermodynamicQuantities/pressure",
            "md/compute/ThermodynamicQuantities/pressure_tensor",
            "md/pair/Ewald/energy",
            "md/pair/LJ/energy",
            "md/long_range/pppm/Coulomb/energy",
            "md/special_pair/Coulomb/energy",
            "md/special_pair/LJ/energy",
            "md/bond/Harmonic/energy",
            "md/angle/Harmonic/energy",
            "md/dihedral/OPLS/energy",
        ]

        with gsd.hoomd.open("trajectory.gsd") as traj:
            snap = traj[-1]
            log_keys = list(snap.log.keys())

        assert sorted(expected_gsd_quantities) == sorted(log_keys)
        expected_table_quantities = [
            "hoomd_organicsbasesimulationSimulationtimestep",
            "hoomd_organicsbasesimulationSimulationtps",
            "mdcomputeThermodynamicQuantitieskinetic_temperature",
            "mdcomputeThermodynamicQuantitiespotential_energy",
            "mdcomputeThermodynamicQuantitieskinetic_energy",
            "mdcomputeThermodynamicQuantitiesvolume",
            "mdcomputeThermodynamicQuantitiespressure",
            "mdpairEwaldenergy",
            "mdpairLJenergy",
            "mdlong_rangepppmCoulombenergy",
            "mdspecial_pairCoulombenergy",
            "mdspecial_pairLJenergy",
            "mdbondHarmonicenergy",
            "mdangleHarmonicenergy",
            "mddihedralOPLSenergy",
        ]
        table = np.genfromtxt("sim_data.txt", names=True)
        table_keys = list(table.dtype.fields.keys())
        assert sorted(expected_table_quantities) == sorted(table_keys)

        os.remove("trajectory.gsd")
