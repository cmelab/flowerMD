import copy
import os
import pickle

import gsd.hoomd
import hoomd
import numpy as np
import pytest
import unyt as u

from flowermd import Simulation
from flowermd.base import Pack
from flowermd.library import OPLS_AA_PPS
from flowermd.library.forcefields import EllipsoidForcefield
from flowermd.library.polymers import EllipsoidChain
from flowermd.tests import BaseTest
from flowermd.utils import get_target_box_mass_density, set_bond_constraints


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

    def test_initialize_from_simulation_pickle(self, benzene_system):
        sim = Simulation.from_snapshot_forces(
            initial_state=benzene_system.hoomd_snapshot,
            forcefield=benzene_system.hoomd_forcefield,
            reference_values=benzene_system.reference_values,
        )
        sim.run_NVT(n_steps=1e3, kT=1.0, tau_kt=0.001)
        sim.save_simulation("simulation.pickle")
        sim.save_restart_gsd("sim.gsd")
        new_sim = Simulation.from_simulation_pickle("simulation.pickle")
        new_sim.save_restart_gsd("new_sim.gsd")
        assert new_sim.dt == sim.dt
        assert new_sim.gsd_write_freq == sim.gsd_write_freq
        assert new_sim.log_write_freq == sim.log_write_freq
        assert new_sim.seed == sim.seed
        assert (
            new_sim.maximum_write_buffer_size == sim.maximum_write_buffer_size
        )
        assert new_sim.volume_reduced == sim.volume_reduced
        assert new_sim.mass_reduced == sim.mass_reduced
        assert new_sim.reference_mass == sim.reference_mass
        assert new_sim.reference_energy == sim.reference_energy
        assert new_sim.reference_length == sim.reference_length
        with gsd.hoomd.open("sim.gsd") as sim_traj:
            with gsd.hoomd.open("new_sim.gsd") as new_sim_traj:
                assert np.array_equal(
                    sim_traj[0].particles.position,
                    new_sim_traj[0].particles.position,
                )
        new_sim.run_NVT(n_steps=2, kT=1.0, tau_kt=0.001)

    def test_initialize_from_simulation_pickle_with_walls(self, benzene_system):
        sim = Simulation.from_snapshot_forces(
            initial_state=benzene_system.hoomd_snapshot,
            forcefield=benzene_system.hoomd_forcefield,
            reference_values=benzene_system.reference_values,
        )
        sim.add_walls(wall_axis=(1, 0, 0), sigma=1, epsilon=1, r_cut=1)
        sim.save_simulation("simulation.pickle")
        new_sim = Simulation.from_simulation_pickle("simulation.pickle")
        assert len(new_sim.forces) == len(sim.forces)
        new_sim.run_NVT(n_steps=2, kT=1.0, tau_kt=0.001)

    def test_initialize_from_bad_pickle(self, benzene_system):
        sim = Simulation.from_snapshot_forces(
            initial_state=benzene_system.hoomd_snapshot,
            forcefield=benzene_system.hoomd_forcefield,
            reference_values=benzene_system.reference_values,
        )
        sim.pickle_forcefield("forces.pickle")
        with pytest.raises(ValueError):
            Simulation.from_simulation_pickle("forces.pickle")

    def test_save_forces_with_walls(self, benzene_system):
        sim = Simulation.from_snapshot_forces(
            initial_state=benzene_system.hoomd_snapshot,
            forcefield=benzene_system.hoomd_forcefield,
            reference_values=benzene_system.reference_values,
        )
        sim.add_walls(wall_axis=(1, 0, 0), sigma=1.0, epsilon=1.0, r_cut=2.0)
        assert len(sim._wall_forces[(1, 0, 0)]) == 2

        # Test without saving walls
        sim.pickle_forcefield("forces_no_walls.pickle", save_walls=False)
        found_wall_force = False
        with open("forces_no_walls.pickle", "rb") as f:
            forces = pickle.load(f)
            for force in forces:
                if isinstance(force, hoomd.md.external.wall.LJ):
                    found_wall_force = True
        assert found_wall_force is False
        # Make sure wall force is still in sim object
        assert len(sim._wall_forces[(1, 0, 0)]) == 2
        # Test with saving walls
        sim.pickle_forcefield("forces_walls.pickle", save_walls=True)
        found_wall_force = False
        with open("forces_walls.pickle", "rb") as f:
            forces = pickle.load(f)
            for force in forces:
                if isinstance(force, hoomd.md.external.wall.LJ):
                    found_wall_force = True
        assert found_wall_force is True

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
        assert np.isclose(
            float(sim.mass.value), benzene_system.mass.value, atol=1e-4
        )
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
        assert isinstance(sim.method, hoomd.md.methods.ConstantVolume)

    def test_NPT(self, benzene_system):
        sim = Simulation.from_system(benzene_system)
        sim.run_NPT(
            kT=1.0,
            n_steps=500,
            pressure=0.0001,
            tau_kt=0.001,
            tau_pressure=0.01,
        )
        assert isinstance(sim.method, hoomd.md.methods.ConstantPressure)

    def test_langevin(self, benzene_system):
        sim = Simulation.from_system(benzene_system)
        sim.run_langevin(n_steps=500, kT=1.0)
        assert isinstance(sim.method, hoomd.md.methods.Langevin)

    def test_NVE(self, benzene_system):
        sim = Simulation.from_system(benzene_system)
        sim.run_NVE(n_steps=500)
        assert isinstance(sim.method, hoomd.md.methods.ConstantVolume)

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

    def test_update_volume_no_units(self, benzene_system):
        sim = Simulation(
            initial_state=benzene_system.hoomd_snapshot,
            forcefield=benzene_system.hoomd_forcefield,
            reference_values=dict(),
        )
        init_box = sim.box_lengths_reduced
        sim.run_update_volume(
            final_box_lengths=init_box / 2,
            kT=1.0,
            tau_kt=0.01,
            n_steps=500,
            period=1,
        )
        assert np.allclose(sim.box_lengths_reduced * 2, init_box)

    def test_update_volume_density(self, benzene_system):
        sim = Simulation.from_system(benzene_system)
        target_box = get_target_box_mass_density(
            density=0.05 * u.Unit("g") / u.Unit("cm**3"), mass=sim.mass.to(u.g)
        )
        sim.run_update_volume(
            kT=1.0,
            tau_kt=0.01,
            n_steps=500,
            period=1,
            final_box_lengths=target_box,
        )
        assert np.isclose(
            sim.density.to(u.g / u.cm**3).value,
            (0.05 * (u.g / u.cm**3)).value,
            atol=1e-4,
        )

    def test_update_volume_by_density_factor(self, benzene_system):
        sim = Simulation.from_system(benzene_system)
        init_density = copy.deepcopy(sim.density)
        target_box = get_target_box_mass_density(
            density=init_density * 5, mass=sim.mass.to(u.g)
        )
        sim.run_update_volume(
            kT=1.0,
            tau_kt=0.01,
            n_steps=500,
            period=1,
            final_box_lengths=target_box,
        )
        assert np.isclose(
            sim.density.value, (init_density * 5).value, atol=1e-4
        )

    def test_change_methods(self, benzene_system):
        sim = Simulation.from_system(benzene_system)
        sim.run_NVT(kT=1.0, tau_kt=0.01, n_steps=0)
        assert isinstance(sim.method, hoomd.md.methods.ConstantVolume)
        sim.run_NPT(
            kT=1.0, tau_kt=0.01, tau_pressure=0.1, pressure=0.001, n_steps=0
        )
        assert isinstance(sim.method, hoomd.md.methods.ConstantPressure)

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

    def test_bad_constraint(self, benzene_system):
        with pytest.raises(ValueError):
            Simulation.from_system(benzene_system, constraint="A")

    def test_d_constrain_sim(self):
        ellipsoid_chain = EllipsoidChain(
            lengths=4,
            num_mols=2,
            lpar=1.0,
            bead_mass=100,
        )
        system = Pack(
            molecules=ellipsoid_chain,
            density=0.1,
            base_units=dict(),
            fix_orientation=True,
        )
        ellipsoid_ff = EllipsoidForcefield(
            lpar=1,
            lperp=0.5,
            epsilon=1.0,
            r_cut=2.5,
            angle_k=50,
            angle_theta0=2.2,
        )
        constrain_snap, d_constraint = set_bond_constraints(
            system.hoomd_snapshot, constrain_value=1.0, bond_type="_C-_H"
        )
        sim = Simulation(
            initial_state=constrain_snap,
            forcefield=ellipsoid_ff.hoomd_forces,
            constraint=d_constraint,
        )
        assert isinstance(sim._distance_constraint, hoomd.md.constrain.Distance)
        assert sim._rigid_constraint is None
        sim.run_NVT(n_steps=10, kT=1.0, tau_kt=sim.dt * 100)
        assert sim.integrator.integrate_rotational_dof is True

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
        sim.operations.writers[-2].flush()
        expected_gsd_quantities = [
            "flowermd/base/simulation/Simulation/timestep",
            "flowermd/base/simulation/Simulation/tps",
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
            "flowermdbasesimulationSimulationtimestep",
            "flowermdbasesimulationSimulationtps",
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
        os.remove("sim_data.txt")

    def test_bad_ff(self, benzene_system):
        with pytest.raises(ValueError):
            Simulation(
                initial_state=benzene_system.hoomd_snapshot, forcefield="gaff"
            )
        with pytest.raises(ValueError):
            Simulation(
                initial_state=benzene_system.hoomd_snapshot,
                forcefield=OPLS_AA_PPS,
            )
        with pytest.raises(ValueError):
            Simulation(
                initial_state=benzene_system.hoomd_snapshot,
                forcefield=[1, 2, 3],
            )

    def test_flush(self, benzene_system):
        sim = Simulation.from_system(benzene_system, gsd_write_freq=100)
        sim.run_NVT(kT=1.0, tau_kt=0.01, n_steps=500, write_at_start=False)
        sim.flush_writers()
        with gsd.hoomd.open("trajectory.gsd") as traj:
            assert len(traj) > 0
        os.remove("trajectory.gsd")
