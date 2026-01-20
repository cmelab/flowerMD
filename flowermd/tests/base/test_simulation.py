import copy
import os
import pickle
from enum import Enum

import gsd.hoomd
import hoomd
import mbuild
import numpy as np
import pytest
import unyt as u
from numpy.typing import NDArray

from flowermd import Simulation
from flowermd.base import Pack
from flowermd.library import OPLS_AA_PPS
from flowermd.library.forcefields import BeadSpring, EllipsoidForcefield
from flowermd.library.polymers import EllipsoidChain, LJChain
from flowermd.tests import BaseTest
from flowermd.utils import (
    create_rigid_ellipsoid_chain,
    get_target_box_mass_density,
    set_bond_constraints,
)


class TestSimulate(BaseTest):
    def test_initialize_from_system(self, benzene_system):
        sim = Simulation.from_system(benzene_system)
        sim.run_NVT(kT=1.0, tau_kt=0.01, n_steps=500)
        assert len(sim.forces) == len(benzene_system.hoomd_forcefield)
        assert sim.reference_values == benzene_system.reference_values

    def test_update_nlist(self, benzene_system):
        sim = Simulation.from_system(benzene_system)
        tree = hoomd.md.nlist.Tree(buffer=0.40)
        sim.nlist = tree
        sim.run_NVT(kT=1.0, tau_kt=0.01, n_steps=100)
        assert isinstance(sim.nlist[0], hoomd.md.nlist.Tree)

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
        for force in sim._pair_force():
            try:
                force.params["epsilon"]
            except KeyError:
                continue
            for param in force.params:
                epsilons.append(sim._pair_force().params[param]["epsilon"])
            sim.adjust_epsilon(scale_by=0.5)

        epsilons_scaled = []

        for force in sim._pair_force():
            try:
                force.params["epsilon"]
            except KeyError:
                continue
            for param in force.params:
                epsilons_scaled.append(sim._pair_force().params[param]["epsilon"])
        for i, j in zip(epsilons, epsilons_scaled):
            assert np.allclose(i * 0.5, j, atol=1e-3)

    def test_shift_epsilon(self, benzene_system):
        sim = Simulation.from_system(benzene_system)
        epsilons = []
        for force in sim._pair_force():
            try:
                force.params["epsilon"]
            except KeyError:
                continue
            for param in force.params:
                epsilons.append(force.params[param]["epsilon"])

        sim.adjust_epsilon(shift_by=1.0)
        epsilons_scaled = []
        for force in sim._pair_force():
            try:
                force.params["epsilon"]
            except KeyError:
                continue
            for param in force.params:
                epsilons_scaled.append(force.params[param]["epsilon"])
        for i, j in zip(epsilons, epsilons_scaled):
            assert np.allclose(i + 1, j, atol=1e-3)

    def test_scale_sigma(self, benzene_system):
        sim = Simulation.from_system(benzene_system)
        sigmas = []
        for force in sim._pair_force():
            try:
                force.params["sigma"]
            except KeyError:
                continue
            for param in force.params:
                sigmas.append(force.params[param]["sigma"])

        sim.adjust_sigma(scale_by=0.5)
        sigmas_scaled = []
        for force in sim._pair_force():
            try:
                force.params["sigma"]
            except KeyError:
                continue
            for param in force.params:
                sigmas_scaled.append(force.params[param]["sigma"])
        for i, j in zip(sigmas, sigmas_scaled):
            assert np.allclose(i * 0.5, j, atol=1e-3)

    def test_shift_sigma(self, benzene_system):
        sim = Simulation.from_system(benzene_system)
        sigmas = []
        for force in sim._pair_force():
            try:
                force.params["sigma"]
            except KeyError:
                continue
            for param in force.params:
                sigmas.append(force.params[param]["sigma"])

        sim.adjust_sigma(shift_by=1.0)
        sigmas_scaled = []
        for force in sim._pair_force():
            try:
                force.params["sigma"]
            except KeyError:
                continue
            for param in force.params:
                sigmas_scaled.append(force.params[param]["sigma"])
        for i, j in zip(sigmas, sigmas_scaled):
            assert np.allclose(i + 1, j, atol=1e-3)

    def test_remove_force(self, benzene_system):
        sim = Simulation.from_system(benzene_system)
        lj_force = [i for i in sim._pair_force() if isinstance(i, hoomd.md.pair.LJ)]
        sim.remove_force(lj_force[0])
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
        chains = LJChain(lengths=10, num_mols=1)
        system = Pack(molecules=chains, density=0.001, base_units=dict())
        snap, d = set_bond_constraints(
            snapshot=system.hoomd_snapshot,
            bond_types=["A-A"],
            constraint_values=[1.0],
        )
        ff = BeadSpring(
            beads={"A": dict(epsilon=1.0, sigma=1.0)},
            angles={"A-A-A": dict(t0=2.2, k=100)},
            r_cut=2.5,
        )
        sim = Simulation(
            initial_state=snap, forcefield=ff.hoomd_forces, constraint=d
        )
        assert isinstance(sim._distance_constraint, hoomd.md.constrain.Distance)
        assert sim._rigid_constraint is None
        sim.run_NVT(n_steps=10, kT=1.0, tau_kt=sim.dt * 100)
        assert sim.integrator.integrate_rotational_dof is True

    def test_ellipsoid_chain_sim(self):
        LPAR = 1.0
        LPERP = 0.5

        chain = EllipsoidChain(
            lengths=15, num_mols=15, bead_mass=1, lpar=LPAR, bond_L=0.0
        )
        system = Pack(
            molecules=chain,
            density=0.0005,
            edge=5,
            overlap=1,
            fix_orientation=True,
        )
        rigid_snap, rigid = create_rigid_ellipsoid_chain(
            system.hoomd_snapshot, LPAR, LPERP
        )
        forces = EllipsoidForcefield(
            angle_k=25,
            angle_theta0=2.2,
            bond_r0=0.0,
            lpar=LPAR,
            lperp=LPERP,
            epsilon=1,
            r_cut=3.0,
        )
        sim = Simulation(
            initial_state=rigid_snap,
            constraint=rigid,
            gsd_file_name="traj.gsd",
            gsd_write_freq=100,
            forcefield=forces.hoomd_forces,
        )
        assert isinstance(sim._rigid_constraint, hoomd.md.constrain.Rigid)
        assert sim._distance_constraint is None
        sim.run_NVT(n_steps=10, kT=1.0, tau_kt=sim.dt * 100)
        assert sim.integrator.integrate_rotational_dof is True

    # verify that different orientations have different PE values for the same
    # center-to-center distance.
    # e.g. two ellipsoids that look like ()() should not have the same PE as two
    # ellipsoids that look like ()<> given the same center-to-center distance
    def test_ellipsoid_chain_orientations(self):
        class Axis(Enum):
            X = 0
            Y = 1
            Z = 2

        def translate_ellipsoid_by(
            ellipsoid: mbuild.compound.Compound,
            translation: [float, float, float],
        ) -> mbuild.compound.Compound:
            """
            Translate an ellipsoid by the given translation

            Parameters
            ----------
            ellipsoid: mbuild.compound.Compound, The compound representing the
            ellipsoid body. See how this parameter is passed in in the example

            translation: [float, float, float], How the ellipsoid should be
            translated, where the array is formatted as [dx, dy, dz]

            Example
            -------
            ellipsoids = EllipsoidChain(num_mols=2, lpar=1.0, bead_mass=1.0, lengths=1)
            system = Pack(density=0.01*u.Unit("nm**-3"), molecules=ellipsoids)
            translate_ellipsoid_by(system.system.children[0].children[0], [1.0, 0.2, 0])

            """

            for child in ellipsoid.children:
                child.pos += translation
            return ellipsoid

        def rotation_matrix_x(val):
            return np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(val), -np.sin(val)],
                    [0, np.sin(val), np.cos(val)],
                ]
            )

        def rotation_matrix_y(val):
            return np.array(
                [
                    [np.cos(val), 0, np.sin(val)],
                    [0, 1, 0],
                    [-np.sin(val), 0, np.cos(val)],
                ]
            )

        def rotation_matrix_z(val):
            return np.array(
                [
                    [np.cos(val), -np.sin(val), 0],
                    [np.sin(val), np.cos(val), 0],
                    [0, 0, 1],
                ]
            )

        def rotate_ellipsoid_by(
            ellipsoid: mbuild.compound.Compound, rotation: NDArray[np.float64]
        ) -> mbuild.compound.Compound:
            """
            Rotate an ellipsoid by the given rotation about its center, where
            the rotation is in quaternion form

            Parameters
            ----------
            ellipsoid: mbuild.compound.Compound, The compound representing the
            ellipsoid body. See how this parameter is passed in in the example

            rotation: NDArray[np.float64], shape: (4,), The quaternion used to
            rotate the ellipsoid about its center

            Example
            -------
            ellipsoids = EllipsoidChain(num_mols=2, lpar=1.0, bead_mass=1.0, lengths=1)
            system = Pack(density=0.01*u.Unit("nm**-3"), molecules=ellipsoids)
            rotate_ellipsoid_by(
                system.system.children[0].children[0],
                euler_to_quaternion(np.pi/2, 0, 0)
            )
            """

            center_particle = ellipsoid.children[0]
            for child in ellipsoid.children:
                child_rel_to_center = child.pos - center_particle.pos

                child_rel_to_center = rotation_matrix_x(rotation[0]).dot(
                    child_rel_to_center
                )
                child_rel_to_center = rotation_matrix_y(rotation[1]).dot(
                    child_rel_to_center
                )
                child_rel_to_center = rotation_matrix_z(rotation[2]).dot(
                    child_rel_to_center
                )

                child.pos = center_particle.pos + child_rel_to_center
            return ellipsoid

        def ellipsoid_to_origin(ellipsoid):
            """
            Move an ellipsoid such that it's center (X particle)
            is at (0, 0, 0), and is parallel with the Z-axis

            Parameters
            ----------
            ellipsoid: mbuild.compound.Compound, The compound representing the
            ellipsoid body. See how this parameter is passed in in the example

            Example
            -------
            ellipsoids = EllipsoidChain(num_mols=2, lpar=1.0, bead_mass=1.0, lengths=1)
            system = Pack(density=0.01*u.Unit("nm**-3"), molecules=ellipsoids)
            ellipsoid_to_origin(system.system.children[0].children[0])
            """
            center = ellipsoid.children[0].pos
            bond = ellipsoid.children[1].pos
            head = ellipsoid.children[2].pos

            lpar = np.linalg.norm(head - center)
            bond_from_center = np.linalg.norm(bond - center)

            ellipsoid.children[0].pos = np.array([0.0, 0.0, 0.0])
            ellipsoid.children[1].pos = np.array([0.0, 0.0, bond_from_center])
            ellipsoid.children[2].pos = np.array([0.0, 0.0, lpar])
            ellipsoid.children[3].pos = np.array([0.0, 0.0, -lpar])

            return ellipsoid

        # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Source_code
        def euler_to_quaternion(
            roll: float, pitch: float, yaw: float
        ) -> NDArray[np.float64]:
            """
            Convert the euler angle represented by the given roll, pitch, and
            yaw to a unit quaternion that represents the same rotation in 3D
            space
            """
            cr = np.cos(roll * 0.5)
            sr = np.sin(roll * 0.5)
            cp = np.cos(pitch * 0.5)
            sp = np.sin(pitch * 0.5)
            cy = np.cos(yaw * 0.5)
            sy = np.sin(yaw * 0.5)

            q = np.zeros(4)
            q[0] = cr * cp * cy + sr * sp * sy
            q[1] = sr * cp * cy - cr * sp * sy
            q[2] = cr * sp * cy + sr * cp * sy
            q[3] = cr * cp * sy - sr * sp * cy

            return q

        def run_sim(
            axis: Axis,
            dist: float,
            rotation: np.ndarray[
                np.float64
            ] = None,  # quaternion representing rotation of the second ellipsoid
        ) -> float:
            """
            Do a simulation of two ellipsoids, where the first is located at (0,
            0, 0) and parallel to the z-axis, and the other is placed dist away
            from the first along the specified axis, and rotated around its
            center according to the supplied quaternion

            Parameters
            ----------
            axis: Axis, The axis for the second ellipsoid to be translated along

            dist: float, Center-to-center distance between both ellipsoids

            rotation: NDArray[np.float64], An array representing a quaternion in
            the format [w, x, y, z]

            Return Value
            ------------
            The potential energy for the input configuration
            """

            LPAR = 1.0
            LPERP = 0.5

            ellipsoid = EllipsoidChain(
                num_mols=2, lpar=LPAR, bead_mass=1.0, lengths=1
            )
            system = Pack(density=0.1 * u.Unit("nm**-3"), molecules=ellipsoid)

            ellipsoid_to_origin(system.system.children[0].children[0])
            ellipsoid_to_origin(system.system.children[1].children[0])
            translation = [0.0, 0.0, 0.0]
            translation[axis.value] = dist
            translate_ellipsoid_by(
                system.system.children[1].children[0], translation
            )
            if rotation is not None:
                rotate_ellipsoid_by(
                    system.system.children[1].children[0], rotation
                )
            system.gmso_system = system._convert_to_gmso()
            system._hoomd_snapshot = system._create_hoomd_snapshot()

            ff = EllipsoidForcefield(
                epsilon=1.0,
                lpar=LPAR,
                lperp=LPERP,
                r_cut=10,
            )

            rigid_frame, rigid_constraint = create_rigid_ellipsoid_chain(
                system.hoomd_snapshot, LPAR, LPERP
            )

            # apply quaternion to particle orientations
            if rotation is not None:
                rigid_frame.particles.orientation = [
                    # rigid body orientations
                    [1, 0, 0, 0],
                    rotation,
                    # constituent particle orientations
                    [1, 0, 0, 0],
                    [1, 0, 0, 0],
                    [1, 0, 0, 0],
                    [1, 0, 0, 0],
                    rotation,
                    rotation,
                    rotation,
                    rotation,
                ]

            ellipsoid_sim = Simulation(
                initial_state=rigid_frame,
                forcefield=ff.hoomd_forces,
                constraint=rigid_constraint,
                gsd_write_freq=1,
                gsd_file_name="traj.gsd",
                log_write_freq=1,
                log_file_name="log.txt",
                dt=0.001,
            )

            ellipsoid_sim.run_NVT(
                n_steps=0, kT=1.0, tau_kt=1.0, thermalize_particles=False
            )
            return ellipsoid_sim.operations.computes[:][0].potential_energy

        dist = 2 ** (1 / 6)
        parallel_pe = run_sim(Axis.Y, dist)  # ()()
        parallel_long_pe = run_sim(Axis.Z, dist)  # <><>
        perpendicular_pe = run_sim(
            Axis.Y, dist, euler_to_quaternion(np.pi / 2, 0, 0)
        )  # ()<>

        assert parallel_pe != perpendicular_pe
        assert parallel_pe != parallel_long_pe
        assert perpendicular_pe != parallel_long_pe

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
