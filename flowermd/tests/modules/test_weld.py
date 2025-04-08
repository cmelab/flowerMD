import os

import gsd.hoomd
import hoomd
import numpy as np

from flowermd import Simulation
from flowermd.modules.welding import Interface, SlabSimulation, WeldSimulation
from flowermd.modules.welding.utils import add_void_particles
from flowermd.tests.base_test import BaseTest


class TestWelding(BaseTest):
    def test_interface(self, polyethylene_system):
        sim = Simulation(
            initial_state=polyethylene_system.hoomd_snapshot,
            forcefield=polyethylene_system.hoomd_forcefield,
            log_write_freq=2000,
        )
        sim.add_walls(wall_axis=(1, 0, 0), sigma=1, epsilon=1, r_cut=2)
        sim.run_update_volume(
            duration=1000,
            period=10,
            temperature=2.0,
            tau_kt=0.01,
            final_box_lengths=sim.box_lengths / 2,
        )
        sim.save_restart_gsd()
        interface = Interface(
            gsd_files="restart.gsd", interface_axis=(1, 0, 0), gap=0.1
        )
        interface_snap = interface.hoomd_snapshot
        with gsd.hoomd.open("restart.gsd", "r") as traj:
            slab_snap = traj[0]

        assert interface_snap.particles.N == slab_snap.particles.N * 2
        assert interface_snap.bonds.N == slab_snap.bonds.N * 2
        assert interface_snap.bonds.M == slab_snap.bonds.M
        assert interface_snap.angles.N == slab_snap.angles.N * 2
        assert interface_snap.angles.M == slab_snap.angles.M
        assert interface_snap.dihedrals.N == slab_snap.dihedrals.N * 2
        assert interface_snap.dihedrals.M == slab_snap.dihedrals.M
        assert interface_snap.pairs.N == slab_snap.pairs.N * 2
        assert interface_snap.pairs.M == slab_snap.pairs.M

        if os.path.isfile("restart.gsd"):
            os.remove("restart.gsd")

    def test_interface_2_files(self, polyethylene_system):
        sim = Simulation(
            initial_state=polyethylene_system.hoomd_snapshot,
            forcefield=polyethylene_system.hoomd_forcefield,
            log_write_freq=2000,
        )
        sim.add_walls(wall_axis=(1, 0, 0), sigma=1, epsilon=1, r_cut=2)
        sim.run_update_volume(
            duration=1000,
            period=10,
            temperature=2.0,
            tau_kt=0.01,
            final_box_lengths=sim.box_lengths / 2,
        )
        sim.save_restart_gsd("restart.gsd")
        sim.save_restart_gsd("restart2.gsd")
        interface = Interface(
            gsd_files=["restart.gsd", "restart2.gsd"],
            interface_axis=(1, 0, 0),
            gap=0.1,
        )
        interface_snap = interface.hoomd_snapshot
        with gsd.hoomd.open("restart.gsd", "r") as traj:
            slab_snap = traj[0]

        assert interface_snap.particles.N == slab_snap.particles.N * 2
        assert interface_snap.bonds.N == slab_snap.bonds.N * 2
        assert interface_snap.bonds.M == slab_snap.bonds.M
        assert interface_snap.angles.N == slab_snap.angles.N * 2
        assert interface_snap.angles.M == slab_snap.angles.M
        assert interface_snap.dihedrals.N == slab_snap.dihedrals.N * 2
        assert interface_snap.dihedrals.M == slab_snap.dihedrals.M
        assert interface_snap.pairs.N == slab_snap.pairs.N * 2
        assert interface_snap.pairs.M == slab_snap.pairs.M

        if os.path.isfile("restart.gsd"):
            os.remove("restart.gsd")

    def test_slab_sim_xaxis(self, polyethylene_system):
        sim = SlabSimulation(
            initial_state=polyethylene_system.hoomd_snapshot,
            forcefield=polyethylene_system.hoomd_forcefield,
            log_write_freq=2000,
        )
        assert any(
            [isinstance(i, hoomd.md.external.wall.LJ) for i in sim.forces]
        )
        sim.run_NVT(temperature=1.0, tau_kt=0.01, duration=500)

    def test_slab_sim_yaxis(self, polyethylene_system):
        sim = SlabSimulation(
            initial_state=polyethylene_system.hoomd_snapshot,
            forcefield=polyethylene_system.hoomd_forcefield,
            interface_axis=(0, 1, 0),
            log_write_freq=2000,
        )
        assert any(
            [isinstance(i, hoomd.md.external.wall.LJ) for i in sim.forces]
        )
        sim.run_NVT(temperature=1.0, tau_kt=0.01, duration=500)

    def test_slab_sim_zaxis(self, polyethylene_system):
        sim = SlabSimulation(
            initial_state=polyethylene_system.hoomd_snapshot,
            forcefield=polyethylene_system.hoomd_forcefield,
            interface_axis=(0, 0, 1),
            log_write_freq=2000,
        )
        assert any(
            [isinstance(i, hoomd.md.external.wall.LJ) for i in sim.forces]
        )
        sim.run_NVT(temperature=1.0, tau_kt=0.01, duration=500)

    def test_weld_sim(self, polyethylene_system):
        sim = SlabSimulation(
            initial_state=polyethylene_system.hoomd_snapshot,
            forcefield=polyethylene_system.hoomd_forcefield,
            log_write_freq=2000,
        )
        sim.run_NVT(temperature=1.0, tau_kt=0.01, duration=500)
        sim.save_restart_gsd()
        # Create interface system from slab restart.gsd file
        interface = Interface(
            gsd_files="restart.gsd", interface_axis="x", gap=0.1
        )
        sim = WeldSimulation(
            initial_state=interface.hoomd_snapshot,
            forcefield=polyethylene_system.hoomd_forcefield,
        )
        if os.path.isfile("restart.gsd"):
            os.remove("restart.gsd")

    def test_void_particle(self, polyethylene_system):
        init_snap = polyethylene_system.hoomd_snapshot
        init_num_particles = init_snap.particles.N
        init_types = init_snap.particles.types
        void_snap, ff = add_void_particles(
            init_snap,
            polyethylene_system.hoomd_forcefield,
            void_diameter=0.4,
            num_voids=1,
            void_axis=(1, 0, 0),
            epsilon=1,
            r_cut=0.4,
        )
        assert init_num_particles == void_snap.particles.N - 1
        lj = [i for i in ff if isinstance(i, hoomd.md.pair.LJ)][0]
        for p_type in init_types:
            assert lj.params[(p_type, "VOID")]["sigma"] == 0.4
            assert lj.params[(p_type, "VOID")]["epsilon"] == 1

    def test_interface_with_void_particle(self, polyethylene_system):
        init_snap = polyethylene_system.hoomd_snapshot
        init_N = np.copy(init_snap.particles.N)
        void_snap, ff = add_void_particles(
            init_snap,
            polyethylene_system.hoomd_forcefield,
            void_diameter=0.10,
            num_voids=1,
            void_axis=(1, 0, 0),
            epsilon=0.7,
            r_cut=0.7,
        )
        assert void_snap.particles.N == init_N + 1
        sim = SlabSimulation(
            initial_state=void_snap,
            forcefield=ff,
            log_write_freq=2000,
        )
        sim.run_update_volume(
            duration=1000,
            period=10,
            temperature=2.0,
            tau_kt=0.01,
            final_box_lengths=sim.box_lengths / 2,
        )
        sim.save_restart_gsd("restart.gsd")
        interface = Interface(
            gsd_files=["restart.gsd"],
            interface_axis=(1, 0, 0),
            gap=0.1,
        )

        interface_snap = interface.hoomd_snapshot
        with gsd.hoomd.open("restart.gsd", "r") as traj:
            slab_snap = traj[0]

        assert "VOID" not in interface_snap.particles.types
        assert interface_snap.particles.N == (slab_snap.particles.N * 2) - 2
        assert interface_snap.bonds.N == slab_snap.bonds.N * 2
        assert interface_snap.bonds.M == slab_snap.bonds.M
        assert interface_snap.angles.N == slab_snap.angles.N * 2
        assert interface_snap.angles.M == slab_snap.angles.M
        assert interface_snap.dihedrals.N == slab_snap.dihedrals.N * 2
        assert interface_snap.dihedrals.M == slab_snap.dihedrals.M
        assert interface_snap.pairs.N == slab_snap.pairs.N * 2
        assert interface_snap.pairs.M == slab_snap.pairs.M

        if os.path.isfile("restart.gsd"):
            os.remove("restart.gsd")
