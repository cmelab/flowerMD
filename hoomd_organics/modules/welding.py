import gsd.hoomd
import hoomd
import numpy as np

from hoomd_organics.base.simulation import Simulation


class Interface:
    def __init__(self, gsd_file, interface_axis, gap, wall_sigma=1.0):
        self.gsd_file = gsd_file
        self.interface_axis = interface_axis
        self.gap = gap
        self.wall_sigma = wall_sigma
        self.hoomd_snapshot = self._build()

    def _build(self):
        gsd_file = gsd.hoomd.open(self.gsd_file)
        snap = gsd_file[-1]
        gsd_file.close()
        axis_index = np.where(self.interface_axis != 0)[0]

        interface = gsd.hoomd.Snapshot()
        interface.particles.N = snap.particles.N * 2
        interface.bonds.N = snap.bonds.N * 2
        interface.bonds.M = snap.bonds.M
        interface.angles.N = snap.angles.N * 2
        interface.angles.M = snap.angles.M
        interface.dihedrals.N = snap.dihedrals.N * 2
        interface.dihedrals.M = snap.dihedrals.M
        interface.pairs.N = snap.pairs.N * 2

        # Set up box. Box edge is doubled along the interface axis direction,
        # plus the gap
        interface.configuration.box = np.copy(snap.configuration.box)
        interface.configuration.box[axis_index] *= 2
        interface.configuration.box[axis_index] += self.gap - self.wall_sigma

        # Set up snapshot.particles info:
        # Get set of new coordiantes, shifted along interface axis
        shift = (
            snap.configuration.box[axis_index] + self.gap - self.wall_sigma
        ) / 2
        right_pos = np.copy(snap.particles.position)
        right_pos[:, axis_index] += shift
        left_pos = np.copy(snap.particles.position)
        left_pos[:, axis_index] -= shift

        pos = np.concatenate((left_pos, right_pos), axis=None)
        mass = np.concatenate(
            (snap.particles.mass, snap.particles.mass), axis=None
        )
        charges = np.concatenate(
            (snap.particles.charge, snap.particles.charge), axis=None
        )
        type_ids = np.concatenate(
            (snap.particles.typeid, snap.particles.typeid), axis=None
        )
        interface.particles.position = pos
        interface.particles.mass = mass
        interface.particles.charge = charges
        interface.particles.types = snap.particles.types
        interface.particles.typeid = type_ids

        # Set up bonds:
        bond_group_left = np.copy(snap.bonds.group)
        bond_group_right = np.copy(snap.bonds.group) + snap.particles.N
        bond_group = np.concatenate(
            (bond_group_left, bond_group_right), axis=None
        )
        bond_type_ids = np.concatenate(
            (snap.bonds.typeid, snap.bonds.typeid), axis=None
        )
        interface.bonds.group = bond_group
        interface.bonds.typeid = bond_type_ids
        interface.bonds.types = snap.bonds.types

        # Set up angles:
        angle_group_left = np.copy(snap.angles.group)
        angle_group_right = np.copy(snap.angles.group) + snap.particles.N
        angle_group = np.concatenate(
            (angle_group_left, angle_group_right), axis=None
        )
        angle_type_ids = np.concatenate(
            (snap.angles.typeid, snap.angles.typeid), axis=None
        )
        interface.angles.group = angle_group
        interface.angles.typeid = angle_type_ids
        interface.angles.types = snap.angles.types

        # Set up dihedrals:
        dihedral_group_left = np.copy(snap.dihedrals.group)
        dihedral_group_right = np.copy(snap.dihedrals.group) + snap.particles.N
        dihedral_group = np.concatenate(
            (dihedral_group_left, dihedral_group_right), axis=None
        )
        dihedral_type_ids = np.concatenate(
            (snap.dihedrals.typeid, snap.dihedrals.typeid), axis=None
        )
        interface.dihedrals.group = dihedral_group
        interface.dihedrals.typeid = dihedral_type_ids
        interface.dihedrals.types = snap.dihedrals.types

        # Set up pairs:
        if snap.pairs.N > 0:
            pair_group_left = np.copy(snap.pairs.group)
            pair_group_right = np.copy(snap.pairs.group) + snap.particles.N
            pair_group = np.concatenate((pair_group_left, pair_group_right))
            pair_type_ids = np.concatenate(
                (snap.pairs.typeid, snap.pairs.typeid), axis=None
            )
            interface.pairs.group = pair_group
            interface.pairs.typeid = pair_type_ids
            interface.pairs.types = snap.pairs.types
        return interface


class SlabSimulation(Simulation):
    def __init__(
        self,
        initial_state,
        forcefield,
        interface_axis=(1, 0, 0),
        wall_sigma=1.0,
        wall_epsilon=1.0,
        wall_r_cut=2.5,
        wall_r_extrap=0,
        reference_values=dict(),
        dt=0.0001,
        r_cut=2.5,
        seed=42,
        gsd_write_freq=1e4,
        gsd_file_name="trajectory.gsd",
        log_write_freq=1e3,
        log_file_name="log.txt",
    ):
        super(SlabSimulation, self).__init__(
            initial_state=initial_state,
            forcefield=forcefield,
            reference_values=reference_values,
            r_cut=r_cut,
            seed=seed,
            gsd_write_freq=gsd_write_freq,
            gsd_file_name=gsd_file_name,
            log_write_freq=log_write_freq,
            log_file_name=log_file_name,
        )
        self.interface_axis = np.asarray(interface_axis)
        self.add_walls(
            self.interface_axis,
            wall_sigma,
            wall_epsilon,
            wall_r_cut,
            wall_r_extrap,
        )

        snap = self.state.get_snapshot()
        integrate_types = [i for i in snap.particles.types if i != "VOID"]
        self.integrate_group = hoomd.filter.Type(integrate_types)


class WeldSimulation(Simulation):
    def __init__(
        self,
        initial_state,
        forcefield,
        interface_axis=(1, 0, 0),
        wall_sigma=1.0,
        wall_epsilon=1.0,
        wall_r_cut=2.5,
        wall_r_extrap=0,
        reference_values=dict(),
        dt=0.0001,
        r_cut=2.5,
        seed=42,
        gsd_write_freq=1e4,
        gsd_file_name="weld.gsd",
        log_write_freq=1e3,
        log_file_name="sim_data.txt",
    ):
        super(WeldSimulation, self).__init__(
            initial_state=initial_state,
            forcefield=forcefield,
            reference_values=reference_values,
            dt=dt,
            r_cut=r_cut,
            seed=seed,
            gsd_write_freq=gsd_write_freq,
            gsd_file_name=gsd_file_name,
            log_write_freq=log_write_freq,
            log_file_name=log_file_name,
        )
        self.interface_axis = interface_axis
        self.add_walls(
            self.interface_axis,
            wall_sigma,
            wall_epsilon,
            wall_r_cut,
            wall_r_extrap
        )
