"""Module for simulating interfaces and welding."""

import gsd.hoomd
import hoomd
import numpy as np

from flowermd.base.simulation import Simulation
from flowermd.internal import check_return_iterable
from flowermd.utils import HOOMDThermostats


class Interface:
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

    def __init__(
        self,
        gsd_files,
        interface_axis,
        gap,
        wall_sigma=1.0,
        remove_void_particles=True,
    ):
        self.gsd_files = check_return_iterable(gsd_files)
        self.interface_axis = np.asarray(interface_axis)
        self.gap = gap
        self.wall_sigma = wall_sigma
        self._remove_void_particles = remove_void_particles
        self.hoomd_snapshot = self._build()

    def _build(self):
        """Duplicates the slab and builds the interface."""
        if len(self.gsd_files) == 1:
            gsd_file_L = gsd.hoomd.open(self.gsd_files[0])
            gsd_file_R = gsd.hoomd.open(self.gsd_files[0])
        else:
            gsd_file_L = gsd.hoomd.open(self.gsd_files[0])
            gsd_file_R = gsd.hoomd.open(self.gsd_files[1])
        # Get snapshots
        snap_L = gsd_file_L[-1]
        snap_R = gsd_file_R[-1]
        gsd_file_L.close()
        gsd_file_R.close()
        # Remove VOID partilces if needed
        if self._remove_void_particles:
            for snap in [snap_L, snap_R]:
                if "VOID" in snap.particles.types:
                    snap.particles.N -= 1
                    void_id = snap.particles.types.index("VOID")
                    snap.particles.types.remove("VOID")
                    keep_indices = np.where(snap.particles.typeid != void_id)[0]
                    snap.particles.position = snap.particles.position[
                        keep_indices
                    ]
                    snap.particles.mass = snap.particles.mass[keep_indices]
                    snap.particles.charge = snap.particles.charge[keep_indices]
                    snap.particles.orientation = snap.particles.orientation[
                        keep_indices
                    ]
                    snap.particles.typeid = snap.particles.typeid[keep_indices]

        interface = gsd.hoomd.Frame()
        interface.particles.N = snap_L.particles.N + snap_R.particles.N
        interface.bonds.N = snap_L.bonds.N + snap_R.bonds.N
        interface.bonds.M = 2
        interface.angles.N = snap_L.angles.N + snap_R.angles.N
        interface.angles.M = 3
        interface.dihedrals.N = snap_L.dihedrals.N + snap_R.dihedrals.N
        interface.dihedrals.M = 4
        interface.pairs.N = snap_L.pairs.N + snap_R.pairs.N
        # Box edge is doubled along the interface axis plus the gap
        axis_index = np.where(self.interface_axis != 0)[0]
        interface.configuration.box = np.copy(snap_L.configuration.box)
        interface.configuration.box[axis_index] *= 2
        interface.configuration.box[axis_index] += self.gap - self.wall_sigma

        # Set up snapshot.particles info:
        # Get set of new coordiantes, shifted along interface axis
        shift = (
            snap_L.configuration.box[axis_index] + self.gap - self.wall_sigma
        ) / 2
        right_pos = np.copy(snap_R.particles.position)
        right_pos[:, axis_index] += shift
        left_pos = np.copy(snap_L.particles.position)
        left_pos[:, axis_index] -= shift

        pos = np.concatenate((left_pos, right_pos), axis=None)
        mass = np.concatenate(
            (snap_L.particles.mass, snap_R.particles.mass), axis=None
        )
        charges = np.concatenate(
            (snap_L.particles.charge, snap_R.particles.charge), axis=None
        )
        type_ids = np.concatenate(
            (snap_L.particles.typeid, snap_R.particles.typeid), axis=None
        )
        interface.particles.position = pos
        interface.particles.mass = mass
        interface.particles.charge = charges
        interface.particles.types = snap_L.particles.types
        interface.particles.typeid = type_ids

        # Set up bonds:
        bond_group_left = np.copy(snap_L.bonds.group)
        bond_group_right = np.copy(snap_R.bonds.group) + snap_R.particles.N
        bond_group = np.concatenate(
            (bond_group_left, bond_group_right), axis=None
        )
        bond_type_ids = np.concatenate(
            (snap_L.bonds.typeid, snap_R.bonds.typeid), axis=None
        )
        interface.bonds.group = bond_group
        interface.bonds.typeid = bond_type_ids
        interface.bonds.types = snap_L.bonds.types

        # Set up angles:
        angle_group_left = np.copy(snap_L.angles.group)
        angle_group_right = np.copy(snap_R.angles.group) + snap_L.particles.N
        angle_group = np.concatenate(
            (angle_group_left, angle_group_right), axis=None
        )
        angle_type_ids = np.concatenate(
            (snap_L.angles.typeid, snap_R.angles.typeid), axis=None
        )
        interface.angles.group = angle_group
        interface.angles.typeid = angle_type_ids
        interface.angles.types = snap_L.angles.types

        # Set up dihedrals:
        dihedral_group_left = np.copy(snap_L.dihedrals.group)
        dihedral_group_right = (
            np.copy(snap_R.dihedrals.group) + snap_L.particles.N
        )
        dihedral_group = np.concatenate(
            (dihedral_group_left, dihedral_group_right), axis=None
        )
        dihedral_type_ids = np.concatenate(
            (snap_L.dihedrals.typeid, snap_R.dihedrals.typeid), axis=None
        )
        interface.dihedrals.group = dihedral_group
        interface.dihedrals.typeid = dihedral_type_ids
        interface.dihedrals.types = snap_L.dihedrals.types

        # Set up pairs:
        if snap_L.pairs.N > 0:
            pair_group_left = np.copy(snap_L.pairs.group)
            pair_group_right = np.copy(snap_R.pairs.group) + snap_L.particles.N
            pair_group = np.concatenate((pair_group_left, pair_group_right))
            pair_type_ids = np.concatenate(
                (snap_L.pairs.typeid, snap_R.pairs.typeid), axis=None
            )
            interface.pairs.group = pair_group
            interface.pairs.typeid = pair_type_ids
            interface.pairs.types = snap_L.pairs.types
        return interface


class SlabSimulation(Simulation):
    """Simulation which creates a slab for interface systems.

    Parameters
    ----------
    interface_axis : tuple, default=(1, 0, 0)
        Axis along which the interface is to be created.
        The box edges along this axis will have a flat surface
    wall_sigma : float, default 1.0
        Sigma parameter for the wall potential.
    wall_epsilon : float, default 1.0
        Epsilon parameter for the wall potential.
    wall_r_cut : float, default 2.5
        Cutoff radius for the wall potential.
    wall_r_extrap : float, default 0
        Extrapolation distance for the wall potential.

    """

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
        device=hoomd.device.auto_select(),
        seed=42,
        gsd_write_freq=1e4,
        gsd_file_name="trajectory.gsd",
        log_write_freq=1e3,
        log_file_name="log.txt",
        thermostat=HOOMDThermostats.MTTK,
    ):
        super(SlabSimulation, self).__init__(
            initial_state=initial_state,
            forcefield=forcefield,
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
    """For simulating welding of an interface joint."""

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
        device=hoomd.device.auto_select(),
        seed=42,
        gsd_write_freq=1e4,
        gsd_file_name="weld.gsd",
        log_write_freq=1e3,
        log_file_name="sim_data.txt",
        thermostat=HOOMDThermostats.MTTK,
    ):
        super(WeldSimulation, self).__init__(
            initial_state=initial_state,
            forcefield=forcefield,
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
        self.interface_axis = interface_axis
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
