import numpy as np
import gsd.hoomd


class Interface:
    def __init__(self, gsd_file, interface_axis, gap, wall_sigma=1.0):
        self.gsd_file = gsd_file,
        self.axis = interface_axis.lower()
        self.gap = gap
        self.wall_sigma = wall_sigma
        self.hoomd_snapshot = self._build()

    def _build(self):
    axis_dict = {
        "x": 0,
        "y": 1,
        "z": 2,
    }
    gsd_file = gsd.hoomd.open(self.gsd_file)
    snap = gsd_file[-1]
    gsd_file.close()
    axis_index = axis_dict[self.axis]

    interface = gsd.hoomd.Snapshot()
    interface.particles.N = snap.particles.N * 2
    interface.bonds.N = snap.bonds.N * 2
    interface.bonds.M = snap.bonds.M
    interface.angles.N = snap.angles.N * 2
    interface.angles.M = snap.angles.M
    interface.dihedrals.N = snap.dihedrals.N * 2
    interface.dihedrals.M = snap.dihedrals.M

    # Set up box. Box edge is doubled along the interface axis direction, plus the gap
    interface.configuration.box = np.copy(snap.configuration.box)
    interface.configuration.box[axis_index] *= 2
    interface.configuration.box[axis_index] += (gap - wall_sigma)
    
    # Set up snapshot.particles info:
    # Get set of new coordiantes, shifted along interface axis
    shift = (snap.configuration.box[axis_index] + self.gap - self.wall_sigma)/2
    right_pos = np.copy(snap.particles.position)
    right_pos[:,axis_index] += shift
    left_pos = np.copy(snaps.particles.position)
    left_pos[:,axis_index] -= shift
    
    pos = np.concatenate((left_pos, right_pos), axis=None)
    mass = np.concatenate((snap.particles.mass, snap.particles.mass), axis=None)
    type_ids = np.concatenate(
            (snap.particles.typeid, snap.particles.typeid), axis=None
    )
    interface.particles.position = pos
    interface.particles.mass = mass
    interface.particles.types = np.copy(snap.particles.types)
    interface.particles.typeid = type_ids
    
    # Set up bonds:
    bond_group_left = np.copy(snap.bonds.group)
    bond_group_right = np.copy(snap.bonds.group) + snap.particles.N
    bond_group = np.concatenate((bond_group_left, bond_group_right), axis=None)
    bond_type_ids = np.concatenate(
            (snap.bonds.typeid, snap.bonds.typeid), axis=None
    )
    interface.bonds.group = bond_group
    interface.bonds.typeid = bond_type_ids
    interface.bonds.types = np.copy(snap.bonds.types)
    
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
    interface.angles.types = np.copy(snap.angles.types)
    
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
    interface.dihedrals.types = np.copy(snap.dihedrals.types)
    return interface
