import gsd
import gsd.hoomd
import hoomd
import numpy as np
from cmeutils.geometry import moit


def create_rigid_body(snapshot, bead_constituents_types, bead_name="R"):
    """Create rigid bodies from a snapshot.

    Parameters
    ----------
    snapshot : gsd.hoomd.Snapshot; required
        The snapshot of the system.
    bead_constituents_types : list of particle types; required
        The list of particle types that make up a rigid body.

    Returns
    -------
    rigid_frame : gsd.hoomd.Frame
        The snapshot of the rigid bodies.
    rigid_constrain : hoomd.md.constrain.Rigid
        The rigid body constrain object.
    """
    # find typeid sequence of the constituent particles types in a rigid bead
    p_types = np.array(snapshot.particles.types)
    constituent_type_ids = np.where(
        p_types[:, None] == bead_constituents_types
    )[0]

    # find indices that matches the constituent particle types
    typeids = snapshot.particles.typeid
    bead_len = len(bead_constituents_types)
    matches = np.where((typeids.reshape(-1, bead_len) == constituent_type_ids))
    if len(matches[0]) == 0:
        raise ValueError(
            "No matches found between particle types in the system"
            " and bead constituents particles"
        )
    rigid_const_idx = (matches[0] * bead_len + matches[1]).reshape(-1, bead_len)

    n_rigid = rigid_const_idx.shape[0]  # number of rigid bodies

    # calculate center of mass and its position for each rigid body
    com_mass, com_position, com_moi = _get_com_mass_pos_moi(
        snapshot, rigid_const_idx
    )

    rigid_frame = gsd.hoomd.Frame()
    rigid_frame.particles.types = [bead_name] + snapshot.particles.types
    rigid_frame.particles.N = n_rigid + snapshot.particles.N
    rigid_frame.particles.typeid = np.concatenate(
        (([0] * n_rigid), snapshot.particles.typeid + 1)
    )
    rigid_frame.particles.mass = np.concatenate(
        (com_mass, snapshot.particles.mass)
    )
    rigid_frame.particles.position = np.concatenate(
        (com_position, snapshot.particles.position)
    )
    rigid_frame.particles.moment_inertia = np.concatenate(
        (com_moi, np.zeros((snapshot.particles.N, 3)))
    )
    rigid_frame.particles.orientation = [(1.0, 0.0, 0.0, 0.0)] * (
        n_rigid + snapshot.particles.N
    )
    rigid_frame.particles.body = np.concatenate(
        (
            np.arange(n_rigid),
            np.arange(n_rigid).repeat(rigid_const_idx.shape[1]),
        )
    )
    rigid_frame.configuration.box = snapshot.configuration.box

    # set up bonds
    if snapshot.bonds.N > 0:
        rigid_frame.bonds.N = snapshot.bonds.N
        rigid_frame.bonds.types = snapshot.bonds.types
        rigid_frame.bonds.typeid = snapshot.bonds.typeid
        rigid_frame.bonds.group = [
            list(np.add(g, n_rigid)) for g in snapshot.bonds.group
        ]
    # set up angles
    if snapshot.angles.N > 0:
        rigid_frame.angles.N = snapshot.angles.N
        rigid_frame.angles.types = snapshot.angles.types
        rigid_frame.angles.typeid = snapshot.angles.typeid
        rigid_frame.angles.group = [
            list(np.add(g, n_rigid)) for g in snapshot.angles.group
        ]

    # set up dihedrals
    if snapshot.dihedrals.N > 0:
        rigid_frame.dihedrals.N = snapshot.dihedrals.N
        rigid_frame.dihedrals.types = snapshot.dihedrals.types
        rigid_frame.dihedrals.typeid = snapshot.dihedrals.typeid
        rigid_frame.dihedrals.group = [
            list(np.add(g, n_rigid)) for g in snapshot.dihedrals.group
        ]

    # find local coordinates of the particles in the first rigid body
    # only need to find the local coordinates for the first rigid body
    local_coords = (
        snapshot.particles.position[rigid_const_idx[0]] - com_position[0]
    )

    rigid_constrain = hoomd.md.constrain.Rigid()
    rigid_constrain.body["R"] = {
        "constituent_types": bead_constituents_types,
        "positions": local_coords,
        "orientations": [(1.0, 0.0, 0.0, 0.0)] * len(local_coords),
    }
    return rigid_frame, rigid_constrain


def _get_com_mass_pos_moi(snapshot, rigid_const_idx):
    com_mass = []
    com_position = []
    com_moi = []
    for idx in rigid_const_idx:
        constituents_mass = np.array(snapshot.particles.mass)
        constituents_pos = np.array(snapshot.particles.position)
        total_mass = np.sum(constituents_mass[idx])
        com_mass.append(total_mass)
        com = (
            np.sum(
                constituents_pos[idx] * constituents_mass[idx, np.newaxis],
                axis=0,
            )
            / total_mass
        )
        com_position.append(com)
        com_moi.append(
            moit(
                points=constituents_pos[idx],
                masses=constituents_mass[idx],
                center=com,
            )
        )
    return com_mass, com_position, com_moi
