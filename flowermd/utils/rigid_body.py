import gsd
import gsd.hoomd
import hoomd
import numpy as np
from cmeutils.geometry import moit


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


def create_rigid_body(snapshot, bead_constituents_types):
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
    rigid_frame.particles.types = ["R"] + snapshot.particles.types
    rigid_frame.particles.N = n_rigid
    rigid_frame.particles.typeid = [0] * n_rigid
    rigid_frame.particles.mass = com_mass
    rigid_frame.particles.position = com_position
    rigid_frame.particles.moment_inertia = com_moi
    rigid_frame.particles.orientation = [(1.0, 0.0, 0.0, 0.0)] * n_rigid
    rigid_frame.configuration.box = snapshot.configuration.box

    # find local coordinates of the particles in the first rigid body
    # only need to find the local coordinates for the first rigid body
    local_coords = (
        snapshot.particles.position[rigid_const_idx[0]] - com_position[0]
    )

    rigid_constrain = hoomd.md.constrain.Rigid()
    rigid_constrain.body["rigid"] = {
        "constituent_types": bead_constituents_types,
        "positions": local_coords,
        "orientations": [(1.0, 0.0, 0.0, 0.0)] * len(local_coords),
    }
    return rigid_frame, rigid_constrain
