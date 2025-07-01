"""Utility functions for the Welding module."""

import hoomd
import numpy as np


def add_void_particles(
    snapshot, forcefield, num_voids, void_axis, void_diameter, epsilon, r_cut
):
    """Add void particles to a simulation.

    This method adds void particles to the snapshot and creates the necessary
    LJ force objects to simulate them.

    Parameters
    ----------
    snapshot: hoomd.snapshot.Snapshot, required
        The snapshot to add void particles to.
    forcefield: List of hoomd.md.force.Force, required
        The simulation forces to add void parameters to.
    num_voids: int, required
        The number of void particles to add.
    void_axis: tuple of int, required
        The axis along which to add void particles.
    void_diameter: float, required
        The diameter of the void particles.
    epsilon: float, required
        The epsilon parameter for the void particles LJ.
    r_cut: float, required
        The cutoff radius for the void particles LJ.

    """
    void_axis = np.asarray(void_axis)
    snapshot.particles.N += num_voids
    # Set updated positions
    void_pos = void_axis * snapshot.configuration.box[0:3] / 2
    init_pos = snapshot.particles.position
    new_pos = np.empty((init_pos.shape[0] + 1, 3))
    new_pos[: init_pos.shape[0]] = init_pos
    new_pos[-1] = void_pos
    snapshot.particles.position = np.concatenate(
        (init_pos, void_pos.reshape(1, 3)), axis=0
    )
    # Set updated types and type IDs
    if "VOID" not in snapshot.particles.types:
        snapshot.particles.types.append("VOID")
    void_id = len(snapshot.particles.types) - 1
    init_ids = snapshot.particles.typeid
    snapshot.particles.typeid = np.concatenate(
        (init_ids, np.array([void_id])), axis=None
    )
    # Set updated mass and charges
    init_mass = snapshot.particles.mass
    snapshot.particles.mass = np.concatenate(
        (init_mass, np.array([1])), axis=None
    )
    init_charges = snapshot.particles.charge
    snapshot.particles.charge = np.concatenate(
        (init_charges, np.array([0])), axis=None
    )
    init_orientation = snapshot.particles.orientation
    snapshot.particles.orientation = np.concatenate(
        (init_orientation, np.array([1, 0, 0, 0])), axis=None
    )
    # Updated LJ params
    lj = [i for i in forcefield if isinstance(i, hoomd.md.pair.LJ)][0]
    for ptype in snapshot.particles.types:
        lj.params[(ptype, "VOID")] = {
            "sigma": void_diameter,
            "epsilon": epsilon,
        }
        lj.r_cut[(ptype, "VOID")] = r_cut
    return snapshot, forcefield
