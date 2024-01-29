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
    snapshot.particles.position[-1] = (
        void_axis * snapshot.configuration.box[0:3] / 2
    )
    snapshot.particles.types = snapshot.particles.types + ["VOID"]
    snapshot.particles.typeid[-1] = len(snapshot.particles.types) - 1
    snapshot.particles.mass[-1] = 1
    lj = [i for i in forcefield if isinstance(i, hoomd.md.pair.LJ)][0]
    for ptype in snapshot.particles.types:
        lj.params[(ptype, "VOID")] = {
            "sigma": void_diameter,
            "epsilon": epsilon,
        }
        lj.r_cut[(ptype, "VOID")] = r_cut

    return snapshot, forcefield
