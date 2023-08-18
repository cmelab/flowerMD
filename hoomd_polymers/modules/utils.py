import hoomd
import numpy as np


def add_void_particles(
    snapshot, forcefield, num_voids, void_axis, void_diameter, epsilon, r_cut
):
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
