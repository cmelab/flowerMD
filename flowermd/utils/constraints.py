import gsd
import gsd.hoomd
import hoomd
import numpy as np
from cmeutils.geometry import moit


def set_bond_constraints(
    snapshot, bond_types, constraint_values, tolerance=1e-5
):
    """Helper method to add fixed bond constraints to a gsd.hoomd.Frame.

    Parameters
    ----------
    snapshot : gsd.hoomd.Frame, required
        Snapshot of complete topology that will have constraints added.
    bond_type : list of str, required
        The bond type to add constraints for. Must match snapshot.bonds.types.
    constrain_value : list of float, required
        The value to use for the constrained bond length.
        Must be close to the exisitng bond lenghts in snapshot.bonds
    tolerance : float, default 1e-5
        Used to compare actual bond lengths vs `constraint_value`
        Sets the tolerance property in hoomd.md.constrain.Distance

    Returns
    -------
    snapshot : gsd.hoomd.Frame
        The modified snapshot with populated snapshot.constraints
    d : gsd.hoomd.constrain.Distance
        Used when initializing a simulation in flowermd.base.Simulation

    Notes
    -----
    Pass the snapshot and constraint object into flowermd.base.Simulation
    in order for the fixed bond lengths to take effect.

    Examples
        --------
        This example demonstrates how to create a snapshot with fixed bonds
        from a snapshot of a bead-spring system.
        The ellipsoids are created using the EllipsoidChain class from
        the `flowermd.library.polymers`.
        The snapshot, and the constraint object are passed into flowermd.Simulation
        and used by Hoomd-Blue to constrain the bond lengths between
        bonded ellipsoids.

        ::
            from flowermd.library import EllipsoidChain, EllipsoidForcefield
            from flowermd import Pack, Simulation
            from flowermd.utils import set_bond_constraints

            chain = EllipsoidChain(lengths=30, num_mols=1, lpar=1, bead_mass=1)
            system = Pack(molecules=chain, density=0.01, packing_expand_factor=25)
            constrain_snap, d_constraint = set_bond_constraints(
                    system.hoomd_snapshot, constrain_value=1.0, bond_type="_C-_H"
            )
            forces = EllipsoidForcefield(
                    epsilon=1.0, lpar=1.0, lperp=0.5, r_cut=3.0, angle_k=10, angle_theta0=2.2
            )

            sim = Simulation(
                initial_state=constrain_snap,
                forcefield=forces.hoomd_forces,
                constraint=d_constraint,
                gsd_write_freq=5000
            )
            sim.run_NVT(kT=1.0, n_steps=500000, tau_kt=100*sim.dt)
            sim.flush_writers()

    """
    constraint_values_list = []
    constraint_groups_list = []
    for b_type, val in zip(bond_types, constraint_values):
        type_values = []
        type_groups = []
        b_type_id = snapshot.bonds.types.index(b_type)
        indices = np.where(snapshot.bonds.typeid == np.array(b_type_id))[
            0
        ].astype(int)
        for idx in indices:
            group = snapshot.bonds.group[idx]
            bond_len = np.linalg.norm(
                snapshot.particles.position[group[1]]
                - snapshot.particles.position[group[0]]
            )
            if not np.isclose(val, bond_len, atol=tolerance):
                raise ValueError("Values found not within the given tolerance")
            type_values.append(val)
            type_groups.append(group)
        constraint_values_list.extend(type_values)
        constraint_groups_list.extend(type_groups)

    snapshot.constraints.N = len(constraint_values_list)
    snapshot.constraints.value = constraint_values_list
    snapshot.constraints.group = constraint_groups_list
    d = hoomd.md.constrain.Distance(tolerance=tolerance)
    return snapshot, d


def create_rigid_ellipsoid_chain(snapshot):
    """Create rigid bodies from a snapshot.

    This is designed to be used with flowerMD's built in library
    for simulating ellipsoidal chains.
    As a result, this will not work for setting up rigid bodies
    for other kinds of systems.

    See `flowermd.library.polymer.EllipsoidChain` and
    `flowermd.library.forcefields.EllipsoidForcefield`.

    Parameters
    ----------
    snapshot : gsd.hoomd.Snapshot; required
        The snapshot of the system.
        Pass in `flowermd.base.System.hoomd_snapshot()`.

    Returns
    -------
    rigid_frame : gsd.hoomd.Frame
        The snapshot of the rigid bodies.
    rigid_constrain : hoomd.md.constrain.Rigid
        The rigid body constrain object.

    """
    bead_len = 4
    typeids = snapshot.particles.typeid.reshape(-1, bead_len)
    matches = np.where((typeids == typeids))
    rigid_const_idx = (matches[0] * bead_len + matches[1]).reshape(-1, bead_len)
    n_rigid = rigid_const_idx.shape[0]  # number of ellipsoid monomers

    rigid_masses = []
    rigid_pos = []
    rigid_moi = []
    # Find the mass, position and MOI for reach rigid center
    for idx in rigid_const_idx:
        mass = np.sum(np.array(snapshot.particles.mass)[idx])
        pos = snapshot.particles.position[idx][0]
        moi = [0, 2, 2]
        rigid_masses.append(mass)
        rigid_pos.append(pos)
        rigid_moi.append(moi)

    rigid_frame = gsd.hoomd.Frame()
    rigid_frame.particles.types = ["R"] + snapshot.particles.types
    rigid_frame.particles.N = n_rigid + snapshot.particles.N
    rigid_frame.particles.typeid = np.concatenate(
        (([0] * n_rigid), snapshot.particles.typeid + 1)
    )
    rigid_frame.particles.mass = np.concatenate(
        (rigid_masses, snapshot.particles.mass)
    )
    rigid_frame.particles.position = np.concatenate(
        (rigid_pos, snapshot.particles.position)
    )
    rigid_frame.particles.moment_inertia = np.concatenate(
        (rigid_moi, np.zeros((snapshot.particles.N, 3)))
    )
    rigid_frame.particles.orientation = [[1, 0, 0, 0]] * (
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
    # set up constraints
    if snapshot.constraints.N > 0:
        rigid_frame.constraints.N = snapshot.constraints.N
        rigid_frame.constraints.value = snapshot.constraints.value
        rigid_frame.constraints.group = [
            list(np.add(g, n_rigid)) for g in snapshot.constraints.group
        ]

    # find local coordinates of the particles in the first rigid body
    # only need to find the local coordinates for the first rigid body
    local_coords = (
        snapshot.particles.position[rigid_const_idx[0]] - rigid_pos[0]
    )

    rigid_constrain = hoomd.md.constrain.Rigid()
    rigid_constrain.body["R"] = {
        "constituent_types": ["X", "A", "T", "T"],
        "positions": local_coords,
        "orientations": [[1, 0, 0, 0]] * len(local_coords),
    }
    return rigid_frame, rigid_constrain


def _get_com_mass_pos_moi(snapshot, rigid_const_idx):
    com_mass = []
    com_position = []
    com_moi = []
    for idx in rigid_const_idx:
        constituents_mass = np.array(snapshot.particles.mass)[idx][0]
        constituents_pos = np.array(snapshot.particles.position)[idx]
        total_mass = np.sum(constituents_mass)
        com_mass.append(total_mass)
        com = (
            np.sum(
                constituents_pos * constituents_mass,
                axis=0,
            )
            / total_mass
        )
        com_position.append(com)
        moi = moit(
            points=constituents_pos,
            masses=constituents_mass,
            center=com,
        )
        com_moi.append(moi)
    return com_mass, com_position, com_moi
