import freud
import gsd
import gsd.hoomd
import hoomd
import numpy as np


def initialize_snapshot_rand_walk(
    num_pol, num_mon, density, bond_length=1.0, seed=1234
):
    """
    Create a HOOMD snapshot of a cubic box with the number density given by input parameters. Configure particles using a random walk.

    """
    rng = np.random.default_rng(seed)

    N = num_pol * num_mon
    L = np.cbrt(N / density)

    positions = np.empty((N, 3))
    starts = rng.uniform(0, L, size=(num_pol, 3))

    thetas = rng.uniform(0, 2 * np.pi, size=(num_pol, num_mon - 1))
    phis = np.arccos(rng.uniform(-1, 1, size=(num_pol, num_mon - 1)))
    x = np.sin(phis) * np.cos(thetas)
    y = np.sin(phis) * np.sin(thetas)
    z = np.cos(phis)

    deltas = np.stack([x, y, z], axis=2) * bond_length
    displacements = np.cumsum(deltas, axis=1)

    positions_view = positions.reshape(num_pol, num_mon, 3)
    positions_view[:, 0, :] = starts
    positions_view[:, 1:, :] = starts[:, None, :] + displacements

    # pbc
    positions %= L
    positions -= L / 2

    indices = np.arange(N).reshape(num_pol, num_mon)
    bonds = np.column_stack([indices[:, :-1].ravel(), indices[:, 1:].ravel()])

    frame = gsd.hoomd.Frame()
    frame.particles.types = ["A"]
    frame.particles.N = N
    frame.particles.position = positions
    frame.bonds.N = len(bonds)
    frame.bonds.group = bonds
    frame.bonds.types = ["b"]
    frame.configuration.box = [L, L, L, 0, 0, 0]

    return frame


def check_bond_length_equilibration(
    snap, num_mon, num_pol, max_bond_length=1.1, min_bond_length=0.95
):
    """
    Check the bond distances.

    """
    frame_ds = []
    for j in range(num_pol):
        idx = j * num_mon
        d1 = (
            snap.particles.position[idx : idx + num_mon - 1]
            - snap.particles.position[idx + 1 : idx + num_mon]
        )
        L = snap.configuration.box[0]
        d1 -= L * np.round(d1 / L)
        bond_l = np.linalg.norm(d1, axis=1)
        frame_ds.append(bond_l)
    max_frame_bond_l = np.max(np.array(frame_ds))
    min_frame_bond_l = np.min(np.array(frame_ds))
    print("max: ", max_frame_bond_l, " min: ", min_frame_bond_l)
    if (
        max_frame_bond_l <= max_bond_length
        and min_frame_bond_l >= min_bond_length
    ):
        print("Bonds relaxed.")
        return True
    if max_frame_bond_l > max_bond_length or min_frame_bond_l < min_bond_length:
        return False


def check_inter_particle_distance(snap, minimum_distance=0.95):
    """
    Check particle separations.

    """
    positions = snap.particles.position
    box = snap.configuration.box
    aq = freud.locality.AABBQuery(box, positions)
    aq_query = aq.query(
        query_points=positions,
        query_args=dict(r_min=0.0, r_max=minimum_distance, exclude_ii=True),
    )
    nlist = aq_query.toNeighborList()
    if len(nlist) == 0:
        print("Inter-particle separation reached.")
        return True
    else:
        return False


def add_hoomd_writers(
    sim,
    gsd_file_name="trajectory.gsd",
    gsd_write_freq=10,
    log_file_name="log.txt",
    log_write_freq=10,
):
    """Add GSD trajectory and log writers to a HOOMD simulation.

    This function creates:
    - a GSD trajectory writer for particle configurations
    - a table logger for thermodynamic and force quantities
    - thermodynamic compute operations for system properties

    Parameters
    ----------
    sim : hoomd.Simulation
        HOOMD simulation object to which writers and
        computes will be attached.
    gsd_file_name : str, default 'trajectory.gsd'
        the file that the gsd trajectory data will be saved to
    gsd_write_freq : int, default 10
        Period to write simulation data to the gsd file.
    log_file_name : str, default 'log.txt'
        the file that the .txt log file will be saved to
    log_write_freq : int, default 10
        Period to write simulation data to the log file.

    Returns
    -------
    None
        This function modifies the simulation object in place
        and does not return a value.

    """
    gsd_logger = hoomd.logging.Logger(
        categories=["scalar", "string", "sequence"]
    )
    logger = hoomd.logging.Logger(categories=["scalar", "string"])
    gsd_logger.add(sim, quantities=["timestep", "tps"])
    logger.add(sim, quantities=["timestep", "tps"])
    thermo_props = hoomd.md.compute.ThermodynamicQuantities(
        filter=hoomd.filter.All()
    )
    sim.operations.computes.append(thermo_props)
    log_quantities = [
        "kinetic_temperature",
        "potential_energy",
        "kinetic_energy",
        "volume",
        "pressure",
        "pressure_tensor",
    ]
    gsd_logger.add(thermo_props, quantities=log_quantities)
    logger.add(thermo_props, quantities=log_quantities)

    for f in sim.operations.integrator.forces:
        logger.add(f, quantities=["energy"])
        gsd_logger.add(f, quantities=["energy"])

    gsd_trigger = hoomd.trigger.Or(
        [hoomd.trigger.Before(2), hoomd.trigger.Periodic(int(gsd_write_freq))]
    )

    gsd_writer = hoomd.write.GSD(
        filename=gsd_file_name,
        trigger=gsd_trigger,
        mode="wb",
        dynamic=["momentum", "property"],
        filter=hoomd.filter.All(),
        logger=gsd_logger,
    )
    gsd_writer.maximum_write_buffer_size = 64 * 1024 * 1024
    log_trigger = hoomd.trigger.Or(
        [hoomd.trigger.Before(2), hoomd.trigger.Periodic(int(log_write_freq))]
    )

    table_file = hoomd.write.Table(
        output=open(log_file_name, mode="w", newline="\n"),
        trigger=log_trigger,
        logger=logger,
        max_header_len=None,
    )
    sim.operations.writers.append(gsd_writer)
    sim.operations.writers.append(table_file)


def check_pair_energy(energy_idx=-1, log_file_name="log.txt"):
    """Check whether the pair interaction energy has equilibrated.

    Pair energies are read from the HOOMD log file and analyzed
    using pymbar timeseries equilibration detection.

    Parameters
    ----------
    energy_idx : int, default -1
        Number of initial simulation steps to discard before
        performing equilibration analysis. Default is to return the last frame.

    Returns
    -------
    float, energy of last frame(s) of dpd simulation

    """
    log = np.genfromtxt(log_file_name, names=True)
    pairs = log["mdpairDPDenergy"]
    if pairs.size > 1:
        return np.mean(pairs[energy_idx:])
    elif pairs.size == 1:
        return pairs


def calculate_pair_energy(A, r, r_cut, num_pol, num_mon, density):
    """
    Calculate the minimum energy for the conservative force to reach at the given radius.
    energy for each pair in the system
    """
    density_scaling = (density / 1.414) ** 2
    constant = (1 / 2) * A * r_cut
    U = ((A * (r**2)) / (2 * r_cut)) - (A * r) + constant
    pair_energy = (13 * U * num_pol * num_mon * density_scaling) / 2

    return pair_energy


def simulation_energy_end(
    A,
    r,
    r_cut,
    num_pol,
    num_mon,
    density,
    energy_idx=-1,
    log_file_name="log.txt",
):
    """
    Calculate the minimum energy for the conservative force to reach at the given radius.
    energy for each pair in the system
    """
    U_goal = calculate_pair_energy(
        A=A, r=r, r_cut=r_cut, num_pol=num_pol, num_mon=num_mon, density=density
    )
    last_U = check_pair_energy(
        energy_idx=energy_idx, log_file_name=log_file_name
    )
    if last_U <= U_goal:
        return True
    else:
        return False
