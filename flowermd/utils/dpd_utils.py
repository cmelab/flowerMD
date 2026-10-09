import freud
import networkx as nx
import numpy as np


def compute_closest_rdf(sim, bins=100, r_max=1.0):
    """Compute the RDF once from the simulation's current state.
    Returns the first non-zero bin, indicating the closest particles.

    Parameters
    ----------
    sim : hoomd.Simulation
        HOOMD simulation object to compute the RDF from.
    bins : int, default 100
        Number of bins for the RDF histogram.
    r_max : float, default 1.0
        Maximum radius for the RDF calculation.

    Returns
    -------
    radius : float
        first non-zero bin
    """
    rdf = freud.density.RDF(bins=bins, r_max=r_max)
    snap = sim.state.get_snapshot()
    rdf.compute(system=snap, reset=True)
    b = (rdf.rdf != 0).argmax()
    return rdf.bin_centers[b]


def random_walk_positions_from_bonds(
    bonds, n_particles, bond_length, box_lengths, buffer=0.0, rng=None
):
    """Random-walk coordinates using bond connectivity.

    BFS from the lowest-id particle in each connected component ("molecule").
    Each particle is placed exactly once, bond_length from the parent that
    discovered it; already-placed particles are never regenerated.

    Parameters
    ----------
    bonds : (n_bonds, 2) array-like of int, required
        Particle-id pairs, consistent with list(compound.particles()) order.
    n_particles : int, required
    bond_length : float, required
    box_lengths : float or (3,) array-like, required
    buffer : float, default 0.0
    rng : np.random.Generator, optional

    Returns
    -------
    positions : (n_particles, 3) ndarray, row i = position of particle id i.
    """
    if rng is None:
        rng = np.random.default_rng()

    bonds = np.asarray(bonds, dtype=int)
    box_lengths = np.broadcast_to(
        np.asarray(box_lengths, dtype=float), (3,)
    ).copy()

    graph = nx.Graph()
    graph.add_nodes_from(range(n_particles))  # keeps isolated particles too
    graph.add_edges_from(bonds)

    components = list(nx.connected_components(graph))
    roots = np.array([min(comp) for comp in components])

    depth = np.full(n_particles, -1, dtype=int)
    parent = np.full(n_particles, -1, dtype=int)
    depth[roots] = 0
    for root in roots:
        for p, c in nx.bfs_edges(graph, root):
            depth[c] = depth[p] + 1
            parent[c] = p

    positions = np.empty((n_particles, 3))
    positions[roots] = rng.uniform(
        buffer, box_lengths - buffer, size=(len(roots), 3)
    )

    for d in range(1, depth.max() + 1):
        children = np.where(depth == d)[0]
        if len(children) == 0:
            continue
        parents = parent[children]

        theta = rng.uniform(0, 2 * np.pi, size=len(children))
        phi = np.arccos(rng.uniform(-1, 1, size=len(children)))
        step = (
            np.column_stack(
                [
                    np.sin(phi) * np.cos(theta),
                    np.sin(phi) * np.sin(theta),
                    np.cos(phi),
                ]
            )
            * bond_length
        )

        positions[children] = positions[parents] + step

    positions %= box_lengths
    positions -= box_lengths / 2
    return positions
