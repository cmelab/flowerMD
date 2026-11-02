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

    Breadth-first search from the lowest-id particle in each connected component ("molecule").
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
    graph.add_nodes_from(range(n_particles))
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


class EnergyStationarity:
    """Stop criterion: every force's energy has stopped changing.

    Each call compares the per-particle energy of each force (averaged
    over the samples taken since the last call) with the previous call, and
    returns True once `consecutive` comparisons in a row changed by at most
    `tol` (relative). A non-finite energy restarts the count.

    Parameters
    ----------
    tol : float, default 0.02
        Largest allowed relative change, ``|new - old| / max(|new|, |old|)``.
    consecutive : int, default 2
        Passing comparisons in a row required.
    forces : list of hoomd.md.force.Force, optional
        Forces to monitor. Defaults to every force of the simulation.

    Notes
    -----
    ``history`` holds the energies compared at each call. Call `reset`
    before reusing an instance for another run.

    """

    def __init__(self, tol=0.02, consecutive=2, forces=None):
        if tol <= 0 or consecutive < 1:
            raise ValueError("tol must be positive and consecutive >= 1.")
        self.tol = tol
        self.consecutive = consecutive
        self.forces = forces
        self.reset()

    def reset(self):
        """Forget the energies, samples and passes seen so far."""
        self.history, self._samples, self._passes = [], [], 0

    def sample(self, sim):
        """Record the current energies; the next call averages them."""
        n = sim.state.N_particles
        forces = sim.forces if self.forces is None else self.forces
        self._samples.append([force.energy / n for force in forces])

    def __call__(self, sim):
        """Return True when the energies are stationary."""
        self.sample(sim)
        current = np.mean(self._samples, axis=0)
        self._samples = []
        if not np.all(np.isfinite(current)):
            self._passes = 0
        elif self.history and np.all(np.isfinite(self.history[-1])):
            old = self.history[-1]
            scale = np.maximum(np.abs(current), np.abs(old))
            change = np.abs(current - old) / np.maximum(scale, 1e-300)
            self._passes = self._passes + 1 if np.all(change <= self.tol) else 0
        self.history.append(current)
        return self._passes >= self.consecutive
