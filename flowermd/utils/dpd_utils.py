import freud


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
