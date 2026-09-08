"""Examples for the Systems class."""

import mbuild as mb
import numpy as np
import unyt as u
from scipy.spatial.distance import pdist

from flowermd.base.system import System
from flowermd.utils import (
    get_target_box_mass_density,
    get_target_box_number_density,
)


class SingleChainSystem(System):
    """Builds a vacuum box around a single chain.

    The box lengths are chosen so they are at least as long as the largest particle distance.
    The maximum distance of the chain is calculated using scipy.spatial.distance.pdist().
    This distance multiplied by a buffer defines the box dimensions. The chain is centered in the box.

    Parameters
    ----------
    buffer : float, default 1.05
        A factor to multiply box dimensions. Must be greater than 1 so that the particles are inside the box.

    """

    def __init__(self, molecules, base_units=dict(), buffer=1.05):
        self.buffer = buffer
        super(SingleChainSystem, self).__init__(
            molecules=molecules, base_units=base_units
        )

    def _build_system(self):
        if len(self.all_molecules) > 1:
            raise ValueError(
                "This system class only works for systems contianing a single molecule."
            )
        chain = self.all_molecules[0]
        eucl_dist = pdist(self.all_molecules[0].xyz)
        chain_length = np.max(eucl_dist)
        box = mb.Box(lengths=np.array([chain_length] * 3) * self.buffer)
        comp = mb.Compound()
        comp.add(chain)
        comp.box = box
        chain.translate_to((box.Lx / 2, box.Ly / 2, box.Lz / 2))
        return comp


class mbuildSystem(System):
    """Builds a system using mbuild box and mbuild positions.

    The box lengths and positions are read from the input mbuild compound. This is intended to be used with mbuild intialization methods,
    like translating polymer contiuents within the box, or a random walk cuboid constraint in mbuild 2.0.

    """

    def __init__(self, molecules, base_units=dict()):
        self.box_temp = molecules.box
        super(mbuildSystem, self).__init__(
            molecules=molecules, base_units=base_units
        )

    def _build_system(self):
        chain = self.all_molecules
        comp = mb.Compound()
        comp.add(chain)
        comp.box = self.box_temp
        return comp


class RandomWalk(System):
    """Places molecules in the system using in a random walk.

    Assumes all beads in each molecule are identical type (e.g., all "A" beads).
    All molecules are positioned in a single vectorized computation:
    - Each chain's first bead is placed at a random location within the box
    - Subsequent beads are placed at fixed bond length steps in random directions
    - All molecules are processed simultaneously using numpy broadcasting

    Parameters
    ----------
    molecules : Polymer or list of Polymer
        The molecule(s) to place in the system. All molecules must be identical.
    density : float or unyt_quantity
        The target density for the system (g/cm^3 or m^-3).
    base_units : dict, default {}
        Base units for the system.
    seed : int, default 12345
        Random seed for reproducibility.
    unique_molecules : bool, default True
        If True, each molecule in the list is treated as unique.
        If False, the single molecule is replicated.
    **kwargs
        Additional keyword arguments passed to System.

    """

    def __init__(
        self,
        molecules,
        buffer,
        bond_length,
        density: float,
        base_units=dict(),
        seed=12345,
        unique_molecules=True,
        **kwargs,
    ):
        if not isinstance(density, u.array.unyt_quantity):
            self.density = density * u.Unit("g") / u.Unit("cm**3")
            warnings.warn(
                "Units for density were not given, assuming units of g/cm**3."
            )
        else:
            self.density = density

        self.seed = seed
        self.unique_molecules = unique_molecules
        self.bond_length = bond_length
        self.n_mols = molecules.n_mols[0]
        self.lengths = molecules.lengths[0]
        self.buffer = buffer
        super(RandomWalk, self).__init__(
            molecules=molecules, base_units=base_units, **kwargs
        )

    def _build_system(self, **kwargs):
        """Build the system by placing molecules using random walk algorithm.

        Uses vectorized computation to generate all positions at once.
        """

        mass_density = u.Unit("kg") / u.Unit("m**3")
        number_density = u.Unit("nm**-3")

        if self.density.units.dimensions == mass_density.dimensions:
            target_box = get_target_box_mass_density(
                density=self.density, mass=self.mass
            ).to("nm")
        elif self.density.units.dimensions == number_density.dimensions:
            target_box = get_target_box_number_density(
                density=self.density, n_beads=self.n_particles
            ).to("nm")
        else:
            raise ValueError(
                f"Density dimensions of {self.density.units.dimensions} "
                "were given, but only mass density "
                f"({mass_density.dimensions}) and "
                f"number density ({number_density.dimensions}) are supported."
            )

        if not self.unique_molecules and len(self._molecules) > 1:
            raise ValueError(
                f"unique_molecules kwarg was set to {self.unique_molecules}, "
                "which doesn't match the length of molecules given: "
                f"{len(self._molecules)} molecules"
            )

        if len(self._molecules) == 1:
            if not self.unique_molecules and self._molecules[0].n_mols > 1:
                raise ValueError(
                    f"unique_molecules kwarg was set to {self.unique_molecules}, "
                    "which doesn't match the polydisperse system given: "
                    f"{self._molecules[0].n_mols}"
                )

        box_lengths = target_box.to_value("nm")
        Lx = box_lengths[0]

        rng = np.random.default_rng(self.seed)
        all_positions = self._generate_all_random_walks_vectorized(
            num_molecules=self.n_mols,
            beads_per_molecule=self.lengths,
            bond_length=self.bond_length,
            box_lengths=Lx,
            buffer=self.buffer,
            rng=rng,
        )

        system = mb.Compound()

        # Apply positions to all molecules and add to system
        for idx, chain in enumerate(self.all_molecules):
            for bead_idx, bead in enumerate(chain):
                flat_idx = idx * self.lengths + bead_idx
                bead.translate_to(all_positions[flat_idx])
            system.add(chain)

        # Set system box from density calculation
        system.box = mb.box.Box(box_lengths)

        return system

    def _generate_all_random_walks_vectorized(
        self,
        num_molecules,
        beads_per_molecule,
        bond_length,
        box_lengths,
        buffer,
        rng,
    ):
        """Generate random walk positions for ALL identical molecules at once.

        Uses vectorized numpy operations for efficiency. Each molecule has
        identical structure (same bead type throughout), differing only in position.

        Parameters
        ----------
        num_molecules : int
            Number of molecules to generate positions for.
        beads_per_molecule : int
            Number of beads in each molecule.
        bond_length : float
            Bond length between consecutive beads (nm).
        box_lengths : np.ndarray
            The box dimensions [Lx, Ly, Lz] in nm.
        rng : np.random.Generator
            Random number generator instance.

        Returns
        -------
        np.ndarray
            Array of shape (num_molecules, beads_per_molecule, 3) with all
            positions in nm.
        """
        positions = np.empty((num_molecules * beads_per_molecule, 3))
        starts = rng.uniform(
            buffer, box_lengths - buffer, size=(num_molecules, 3)
        )

        thetas = rng.uniform(
            0, 2 * np.pi, size=(num_molecules, beads_per_molecule - 1)
        )
        phis = np.arccos(
            rng.uniform(-1, 1, size=(num_molecules, beads_per_molecule - 1))
        )
        x = np.sin(phis) * np.cos(thetas)
        y = np.sin(phis) * np.sin(thetas)
        z = np.cos(phis)

        deltas = np.stack([x, y, z], axis=2) * bond_length
        displacements = np.cumsum(deltas, axis=1)

        positions_view = positions.reshape(num_molecules, beads_per_molecule, 3)
        positions_view[:, 0, :] = starts
        positions_view[:, 1:, :] = starts[:, None, :] + displacements

        # pbc
        positions %= box_lengths
        positions -= box_lengths / 2
        print(positions.shape)
        return positions
