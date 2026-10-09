"""Examples for the Systems class."""

import warnings

import mbuild as mb
import numpy as np
import unyt as u
from scipy.spatial.distance import pdist

from flowermd.base.system import System
from flowermd.utils import (
    get_target_box_mass_density,
    get_target_box_number_density,
    random_walk_positions_from_bonds,
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
        self.buffer = buffer
        super(RandomWalk, self).__init__(
            molecules=molecules, base_units=base_units, **kwargs
        )

    def _build_system(self, **kwargs):
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
                f"Density dimensions of {self.density.units.dimensions} were given, "
                f"but only mass density ({mass_density.dimensions}) and "
                f"number density ({number_density.dimensions}) are supported."
            )

        box_lengths = target_box.to_value("nm")
        rng = np.random.default_rng(self.seed)

        system = mb.Compound()
        system.add(self.all_molecules)

        particles = list(system.particles())
        idx_map = {p: i for i, p in enumerate(particles)}
        bonds = np.array(
            [(idx_map[b[0]], idx_map[b[1]]) for b in system.bonds()], dtype=int
        )

        positions = random_walk_positions_from_bonds(
            bonds=bonds,
            n_particles=len(particles),
            bond_length=self.bond_length,
            box_lengths=box_lengths[
                0
            ],  # cubic box; pass box_lengths directly if not
            buffer=self.buffer,
            rng=rng,
        )
        system.xyz = positions
        system.box = mb.box.Box(box_lengths)
        return system
