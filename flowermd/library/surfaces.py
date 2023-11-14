"""Recipes to generate surfaces using mBuild."""

import mbuild as mb
from mbuild.compound import Compound
from mbuild.lattice import Lattice

from flowermd.base import Molecule


class Graphene(Molecule):
    """Create a rectangular graphene layer or multiple layers.

    Parameters
    ----------
    x_repeat : int, required
        Number of times to repeat graphene lattice in the x-direciton.
    y_repeat: int, required
        Number of times to repeat graphene lattice in the y-direciton.
    n_layers: int, optional, default 1
        Number of times to repeat the complete layer in the normal direction.
    periodicity : tuple of bools, length=3, optional, default=(True, True, False) # noqa: E501
        Whether the Compound is periodic in the x, y, and z directions.
        If None is provided, the periodicity is set to `(False, False, False)`
        which is non-periodic in all directions.

    Notes
    -----
    To create bonds along periodic boundaries of the layers in the x and y
    directions, set `periodicity = (True, True, False)`

    """

    def __init__(
        self,
        x_repeat,
        y_repeat,
        n_layers,
        force_field=None,
        periodicity=(True, True, False),
    ):
        surface = mb.Compound(periodicity=periodicity)
        spacings = [0.425, 0.246, 0.35]
        points = [[1 / 6, 0, 0], [1 / 2, 0, 0], [0, 0.5, 0], [2 / 3, 1 / 2, 0]]
        lattice = Lattice(
            lattice_spacing=spacings,
            angles=[90, 90, 90],
            lattice_points={"A": points},
        )
        carbon = Compound(name="C", element="C")
        layers = lattice.populate(
            compound_dict={"A": carbon}, x=x_repeat, y=y_repeat, z=n_layers
        )
        surface.add(layers)
        surface.freud_generate_bonds("C", "C", dmin=0.14, dmax=0.145)
        super(Graphene, self).__init__(
            compound=surface, num_mols=1, force_field=force_field
        )
