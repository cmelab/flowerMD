import mbuild as mb
from mbuild.formats.hoomd_forcefield import create_hoomd_forcefield
import numpy as np
import unyt
from gmso.external import from_mbuild, to_gsd_snapshot


from hoomd_polymers import System
from hoomd_polymers.utils import scale_charges


class Pack(System):
    """Uses PACKMOL via mbuild.packing.fill_box.
    The box used for packing is expanded to allow PACKMOL to place all of the molecules.

    Parameters
    ----------
    packing_expand_factor : int; optional, default 5

    """
    def __init__(
            self,
            molecules,
            density,
            packing_expand_factor=5,
            edge=0.2
    ):
        super(Pack, self).__init__(molecules=molecules, density=density)
        self.packing_expand_factor = packing_expand_factor
        self.edge = edge
        self._build()

    def _build(self):
        self.set_target_box()
        self.system = mb.packing.fill_box(
                compound=self.molecules,
                n_compounds=[1 for i in self.molecules],
                box=list(self.target_box*self.packing_expand_factor),
                overlap=0.2,
                edge=self.edge
        )


class Lattice(System):
    """Places the molecules in a lattice configuration.
    Assumes two molecules per unit cell.

    Parameters
    ----------
    x : float; required
        The distance (nm) between lattice points in the x direction. 
    y : float; required
        The distance (nm) between lattice points in the y direction. 
    n : int; required
        The number of times to repeat the unit cell in x and y
    lattice_vector : array-like
        The vector between points in the unit cell 
    """
    def __init__(
            self,
            molecules,
            density,
            x,
            y,
            n,
            basis_vector=[0.5, 0.5, 0],
            z_adjust=1.0,
    ):
        super(Lattice, self).__init__(molecules=molecules, density=density)
        self.x = x
        self.y = y
        self.n = n
        self.basis_vector = basis_vector
        self._build()

    def _build(self):
        next_idx = 0
        self.system = mb.Compound()
        for i in range(self.n):
            layer = mb.Compound()
            for j in range(self.n):
                try:
                    comp1 = self.molecules[next_idx]
                    comp2 = self.molecules[next_idx + 1]
                    comp2.translate(self.basis_vector)
                    unit_cell = mb.Compound(subcompounds=[comp1, comp2])
                    unit_cell.translate((0, self.y*j, 0))
                    layer.add(unit_cell)
                    next_idx += 2
                except IndexError:
                    pass
            layer.translate((self.x*i, 0, 0))
            self.system.add(layer)
        bounding_box = self.system.get_boundingbox()
        x_len = bounding_box.lengths[0]
        y_len = bounding_box.lengths[1]
        self.set_target_box(x_constraint=x_len, y_constraint=y_len)
