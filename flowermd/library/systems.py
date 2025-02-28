"""Examples for the Systems class."""

import numpy as np
from scipy.spatial.distance import pdist

from flowermd.base.system import System


class SingleChainSystem(System):
    """Builds a box around a single chain.

    Calculates the maximum distance of the chain using scipy.spatial.distance.pdist().

    Parameters
    ----------
    See System class.

    """

	def __init__(self, molecules, base_units=dict(),buffer=1.05):
       		self.buffer = buffer
		super(SingleChainSystem, self).__init__(
			molecules=molecules,
			base_units=base_units
           	 )

        def _build_system(self):
            chain = self.all_molecules[0]
            eucl_dist = pdist(self.all_molecules[0].xyz)
            chain_length = np.max(eucl_dist)
            box = mb.Box(lengths=np.array([chain_length] * 3) * self.buffer)
            comp = mb.Compound()
            comp.add(chain)
            comp.box = box
            chain.translate_to((box.Lx / 2, box.Ly / 2, box.Lz / 2))
            return comp
