"""Examples for the Systems class."""

# This is a placeholder for any class that inherits from base.system
import numpy
from scipy.spatial.distance import pdist


class SingleChainSystem(System):
    """Builds a box around a single chain.

    Calculates the maximum distance of the chain using scipy.spatial.distance.pdist().

    Parameters
    ----------
    See System class.

    """

	def __init__(self, molecules, base_units=dict()):
		super(SingleChainSystem, self).__init__(
			molecules=molecules,
			base_units=base_units
           	 )

        def _build_system(self):
            chain = self.all_molecules[0]
            children_pos_array = np.zeros((len(chain.children),3))
            for i in range(len(chain.children)):
                children_pos_array[i] = chain.children[i].pos
            eucl_dist = pdist(children_pos_array)
            chain_length = np.max(eucl_dist)
            box = mb.Box(lengths=np.array([chain_length] * 3) * 1.05)
            comp = mb.Compound()
            comp.add(chain)
            comp.box = box
            chain.translate_to((box.Lx / 2, box.Ly / 2, box.Lz / 2))
            return comp
