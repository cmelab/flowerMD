"""Recipes to generate surfaces using mBuild."""

import mbuild as mb
from mbuild.compound import Compound
from mbuild.lattice import Lattice

from flowermd.base import Molecule, System


class Graphene(System):
    """Create a rectangular graphene layer or multiple layers.

    Parameters
    ----------
    x_repeat : int, required
        Number of times to repeat graphene lattice in the x-direciton.
    y_repeat: int, required
        Number of times to repeat graphene lattice in the y-direciton.
    n_layers: int, optional, default 1
        Number of times to repeat the complete layer in the normal direction.
    force_field: force_field : flowermd.ForceField
        The force field to be applied to the surface for paramaterizaiton.
        Note that setting `force_field` does not actually apply the forcefield
        to the molecule. The forcefield in this step is mainly used for
        validation purposes.
    reference_values : dict, required
        A dictionary of reference values for the surface. The keys of the
        dictionary are "length", "energy", and "mass". The values of the
        dictionary are unyt quantities.
    r_cut : float, required
        The cutoff radius for the Lennard-Jones interactions.
    periodicity : tuple of bools, length=3, optional, default=(True, True, False) # noqa: E501
        Whether the Compound is periodic in the x, y, and z directions.
        If None is provided, the periodicity is set to `(False, False, False)`
        which is non-periodic in all directions.
        auto_scale : bool, default=False
        Set to true to use reduced simulation units.
        distance, mass, and energy are scaled by the largest value
        present in the system for each.
    scale_charges : bool, default False
        Set to true to scale charges to net zero.
    remove_charges : bool, default False
        Set to true to remove charges from the system.
    remove_hydrogens : bool, default False
        Set to true to remove hydrogen atoms from the system.
        The masses and charges of the hydrogens are absorbed into
        the heavy atoms they were bonded to.
    pppm_resolution : tuple, default=(8, 8, 8)
        The resolution used in
        `hoomd.md.long_range.pppm.make_pppm_coulomb_force` representing
        number of grid points in the x, y, and z directions.
    ppmp_order : int, default=4
        The order used in
        `hoomd.md.long_range.pppm.make_pppm_coulomb_force` representing
        number of grid points in each direction to assign charges to.
    nlist_buffer : float, default=0.4
        Neighborlist buffer for simulation cell.

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
        force_field,
        reference_values,
        r_cut,
        periodicity=(True, True, False),
        auto_scale=False,
        scale_charges=False,
        remove_charges=False,
        remove_hydrogens=False,
        pppm_resolution=(8, 8, 8),
        pppm_order=4,
        nlist_buffer=0.4,
    ):
        surface = mb.Compound(periodicity=periodicity)
        spacings = [0.425, 0.246, 0.35]
        points = [
            [1 / 6, 0, 0],
            [1 / 2, 0, 0],
            [0, 1 / 2, 0],
            [2 / 3, 1 / 2, 0],
        ]
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
        surface_mol = Molecule(num_mols=1, compound=surface)
        super(Graphene, self).__init__(
            molecules=[surface_mol],
            density=1.0,
            base_units=reference_values,
        )

        # apply forcefield to surface
        self.apply_forcefield(
            r_cut=r_cut,
            force_field=force_field,
            auto_scale=auto_scale,
            scale_charges=scale_charges,
            remove_charges=remove_charges,
            remove_hydrogens=remove_hydrogens,
            pppm_resolution=pppm_resolution,
            pppm_order=pppm_order,
            nlist_buffer=nlist_buffer,
        )

    def _build_system(self):
        return self.all_molecules[0]
