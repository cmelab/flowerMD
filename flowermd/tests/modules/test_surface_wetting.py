import gsd.hoomd
import numpy as np
import unyt as u

from flowermd.base import Pack
from flowermd.library.forcefields import OPLS_AA
from flowermd.library.surfaces import Graphene
from flowermd.modules.surface_wetting import (
    DropletSimulation,
    InterfaceBuilder,
    WettingSimulation,
)
from flowermd.tests.base_test import BaseTest


class TestDropletSimulation(BaseTest):
    def test_droplet_sim(self, polyethylene):
        drop_mol = polyethylene(num_mols=10, lengths=5)
        drop_system = Pack(molecules=drop_mol, density=0.1)
        drop_system.apply_forcefield(
            force_field=OPLS_AA(),
            auto_scale=True,
            remove_charges=True,
            remove_hydrogens=True,
            r_cut=2.5,
        )
        drop_sim = DropletSimulation(
            initial_state=drop_system.hoomd_snapshot,
            forcefield=drop_system.hoomd_forcefield,
            reference_values=drop_system.reference_values,
        )
        drop_sim.run_droplet(
            shrink_kT=5.0,
            shrink_steps=200,
            shrink_period=10,
            shrink_density=0.2,
            expand_kT=0.5,
            expand_steps=200,
            expand_period=10,
            hold_kT=1.0,
            hold_steps=100,
            final_density=0.05,
            tau_kt=drop_sim.dt * 100,
        )
        assert np.isclose(
            drop_sim.density.to(u.g / u.cm**3).value, 0.05, atol=1e-2
        )


class TestSurfaceWetting(BaseTest):
    def test_surface_wetting(self, polyethylene):
        pass
        # load the droplet snapshot

        # recreate droplet forcefield

        # load graphene surface

        # recreate graphene forcefield

        # create interface
        # interface = InterfaceBuilder(
        #     surface_snapshot=surface.hoomd_snapshot,
        #     surface_ff=surface.hoomd_forcefield,
        #     drop_snapshot="droplet_restart.gsd",
        #     drop_ff=drop_ff,
        #     drop_ref_values=drop_sim.reference_values,
        #     box_height=15 * drop_sim.reference_values["length"],
        #     gap=0.4 * drop_sim.reference_values["length"],
        # )
        #
        # # create wetting simulation
        # wetting_sim = WettingSimulation(
        #     initial_state=interface.hoomd_snapshot,
        #     forcefield=interface.hoomd_forces,
        #     reference_values=interface.reference_values,
        # )
        # wetting_sim.run_NVT(n_steps=200, kT=1.0, tau_kt=wetting_sim.dt * 100)
        # assert (
        #         interface.hoomd_snapshot.particles.N
        #         == drop_snapshot.particles.N + surface.surface_snapshot.particles.N
        # )
        # surface_types = [
        #     f"surface_{ptype}"
        #     for ptype in surface.surface_snapshot.particles.types
        # ]
        # assert (
        #         interface.hoomd_snapshot.particles.types
        #         == surface_types + drop_snapshot.particles.types
        # )
