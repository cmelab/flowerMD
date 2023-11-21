from flowermd.base import Pack
from flowermd.library.forcefields import OPLS_AA
from flowermd.modules.surface_wetting import DropletSimulation
from flowermd.tests.base_test import BaseTest


class TestDropletSimulation(BaseTest):
    def test_run_droplet(self, polyethylene):
        mol = polyethylene(num_mols=50, lengths=5)
        system = Pack(molecules=mol, density=1.0)
        system.apply_forcefield(
            force_field=OPLS_AA(),
            auto_scale=True,
            remove_charges=True,
            remove_hydrogens=True,
            r_cut=2.5,
        )
        sim = DropletSimulation(
            initial_state=system.hoomd_snapshot,
            forcefield=system.hoomd_forcefield,
            reference_values=system.reference_values,
        )
        sim.run_droplet(
            shrink_kT=5.0,
            expand_kT=0.5,
            hold_kT=1.5,
            tau_kt=sim.dt * 100,
            shrink_steps=1e4,
            expand_steps=1e4,
            hold_steps=1e4,
            shrink_period=10,
            expand_period=10,
            final_density=0.05,
        )


class TestInterfaceBuilder(BaseTest):
    pass


class TestWettingSimulation(BaseTest):
    pass
