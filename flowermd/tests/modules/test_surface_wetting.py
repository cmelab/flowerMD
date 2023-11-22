import gsd.hoomd

from flowermd.base import Pack
from flowermd.library.forcefields import OPLS_AA
from flowermd.library.surfaces import Graphene
from flowermd.modules.surface_wetting import (
    DropletSimulation,
    InterfaceBuilder,
    WettingSimulation,
)
from flowermd.tests.base_test import BaseTest


class TestSurfaceWetting(BaseTest):
    def test_surface_wetting(self, polyethylene):
        # create droplet
        drop_mol = polyethylene(num_mols=50, lengths=5)
        drop_system = Pack(molecules=drop_mol, density=1.0)
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
            expand_kT=0.5,
            hold_kT=1.0,
            tau_kt=drop_sim.dt * 100,
            shrink_steps=2e2,
            expand_steps=2e2,
            hold_steps=1e3,
            shrink_period=10,
            expand_period=10,
            final_density=0.05,
        )
        # create graphene surface
        surface = Graphene(
            x_repeat=2,
            y_repeat=2,
            n_layers=2,
            force_field=OPLS_AA(),
            reference_values=drop_sim.reference_values,
        )
        drop_sim.save_restart_gsd("droplet_restart.gsd")
        drop_sim.pickle_forcefield("droplet_restart_ff.pkl")
        # load drop ff
        import pickle

        with open("droplet_restart_ff.pkl", "rb") as handle:
            drop_ff = pickle.load(handle)
        # load drop snapshot
        with gsd.hoomd.open("droplet_restart.gsd", "r") as traj:
            drop_snapshot = traj[0]
        # create interface
        interface = InterfaceBuilder(
            surface_snapshot=surface.surface_snapshot,
            surface_ff=surface.surface_ff,
            drop_snapshot="droplet_restart.gsd",
            drop_ff=drop_ff,
            drop_ref_values=drop_sim.reference_values,
            box_height=15 * drop_sim.reference_values["length"],
            gap=0.4 * drop_sim.reference_values["length"],
        )

        # create wetting simulation
        wetting_sim = WettingSimulation(
            initial_state=interface.hoomd_snapshot,
            forcefield=interface.hoomd_forces,
            reference_values=interface.reference_values,
        )
        wetting_sim.run_NVT(n_steps=200, kT=1.0, tau_kt=wetting_sim.dt * 100)
        assert (
            interface.hoomd_snapshot.particles.N
            == drop_snapshot.particles.N + surface.surface_snapshot.particles.N
        )
        surface_types = [
            f"surface_{ptype}"
            for ptype in surface.surface_snapshot.particles.types
        ]
        assert (
            interface.hoomd_snapshot.particles.types
            == surface_types + drop_snapshot.particles.types
        )
